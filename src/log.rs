//! The request log: every finished call, written to Postgres by a task of its own.
//!
//! A call ends by handing its [`Call`] to a channel, which costs it nothing that waits. This task
//! holds what arrives and writes it as one `COPY` every [`FLUSH_EVERY`] or every [`FLUSH_ROWS`]
//! rows, whichever comes first, together with the day's totals in `psyche.usage_daily`, in one
//! transaction. So the database is never on a request's path, and what a crash loses is the batch
//! not yet written, which those two numbers bound.
//!
//! When the database cannot be reached the rows are held and written once it can, up to
//! [`MAX_HELD`]; past that the oldest are dropped, counted and reported, so a long outage costs a
//! bounded amount of memory rather than the process.
//!
//! The log is partitioned by day (UTC). The task makes the partitions from yesterday to two days
//! ahead whenever the date changes, and drops those past the retention, so an expired day is one
//! `DROP TABLE` rather than a delete of every row.

use std::collections::{BTreeMap, VecDeque};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tokio::sync::mpsc::UnboundedReceiver;
use tokio_postgres::binary_copy::BinaryCopyInWriter;
use tokio_postgres::types::Type;
use tokio_postgres::{Client, NoTls};

use crate::price::Dollars;
use crate::proxy::{Call, Outcome};

/// How often what is held is written.
pub const FLUSH_EVERY: Duration = Duration::from_millis(500);
/// How many held rows make a write happen at once.
pub const FLUSH_ROWS: usize = 1000;
/// The most rows held while the database cannot be reached.
pub const MAX_HELD: usize = 100_000;
/// How long a failed connection waits before the next attempt.
const RECONNECT_AFTER: Duration = Duration::from_secs(5);

const SECONDS_PER_DAY: u64 = 86_400;

#[derive(Clone)]
pub struct Settings {
	pub database_url: String,
	/// Days of rows kept; the totals in `usage_daily` are kept for good.
	pub retention_days: u32,
}

impl std::fmt::Debug for Settings {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("Settings")
			.field("retention_days", &self.retention_days)
			.finish_non_exhaustive()
	}
}

/// A day in UTC, by its first second since the epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Day(u64);

impl Day {
	fn of(at: SystemTime) -> Self {
		let unix = at.duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
		Day(unix - unix % SECONDS_PER_DAY)
	}

	fn plus(self, days: i64) -> Self {
		Day(self.0.saturating_add_signed(days * SECONDS_PER_DAY as i64))
	}

	/// `20261018`, the partition's suffix.
	fn compact(self) -> String {
		crate::sigv4::amz_date(self.0)[..8].to_string()
	}

	/// `2026-10-18`, as Postgres reads a date.
	fn iso(self) -> String {
		let d = self.compact();
		format!("{}-{}-{}", &d[..4], &d[4..6], &d[6..])
	}
}

/// Connects as Psyche, for the migrations at start and for the log.
pub async fn connect(url: &str) -> Result<Client, tokio_postgres::Error> {
	let mut config: tokio_postgres::Config = url.parse()?;
	config.application_name("snout-psyche");
	let (client, connection) = config.connect(NoTls).await?;
	tokio::spawn(async move {
		if let Err(error) = connection.await {
			tracing::debug!(%error, "the request log's connection closed");
		}
	});
	Ok(client)
}

/// Writes the calls that arrive on `calls` until the channel closes, then writes what is left and
/// returns.
pub async fn run(settings: Settings, mut calls: UnboundedReceiver<Call>) {
	let mut writer = Writer::new(settings);
	let mut tick = tokio::time::interval(FLUSH_EVERY);
	tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
	loop {
		tokio::select! {
			call = calls.recv() => match call {
				Some(call) => {
					writer.hold(call);
					if writer.held.len() >= FLUSH_ROWS {
						writer.flush().await;
					}
				}
				None => {
					writer.flush().await;
					if !writer.held.is_empty() {
						tracing::error!(rows = writer.held.len(), "the request log could not write its last rows before stopping");
					}
					return;
				}
			},
			_ = tick.tick() => writer.flush().await,
		}
	}
}

struct Writer {
	settings: Settings,
	client: Option<Client>,
	/// When the last connection attempt failed.
	failed_at: Option<tokio::time::Instant>,
	held: VecDeque<Call>,
	/// Rows dropped because too many were held, not yet reported.
	dropped: u64,
	/// The day whose partitions were last made.
	maintained: Option<Day>,
	/// Whether the last write failed, so a run of failures is reported once.
	failing: bool,
}

/// A batch's totals for one day, operation, model and provider.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct Totals {
	requests: i64,
	failed: i64,
	cancelled: i64,
	uncounted: i64,
	prompt_tokens: i64,
	completion_tokens: i64,
	cached_tokens: i64,
	cache_write_tokens: i64,
	reasoning_tokens: i64,
	/// Picodollars.
	cost: u64,
	/// Calls counted whose model has no price, so they are missing from `cost`.
	unpriced: i64,
	/// Picodollars of `cost` paid by credits.
	credits_cost: u64,
}

type Id = Option<crate::keys::KeyId>;
type Key = (Day, &'static str, String, String, Id, Id, Id);

fn totals(batch: &[Call]) -> BTreeMap<Key, Totals> {
	let mut out: BTreeMap<Key, Totals> = BTreeMap::new();
	for call in batch {
		let key = (
			Day::of(call.at),
			call.operation.name(),
			call.model.clone(),
			call.provider.clone(),
			call.who.key,
			call.who.user,
			call.who.group,
		);
		let t = out.entry(key).or_default();
		t.requests += 1;
		match call.outcome {
			Outcome::Completed => {}
			Outcome::Cancelled => t.cancelled += 1,
			Outcome::Failed(_) => t.failed += 1,
		}
		match call.usage {
			None => t.uncounted += 1,
			Some(u) => {
				let n = |v: u64| i64::try_from(v).unwrap_or(i64::MAX);
				t.prompt_tokens += n(u.prompt_tokens);
				t.completion_tokens += n(u.completion_tokens);
				t.cached_tokens += n(u.cached_tokens.unwrap_or(0));
				t.cache_write_tokens += n(u.cache_write_tokens.unwrap_or(0));
				t.reasoning_tokens += n(u.reasoning_tokens.unwrap_or(0));
			}
		}
		match (call.usage.is_some(), call.cost()) {
			(_, Some(cost)) => {
				t.cost = t.cost.saturating_add(cost.0);
				if call.credits {
					t.credits_cost = t.credits_cost.saturating_add(cost.0);
				}
			}
			(true, None) => t.unpriced += 1,
			(false, None) => {}
		}
	}
	out
}

fn outcome(outcome: Outcome) -> &'static str {
	match outcome {
		Outcome::Completed => "completed",
		Outcome::Cancelled => "cancelled",
		Outcome::Failed(class) => class.code(),
	}
}

fn int(n: u64) -> i32 {
	i32::try_from(n).unwrap_or(i32::MAX)
}

fn millis(d: Duration) -> i32 {
	i32::try_from(d.as_millis()).unwrap_or(i32::MAX)
}

/// The batch's rows, by COPY, and its totals, in one transaction.
async fn write(client: &mut Client, batch: &[Call]) -> Result<(), tokio_postgres::Error> {
	let tx = client.transaction().await?;
	let sink = tx
		.copy_in(
			"COPY psyche.requests (at, operation, model, provider, streamed, outcome, \
			 prompt_tokens, completion_tokens, cached_tokens, cache_write_tokens, reasoning_tokens, \
			 first_byte_ms, total_ms, cost, key_id, user_id, group_id, credits, budget_warning) \
			 FROM STDIN BINARY",
		)
		.await?;
	let writer = BinaryCopyInWriter::new(
		sink,
		&[
			Type::TIMESTAMPTZ,
			Type::TEXT,
			Type::TEXT,
			Type::TEXT,
			Type::BOOL,
			Type::TEXT,
			Type::INT4,
			Type::INT4,
			Type::INT4,
			Type::INT4,
			Type::INT4,
			Type::INT4,
			Type::INT4,
			Type::NUMERIC,
			Type::UUID,
			Type::UUID,
			Type::UUID,
			Type::BOOL,
			Type::TEXT,
		],
	);
	let mut writer = std::pin::pin!(writer);
	for call in batch {
		let usage = call.usage;
		writer
			.as_mut()
			.write(&[
				&call.at,
				&call.operation.name(),
				&call.model,
				&call.provider,
				&call.streamed,
				&outcome(call.outcome),
				&usage.map(|u| int(u.prompt_tokens)),
				&usage.map(|u| int(u.completion_tokens)),
				&usage.and_then(|u| u.cached_tokens).map(int),
				&usage.and_then(|u| u.cache_write_tokens).map(int),
				&usage.and_then(|u| u.reasoning_tokens).map(int),
				&millis(call.first_byte),
				&millis(call.total),
				&call.cost(),
				&call.who.key,
				&call.who.user,
				&call.who.group,
				&call.credits,
				&call.budget_warning,
			])
			.await?;
	}
	writer.finish().await?;

	let totals = totals(batch);
	let mut columns: (Vec<String>, Vec<&str>, Vec<&str>, Vec<&str>) = Default::default();
	let mut counts: [Vec<i64>; 10] = Default::default();
	let mut costs: Vec<Dollars> = Vec::new();
	let mut credit_costs: Vec<Dollars> = Vec::new();
	let mut ids: [Vec<Id>; 3] = Default::default();
	for ((day, operation, model, provider, key, user, group), t) in &totals {
		ids[0].push(*key);
		ids[1].push(*user);
		ids[2].push(*group);
		columns.0.push(day.iso());
		columns.1.push(operation);
		columns.2.push(model);
		columns.3.push(provider);
		for (list, n) in counts.iter_mut().zip([
			t.requests,
			t.failed,
			t.cancelled,
			t.uncounted,
			t.prompt_tokens,
			t.completion_tokens,
			t.cached_tokens,
			t.cache_write_tokens,
			t.reasoning_tokens,
			t.unpriced,
		]) {
			list.push(n);
		}
		costs.push(Dollars(t.cost));
		credit_costs.push(Dollars(t.credits_cost));
	}
	let [r, f, c, un, pt, ct, cr, cw, rt, up] = &counts;
	tx.execute(
		"INSERT INTO psyche.usage_daily AS u (day, operation, model, provider, requests, failed, \
		 cancelled, uncounted, prompt_tokens, completion_tokens, cached_tokens, cache_write_tokens, \
		 reasoning_tokens, unpriced, cost, key_id, user_id, group_id, credits_cost) \
		 SELECT d::date, o, m, p, r, f, c, un, pt, ct, cr, cw, rt, up, co, k, us, g, cc \
		 FROM unnest($1::text[], $2::text[], \
		 $3::text[], $4::text[], $5::int8[], $6::int8[], $7::int8[], $8::int8[], $9::int8[], \
		 $10::int8[], $11::int8[], $12::int8[], $13::int8[], $14::int8[], $15::numeric[], $16::uuid[], \
		 $17::uuid[], $18::uuid[], $19::numeric[]) \
		 AS t(d, o, m, p, r, f, c, un, pt, ct, cr, cw, rt, up, co, k, us, g, cc) \
		 ON CONFLICT (day, operation, model, provider, \
		 (coalesce(key_id, '00000000-0000-0000-0000-000000000000'::uuid)), \
		 (coalesce(user_id, '00000000-0000-0000-0000-000000000000'::uuid)), \
		 (coalesce(group_id, '00000000-0000-0000-0000-000000000000'::uuid))) DO UPDATE SET \
		 requests = u.requests + excluded.requests, failed = u.failed + excluded.failed, \
		 cancelled = u.cancelled + excluded.cancelled, uncounted = u.uncounted + excluded.uncounted, \
		 prompt_tokens = u.prompt_tokens + excluded.prompt_tokens, \
		 completion_tokens = u.completion_tokens + excluded.completion_tokens, \
		 cached_tokens = u.cached_tokens + excluded.cached_tokens, \
		 cache_write_tokens = u.cache_write_tokens + excluded.cache_write_tokens, \
		 reasoning_tokens = u.reasoning_tokens + excluded.reasoning_tokens, \
		 unpriced = u.unpriced + excluded.unpriced, cost = u.cost + excluded.cost, \
		 credits_cost = u.credits_cost + excluded.credits_cost",
		&[
			&columns.0,
			&columns.1,
			&columns.2,
			&columns.3,
			r,
			f,
			c,
			un,
			pt,
			ct,
			cr,
			cw,
			rt,
			up,
			&costs,
			&ids[0],
			&ids[1],
			&ids[2],
			&credit_costs,
		],
	)
	.await?;
	tx.commit().await
}

/// Makes each day's partition, if it is not there. Two Psyches making the same one at once is
/// harmless: the one that loses is told it exists.
async fn partitions(client: &Client, days: &[Day]) -> Result<(), tokio_postgres::Error> {
	for day in days {
		let made = client
			.batch_execute(&format!(
				"CREATE TABLE IF NOT EXISTS psyche.requests_{} PARTITION OF psyche.requests \
				 FOR VALUES FROM ('{} 00:00:00+00') TO ('{} 00:00:00+00')",
				day.compact(),
				day.iso(),
				day.plus(1).iso()
			))
			.await;
		if let Err(error) = made {
			let raced = error.code().is_some_and(|code| {
				*code == tokio_postgres::error::SqlState::DUPLICATE_TABLE
					|| *code == tokio_postgres::error::SqlState::UNIQUE_VIOLATION
			});
			if !raced {
				return Err(error);
			}
		}
	}
	Ok(())
}

/// Drops the partitions whose day is older than `keep` days before `today`. Returns their names.
async fn expire(
	client: &Client,
	today: Day,
	keep: u32,
) -> Result<Vec<String>, tokio_postgres::Error> {
	let oldest = format!("requests_{}", today.plus(-i64::from(keep)).compact());
	let rows = client
		.query(
			"SELECT c.relname FROM pg_inherits i JOIN pg_class c ON c.oid = i.inhrelid \
			 WHERE i.inhparent = 'psyche.requests'::regclass ORDER BY 1",
			&[],
		)
		.await?;
	let mut dropped = Vec::new();
	for row in rows {
		let name: String = row.get(0);
		let dated = name.len() == "requests_".len() + 8
			&& name.starts_with("requests_")
			&& name["requests_".len()..]
				.bytes()
				.all(|b| b.is_ascii_digit());
		if dated && name < oldest {
			client
				.batch_execute(&format!("DROP TABLE IF EXISTS psyche.{name}"))
				.await?;
			dropped.push(name);
		}
	}
	Ok(dropped)
}

fn no_partition(error: &tokio_postgres::Error) -> bool {
	error
		.as_db_error()
		.is_some_and(|e| e.message().starts_with("no partition of relation"))
}

impl Writer {
	fn new(settings: Settings) -> Self {
		Self {
			settings,
			client: None,
			failed_at: None,
			held: VecDeque::new(),
			dropped: 0,
			maintained: None,
			failing: false,
		}
	}

	fn hold(&mut self, call: Call) {
		self.held.push_back(call);
		if self.held.len() > MAX_HELD {
			self.held.pop_front();
			self.dropped += 1;
		}
	}

	/// The connection, made again if it closed, but not more often than every [`RECONNECT_AFTER`].
	async fn client(&mut self) -> Option<&mut Client> {
		if self.client.as_ref().is_some_and(Client::is_closed) {
			self.client = None;
			self.maintained = None;
		}
		if self.client.is_none() {
			if self
				.failed_at
				.is_some_and(|at| at.elapsed() < RECONNECT_AFTER)
			{
				return None;
			}
			match connect(&self.settings.database_url).await {
				Ok(client) => {
					self.client = Some(client);
					self.failed_at = None;
				}
				Err(error) => {
					self.failed_at = Some(tokio::time::Instant::now());
					self.report(&format!("cannot connect: {error}"));
					return None;
				}
			}
		}
		self.client.as_mut()
	}

	fn report(&mut self, why: &str) {
		if !self.failing {
			tracing::warn!(
				held = self.held.len(),
				"the request log cannot write: {why}; rows are held until it can"
			);
		}
		self.failing = true;
	}

	/// Makes the partitions around today and drops the expired ones, once per day.
	async fn maintain(&mut self) {
		let today = Day::of(SystemTime::now());
		if self.maintained == Some(today) {
			return;
		}
		let keep = self.settings.retention_days;
		let Some(client) = self.client().await else {
			return;
		};
		let days: Vec<Day> = (-1..=2).map(|n| today.plus(n)).collect();
		let result = match partitions(client, &days).await {
			Ok(()) => expire(client, today, keep).await,
			Err(error) => Err(error),
		};
		match result {
			Ok(dropped) => {
				if !dropped.is_empty() {
					tracing::info!(
						?dropped,
						"the request log dropped the days past its retention"
					);
				}
				self.maintained = Some(today);
			}
			Err(error) => {
				tracing::warn!(%error, "the request log could not make or expire its partitions")
			}
		}
	}

	async fn flush(&mut self) {
		self.maintain().await;
		if self.dropped > 0 {
			tracing::error!(
				rows = self.dropped,
				"the request log dropped its oldest rows: it held {MAX_HELD} it could not write"
			);
			self.dropped = 0;
		}
		while !self.held.is_empty() {
			let n = self.held.len().min(10 * FLUSH_ROWS);
			let batch: Vec<Call> = self.held.drain(..n).collect();
			let Some(client) = self.client().await else {
				self.put_back(batch);
				return;
			};
			let mut result = write(client, &batch).await;
			if result.as_ref().is_err_and(no_partition) {
				// A day the batch holds has no partition (rows held across midnight while the database
				// was away, or a clock that jumped): made, and the batch written again.
				let mut days: Vec<Day> = batch.iter().map(|c| Day::of(c.at)).collect();
				days.sort();
				days.dedup();
				result = match partitions(client, &days).await {
					Ok(()) => write(client, &batch).await,
					Err(error) => Err(error),
				};
			}
			match result {
				Ok(()) => {
					if self.failing {
						tracing::info!("the request log is writing again");
					}
					self.failing = false;
				}
				Err(error) => {
					let why = error
						.as_db_error()
						.map_or_else(|| error.to_string(), |e| e.message().to_string());
					self.put_back(batch);
					self.report(&why);
					return;
				}
			}
		}
	}

	fn put_back(&mut self, batch: Vec<Call>) {
		for call in batch.into_iter().rev() {
			self.held.push_front(call);
		}
		while self.held.len() > MAX_HELD {
			self.held.pop_front();
			self.dropped += 1;
		}
	}
}

#[cfg(test)]
pub(crate) mod tests {
	use super::*;
	use crate::error::Class;
	use crate::proxy::Operation;
	use crate::usage::Usage;

	pub(crate) fn call(
		at: SystemTime,
		model: &str,
		outcome: Outcome,
		usage: Option<Usage>,
	) -> Call {
		Call {
			at,
			operation: Operation::Chat,
			model: model.into(),
			provider: "bedrock".into(),
			streamed: true,
			outcome,
			usage,
			price: None,
			who: Default::default(),
			first_byte: Duration::from_millis(120),
			total: Duration::from_millis(900),
			credits: false,
			budget_warning: None,
		}
	}

	#[test]
	fn days_are_utc_and_named_for_their_partition() {
		let at = UNIX_EPOCH + Duration::from_secs(1_792_324_800 + 3_600 * 11 + 59);
		let day = Day::of(at);
		assert_eq!(day.compact(), "20261018");
		assert_eq!(day.iso(), "2026-10-18");
		assert_eq!(day.plus(1).iso(), "2026-10-19");
		assert_eq!(day.plus(-18).iso(), "2026-09-30");
		assert_eq!(
			Day::of(UNIX_EPOCH + Duration::from_secs(1_792_367_999)),
			day
		);
	}

	#[test]
	fn a_batch_is_totalled_per_day_model_and_provider() {
		let today = UNIX_EPOCH + Duration::from_secs(1_792_324_800);
		let tomorrow = today + Duration::from_secs(SECONDS_PER_DAY);
		let used = Usage {
			prompt_tokens: 100,
			completion_tokens: 20,
			cached_tokens: Some(80),
			cache_write_tokens: None,
			reasoning_tokens: Some(5),
		};
		let batch = vec![
			call(today, "a", Outcome::Completed, Some(used)),
			call(today, "a", Outcome::Failed(Class::RateLimited), None),
			call(today, "a", Outcome::Cancelled, Some(used)),
			call(tomorrow, "a", Outcome::Completed, Some(used)),
			call(today, "b", Outcome::Completed, Some(used)),
		];
		let totals = totals(&batch);
		assert_eq!(totals.len(), 3);
		let a = totals[&(
			Day::of(today),
			"chat",
			"a".into(),
			"bedrock".into(),
			None,
			None,
			None,
		)];
		assert_eq!(
			a,
			Totals {
				requests: 3,
				failed: 1,
				cancelled: 1,
				uncounted: 1,
				prompt_tokens: 200,
				completion_tokens: 40,
				cached_tokens: 160,
				cache_write_tokens: 0,
				reasoning_tokens: 10,
				cost: 0,
				unpriced: 2,
				credits_cost: 0,
			}
		);
	}

	#[test]
	fn held_rows_are_bounded_and_the_oldest_go_first() {
		let mut writer = Writer::new(Settings {
			database_url: "postgres://nowhere".into(),
			retention_days: 30,
		});
		for i in 0..MAX_HELD + 3 {
			let at = UNIX_EPOCH + Duration::from_secs(i as u64);
			writer.hold(call(at, "a", Outcome::Completed, None));
		}
		assert_eq!(writer.held.len(), MAX_HELD);
		assert_eq!(writer.dropped, 3);
		assert_eq!(
			writer.held.front().unwrap().at,
			UNIX_EPOCH + Duration::from_secs(3)
		);
	}

	#[test]
	fn the_settings_never_print_the_url() {
		let settings = Settings {
			database_url: "postgres://psyche:hunter2@db/postgres".into(),
			retention_days: 30,
		};
		assert!(!format!("{settings:?}").contains("hunter2"));
	}

	/// A fresh, migrated database of its own for one test, or `None` when `PSYCHE_TEST_DATABASE_URL`
	/// is not set (tests/db.sh sets it). Returns the client and the database's URL.
	pub(crate) async fn fresh_database(name: &str) -> Option<(Client, String)> {
		let admin_url = std::env::var("PSYCHE_TEST_DATABASE_URL").ok()?;
		let admin = connect(&admin_url).await.unwrap();
		let database = format!("psyche_{name}_{}", std::process::id());
		for statement in [
			format!("DROP DATABASE IF EXISTS {database} WITH (FORCE)"),
			format!("CREATE DATABASE {database}"),
		] {
			admin.batch_execute(&statement).await.unwrap();
		}
		// The signed-in role a SnoutData project has, so the owners' grants are made. Roles belong
		// to the cluster, and tests make databases at once.
		admin
			.batch_execute(
				"DO $$ BEGIN CREATE ROLE authenticated NOLOGIN; \
				 EXCEPTION WHEN duplicate_object OR unique_violation THEN NULL; END $$",
			)
			.await
			.unwrap();
		let url = format!("{}/{database}", admin_url.rsplit_once('/').unwrap().0);
		let mut client = connect(&url).await.unwrap();
		let migrations = crate::migrations::MIGRATIONS;
		assert_eq!(
			crate::migrations::migrate(&mut client, migrations)
				.await
				.unwrap(),
			migrations.iter().map(|m| m.id).collect::<Vec<_>>()
		);
		assert!(
			crate::migrations::migrate(&mut client, migrations)
				.await
				.unwrap()
				.is_empty(),
			"a second run runs nothing"
		);
		Some((client, url))
	}

	async fn count(client: &Client, sql: &str) -> i64 {
		client.query_one(sql, &[]).await.unwrap().get(0)
	}

	#[tokio::test]
	async fn a_changed_migration_is_refused() {
		let Some((mut client, _)) = fresh_database("ledger").await else {
			return;
		};
		client
			.batch_execute("UPDATE psyche.migrations SET hash = repeat('0', 64) WHERE id = 1")
			.await
			.unwrap();
		let error = crate::migrations::migrate(&mut client, crate::migrations::MIGRATIONS)
			.await
			.unwrap_err();
		assert!(
			error.to_string().contains("changed after it ran"),
			"{error}"
		);
	}

	#[tokio::test]
	async fn calls_are_written_in_batches_with_their_totals() {
		let Some((client, url)) = fresh_database("write").await else {
			return;
		};
		let now = SystemTime::now();
		let tomorrow = now + Duration::from_secs(SECONDS_PER_DAY);
		let used = Usage {
			prompt_tokens: 10,
			completion_tokens: 3,
			cached_tokens: Some(4),
			cache_write_tokens: None,
			reasoning_tokens: None,
		};
		let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
		let task = tokio::spawn(run(
			Settings {
				database_url: url,
				retention_days: 30,
			},
			rx,
		));
		let price = serde_json::from_str::<crate::price::PriceEntry>(
			r#"{"input": 2.2, "output": 11, "cache_read": 0.11}"#,
		)
		.unwrap()
		.price()
		.unwrap();
		for _ in 0..2500 {
			let mut priced = call(now, "sonnet", Outcome::Completed, Some(used));
			priced.price = Some(price);
			tx.send(priced).unwrap();
		}
		tx.send(call(
			now,
			"sonnet",
			Outcome::Failed(Class::RateLimited),
			None,
		))
		.unwrap();
		tx.send(call(tomorrow, "luna", Outcome::Cancelled, Some(used)))
			.unwrap();
		// Written as they arrive, a thousand at a time, without waiting for the channel to close.
		let mut written = 0;
		for _ in 0..50 {
			written = count(&client, "SELECT count(*) FROM psyche.requests").await;
			if written >= 2000 {
				break;
			}
			tokio::time::sleep(Duration::from_millis(100)).await;
		}
		assert!(written >= 2000, "{written}");
		drop(tx);
		task.await.unwrap();

		assert_eq!(
			count(&client, "SELECT count(*) FROM psyche.requests").await,
			2502
		);
		let today = Day::of(now);
		let partitions = count(
			&client,
			"SELECT count(*) FROM pg_inherits WHERE inhparent = 'psyche.requests'::regclass",
		)
		.await;
		assert_eq!(partitions, 4, "yesterday to two days ahead");
		let row = client
			.query_one(
				"SELECT requests, failed, cancelled, uncounted, prompt_tokens, completion_tokens, cached_tokens \
				 FROM psyche.usage_daily WHERE day = $1::text::date AND model = 'sonnet'",
				&[&today.iso()],
			)
			.await
			.unwrap();
		let got: Vec<i64> = (0..7).map(|i| row.get(i)).collect();
		assert_eq!(got, vec![2501, 1, 0, 1, 25_000, 7_500, 10_000]);
		// Each call: 6 plain tokens at $2.20, 4 cached at $0.11 and 3 out at $11 per million, so
		// $0.00004664, exactly, 2,500 times; the failed call reported nothing and costs nothing.
		let cost = |sql: &'static str| {
			let client = &client;
			async move {
				client
					.query_one(sql, &[])
					.await
					.unwrap()
					.get::<_, String>(0)
			}
		};
		assert_eq!(
			cost("SELECT sum(cost)::text FROM psyche.requests WHERE model = 'sonnet'").await,
			"0.116600000000"
		);
		assert_eq!(
			cost("SELECT min(cost)::text FROM psyche.requests WHERE model = 'sonnet'").await,
			"0.000046640000"
		);
		assert_eq!(
			cost(
				"SELECT cost::text || ' ' || unpriced FROM psyche.usage_daily WHERE model = 'sonnet'"
			)
			.await,
			"0.116600000000 0"
		);
		assert_eq!(
			cost("SELECT coalesce(max(cost)::text, 'null') || ' ' || (SELECT unpriced FROM psyche.usage_daily WHERE model = 'luna') FROM psyche.requests WHERE model = 'luna'").await,
			"null 1",
			"a model with no price: the row has no cost and the day counts it"
		);
		let failed = client
			.query_one(
				"SELECT outcome, prompt_tokens, first_byte_ms, total_ms FROM psyche.requests WHERE outcome <> 'completed' AND model = 'sonnet'",
				&[],
			)
			.await
			.unwrap();
		assert_eq!(failed.get::<_, String>(0), "rate_limited");
		assert_eq!(failed.get::<_, Option<i32>>(1), None);
		assert_eq!(
			(failed.get::<_, i32>(2), failed.get::<_, i32>(3)),
			(120, 900)
		);
		assert_eq!(
			count(
				&client,
				"SELECT cancelled FROM psyche.usage_daily WHERE model = 'luna'"
			)
			.await,
			1
		);
	}

	#[tokio::test]
	async fn rows_wait_out_an_outage_and_old_days_are_dropped() {
		let Some((client, url)) = fresh_database("outage").await else {
			return;
		};
		let mut writer = Writer::new(Settings {
			database_url: format!("{url}_not_there"),
			retention_days: 7,
		});
		let now = SystemTime::now();
		for _ in 0..5 {
			writer.hold(call(now, "a", Outcome::Completed, None));
		}
		writer.flush().await;
		assert_eq!(writer.held.len(), 5, "held while the database is away");
		assert!(writer.failing);

		// An old day, as if left from before the retention was shortened.
		let old = Day::of(now).plus(-8);
		partitions(&client, &[old, Day::of(now).plus(-7)])
			.await
			.unwrap();
		writer.settings.database_url = url;
		writer.failed_at = None;
		writer.flush().await;
		assert!(writer.held.is_empty());
		assert!(!writer.failing);
		assert_eq!(
			count(&client, "SELECT count(*) FROM psyche.requests").await,
			5
		);
		let names: Vec<String> = client
			.query(
				"SELECT c.relname::text FROM pg_inherits i JOIN pg_class c ON c.oid = i.inhrelid \
				 WHERE i.inhparent = 'psyche.requests'::regclass ORDER BY 1",
				&[],
			)
			.await
			.unwrap()
			.iter()
			.map(|r| r.get(0))
			.collect();
		assert!(
			!names.contains(&format!("requests_{}", old.compact())),
			"{names:?}"
		);
		assert!(names.contains(&format!("requests_{}", Day::of(now).plus(-7).compact())));
	}

	#[tokio::test]
	async fn nobody_but_psyche_reads_the_log_or_the_keys() {
		let Some((client, _)) = fresh_database("grants").await else {
			return;
		};
		client
			.batch_execute(
				"DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'psyche_test_reader') \
				 THEN CREATE ROLE psyche_test_reader; END IF; END $$",
			)
			.await
			.unwrap();
		for table in [
			"psyche.requests",
			"psyche.usage_daily",
			"psyche.keys",
			"psyche.budgets",
			"psyche.limits",
			"psyche.groups",
			"psyche.members",
			"psyche.credentials",
		] {
			let error = client
				.batch_execute(&format!(
					"BEGIN; SET LOCAL ROLE psyche_test_reader; SELECT * FROM {table}; COMMIT"
				))
				.await
				.unwrap_err();
			let _ = client.batch_execute("ROLLBACK").await;
			assert!(
				error
					.as_db_error()
					.unwrap()
					.message()
					.contains("permission denied"),
				"{table}: {error}"
			);
		}
	}

	#[tokio::test]
	async fn a_key_lists_when_it_was_last_used() {
		let Some((mut client, _)) = fresh_database("last_used").await else {
			return;
		};
		client
			.batch_execute(
				"INSERT INTO psyche.keys (name, hash, hint) VALUES
				   ('used', decode(repeat('a1', 32), 'hex'), 'psy_a'),
				   ('idle', decode(repeat('b0', 32), 'hex'), 'psy_b')",
			)
			.await
			.unwrap();
		let used = crate::keys::list(&client)
			.await
			.unwrap()
			.into_iter()
			.find(|k| k["name"] == "used")
			.unwrap()["id"]
			.as_str()
			.unwrap()
			.to_string();
		let now = SystemTime::now();
		let earlier = now - Duration::from_secs(60);
		partitions(&client, &[Day::of(now), Day::of(earlier)])
			.await
			.unwrap();
		let by_key = |at: SystemTime, outcome: Outcome| {
			let mut call = call(at, "a", outcome, Some(Usage::default()));
			call.who.key = crate::keys::KeyId::parse(&used);
			call
		};
		write(
			&mut client,
			&[
				by_key(earlier, Outcome::Completed),
				by_key(now, Outcome::Cancelled),
			],
		)
		.await
		.unwrap();

		let keys = crate::keys::list(&client).await.unwrap();
		let find = |name: &str| keys.iter().find(|k| k["name"] == name).unwrap().clone();
		let (used, idle) = (find("used"), find("idle"));
		// The newest call, to the second, and its day from the totals.
		let second = now.duration_since(UNIX_EPOCH).unwrap().as_secs();
		assert_eq!(used["last_used_at"].as_u64(), Some(second));
		assert_eq!(
			used["last_used_on"].as_str(),
			Some(Day::of(now).iso().as_str())
		);
		assert!(idle["last_used_at"].is_null() && idle["last_used_on"].is_null());

		// Past the log's retention the call's row is gone, and the day is still known.
		client
			.batch_execute("DELETE FROM psyche.requests")
			.await
			.unwrap();
		let keys = crate::keys::list(&client).await.unwrap();
		let used = keys.iter().find(|k| k["name"] == "used").unwrap();
		assert!(used["last_used_at"].is_null());
		assert_eq!(
			used["last_used_on"].as_str(),
			Some(Day::of(now).iso().as_str())
		);
	}

	#[tokio::test]
	async fn owners_see_their_own_keys_calls_totals_and_budgets_and_nothing_else() {
		let Some((mut client, _)) = fresh_database("owners").await else {
			return;
		};
		let alice = "00000000-0000-4000-8000-00000000a11c";
		let bob = "00000000-0000-4000-8000-000000000b0b";
		let id = |text: &str| crate::keys::KeyId::parse(text).unwrap();
		let team: String = client
			.query_one(
				"INSERT INTO psyche.groups (name) VALUES ('team') RETURNING id::text",
				&[],
			)
			.await
			.unwrap()
			.get(0);
		client
			.batch_execute(&format!(
				"INSERT INTO psyche.members (user_id, group_id) VALUES ('{alice}', '{team}');
				 INSERT INTO psyche.keys (name, hash, hint, owner) VALUES
				   ('alice', decode(repeat('a1', 32), 'hex'), 'psy_a', '{alice}'),
				   ('bob', decode(repeat('b0', 32), 'hex'), 'psy_b', '{bob}'),
				   ('app', decode(repeat('00', 32), 'hex'), 'psy_0', null);
				 INSERT INTO psyche.budgets (name, amount, period, every_user) VALUES ('each', 1, 'day', true);
				 INSERT INTO psyche.budgets (name, amount, period, user_id) VALUES ('alice', 5, 'day', '{alice}');
				 INSERT INTO psyche.budgets (name, amount, period, user_id) VALUES ('bob', 5, 'day', '{bob}');
				 INSERT INTO psyche.budgets (name, amount, period, group_id) VALUES ('team', 9, 'day', '{team}');
				 INSERT INTO psyche.budgets (name, amount, period) VALUES ('project', 99, 'month');"
			))
			.await
			.unwrap();
		let now = SystemTime::now();
		partitions(&client, &[Day::of(now)]).await.unwrap();
		let by = |user: &str, group: Option<&str>| {
			let mut call = call(now, "a", Outcome::Completed, Some(Usage::default()));
			call.who.user = Some(id(user));
			call.who.group = group.map(id);
			call
		};
		let batch = [
			by(alice, Some(&team)),
			by(alice, Some(&team)),
			by(bob, None),
		];
		write(&mut client, &batch).await.unwrap();

		let seen = |claims: Option<String>| {
			let client = &client;
			async move {
				let set = claims.map_or(String::new(), |c| {
					format!("SET LOCAL request.jwt.claims = '{c}';")
				});
				client
					.batch_execute(&format!("BEGIN; SET LOCAL ROLE authenticated; {set}"))
					.await
					.unwrap();
				let mut counts = Vec::new();
				for sql in [
					"SELECT count(*) FROM psyche.keys",
					"SELECT count(*) FROM psyche.requests",
					"SELECT coalesce(sum(requests), 0)::bigint FROM psyche.usage_daily",
					"SELECT count(*) FROM psyche.budgets",
					"SELECT count(*) FROM psyche.groups",
					"SELECT count(*) FROM psyche.members",
				] {
					counts.push(count(client, sql).await);
				}
				let hash = client.batch_execute("SELECT hash FROM psyche.keys").await;
				client.batch_execute("ROLLBACK").await.unwrap();
				(counts, hash.is_ok())
			}
		};
		let claims = |sub: &str| Some(format!(r#"{{"sub": "{sub}", "role": "authenticated"}}"#));
		// Alice: her key, her two calls, their total, every user's budget, hers and her group's,
		// her group and her membership. Never a hash.
		let (alice_sees, hash) = seen(claims(alice)).await;
		assert_eq!(alice_sees, vec![1, 2, 2, 3, 1, 1]);
		assert!(!hash, "the hash is not readable");
		let (bob_sees, _) = seen(claims(bob)).await;
		assert_eq!(bob_sees, vec![1, 1, 1, 2, 0, 0]);
		// No user, no rows, but every user's budget, which says nothing about anyone.
		let (nobody_sees, _) = seen(None).await;
		assert_eq!(nobody_sees, vec![0, 0, 0, 1, 0, 0]);
		// The daily totals keep who and which group.
		let rows = count(
			&client,
			&format!(
				"SELECT count(*) FROM psyche.usage_daily WHERE user_id = '{alice}' AND group_id = '{team}'"
			),
		)
		.await;
		assert_eq!(rows, 1);
	}
}
