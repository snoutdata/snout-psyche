//! Health (PS16): what each target did lately, measured from its own calls rather than configured.
//!
//! A target is a model of the catalogue (its provider and its name). Every attempt a call makes is
//! seen here when it ends: the time to the provider's first byte, tokens a second for a streamed
//! reply, and whether it failed for a reason that was the target's (the provider down, slow,
//! unreachable, rate-limiting, refusing Psyche's credential, or out of credit). A caller's own
//! mistake says nothing about the target either way.
//!
//! Three failures in a row cool a target down: a route tries it after every target that is not
//! cooling, and a call that names it alone still gets it, so a cooldown never fails a call by
//! itself. When the cooldown ends the next call to reach it is its probe; the others keep treating
//! it as cooling until the probe ends. A probe that fails cools it again for twice as long, up to
//! five minutes; one that answers puts it back. A provider's 429 with a `retry-after` cools the
//! target for exactly that long, without counting as a failure.
//!
//! The statistics live in memory, and with a database they are written to `psyche.provider_stats`
//! every few seconds and read back at start, so a restart does not start blind.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};
use tokio_postgres::Client;

use crate::error::Class;
use crate::providers::{Model, Sort};
use crate::proxy::{Call, Outcome};

/// How far back the counts reach, in one-minute buckets.
const WINDOW_MINUTES: u64 = 5;
/// Failures in a row that cool a target down.
const OPEN_AFTER: u32 = 3;
const FIRST_COOLDOWN: Duration = Duration::from_secs(10);
const LONGEST_COOLDOWN: Duration = Duration::from_secs(300);
/// A probe that never reported (its call was refused before it was sent) is given up after this.
const PROBE_ABANDONED: Duration = Duration::from_secs(330);
/// The weight of the newest reading in a moving average.
const WEIGHT: f64 = 0.2;
/// A streamed reply shorter than this says little about a target's speed.
const FEWEST_TOKENS_TIMED: u64 = 16;
/// How often the statistics are written.
pub const FLUSH_EVERY: Duration = Duration::from_secs(10);

/// The provider's own reading of what is left on Psyche's credential, from its response headers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Remaining {
	pub requests: Option<u64>,
	pub tokens: Option<u64>,
}

impl Remaining {
	/// OpenAI's `x-ratelimit-remaining-*` and Anthropic's `anthropic-ratelimit-*-remaining`.
	pub fn read(headers: &reqwest::header::HeaderMap) -> Self {
		let number = |names: &[&str]| {
			names.iter().find_map(|n| {
				headers
					.get(*n)
					.and_then(|v| v.to_str().ok())
					.and_then(|v| v.trim().parse::<u64>().ok())
			})
		};
		Self {
			requests: number(&[
				"x-ratelimit-remaining-requests",
				"anthropic-ratelimit-requests-remaining",
			]),
			tokens: number(&[
				"x-ratelimit-remaining-tokens",
				"anthropic-ratelimit-tokens-remaining",
			]),
		}
	}
}

/// What one attempt showed about its target.
#[derive(Debug, Clone, Copy)]
pub struct Seen {
	pub outcome: Outcome,
	pub first_byte: Duration,
	pub total: Duration,
	pub streamed: bool,
	pub output_tokens: Option<u64>,
	/// A provider's `retry-after` on a 429.
	pub retry_after: Option<Duration>,
	pub remaining: Remaining,
}

impl Seen {
	pub fn of(call: &Call, retry_after: Option<Duration>, remaining: Remaining) -> Self {
		Self {
			outcome: call.outcome,
			first_byte: call.first_byte,
			total: call.total,
			streamed: call.streamed,
			output_tokens: call.usage.map(|u| u.completion_tokens),
			retry_after,
			remaining,
		}
	}
}

/// Whether a failure of this class was the target's: the next caller would most likely meet it too.
fn targets_fault(class: Class) -> bool {
	matches!(
		class,
		Class::Upstream
			| Class::Timeout
			| Class::Transport
			| Class::RateLimited
			| Class::ProviderAuth
			| Class::ProviderBilling
			| Class::EgressRefused
	)
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Counts {
	requests: u32,
	failed: u32,
	rate_limited: u32,
}

#[derive(Debug, Clone, Default)]
struct State {
	/// One bucket per minute, the newest last, the oldest dropped past the window.
	buckets: VecDeque<(u64, Counts)>,
	first_byte_ms: Option<f64>,
	tokens_per_second: Option<f64>,
	/// Target failures since the last answer.
	failures: u32,
	/// Cooldowns in a row, which the next one's length doubles with.
	trips: u32,
	cooling_until: Option<SystemTime>,
	/// When a probe started, while one is out.
	probe: Option<SystemTime>,
	remaining: Remaining,
	last_failure: Option<&'static str>,
	updated: Option<SystemTime>,
	/// Changed since it was last written.
	dirty: bool,
}

impl State {
	fn cooling(&self, now: SystemTime) -> bool {
		self.cooling_until.is_some_and(|until| until > now) || self.probing(now)
	}

	fn probing(&self, now: SystemTime) -> bool {
		self.probe.is_some_and(|started| {
			now.duration_since(started)
				.map_or(true, |d| d < PROBE_ABANDONED)
		})
	}

	fn counts(&self, now: SystemTime) -> Counts {
		let minute = minute(now);
		self.buckets
			.iter()
			.filter(|(m, _)| minute.saturating_sub(*m) < WINDOW_MINUTES)
			.fold(Counts::default(), |a, (_, c)| Counts {
				requests: a.requests + c.requests,
				failed: a.failed + c.failed,
				rate_limited: a.rate_limited + c.rate_limited,
			})
	}

	fn count(&mut self, now: SystemTime, add: impl FnOnce(&mut Counts)) {
		let minute = minute(now);
		if self.buckets.back().is_none_or(|(m, _)| *m != minute) {
			self.buckets.push_back((minute, Counts::default()));
		}
		while self
			.buckets
			.front()
			.is_some_and(|(m, _)| minute.saturating_sub(*m) >= WINDOW_MINUTES)
		{
			self.buckets.pop_front();
		}
		if let Some((_, counts)) = self.buckets.back_mut() {
			add(counts);
		}
	}

	fn cool(&mut self, now: SystemTime) {
		let length = FIRST_COOLDOWN
			.saturating_mul(1 << self.trips.min(10))
			.min(LONGEST_COOLDOWN);
		self.trips += 1;
		self.cooling_until = Some(now + length);
	}

	fn word(&self, now: SystemTime) -> &'static str {
		if self.probing(now) {
			"probing"
		} else if self.cooling(now) {
			"cooling"
		} else {
			"ok"
		}
	}
}

fn minute(t: SystemTime) -> u64 {
	t.duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs() / 60)
}

fn average(old: Option<f64>, new: f64) -> f64 {
	match old {
		None => new,
		Some(old) => old + WEIGHT * (new - old),
	}
}

/// Every target's health. Cheap to clone: the clones share it.
#[derive(Debug, Clone, Default)]
pub struct Health {
	targets: Arc<Mutex<HashMap<(String, String), State>>>,
}

impl Health {
	fn with<T>(&self, f: impl FnOnce(&mut HashMap<(String, String), State>) -> T) -> T {
		f(&mut self.targets.lock().unwrap_or_else(|p| p.into_inner()))
	}

	/// The targets in the order to try them: those not cooling first, then those cooling, each
	/// group in the order given, except that a route sorted by `latency` or `throughput` puts the
	/// fastest first among those not cooling (a target never measured goes ahead of them, so it gets
	/// measured).
	pub fn order(
		&self,
		targets: Vec<Arc<Model>>,
		sort: Option<Sort>,
		now: SystemTime,
	) -> Vec<Arc<Model>> {
		self.with(|states| {
			let state = |m: &Model| states.get(&(m.provider.name.clone(), m.name.clone()));
			let (mut ready, cooling): (Vec<_>, Vec<_>) = targets
				.into_iter()
				.partition(|m| !state(m).is_some_and(|s| s.cooling(now)));
			let reading = |m: &Model| match sort {
				Some(Sort::Latency) => state(m).and_then(|s| s.first_byte_ms),
				// Faster is more tokens a second, so it is sorted on the negative.
				Some(Sort::Throughput) => state(m).and_then(|s| s.tokens_per_second).map(|t| -t),
				_ => None,
			};
			if matches!(sort, Some(Sort::Latency | Sort::Throughput)) {
				// A stable sort: equal readings, and every unmeasured target, keep the order given.
				ready.sort_by(|a, b| match (reading(a), reading(b)) {
					(None, None) => std::cmp::Ordering::Equal,
					(None, Some(_)) => std::cmp::Ordering::Less,
					(Some(_), None) => std::cmp::Ordering::Greater,
					(Some(a), Some(b)) => a.total_cmp(&b),
				});
			}
			ready.extend(cooling);
			ready
		})
	}

	/// An attempt is about to be sent to `model`: if its cooldown is over, this attempt is its
	/// probe.
	pub fn begin(&self, model: &Model, now: SystemTime) {
		self.with(|states| {
			if let Some(state) = states.get_mut(&(model.provider.name.clone(), model.name.clone()))
				&& state.cooling_until.is_some_and(|until| until <= now)
				&& !state.probing(now)
			{
				state.probe = Some(now);
				state.dirty = true;
			}
		});
	}

	/// An attempt to `provider`'s `model` ended.
	pub fn seen(&self, provider: &str, model: &str, seen: Seen, now: SystemTime) {
		self.with(|states| {
			let state = states
				.entry((provider.to_string(), model.to_string()))
				.or_default();
			let probing = state.probe.take().is_some();
			if seen.remaining != Remaining::default() {
				state.remaining = seen.remaining;
			}
			state.updated = Some(now);
			state.dirty = true;
			match seen.outcome {
				Outcome::Failed(class) if targets_fault(class) => {
					state.count(now, |c| {
						c.requests += 1;
						c.failed += 1;
						if class == Class::RateLimited {
							c.rate_limited += 1;
						}
					});
					state.last_failure = Some(class.code());
					if class == Class::RateLimited
						&& let Some(wait) = seen.retry_after
					{
						// The provider said when; that is the cooldown, and not a failure in a row.
						let until = now + wait;
						state.cooling_until =
							Some(state.cooling_until.map_or(until, |u| u.max(until)));
						return;
					}
					state.failures += 1;
					if probing || state.failures >= OPEN_AFTER {
						state.cool(now);
					}
				}
				Outcome::Failed(_) | Outcome::Completed => {
					// It answered: a caller's mistake is still an answer.
					state.count(now, |c| c.requests += 1);
					state.failures = 0;
					state.trips = 0;
					state.cooling_until = None;
					state.first_byte_ms = Some(average(
						state.first_byte_ms,
						seen.first_byte.as_secs_f64() * 1000.0,
					));
					if seen.outcome == Outcome::Completed
						&& seen.streamed && let Some(tokens) =
						seen.output_tokens.filter(|t| *t >= FEWEST_TOKENS_TIMED)
						&& let Some(writing) = seen
							.total
							.checked_sub(seen.first_byte)
							.filter(|d| !d.is_zero())
					{
						state.tokens_per_second = Some(average(
							state.tokens_per_second,
							tokens as f64 / writing.as_secs_f64(),
						));
					}
				}
				Outcome::Cancelled => {
					// The caller went away; if the reply had started, the target was answering.
					if !seen.first_byte.is_zero() {
						state.failures = 0;
						state.trips = 0;
						state.cooling_until = None;
					}
				}
			}
		});
	}

	/// Every target's statistics as the admin API gives them.
	pub fn report(&self, now: SystemTime) -> Vec<Value> {
		let mut rows = self.with(|states| {
			states
				.iter()
				.map(|((provider, model), s)| (provider.clone(), model.clone(), row(s, now)))
				.collect::<Vec<_>>()
		});
		rows.sort_by(|a, b| (&a.0, &a.1).cmp(&(&b.0, &b.1)));
		rows.into_iter()
			.map(|(provider, model, mut row)| {
				row["provider"] = provider.into();
				row["model"] = model.into();
				row
			})
			.collect()
	}
}

fn row(s: &State, now: SystemTime) -> Value {
	let counts = s.counts(now);
	let unix = |t: SystemTime| t.duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
	json!({
		"state": s.word(now),
		"requests": counts.requests,
		"failed": counts.failed,
		"rate_limited": counts.rate_limited,
		"error_rate": (counts.requests > 0).then(|| f64::from(counts.failed) / f64::from(counts.requests)),
		"first_byte_ms": s.first_byte_ms.map(|v| v.round()),
		"tokens_per_second": s.tokens_per_second.map(|v| (v * 10.0).round() / 10.0),
		"failures_in_a_row": s.failures,
		"cooling_until": s.cooling_until.filter(|u| *u > now).map(unix),
		"last_failure": s.last_failure,
		"remaining_requests": s.remaining.requests,
		"remaining_tokens": s.remaining.tokens,
		"updated_at": s.updated.map(unix),
	})
}

/// Reads what the last run wrote: the averages, and a cooldown still running.
pub async fn load(client: &Client, health: &Health) -> Result<usize, tokio_postgres::Error> {
	let rows = client
		.query(
			"SELECT provider, model, first_byte_ms, tokens_per_second, trips, \
			 (extract(epoch FROM cooling_until) * 1000)::bigint \
			 FROM psyche.provider_stats",
			&[],
		)
		.await?;
	let n = rows.len();
	health.with(|states| {
		for row in rows {
			let state = states.entry((row.get(0), row.get(1))).or_default();
			state.first_byte_ms = row.get::<_, Option<f64>>(2);
			state.tokens_per_second = row.get::<_, Option<f64>>(3);
			state.trips = u32::try_from(row.get::<_, i32>(4)).unwrap_or(0);
			state.cooling_until = row
				.get::<_, Option<i64>>(5)
				.and_then(|ms| u64::try_from(ms).ok())
				.map(|ms| UNIX_EPOCH + Duration::from_millis(ms));
		}
	});
	Ok(n)
}

/// Writes the targets that changed since the last write.
pub async fn flush(
	client: &Client,
	health: &Health,
	now: SystemTime,
) -> Result<usize, tokio_postgres::Error> {
	let changed: Vec<(String, String, State)> = health.with(|states| {
		states
			.iter_mut()
			.filter(|(_, s)| s.dirty)
			.map(|((p, m), s)| {
				s.dirty = false;
				(p.clone(), m.clone(), s.clone())
			})
			.collect()
	});
	for (provider, model, s) in &changed {
		let counts = s.counts(now);
		let ms = |t: Option<SystemTime>| {
			t.and_then(|t| t.duration_since(UNIX_EPOCH).ok())
				.map(|d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
		};
		let n = |v: Option<u64>| v.map(|v| i64::try_from(v).unwrap_or(i64::MAX));
		let written = client
			.execute(
				"INSERT INTO psyche.provider_stats (provider, model, state, requests, failed, rate_limited, \
				 first_byte_ms, tokens_per_second, failures_in_a_row, trips, cooling_until, last_failure, \
				 remaining_requests, remaining_tokens, updated_at) \
				 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, to_timestamp($11::bigint / 1000.0), $12, $13, $14, \
				 to_timestamp($15::bigint / 1000.0)) \
				 ON CONFLICT (provider, model) DO UPDATE SET state = excluded.state, requests = excluded.requests, \
				 failed = excluded.failed, rate_limited = excluded.rate_limited, \
				 first_byte_ms = excluded.first_byte_ms, tokens_per_second = excluded.tokens_per_second, \
				 failures_in_a_row = excluded.failures_in_a_row, trips = excluded.trips, \
				 cooling_until = excluded.cooling_until, last_failure = excluded.last_failure, \
				 remaining_requests = excluded.remaining_requests, remaining_tokens = excluded.remaining_tokens, \
				 updated_at = excluded.updated_at",
				&[
					provider,
					model,
					&s.word(now),
					&i32::try_from(counts.requests).unwrap_or(i32::MAX),
					&i32::try_from(counts.failed).unwrap_or(i32::MAX),
					&i32::try_from(counts.rate_limited).unwrap_or(i32::MAX),
					&s.first_byte_ms,
					&s.tokens_per_second,
					&i32::try_from(s.failures).unwrap_or(i32::MAX),
					&i32::try_from(s.trips).unwrap_or(i32::MAX),
					&ms(s.cooling_until),
					&s.last_failure,
					&n(s.remaining.requests),
					&n(s.remaining.tokens),
					&ms(s.updated.or(Some(now))),
				],
			)
			.await;
		if let Err(error) = written {
			// Written again at the next flush.
			health.with(|states| {
				if let Some(state) = states.get_mut(&(provider.clone(), model.clone())) {
					state.dirty = true;
				}
			});
			return Err(error);
		}
	}
	Ok(changed.len())
}

/// Writes the statistics every few seconds for as long as Psyche runs, reconnecting after a
/// failure.
pub async fn keep(database_url: String, health: Health) {
	let mut client: Option<Client> = None;
	let mut tick = tokio::time::interval(FLUSH_EVERY);
	tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
	loop {
		tick.tick().await;
		if client.as_ref().is_none_or(Client::is_closed) {
			client = crate::log::connect(&database_url).await.ok();
		}
		let Some(c) = &client else { continue };
		if let Err(error) = flush(c, &health, SystemTime::now()).await {
			tracing::warn!(error = %crate::db_message(&error), "the provider statistics could not be written");
			client = None;
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::providers::{Catalogue, Spec};

	fn models() -> Vec<Arc<Model>> {
		let spec = Spec::from_json(
			r#"{"providers": [{"name": "p", "kind": "openai", "base_url": "https://p.example/v1"}],
				"models": [{"name": "a", "provider": "p"}, {"name": "b", "provider": "p"}, {"name": "c", "provider": "p"}]}"#,
			&HashMap::new(),
			crate::net::Egress::Public,
		)
		.unwrap();
		Catalogue::build(&spec).unwrap().models().to_vec()
	}

	fn names(order: &[Arc<Model>]) -> Vec<&str> {
		order.iter().map(|m| m.name.as_str()).collect()
	}

	fn failed(class: Class) -> Seen {
		Seen {
			outcome: Outcome::Failed(class),
			first_byte: Duration::ZERO,
			total: Duration::from_millis(5),
			streamed: false,
			output_tokens: None,
			retry_after: None,
			remaining: Remaining::default(),
		}
	}

	fn answered(first_byte_ms: u64, tokens: u64, total_ms: u64) -> Seen {
		Seen {
			outcome: Outcome::Completed,
			first_byte: Duration::from_millis(first_byte_ms),
			total: Duration::from_millis(total_ms),
			streamed: true,
			output_tokens: Some(tokens),
			retry_after: None,
			remaining: Remaining::default(),
		}
	}

	#[test]
	fn three_failures_in_a_row_cool_a_target_and_a_probe_decides_when_it_comes_back() {
		let health = Health::default();
		let all = models();
		let t0 = UNIX_EPOCH + Duration::from_secs(1_800_000_000);
		for i in 0..2 {
			health.seen(
				"p",
				"a",
				failed(Class::Upstream),
				t0 + Duration::from_secs(i),
			);
		}
		assert_eq!(
			names(&health.order(all.clone(), None, t0)),
			["a", "b", "c"],
			"two are not enough"
		);
		// The caller's own mistake is an answer: the count starts again.
		health.seen("p", "a", failed(Class::Rejected), t0);
		health.seen("p", "a", failed(Class::Upstream), t0);
		health.seen("p", "a", failed(Class::Upstream), t0);
		assert_eq!(names(&health.order(all.clone(), None, t0)), ["a", "b", "c"]);
		health.seen("p", "a", failed(Class::Timeout), t0);
		assert_eq!(
			names(&health.order(all.clone(), None, t0)),
			["b", "c", "a"],
			"cooling goes last"
		);
		assert_eq!(health.report(t0)[0]["state"], "cooling");

		// Ten seconds on, the first call to reach it is its probe; the others still go elsewhere.
		let t1 = t0 + FIRST_COOLDOWN;
		assert_eq!(names(&health.order(all.clone(), None, t1)), ["a", "b", "c"]);
		health.begin(&all[0], t1);
		assert_eq!(names(&health.order(all.clone(), None, t1)), ["b", "c", "a"]);
		assert_eq!(health.report(t1)[0]["state"], "probing");
		// The probe fails: twice as long this time.
		health.seen("p", "a", failed(Class::Transport), t1);
		assert_eq!(
			health.report(t1)[0]["cooling_until"],
			(t1 + FIRST_COOLDOWN * 2)
				.duration_since(UNIX_EPOCH)
				.unwrap()
				.as_secs()
		);
		let t2 = t1 + FIRST_COOLDOWN * 2;
		health.begin(&all[0], t2);
		health.seen("p", "a", answered(120, 40, 520), t2);
		assert_eq!(
			names(&health.order(all.clone(), None, t2)),
			["a", "b", "c"],
			"the probe answered"
		);
		let report = &health.report(t2)[0];
		assert_eq!(report["state"], "ok");
		assert_eq!(report["failures_in_a_row"], 0);
		assert_eq!(report["requests"], 8);
		assert_eq!(
			report["failed"], 6,
			"the caller's mistake is not the target's failure"
		);
		assert_eq!(report["tokens_per_second"], 100.0);

		// The next run of failures starts from the first cooldown again.
		for _ in 0..3 {
			health.seen("p", "a", failed(Class::Upstream), t2);
		}
		assert_eq!(
			health.report(t2)[0]["cooling_until"],
			(t2 + FIRST_COOLDOWN)
				.duration_since(UNIX_EPOCH)
				.unwrap()
				.as_secs()
		);
	}

	#[test]
	fn a_rate_limit_with_retry_after_cools_for_exactly_that_long_and_is_not_a_failure_in_a_row() {
		let health = Health::default();
		let all = models();
		let t0 = UNIX_EPOCH + Duration::from_secs(1_800_000_000);
		let mut limited = failed(Class::RateLimited);
		limited.retry_after = Some(Duration::from_secs(30));
		health.seen("p", "b", limited, t0);
		assert_eq!(
			names(&health.order(all.clone(), None, t0 + Duration::from_secs(29))),
			["a", "c", "b"]
		);
		assert_eq!(
			names(&health.order(all.clone(), None, t0 + Duration::from_secs(30))),
			["a", "b", "c"]
		);
		let report = &health.report(t0)[0];
		assert_eq!(report["failures_in_a_row"], 0);
		assert_eq!(report["rate_limited"], 1);
		assert_eq!(report["last_failure"], "rate_limited");
	}

	#[test]
	fn a_route_sorted_by_latency_or_throughput_puts_the_unmeasured_then_the_fastest_first() {
		let health = Health::default();
		let all = models();
		let t0 = UNIX_EPOCH + Duration::from_secs(1_800_000_000);
		health.seen("p", "a", answered(900, 100, 1900), t0); // 100 tokens a second
		health.seen("p", "b", answered(200, 100, 5200), t0); // 20
		assert_eq!(
			names(&health.order(all.clone(), Some(Sort::Latency), t0)),
			["c", "b", "a"]
		);
		assert_eq!(
			names(&health.order(all.clone(), Some(Sort::Throughput), t0)),
			["c", "a", "b"]
		);
		assert_eq!(
			names(&health.order(all.clone(), Some(Sort::Order), t0)),
			["a", "b", "c"]
		);
		// A moving average: one slow reply moves it a fifth of the way.
		health.seen("p", "b", answered(1200, 100, 6200), t0);
		assert_eq!(health.report(t0)[1]["first_byte_ms"], 400.0);
	}

	#[test]
	fn counts_cover_the_last_five_minutes_and_an_abandoned_probe_is_given_up() {
		let health = Health::default();
		let all = models();
		let t0 = UNIX_EPOCH + Duration::from_secs(1_800_000_000 / 60 * 60);
		health.seen("p", "a", answered(100, 1, 200), t0);
		health.seen(
			"p",
			"a",
			answered(100, 1, 200),
			t0 + Duration::from_secs(4 * 60),
		);
		assert_eq!(
			health.report(t0 + Duration::from_secs(4 * 60))[0]["requests"],
			2
		);
		assert_eq!(
			health.report(t0 + Duration::from_secs(5 * 60))[0]["requests"],
			1
		);
		assert!(health.report(t0 + Duration::from_secs(10 * 60))[0]["error_rate"].is_null());

		for _ in 0..3 {
			health.seen("p", "b", failed(Class::Upstream), t0);
		}
		let t1 = t0 + FIRST_COOLDOWN;
		health.begin(&all[1], t1);
		let almost = t1 + PROBE_ABANDONED - Duration::from_secs(1);
		assert_eq!(
			names(&health.order(all.clone(), None, almost)),
			["a", "c", "b"]
		);
		assert_eq!(
			names(&health.order(all, None, t1 + PROBE_ABANDONED)),
			["a", "b", "c"]
		);
	}

	#[tokio::test]
	async fn the_statistics_are_written_and_a_restart_reads_them_back() {
		let Some((db, _url)) = crate::log::tests::fresh_database("health").await else {
			return;
		};
		let health = Health::default();
		let now = SystemTime::now();
		let mut remaining = answered(250, 100, 1250);
		remaining.remaining = Remaining {
			requests: Some(42),
			tokens: None,
		};
		health.seen("p", "a", remaining, now);
		for _ in 0..3 {
			health.seen("p", "b", failed(Class::ProviderBilling), now);
		}
		assert_eq!(flush(&db, &health, now).await.unwrap(), 2);
		assert_eq!(
			flush(&db, &health, now).await.unwrap(),
			0,
			"only what changed is written"
		);
		let row = db
			.query_one(
				"SELECT state, requests, failed, first_byte_ms, tokens_per_second, remaining_requests, last_failure \
				 FROM psyche.provider_stats WHERE provider = 'p' AND model = 'b'",
				&[],
			)
			.await
			.unwrap();
		assert_eq!(row.get::<_, &str>(0), "cooling");
		assert_eq!((row.get::<_, i32>(1), row.get::<_, i32>(2)), (3, 3));
		assert_eq!(row.get::<_, Option<&str>>(6), Some("provider_billing"));
		let a = db
			.query_one(
				"SELECT first_byte_ms, tokens_per_second, remaining_requests FROM psyche.provider_stats WHERE model = 'a'",
				&[],
			)
			.await
			.unwrap();
		assert_eq!(a.get::<_, Option<f64>>(0), Some(250.0));
		assert_eq!(a.get::<_, Option<f64>>(1), Some(100.0));
		assert_eq!(a.get::<_, Option<i64>>(2), Some(42));

		// A new start reads the averages and the cooldown still running.
		let restarted = Health::default();
		assert_eq!(load(&db, &restarted).await.unwrap(), 2);
		assert_eq!(
			names(&restarted.order(models(), Some(Sort::Latency), now)),
			["c", "a", "b"],
			"a measured, b still cooling, c never measured"
		);

		// A route may be sorted by what is measured.
		db.execute(
			"INSERT INTO psyche.routes (name, targets, sort) VALUES ('quick', '{a,b}', 'latency'), ('steady', '{a}', 'throughput')",
			&[],
		)
		.await
		.unwrap();
		assert!(
			db.execute(
				"INSERT INTO psyche.routes (name, targets, sort) VALUES ('bad', '{a}', 'random')",
				&[]
			)
			.await
			.is_err()
		);
	}

	#[test]
	fn remaining_quota_is_read_from_either_providers_headers() {
		let mut headers = reqwest::header::HeaderMap::new();
		headers.insert("x-ratelimit-remaining-requests", "499".parse().unwrap());
		headers.insert(
			"anthropic-ratelimit-tokens-remaining",
			"80000".parse().unwrap(),
		);
		assert_eq!(
			Remaining::read(&headers),
			Remaining {
				requests: Some(499),
				tokens: Some(80000)
			}
		);
		assert_eq!(
			Remaining::read(&reqwest::header::HeaderMap::new()),
			Remaining::default()
		);
	}
}
