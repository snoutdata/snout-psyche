//! Budgets: reserve before a call is sent, settle when it ends (PS8).
//!
//! A budget caps what the project, or one key, may spend in a window: a calendar day or month (UTC),
//! or the last N days. Before a call goes to its provider, its worst-case cost (the prompt's upper
//! bound and the reply's cap, at the model's price) is reserved against every budget in its scope,
//! under one lock, and a hard budget it would not fit refuses it with a sentence that names the
//! budget. When the call ends the reservation is replaced by what the provider reported, so the
//! next call sees what this one really cost, and a long reply can never run through a budget
//! because the room for it was taken first.
//!
//! Spend is held in memory as one bucket per subject (the project, each key) per UTC day, read at
//! start from `psyche.usage_daily` (kept for good, unlike the request log) for the last
//! [`HISTORY_DAYS`]. In memory is the whole truth while the process runs (PS7: one process per
//! project). What it can miss is a crash's unwritten log rows (`log.rs`), which a restart reads as
//! unspent; and a call whose provider reported no usage is charged its whole reservation here,
//! but logged with no cost, so a restart forgets that charge.
//!
//! Two limits on "no overshoot", both written down rather than hidden: a soft budget never refuses
//! (it is reported in the `x-psyche-budget-warning` header), and a call whose reply has no cap
//! (an OpenAI-style provider asked without `max_tokens`, and no `max_output_tokens` in the model's
//! entry) is reserved at [`UNCAPPED_OUTPUT`] tokens, so such a reply can overrun by the rest of
//! itself, once, before the next call is refused.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use crate::keys::KeyId;
use crate::price::{Dollars, PICO};

/// How far back spend is held: the longest window a budget may have.
pub const HISTORY_DAYS: u32 = 366;
/// The reply length reserved for a call that names no cap and whose model's entry names none.
pub const UNCAPPED_OUTPUT: u64 = 4096;

const SECONDS_PER_DAY: u64 = 86_400;

/// Days since the epoch, UTC.
pub type Day = u32;

pub fn day_of(at: SystemTime) -> Day {
	let unix = at
		.duration_since(SystemTime::UNIX_EPOCH)
		.map_or(0, |d| d.as_secs());
	(unix / SECONDS_PER_DAY) as Day
}

/// The first day of `day`'s month (Howard Hinnant's civil-from-days, for the day of the month).
fn month_start(day: Day) -> Day {
	let z = i64::from(day) + 719_468;
	let era = z.div_euclid(146_097);
	let doe = z - era * 146_097;
	let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
	let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
	let mp = (5 * doy + 2) / 153;
	let day_of_month = doy - (153 * mp + 2) / 5 + 1;
	day - (day_of_month - 1) as Day
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Subject {
	Project,
	Key(KeyId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Period {
	Day,
	Month,
	/// The last N days, today included.
	Rolling(u32),
}

impl Period {
	/// The first day of the window that holds `today`.
	fn start(self, today: Day) -> Day {
		match self {
			Period::Day => today,
			Period::Month => month_start(today),
			Period::Rolling(days) => today.saturating_sub(days.clamp(1, HISTORY_DAYS) - 1),
		}
	}

	fn describe(self) -> String {
		match self {
			Period::Day => "a day".into(),
			Period::Month => "a month".into(),
			Period::Rolling(1) => "a day".into(),
			Period::Rolling(days) => format!("every {days} days"),
		}
	}
}

#[derive(Debug, Clone, PartialEq)]
pub struct Budget {
	pub id: KeyId,
	pub name: String,
	pub subject: Subject,
	/// Picodollars.
	pub limit: u64,
	pub period: Period,
	/// A hard budget refuses a call that would not fit; a soft one only warns.
	pub hard: bool,
}

#[derive(Default)]
struct Ledger {
	budgets: Vec<Budget>,
	/// Picodollars settled, per subject per day.
	spent: HashMap<(Subject, Day), u64>,
	/// Picodollars reserved by calls in flight, per subject.
	reserved: HashMap<Subject, u64>,
	/// The last day old buckets were dropped on.
	pruned: Day,
}

impl Ledger {
	fn spent(&self, subject: Subject, period: Period, today: Day) -> u64 {
		(period.start(today)..=today)
			.map(|day| self.spent.get(&(subject, day)).copied().unwrap_or(0))
			.sum()
	}
}

/// A refusal: the budget, and the sentence that says why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Over {
	pub budget: String,
	pub message: String,
}

/// Every budget and the spend against them, shared by every call.
#[derive(Clone, Default)]
pub struct Budgets(Arc<Mutex<Ledger>>);

impl Budgets {
	fn ledger(&self) -> std::sync::MutexGuard<'_, Ledger> {
		self.0.lock().unwrap_or_else(|e| e.into_inner())
	}

	pub fn replace(&self, budgets: Vec<Budget>) {
		self.ledger().budgets = budgets;
	}

	/// Whether any budget would apply to a call by `key`.
	pub fn applies(&self, key: Option<KeyId>) -> bool {
		let subjects = subjects(key);
		self.ledger()
			.budgets
			.iter()
			.any(|b| subjects.contains(&b.subject))
	}

	/// Spend already settled (from the log, at start).
	pub fn add_spent(&self, subject: Subject, day: Day, picodollars: u64) {
		let mut ledger = self.ledger();
		let bucket = ledger.spent.entry((subject, day)).or_insert(0);
		*bucket = bucket.saturating_add(picodollars);
	}

	/// Reserves `amount` picodollars for a call by `key`, or refuses it. A soft budget it would not
	/// fit is named in the reservation's warnings instead.
	pub fn reserve(
		&self,
		key: Option<KeyId>,
		amount: u64,
		now: SystemTime,
	) -> Result<Reservation, Over> {
		let today = day_of(now);
		let subjects = subjects(key);
		let mut ledger = self.ledger();
		let mut warnings = Vec::new();
		for budget in ledger
			.budgets
			.iter()
			.filter(|b| subjects.contains(&b.subject))
		{
			let spent = ledger.spent(budget.subject, budget.period, today);
			let reserved = ledger.reserved.get(&budget.subject).copied().unwrap_or(0);
			let used = spent.saturating_add(reserved);
			if used.saturating_add(amount) <= budget.limit {
				continue;
			}
			if !budget.hard {
				warnings.push(budget.name.clone());
				continue;
			}
			let left = budget.limit.saturating_sub(used);
			return Err(Over {
				budget: budget.name.clone(),
				message: format!(
					"The budget {} ({} {}) has {} left, and this call could cost up to {}.",
					budget.name,
					money(budget.limit),
					budget.period.describe(),
					money(left),
					money(amount),
				),
			});
		}
		for subject in &subjects {
			let held = ledger.reserved.entry(*subject).or_insert(0);
			*held = held.saturating_add(amount);
		}
		Ok(Reservation {
			budgets: self.clone(),
			subjects,
			amount,
			warnings,
			settled: false,
		})
	}
}

fn subjects(key: Option<KeyId>) -> Vec<Subject> {
	match key {
		None => vec![Subject::Project],
		Some(key) => vec![Subject::Project, Subject::Key(key)],
	}
}

/// Dollars for a sentence: cents when that is exact enough, more places when it is not.
fn money(picodollars: u64) -> String {
	let cents = picodollars / (PICO / 100);
	if picodollars.is_multiple_of(PICO / 100) || picodollars >= PICO {
		return format!("${}.{:02}", cents / 100, cents % 100);
	}
	let text = Dollars(picodollars).to_string();
	format!("${}", text.trim_end_matches('0'))
}

/// The room a call holds in its budgets until it ends.
#[derive(Debug)]
pub struct Reservation {
	budgets: Budgets,
	subjects: Vec<Subject>,
	amount: u64,
	/// The soft budgets this call goes over.
	pub warnings: Vec<String>,
	settled: bool,
}

impl std::fmt::Debug for Budgets {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.write_str("Budgets(..)")
	}
}

impl Reservation {
	pub fn amount(&self) -> u64 {
		self.amount
	}

	/// Releases the reservation and charges `cost`: what the call cost, or, when its provider
	/// reported nothing, the whole reservation.
	pub fn settle(mut self, cost: Option<u64>, at: SystemTime) {
		self.release(cost.unwrap_or(self.amount), at);
	}

	fn release(&mut self, charge: u64, at: SystemTime) {
		if self.settled {
			return;
		}
		self.settled = true;
		let day = day_of(at);
		let mut ledger = self.budgets.ledger();
		for subject in &self.subjects {
			if let Some(held) = ledger.reserved.get_mut(subject) {
				*held = held.saturating_sub(self.amount);
			}
			let bucket = ledger.spent.entry((*subject, day)).or_insert(0);
			*bucket = bucket.saturating_add(charge);
		}
		// The first settle of each day lets the days past the history go.
		if day > ledger.pruned {
			ledger.pruned = day;
			let oldest = day.saturating_sub(HISTORY_DAYS);
			ledger.spent.retain(|(_, d), _| *d >= oldest);
		}
	}
}

impl Drop for Reservation {
	/// A reservation dropped without a settle (a call that never reached its provider) charges
	/// nothing.
	fn drop(&mut self) {
		self.release(0, SystemTime::now());
	}
}

/// The prompt's upper bound in tokens: its bytes, since a token is at least a byte in every
/// byte-level tokenizer, except an inline image, which is its bytes' worth of base64 and costs a
/// fixed number of tokens instead.
pub fn prompt_bound(body: &[u8]) -> u64 {
	const IMAGE_TOKENS: u64 = 2_000;
	let mut bytes = body.len() as u64;
	let mut images = 0u64;
	let mut from = 0;
	while let Some(at) = body[from..].windows(5).position(|w| w == b"data:") {
		let start = from + at;
		let end = body[start..]
			.iter()
			.position(|b| *b == b'"')
			.map_or(body.len(), |n| start + n);
		bytes = bytes.saturating_sub((end - start) as u64);
		images += 1;
		from = end;
	}
	bytes + images * IMAGE_TOKENS
}

/// Every budget, from `psyche.budgets`.
pub async fn load(client: &tokio_postgres::Client) -> Result<Vec<Budget>, tokio_postgres::Error> {
	let rows = client
		.query(
			"SELECT id::text, name, key_id::text, (amount * 1000000000000)::bigint, period, days, hard \
			 FROM psyche.budgets",
			&[],
		)
		.await?;
	Ok(rows
		.iter()
		.filter_map(|row| {
			let period = match (row.get::<_, &str>(4), row.get::<_, Option<i32>>(5)) {
				("day", _) => Period::Day,
				("month", _) => Period::Month,
				("rolling", Some(days)) if days > 0 => Period::Rolling(days as u32),
				_ => return None,
			};
			Some(Budget {
				id: KeyId::parse(row.get(0))?,
				name: row.get(1),
				subject: match row.get::<_, Option<&str>>(2) {
					None => Subject::Project,
					Some(key) => Subject::Key(KeyId::parse(key)?),
				},
				limit: u64::try_from(row.get::<_, i64>(3)).unwrap_or(0),
				period,
				hard: row.get(6),
			})
		})
		.collect())
}

/// What was spent in the last [`HISTORY_DAYS`], from the daily totals, into `budgets`. Read once,
/// at start, before the first call: from then on the ledger is kept by the calls themselves.
pub async fn load_spent(
	client: &tokio_postgres::Client,
	budgets: &Budgets,
	now: SystemTime,
) -> Result<(), tokio_postgres::Error> {
	let today = day_of(now);
	let rows = client
		.query(
			"SELECT (day - date '1970-01-01')::int, key_id::text, \
			 (sum(cost) * 1000000000000)::bigint FROM psyche.usage_daily \
			 WHERE day >= date '1970-01-01' + $1::int GROUP BY 1, 2",
			&[&(today.saturating_sub(HISTORY_DAYS) as i32)],
		)
		.await?;
	for row in rows {
		let day = row.get::<_, i32>(0).max(0) as Day;
		let spent = u64::try_from(row.get::<_, i64>(2)).unwrap_or(0);
		budgets.add_spent(Subject::Project, day, spent);
		if let Some(key) = row.get::<_, Option<&str>>(1).and_then(KeyId::parse) {
			budgets.add_spent(Subject::Key(key), day, spent);
		}
	}
	Ok(())
}

#[cfg(test)]
mod tests {
	use super::*;
	use std::time::Duration;

	fn budget(name: &str, subject: Subject, dollars: u64, period: Period, hard: bool) -> Budget {
		Budget {
			id: KeyId([name.len() as u8; 16]),
			name: name.into(),
			subject,
			limit: dollars * PICO,
			period,
			hard,
		}
	}

	#[test]
	fn months_start_on_the_first() {
		// 2026-10-18 is day 20,744; October began on day 20,727.
		assert_eq!(month_start(20_744), 20_727);
		assert_eq!(month_start(20_727), 20_727);
		// 2024-02-29, a leap day, is day 19,782; February began on 19,754.
		assert_eq!(month_start(19_782), 19_754);
		assert_eq!(Period::Rolling(7).start(100), 94);
		assert_eq!(Period::Day.start(100), 100);
	}

	#[test]
	fn a_call_that_would_not_fit_is_refused_and_names_the_budget() {
		let budgets = Budgets::default();
		let key = KeyId([7; 16]);
		budgets.replace(vec![budget("ci", Subject::Key(key), 1, Period::Day, true)]);
		let now = SystemTime::now();
		let first = budgets.reserve(Some(key), PICO * 6 / 10, now).unwrap();
		let over = budgets.reserve(Some(key), PICO * 6 / 10, now).unwrap_err();
		assert_eq!(over.budget, "ci");
		assert!(
			over.message.contains("$1.00 a day") && over.message.contains("$0.40 left"),
			"{}",
			over.message
		);
		// Another key, and the service key, are not held by it.
		assert!(
			budgets
				.reserve(Some(KeyId([8; 16])), PICO * 6 / 10, now)
				.is_ok()
		);
		assert!(budgets.reserve(None, PICO * 6 / 10, now).is_ok());
		// Settled at what it really cost, the room comes back.
		first.settle(Some(PICO / 10), now);
		assert!(budgets.reserve(Some(key), PICO * 6 / 10, now).is_ok());
	}

	#[test]
	fn a_long_reply_cannot_run_through_a_budget() {
		// PS8's case: two calls that each fit alone, in flight together.
		let budgets = Budgets::default();
		budgets.replace(vec![budget(
			"project",
			Subject::Project,
			1,
			Period::Month,
			true,
		)]);
		let now = SystemTime::now();
		let long = budgets.reserve(None, PICO * 9 / 10, now).unwrap();
		assert!(
			budgets.reserve(None, PICO / 5, now).is_err(),
			"the room is taken"
		);
		long.settle(Some(PICO * 9 / 10), now);
		assert!(
			budgets.reserve(None, PICO / 5, now).is_err(),
			"and then spent"
		);
		assert!(budgets.reserve(None, PICO / 10, now).is_ok());
	}

	#[test]
	fn a_call_with_no_usage_is_charged_its_reservation_and_one_never_sent_nothing() {
		let budgets = Budgets::default();
		budgets.replace(vec![budget("p", Subject::Project, 1, Period::Day, true)]);
		let now = SystemTime::now();
		budgets
			.reserve(None, PICO / 2, now)
			.unwrap()
			.settle(None, now);
		drop(budgets.reserve(None, PICO / 4, now).unwrap());
		let held = budgets.reserve(None, PICO / 2, now).unwrap();
		assert!(
			budgets.reserve(None, PICO / 100, now).is_err(),
			"$0.50 charged, $0.50 held"
		);
		drop(held);
	}

	#[test]
	fn a_soft_budget_warns_and_windows_forget_old_days() {
		let budgets = Budgets::default();
		budgets.replace(vec![
			budget("soft", Subject::Project, 1, Period::Rolling(7), false),
			budget("day", Subject::Project, 5, Period::Day, true),
		]);
		let now = SystemTime::now();
		let today = day_of(now);
		budgets.add_spent(Subject::Project, today - 7, PICO * 4);
		budgets.add_spent(Subject::Project, today - 6, PICO);
		let reservation = budgets.reserve(None, PICO / 10, now).unwrap();
		assert_eq!(reservation.warnings, vec!["soft".to_string()]);
		drop(reservation);
		let tomorrow = now + Duration::from_secs(SECONDS_PER_DAY);
		assert!(
			budgets
				.reserve(None, PICO / 10, tomorrow)
				.unwrap()
				.warnings
				.is_empty(),
			"day -6 has left the seven days"
		);
	}

	#[tokio::test]
	async fn what_was_spent_is_read_back_from_the_daily_totals() {
		let Some((db, _)) = crate::log::tests::fresh_database("spent").await else {
			return;
		};
		let id = "0190f1c2-7a3b-4c5d-8e9f-a0b1c2d3e4f5";
		db.batch_execute(&format!(
			"INSERT INTO psyche.usage_daily (day, operation, model, provider, cost, key_id) VALUES \
			 (current_date, 'chat', 'm', 'p', 0.75, '{id}'), \
			 (current_date - 1, 'chat', 'm', 'p', 0.5, NULL), \
			 (current_date - 400, 'chat', 'm', 'p', 99, NULL)"
		))
		.await
		.unwrap();
		let budgets = Budgets::default();
		load_spent(&db, &budgets, SystemTime::now()).await.unwrap();
		let key = KeyId::parse(id).unwrap();
		budgets.replace(vec![
			budget("project", Subject::Project, 2, Period::Rolling(2), true),
			budget("key", Subject::Key(key), 1, Period::Day, true),
		]);
		let now = SystemTime::now();
		// The project has spent $1.25 in two days (the day 400 back is past the history), the key $0.75.
		assert!(budgets.reserve(None, PICO * 3 / 4, now).is_ok());
		assert!(budgets.reserve(None, PICO * 4 / 5, now).is_err());
		assert!(budgets.reserve(Some(key), PICO / 4, now).is_ok());
		assert!(budgets.reserve(Some(key), PICO * 3 / 10, now).is_err());
	}

	#[test]
	fn money_reads_as_dollars() {
		assert_eq!(money(PICO * 3 / 2), "$1.50");
		assert_eq!(money(PICO / 100), "$0.01");
		assert_eq!(money(46_640_000), "$0.00004664");
	}

	#[test]
	fn the_prompt_bound_counts_bytes_and_prices_images_flat() {
		let text = br#"{"messages":[{"role":"user","content":"hello"}]}"#;
		assert_eq!(prompt_bound(text), text.len() as u64);
		let image = format!(
			r#"{{"messages":[{{"role":"user","content":[{{"type":"image_url","image_url":{{"url":"data:image/png;base64,{}"}}}}]}}]}}"#,
			"A".repeat(100_000)
		);
		let bound = prompt_bound(image.as_bytes());
		assert!(bound < 2_200, "{bound}");
		assert!(bound >= 2_000);
	}
}
