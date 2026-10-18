//! Budgets: reserve before a call is sent, settle when it ends (PS8).
//!
//! A budget caps what the project, one key, each signed-in user, one user or a group may spend in a
//! window (`who.rs`): a calendar day or month (UTC), or the last N days. Before a call goes to its provider, its worst-case cost (the prompt's upper
//! bound and the reply's cap, at the model's price) is reserved against every budget in its scope,
//! under one lock, and a hard budget it would not fit refuses it with a sentence that names the
//! budget. When the call ends the reservation is replaced by what the provider reported, so the
//! next call sees what this one really cost, and a long reply can never run through a budget
//! because the room for it was taken first.
//!
//! Spend is held in memory as one bucket per subject (the project, each key, user and group) per UTC
//! day, kept for every subject of a call any budget applies to, and read at
//! start from `psyche.usage_daily` (kept for good, unlike the request log) for the last
//! [`HISTORY_DAYS`]. In memory is the whole truth while the process runs (PS7: one process per
//! project). What it can miss is a crash's unwritten log rows (`log.rs`), which a restart reads as
//! unspent; and a call whose provider reported no usage is charged its whole reservation here,
//! but logged with no cost, so a restart forgets that charge.
//!
//! A reply with no cap of its own (the caller named none, and the model's entry names no
//! `max_output_tokens`) is sent one when a hard budget applies: [`UNCAPPED_OUTPUT`] tokens, or what
//! the tightest hard budget still has room for if that is less, so the provider cannot run past what
//! was reserved. One limit on "no overshoot" is written down rather than hidden: a soft budget never
//! refuses (it is reported in the `x-psyche-budget-warning` header), and a call under soft budgets
//! only is sent no cap and reserved at [`UNCAPPED_OUTPUT`].

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use crate::keys::KeyId;
use crate::price::{Dollars, PICO};
pub use crate::who::Subject;
use crate::who::{Scope, UserId, Who, subjects};

/// How far back spend is held: the longest window a budget may have.
pub const HISTORY_DAYS: u32 = 366;
/// The reply cap sent, under a hard budget, for a call that names none and whose model's entry
/// names none; and what such a call is reserved at under soft budgets. The same 4096 Anthropic is
/// always sent (`anthropic::DEFAULT_MAX_TOKENS`).
pub const UNCAPPED_OUTPUT: u64 = 4096;
/// The shortest cap sent when a hard budget is nearly spent. A call whose prompt leaves less room
/// than this is refused rather than sent for a reply too short to use.
pub const MIN_SENT_OUTPUT: u64 = 256;

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
	pub scope: Scope,
	/// Picodollars.
	pub limit: u64,
	pub period: Period,
	/// A hard budget refuses a call that would not fit; a soft one only warns.
	pub hard: bool,
}

#[derive(Default)]
struct Ledger {
	budgets: Vec<Budget>,
	/// The users a budget names one by one, whom the every-user budgets leave alone.
	overridden: HashSet<UserId>,
	/// Picodollars settled, per subject per day.
	spent: HashMap<(Subject, Day), u64>,
	/// Picodollars reserved by calls in flight, per subject.
	reserved: HashMap<Subject, u64>,
	/// The last day old buckets were dropped on.
	pruned: Day,
}

impl Ledger {
	/// The budgets that hold a call by `who`, each with what it holds it to.
	fn over(&self, who: &Who) -> impl Iterator<Item = (&Budget, Subject)> {
		self.budgets
			.iter()
			.filter_map(|b| Some((b, b.scope.subject(who, &self.overridden)?)))
	}

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
		let mut ledger = self.ledger();
		ledger.overridden = crate::who::overridden(budgets.iter().map(|b| b.scope));
		ledger.budgets = budgets;
	}

	/// Whether any budget would apply to a call by `who`.
	pub fn applies(&self, who: &Who) -> bool {
		self.ledger().over(who).next().is_some()
	}

	/// Why a signed-in user may not call yet, if they may not: the anon path takes no call until
	/// the project has a hard budget of its own and a hard one for this user (every user's, or
	/// theirs), so an app open to the world cannot spend without a cap (PS10b).
	pub fn admits(&self, user: UserId) -> Result<(), &'static str> {
		let ledger = self.ledger();
		let who = Who {
			user: Some(user),
			..Who::default()
		};
		let hard = |subject: Subject| ledger.over(&who).any(|(b, s)| b.hard && s == subject);
		if !hard(Subject::Project) {
			return Err(
				"Signed-in users can call models once the project has a hard budget of its own. Add one to psyche.budgets.",
			);
		}
		if !hard(Subject::User(user)) {
			return Err(
				"Signed-in users can call models once each has a hard budget: add one with every_user set to psyche.budgets.",
			);
		}
		Ok(())
	}

	/// Spend already settled (from the log, at start).
	pub fn add_spent(&self, subject: Subject, day: Day, picodollars: u64) {
		let mut ledger = self.ledger();
		let bucket = ledger.spent.entry((subject, day)).or_insert(0);
		*bucket = bucket.saturating_add(picodollars);
	}

	/// What the tightest hard budget over a call by `key` still has room for, in picodollars;
	/// `None` when no hard budget applies.
	pub fn room(&self, who: &Who, now: SystemTime) -> Option<u64> {
		let today = day_of(now);
		let ledger = self.ledger();
		ledger
			.over(who)
			.filter(|(b, _)| b.hard)
			.map(|(b, subject)| {
				let reserved = ledger.reserved.get(&subject).copied().unwrap_or(0);
				b.limit.saturating_sub(
					ledger
						.spent(subject, b.period, today)
						.saturating_add(reserved),
				)
			})
			.min()
	}

	/// Reserves `amount` picodollars for a call by `key`, or refuses it. A soft budget it would not
	/// fit is named in the reservation's warnings instead.
	pub fn reserve(&self, who: &Who, amount: u64, now: SystemTime) -> Result<Reservation, Over> {
		let today = day_of(now);
		let subjects = subjects(who);
		let mut ledger = self.ledger();
		let mut warnings = Vec::new();
		for (budget, subject) in ledger.over(who) {
			let spent = ledger.spent(subject, budget.period, today);
			let reserved = ledger.reserved.get(&subject).copied().unwrap_or(0);
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
				if *held == 0 {
					ledger.reserved.remove(subject);
				}
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
			"SELECT id::text, name, key_id::text, (amount * 1000000000000)::bigint, period, days, hard, \
			 every_user, user_id::text, group_id::text FROM psyche.budgets",
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
				scope: Scope::from_row(row.get(2), row.get(7), row.get(8), row.get(9))?,
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
			"SELECT (day - date '1970-01-01')::int, key_id::text, user_id::text, group_id::text, \
			 (sum(cost) * 1000000000000)::bigint FROM psyche.usage_daily \
			 WHERE day >= date '1970-01-01' + $1::int GROUP BY 1, 2, 3, 4",
			&[&(today.saturating_sub(HISTORY_DAYS) as i32)],
		)
		.await?;
	for row in rows {
		let day = row.get::<_, i32>(0).max(0) as Day;
		let spent = u64::try_from(row.get::<_, i64>(4)).unwrap_or(0);
		let id = |n: usize| row.get::<_, Option<&str>>(n).and_then(KeyId::parse);
		let who = Who {
			key: id(1),
			user: id(2),
			group: id(3),
		};
		for subject in subjects(&who) {
			budgets.add_spent(subject, day, spent);
		}
	}
	Ok(())
}

#[cfg(test)]
mod tests {
	use super::*;
	use std::time::Duration;

	fn by(key: KeyId) -> Who {
		Who {
			key: Some(key),
			..Who::default()
		}
	}

	fn budget(name: &str, scope: Scope, dollars: u64, period: Period, hard: bool) -> Budget {
		Budget {
			id: KeyId([name.len() as u8; 16]),
			name: name.into(),
			scope,
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
		budgets.replace(vec![budget("ci", Scope::Key(key), 1, Period::Day, true)]);
		let now = SystemTime::now();
		let first = budgets.reserve(&by(key), PICO * 6 / 10, now).unwrap();
		let over = budgets.reserve(&by(key), PICO * 6 / 10, now).unwrap_err();
		assert_eq!(over.budget, "ci");
		assert!(
			over.message.contains("$1.00 a day") && over.message.contains("$0.40 left"),
			"{}",
			over.message
		);
		// Another key, and the service key, are not held by it.
		assert!(
			budgets
				.reserve(&by(KeyId([8; 16])), PICO * 6 / 10, now)
				.is_ok()
		);
		assert!(budgets.reserve(&Who::default(), PICO * 6 / 10, now).is_ok());
		// Settled at what it really cost, the room comes back.
		first.settle(Some(PICO / 10), now);
		assert!(budgets.reserve(&by(key), PICO * 6 / 10, now).is_ok());
	}

	#[test]
	fn each_user_has_their_own_room_a_group_shares_one_and_an_override_replaces_the_default() {
		let budgets = Budgets::default();
		let (alice, bob, carol, team) = (
			KeyId([1; 16]),
			KeyId([2; 16]),
			KeyId([3; 16]),
			KeyId([4; 16]),
		);
		let user = |id: KeyId, group: Option<KeyId>| Who {
			key: None,
			user: Some(id),
			group,
		};
		let now = SystemTime::now();
		assert!(budgets.admits(alice).is_err(), "no budgets yet");
		budgets.replace(vec![budget(
			"everyone",
			Scope::EveryUser,
			1,
			Period::Day,
			true,
		)]);
		assert!(
			budgets.admits(alice).unwrap_err().contains("of its own"),
			"a user budget alone is not enough"
		);
		budgets.replace(vec![
			budget("project", Scope::Project, 100, Period::Month, true),
			budget("everyone", Scope::EveryUser, 1, Period::Day, true),
			budget("team", Scope::Group(team), 2, Period::Day, true),
			budget("alice", Scope::User(alice), 5, Period::Day, true),
		]);
		assert!(budgets.admits(bob).is_ok());
		// Each user spends from their own dollar.
		let bob_spends = budgets.reserve(&user(bob, None), PICO, now).unwrap();
		assert!(budgets.reserve(&user(bob, None), PICO / 10, now).is_err());
		let carol_holds = budgets
			.reserve(&user(carol, Some(team)), PICO, now)
			.unwrap();
		bob_spends.settle(Some(PICO), now);
		// Alice's own budget replaces the default: five dollars, not one.
		let over = budgets
			.reserve(&user(alice, None), PICO * 6, now)
			.unwrap_err();
		assert_eq!(over.budget, "alice");
		assert!(budgets.reserve(&user(alice, None), PICO * 4, now).is_ok());
		assert!(budgets.admits(alice).is_ok());
		// Carol's group has two dollars for all of its members, Carol and Dave hold one and a half.
		let dave = KeyId([5; 16]);
		let dave_holds = budgets
			.reserve(&user(dave, Some(team)), PICO / 2, now)
			.unwrap();
		let over = budgets
			.reserve(&user(KeyId([6; 16]), Some(team)), PICO * 6 / 10, now)
			.unwrap_err();
		assert_eq!(over.budget, "team");
		drop((carol_holds, dave_holds));
	}

	#[test]
	fn room_is_the_tightest_hard_budget_less_what_is_spent_and_held() {
		let budgets = Budgets::default();
		let key = KeyId([7; 16]);
		let now = SystemTime::now();
		assert_eq!(budgets.room(&by(key), now), None);
		budgets.replace(vec![
			budget("soft", Scope::Key(key), 1, Period::Day, false),
			budget("project", Scope::Project, 5, Period::Month, true),
			budget("key", Scope::Key(key), 2, Period::Day, true),
		]);
		assert_eq!(budgets.room(&by(key), now), Some(2 * PICO));
		assert_eq!(budgets.room(&Who::default(), now), Some(5 * PICO));
		let held = budgets.reserve(&by(key), PICO / 2, now).unwrap();
		assert_eq!(budgets.room(&by(key), now), Some(PICO * 3 / 2));
		held.settle(Some(PICO), now);
		assert_eq!(budgets.room(&by(key), now), Some(PICO));
		assert_eq!(budgets.room(&Who::default(), now), Some(4 * PICO));
		budgets.replace(vec![budget("soft", Scope::Project, 1, Period::Day, false)]);
		assert_eq!(
			budgets.room(&Who::default(), now),
			None,
			"a soft budget sets no room"
		);
	}

	#[test]
	fn a_long_reply_cannot_run_through_a_budget() {
		// PS8's case: two calls that each fit alone, in flight together.
		let budgets = Budgets::default();
		budgets.replace(vec![budget(
			"project",
			Scope::Project,
			1,
			Period::Month,
			true,
		)]);
		let now = SystemTime::now();
		let long = budgets
			.reserve(&Who::default(), PICO * 9 / 10, now)
			.unwrap();
		assert!(
			budgets.reserve(&Who::default(), PICO / 5, now).is_err(),
			"the room is taken"
		);
		long.settle(Some(PICO * 9 / 10), now);
		assert!(
			budgets.reserve(&Who::default(), PICO / 5, now).is_err(),
			"and then spent"
		);
		assert!(budgets.reserve(&Who::default(), PICO / 10, now).is_ok());
	}

	#[test]
	fn a_call_with_no_usage_is_charged_its_reservation_and_one_never_sent_nothing() {
		let budgets = Budgets::default();
		budgets.replace(vec![budget("p", Scope::Project, 1, Period::Day, true)]);
		let now = SystemTime::now();
		budgets
			.reserve(&Who::default(), PICO / 2, now)
			.unwrap()
			.settle(None, now);
		drop(budgets.reserve(&Who::default(), PICO / 4, now).unwrap());
		let held = budgets.reserve(&Who::default(), PICO / 2, now).unwrap();
		assert!(
			budgets.reserve(&Who::default(), PICO / 100, now).is_err(),
			"$0.50 charged, $0.50 held"
		);
		drop(held);
	}

	#[test]
	fn a_soft_budget_warns_and_windows_forget_old_days() {
		let budgets = Budgets::default();
		budgets.replace(vec![
			budget("soft", Scope::Project, 1, Period::Rolling(7), false),
			budget("day", Scope::Project, 5, Period::Day, true),
		]);
		let now = SystemTime::now();
		let today = day_of(now);
		budgets.add_spent(Subject::Project, today - 7, PICO * 4);
		budgets.add_spent(Subject::Project, today - 6, PICO);
		let reservation = budgets.reserve(&Who::default(), PICO / 10, now).unwrap();
		assert_eq!(reservation.warnings, vec!["soft".to_string()]);
		drop(reservation);
		let tomorrow = now + Duration::from_secs(SECONDS_PER_DAY);
		assert!(
			budgets
				.reserve(&Who::default(), PICO / 10, tomorrow)
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
			budget("project", Scope::Project, 2, Period::Rolling(2), true),
			budget("key", Scope::Key(key), 1, Period::Day, true),
		]);
		let now = SystemTime::now();
		// The project has spent $1.25 in two days (the day 400 back is past the history), the key $0.75.
		assert!(budgets.reserve(&Who::default(), PICO * 3 / 4, now).is_ok());
		assert!(budgets.reserve(&Who::default(), PICO * 4 / 5, now).is_err());
		assert!(budgets.reserve(&by(key), PICO / 4, now).is_ok());
		assert!(budgets.reserve(&by(key), PICO * 3 / 10, now).is_err());
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
