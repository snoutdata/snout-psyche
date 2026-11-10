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
//!
//! **Spreading (PS17).** Each target has a room: the share of its quota left, the tightest of what
//! the provider last said in its headers and of what `Model::quota` allows against what was sent
//! in the last minute (Bedrock sends no such headers). A `weighted` or `balanced` route, and
//! targets of equal price in a `price` route, draw their order in proportion to it, so traffic
//! moves away as a target fills, before the 429s start. A target with no room left goes after
//! every target with some, in any route.
//!
//! **Sticking (PS18).** A prompt cache lives in one account and region, so a conversation that hit
//! one stays with that target while it is healthy and has room ([`Health::stick`], read by
//! [`Health::arrange`]). A conversation is its `session_id` when it has one, else its first system
//! message and first other message (`conversation`), the two that do not change as it grows.

use std::collections::{HashMap, VecDeque};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value, json};
use tokio_postgres::Client;

use crate::error::Class;
use crate::fits::Quota;
use crate::prefs::Preferred;
use crate::providers::{Model, Sort, list_price};
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
/// How long what a provider said about its quota is believed: its buckets refill as time passes.
const READING_FRESH: Duration = Duration::from_secs(30);
/// Below this share of its quota a target has no room, and goes after those that do.
const NO_ROOM: f64 = 0.02;
/// How long a conversation stays with its target after its last answer (OpenRouter's figure; a
/// provider's cache lives five minutes to an hour from its last use).
const STICKY_IDLE: Duration = Duration::from_secs(600);
/// The conversations remembered; past this the expired go, and if none did, all of them.
const MOST_STICKY: usize = 65_536;
/// The longest conversation key taken from a caller.
pub const LONGEST_SESSION: usize = 256;
/// The smallest share a target's price or health leaves it, so a dear or a shaky target keeps
/// being measured and is warm when it is needed.
const SMALLEST_SHARE: f64 = 0.02;
/// The smallest share speed leaves a target of a `balanced` route.
const SLOWEST_SHARE: f64 = 0.25;
/// Calls a target is assumed to have answered before any were counted, so that one early failure
/// does not take a target's whole share.
const ASSUMED_ANSWERS: f64 = 4.0;

/// The provider's own reading of what is left on Psyche's credential, from its response headers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Remaining {
	pub requests: Option<u64>,
	pub tokens: Option<u64>,
	/// The limits those are counted down from, when the provider says.
	pub requests_limit: Option<u64>,
	pub tokens_limit: Option<u64>,
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
			requests_limit: number(&[
				"x-ratelimit-limit-requests",
				"anthropic-ratelimit-requests-limit",
			]),
			tokens_limit: number(&[
				"x-ratelimit-limit-tokens",
				"anthropic-ratelimit-tokens-limit",
			]),
		}
	}

	/// The share left, the tightest of what it says; 1 when it says nothing.
	fn share(self) -> f64 {
		let mut left: f64 = 1.0;
		for (remaining, limit) in [
			(self.requests, self.requests_limit),
			(self.tokens, self.tokens_limit),
		] {
			match (remaining, limit) {
				(Some(0), _) => left = 0.0,
				(Some(remaining), Some(limit)) if limit > 0 => {
					left = left.min(remaining as f64 / limit as f64);
				}
				_ => {}
			}
		}
		left
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
	/// Prompt and reply together, when counted.
	pub tokens: Option<u64>,
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
			tokens: call.usage.map(|u| u.total()),
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
	/// Attempts begun, for a target with a quota: counted when sent, so concurrent calls see them.
	started: u32,
	/// Tokens used, for a target with a quota.
	tokens: u64,
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
	/// When the provider last said what is left.
	remaining_at: Option<SystemTime>,
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
				started: a.started + c.started,
				tokens: a.tokens + c.tokens,
			})
	}

	/// How much was used in about the last 60 s: this minute's count, and the part of the last
	/// minute's that a sliding window would still hold.
	fn last_minute(&self, now: SystemTime, of: impl Fn(&Counts) -> u64) -> f64 {
		let this_minute = minute(now);
		let elapsed = now
			.duration_since(UNIX_EPOCH)
			.map_or(0, |d| d.as_secs() % 60) as f64
			/ 60.0;
		let at = |m: u64| {
			self.buckets
				.iter()
				.find(|(bucket, _)| *bucket == m)
				.map_or(0.0, |(_, c)| of(c) as f64)
		};
		at(this_minute) + at(this_minute.saturating_sub(1)) * (1.0 - elapsed)
	}

	/// The share of its quota this target has left, 0 to 1, by what the provider last said and by
	/// `quota` against what it was sent.
	fn room(&self, quota: Option<Quota>, now: SystemTime) -> f64 {
		let mut room: f64 = 1.0;
		if self
			.remaining_at
			.is_some_and(|at| now.duration_since(at).map_or(true, |d| d < READING_FRESH))
		{
			room = room.min(self.remaining.share());
		}
		if let Some(quota) = quota {
			if let Some(limit) = quota.requests {
				let used = self.last_minute(now, |c| u64::from(c.started));
				room = room.min(1.0 - used / limit as f64);
			}
			if let Some(limit) = quota.tokens {
				let used = self.last_minute(now, |c| c.tokens);
				room = room.min(1.0 - used / limit as f64);
			}
		}
		room.clamp(0.0, 1.0)
	}

	/// How much of its share the last five minutes leave a target: squared, so that failing half
	/// its calls costs a target three quarters of its traffic.
	fn health_share(&self, now: SystemTime) -> f64 {
		let counts = self.counts(now);
		let failing = f64::from(counts.failed) / (f64::from(counts.requests) + ASSUMED_ANSWERS);
		((1.0 - failing).powi(2)).max(SMALLEST_SHARE)
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

/// Which target a conversation's cache is on.
#[derive(Debug, Clone)]
struct Stuck {
	provider: String,
	model: String,
	last: SystemTime,
}

/// How a call's targets are to be ordered.
#[derive(Debug, Clone, Copy, Default)]
pub struct Plan<'a> {
	pub sort: Option<Sort>,
	/// The route's weights, by model name.
	pub weights: Option<&'a HashMap<String, u32>>,
	/// Which of the names sent each target came from, to sort within each one. `None` sorts the
	/// targets as one list.
	pub groups: Option<&'a HashMap<String, usize>>,
	/// The conversation, from [`conversation`].
	pub sticky: Option<u64>,
	/// The caller's `provider.preferred_*`: a target measured as missing one goes after those that
	/// do not, within its name. A preference, so it moves a target and never removes one.
	pub preferred: Option<Preferred>,
}

/// The targets in the order to try them, and the one a conversation stayed on.
#[derive(Debug, Clone)]
pub struct Arranged {
	pub targets: Vec<Arc<Model>>,
	pub stuck: Option<String>,
	/// The targets put behind the others for missing a preference.
	pub behind: Vec<String>,
}

/// Every target's health. Cheap to clone: the clones share it.
#[derive(Debug, Clone, Default)]
pub struct Health {
	targets: Arc<Mutex<HashMap<(String, String), State>>>,
	sticky: Arc<Mutex<HashMap<u64, Stuck>>>,
	/// The state of the draws routing makes (SplitMix64): cheap, and with no syscall on the path.
	draws: Arc<AtomicU64>,
}

impl Health {
	fn with<T>(&self, f: impl FnOnce(&mut HashMap<(String, String), State>) -> T) -> T {
		f(&mut self.targets.lock().unwrap_or_else(|p| p.into_inner()))
	}

	/// The next number of the routing's draws, in (0, 1].
	fn draw(&self) -> f64 {
		let mut z = self
			.draws
			.fetch_add(0x9e37_79b9_7f4a_7c15, Ordering::Relaxed)
			.wrapping_add(0x9e37_79b9_7f4a_7c15);
		z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
		z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
		z ^= z >> 31;
		((z >> 11) as f64 + 1.0) / (1u64 << 53) as f64
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
		let plan = Plan {
			sort,
			..Plan::default()
		};
		self.arrange(targets, &plan, now).targets
	}

	/// The targets in the order to try them, as [`Health::order`] says, and with `plan`: a route
	/// sorted by price puts the cheapest first and draws the order of those of equal price, and a
	/// `weighted` or `balanced` route draws its whole order. A target with no room left goes after
	/// the ready ones with some, and a conversation's own target, when it is ready and has room,
	/// goes first of all.
	///
	/// **The draw.** A target's chance of being first is its share of the sum, and the order of the
	/// rest follows the same way (weighted sampling without replacement, by key `u^(1/share)`):
	///
	/// `share = route weight × health × room × (price factor × speed factor, for `balanced`)`
	///
	/// - The price factor is the inverse square of the price against the cheapest target, the
	///   weighting OpenRouter documents and the Phase 0 triage adopted: a target three times the
	///   price gets a ninth of the traffic. It is steep on purpose, since `balanced` is still meant
	///   to spend less, and a target of the same price (the same model on several accounts) is left
	///   to health and room alone. It stops at 2% rather than at nothing, so a dear target goes on
	///   being measured and is warm when the cheap one fails. A target nothing prices counts as the
	///   dearest priced one, so an unknown price is never a reason to prefer it.
	/// - The health factor is the square of the share of the last five minutes' calls that were
	///   answered, with four answers assumed. A 30 s window with a yes-or-no answer, as OpenRouter
	///   has, is what the cooldown above already does; this covers the target that fails sometimes
	///   and never three times running, which a cooldown cannot see.
	/// - The room factor is linear, so that traffic is in proportion to headroom (PS17).
	/// - The speed factor is the fastest first byte against this target's, stopping at a quarter,
	///   and is 1 for a target not yet measured, so it gets measured. It is `balanced`'s only
	///   difference in kind from `price`: a route that wants only the fastest says `latency`.
	///
	/// No route changes to this unless it asks: a route written `order` keeps its written order
	/// (with its failures, cooldowns and room honoured), since a primary and a backup are a decision
	/// its owner made.
	pub fn arrange(&self, targets: Vec<Arc<Model>>, plan: &Plan<'_>, now: SystemTime) -> Arranged {
		let draw = || self.draw();
		self.arrange_with(targets, plan, now, draw)
	}

	fn arrange_with(
		&self,
		targets: Vec<Arc<Model>>,
		plan: &Plan<'_>,
		now: SystemTime,
		mut draw: impl FnMut() -> f64,
	) -> Arranged {
		let stuck_on = plan.sticky.and_then(|key| self.stuck_on(key, now));
		self.with(|states| {
			let state = |m: &Model| states.get(&(m.provider.name.clone(), m.name.clone()));
			let group = |m: &Model| {
				plan.groups
					.and_then(|g| g.get(&m.name).copied())
					.unwrap_or(0)
			};
			let room = |m: &Model| state(m).map_or(1.0, |s| s.room(m.quota, now));
			let (ready, cooling): (Vec<_>, Vec<_>) = targets
				.into_iter()
				.partition(|m| !state(m).is_some_and(|s| s.cooling(now)));

			let cheapest = ready
				.iter()
				.filter_map(|m| m.price.map(|_| list_price(m)))
				.min();
			let dearest = ready
				.iter()
				.filter_map(|m| m.price.map(|_| list_price(m)))
				.max();
			let fastest = ready
				.iter()
				.filter_map(|m| state(m).and_then(|s| s.first_byte_ms))
				.fold(None, |least: Option<f64>, ms| {
					Some(least.map_or(ms, |l| l.min(ms)))
				});
			let share = |m: &Model| {
				let written = plan
					.weights
					.and_then(|w| w.get(&m.name).copied())
					.unwrap_or(1);
				let mut share =
					f64::from(written) * state(m).map_or(1.0, |s| s.health_share(now)) * room(m);
				if plan.sort == Some(Sort::Balanced) {
					if let (Some(cheapest), Some(dearest)) = (cheapest, dearest) {
						let price = if m.price.is_some() {
							list_price(m)
						} else {
							dearest
						};
						let factor = (cheapest.max(1) as f64 / price.max(1) as f64).powi(2);
						share *= factor.max(SMALLEST_SHARE);
					}
					if let (Some(fastest), Some(own)) =
						(fastest, state(m).and_then(|s| s.first_byte_ms))
					{
						share *= (fastest / own.max(1.0)).clamp(SLOWEST_SHARE, 1.0);
					}
				}
				share
			};
			// The draw's key: a share of nothing goes after every target with a share.
			let mut key = |m: &Model| {
				let share = share(m);
				if share > 0.0 {
					draw().powf(1.0 / share)
				} else {
					-1.0
				}
			};
			let reading = |m: &Model| match plan.sort {
				Some(Sort::Latency) => state(m).and_then(|s| s.first_byte_ms),
				// Faster is more tokens a second, so it is sorted on the negative.
				Some(Sort::Throughput) => state(m).and_then(|s| s.tokens_per_second).map(|t| -t),
				_ => None,
			};
			let mut ready: Vec<(f64, Arc<Model>)> = ready
				.into_iter()
				.map(|m| {
					let drawn = match plan.sort {
						Some(Sort::Weighted | Sort::Balanced) => key(&m),
						// Only ties in price are drawn; a target with no price keeps its place.
						Some(Sort::Price) if m.price.is_some() => key(&m),
						_ => 0.0,
					};
					(drawn, m)
				})
				.collect();
			// Stable sorts throughout: equal readings, and every unmeasured target, keep the order
			// given.
			match plan.sort {
				Some(Sort::Latency | Sort::Throughput) => {
					ready.sort_by(|(_, a), (_, b)| {
						group(a)
							.cmp(&group(b))
							.then_with(|| match (reading(a), reading(b)) {
								(None, None) => std::cmp::Ordering::Equal,
								(None, Some(_)) => std::cmp::Ordering::Less,
								(Some(_), None) => std::cmp::Ordering::Greater,
								(Some(a), Some(b)) => a.total_cmp(&b),
							})
					});
				}
				Some(Sort::Price) => {
					ready.sort_by(|(x, a), (y, b)| {
						group(a)
							.cmp(&group(b))
							.then_with(|| list_price(a).cmp(&list_price(b)))
							.then_with(|| y.total_cmp(x))
					});
				}
				Some(Sort::Weighted | Sort::Balanced) => {
					ready.sort_by(|(x, a), (y, b)| {
						group(a).cmp(&group(b)).then_with(|| y.total_cmp(x))
					});
				}
				_ => {}
			}
			let mut ready: Vec<Arc<Model>> = ready.into_iter().map(|(_, m)| m).collect();
			let mut behind = Vec::new();
			if let Some(preferred) = plan.preferred {
				let misses = |m: &Model| {
					state(m).is_some_and(|s| preferred.misses(s.first_byte_ms, s.tokens_per_second))
				};
				// Stable, so each side keeps the order it was drawn or sorted in.
				ready.sort_by_key(|m| (group(m), misses(m)));
				behind = ready
					.iter()
					.filter(|m| misses(m))
					.map(|m| m.name.clone())
					.collect();
			}
			let (mut out, no_room): (Vec<_>, Vec<_>) =
				ready.into_iter().partition(|m| room(m) >= NO_ROOM);
			let stuck = stuck_on.and_then(|(provider, model)| {
				let at = out
					.iter()
					.position(|m| m.provider.name == provider && m.name == model)?;
				let target = out.remove(at);
				let name = target.name.clone();
				out.insert(0, target);
				Some(name)
			});
			out.extend(no_room);
			out.extend(cooling);
			Arranged {
				targets: out,
				stuck,
				behind,
			}
		})
	}

	/// The target a conversation's cache is on, while it has not gone idle.
	fn stuck_on(&self, key: u64, now: SystemTime) -> Option<(String, String)> {
		let sticky = self.sticky.lock().unwrap_or_else(|p| p.into_inner());
		sticky
			.get(&key)
			.filter(|s| now.duration_since(s.last).map_or(true, |d| d < STICKY_IDLE))
			.map(|s| (s.provider.clone(), s.model.clone()))
	}

	/// A call of a conversation was answered by this target. `proven` is whether that is a reason to
	/// stay: a cache hit was seen, or the caller named the conversation. An answer from the target
	/// it is already on keeps it there either way.
	pub fn stick(&self, key: u64, provider: &str, model: &str, proven: bool, now: SystemTime) {
		let mut sticky = self.sticky.lock().unwrap_or_else(|p| p.into_inner());
		if let Some(stuck) = sticky.get_mut(&key)
			&& stuck.provider == provider
			&& stuck.model == model
		{
			stuck.last = now;
			return;
		}
		if !proven {
			return;
		}
		if sticky.len() >= MOST_STICKY {
			sticky.retain(|_, s| now.duration_since(s.last).map_or(true, |d| d < STICKY_IDLE));
			if sticky.len() >= MOST_STICKY {
				sticky.clear();
			}
		}
		sticky.insert(
			key,
			Stuck {
				provider: provider.to_string(),
				model: model.to_string(),
				last: now,
			},
		);
	}

	/// Forgets every conversation, for a test that measures routing without them.
	#[cfg(test)]
	pub fn forget_conversations(&self) {
		self.sticky
			.lock()
			.unwrap_or_else(|p| p.into_inner())
			.clear();
	}

	/// An attempt is about to be sent to `model`: if its cooldown is over, this attempt is its
	/// probe, and a target with a quota counts it now, so calls in flight together see each other.
	pub fn begin(&self, model: &Model, now: SystemTime) {
		self.with(|states| {
			if model.quota.is_some() {
				states
					.entry((model.provider.name.clone(), model.name.clone()))
					.or_default()
					.count(now, |c| c.started += 1);
			}
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
				state.remaining_at = Some(now);
			}
			if let Some(tokens) = seen.tokens {
				state.count(now, |c| c.tokens += tokens);
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

/// One conversation's key: what the caller named it, or else the first system message and the
/// first other message, the two that stay the same as it grows (OpenRouter's key, from the Phase 0
/// triage). `cache_control` is not part of it, since a client moves its breakpoints between turns.
/// `None` when there is nothing to tell it by. `named` is what the call asked for, so two routes
/// never share a conversation.
pub fn conversation(
	body: &Map<String, Value>,
	named: &str,
	session: Option<&str>,
) -> Option<(u64, bool)> {
	let mut hasher = DefaultHasher::new();
	named.hash(&mut hasher);
	let session = session.or_else(|| body.get("prompt_cache_key").and_then(Value::as_str));
	if let Some(session) = session {
		session.hash(&mut hasher);
		return Some((hasher.finish(), true));
	}
	// The Responses API's conversation is its `input`, and its system prompt may be `instructions`.
	let messages = body
		.get("messages")
		.or_else(|| body.get("input"))
		.and_then(Value::as_array)?;
	let is_system = |m: &&Value| {
		matches!(
			m.get("role").and_then(Value::as_str),
			Some("system" | "developer")
		)
	};
	let first = messages.iter().find(|m| !is_system(m))?;
	if let Some(system) = messages.iter().find(is_system) {
		hash_value(system, &mut hasher);
	}
	if let Some(instructions) = body.get("instructions") {
		hash_value(instructions, &mut hasher);
	}
	hash_value(first, &mut hasher);
	Some((hasher.finish(), false))
}

/// Feeds a JSON value to `hasher`, leaving out any `cache_control`. No key order to normalize: a
/// client sends a conversation's first messages the same way each time.
fn hash_value(value: &Value, hasher: &mut DefaultHasher) {
	match value {
		Value::Null => 0u8.hash(hasher),
		Value::Bool(b) => b.hash(hasher),
		Value::Number(n) => n.to_string().hash(hasher),
		Value::String(s) => s.hash(hasher),
		Value::Array(items) => {
			items.len().hash(hasher);
			for item in items {
				hash_value(item, hasher);
			}
		}
		Value::Object(fields) => {
			for (key, field) in fields.iter().filter(|(k, _)| *k != "cache_control") {
				key.hash(hasher);
				hash_value(field, hasher);
			}
		}
	}
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
			tokens: None,
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
			tokens: None,
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
			..Remaining::default()
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

	/// Two accounts' worth of one model, with a price and a quota of their own.
	fn spread_models() -> Vec<Arc<Model>> {
		let spec = Spec::from_json(
			r#"{"providers": [
				{"name": "acct-a", "kind": "bedrock", "region": "us-west-2", "credential": "a"},
				{"name": "acct-b", "kind": "bedrock", "region": "us-east-1", "credential": "b"}
			], "models": [
				{"name": "near", "provider": "acct-a", "upstream": "m", "quota": {"requests": 100},
					"price": {"input": 3, "output": 15, "cache_read": 0.3}},
				{"name": "far", "provider": "acct-b", "upstream": "m", "quota": {"requests": 100},
					"price": {"input": 3, "output": 15, "cache_read": 0.3}}
			]}"#,
			&HashMap::new(),
			crate::net::Egress::Public,
		)
		.unwrap();
		Catalogue::build(&spec).unwrap().models().to_vec()
	}

	/// How many of 2000 draws put `name` first.
	fn firsts(
		health: &Health,
		models: &[Arc<Model>],
		plan: &Plan<'_>,
		now: SystemTime,
		name: &str,
	) -> u32 {
		(0..2000)
			.filter(|_| health.arrange(models.to_vec(), plan, now).targets[0].name == name)
			.count() as u32
	}

	#[test]
	fn traffic_shifts_off_a_credential_near_its_quota_before_any_429() {
		let health = Health::default();
		let models = spread_models();
		// The start of a minute, so nothing of the last one is in the count.
		let t0 = UNIX_EPOCH + Duration::from_secs(1_800_000_000 / 60 * 60);
		let plan = Plan {
			sort: Some(Sort::Balanced),
			..Plan::default()
		};
		// Level at first: about half each.
		let even = firsts(&health, &models, &plan, t0, "near");
		assert!((800..=1200).contains(&even), "{even} of 2000");

		// The first account is 90% through its quota and has had no 429: its share follows its
		// room, about a tenth against the other's whole.
		for _ in 0..90 {
			health.begin(&models[0], t0);
		}
		let filling = firsts(&health, &models, &plan, t0, "near");
		assert!(
			(90..=320).contains(&filling),
			"{filling} of 2000 with a tenth of its quota left"
		);

		// Full: it goes behind the other in any route, a written order included.
		for _ in 0..10 {
			health.begin(&models[0], t0);
		}
		assert_eq!(firsts(&health, &models, &plan, t0, "near"), 0);
		assert_eq!(
			names(&health.order(models.clone(), Some(Sort::Order), t0)),
			["far", "near"]
		);
		// A target out of room is still tried when nothing else is left.
		for _ in 0..100 {
			health.begin(&models[1], t0);
		}
		assert_eq!(health.order(models.clone(), Some(Sort::Order), t0).len(), 2);

		// The next minute it is coming back, as the sliding window lets go of the last one.
		let later = t0 + Duration::from_secs(90);
		assert!(firsts(&health, &models, &plan, later, "near") > 0);
	}

	#[test]
	fn what_a_provider_says_is_left_moves_traffic_while_it_is_fresh() {
		let health = Health::default();
		let models = spread_models();
		let t0 = UNIX_EPOCH + Duration::from_secs(1_800_000_000);
		let plan = Plan {
			sort: Some(Sort::Weighted),
			..Plan::default()
		};
		let mut reading = answered(100, 10, 200);
		reading.remaining = Remaining {
			requests: Some(5),
			requests_limit: Some(1000),
			..Remaining::default()
		};
		health.seen("acct-a", "near", reading, t0);
		// Half a percent left: under the line, so behind the other.
		assert_eq!(firsts(&health, &models, &plan, t0, "near"), 0);
		// Thirty seconds on, the reading says nothing any more.
		let later = t0 + READING_FRESH;
		let back = firsts(&health, &models, &plan, later, "near");
		assert!((800..=1200).contains(&back), "{back}");
	}

	#[test]
	fn weights_set_the_shares_of_a_weighted_route() {
		let health = Health::default();
		let models = spread_models();
		let t0 = UNIX_EPOCH + Duration::from_secs(1_800_000_000);
		let weights = HashMap::from([("near".to_string(), 3)]);
		let plan = Plan {
			sort: Some(Sort::Weighted),
			weights: Some(&weights),
			..Plan::default()
		};
		let near = firsts(&health, &models, &plan, t0, "near");
		assert!((1350..=1650).contains(&near), "{near} of 2000 at 3 to 1");
	}

	#[test]
	fn balanced_prefers_the_cheap_by_the_inverse_square_of_price_and_the_healthy() {
		let health = Health::default();
		let spec = Spec::from_json(
			r#"{"providers": [{"name": "p", "kind": "openai", "base_url": "https://p.example/v1"}],
				"models": [
					{"name": "cheap", "provider": "p", "price": {"input": 1, "output": 1}},
					{"name": "dear", "provider": "p", "price": {"input": 3, "output": 3}}
				]}"#,
			&HashMap::new(),
			crate::net::Egress::Public,
		)
		.unwrap();
		let models = Catalogue::build(&spec).unwrap().models().to_vec();
		let t0 = UNIX_EPOCH + Duration::from_secs(1_800_000_000);
		let plan = Plan {
			sort: Some(Sort::Balanced),
			..Plan::default()
		};
		// Three times the price: about one call in ten is the dear one's (a ninth against a whole).
		let dear = firsts(&health, &models, &plan, t0, "dear");
		assert!((120..=300).contains(&dear), "{dear} of 2000");

		// The cheap one now fails half its calls (never three in a row): the dear one's share rises.
		for _ in 0..20 {
			health.seen("p", "cheap", failed(Class::Upstream), t0);
			health.seen("p", "cheap", failed(Class::Upstream), t0);
			health.seen("p", "cheap", answered(100, 1, 200), t0);
			health.seen("p", "cheap", answered(100, 1, 200), t0);
		}
		let shaken = firsts(&health, &models, &plan, t0, "dear");
		assert!(shaken > dear + 200, "{shaken} against {dear}");
	}

	#[test]
	fn a_price_route_draws_only_among_equal_prices() {
		let health = Health::default();
		let spec = Spec::from_json(
			r#"{"providers": [{"name": "p", "kind": "openai", "base_url": "https://p.example/v1"}],
				"models": [
					{"name": "a", "provider": "p", "price": {"input": 1, "output": 1}},
					{"name": "b", "provider": "p", "price": {"input": 1, "output": 1}},
					{"name": "c", "provider": "p", "price": {"input": 2, "output": 1}},
					{"name": "unpriced-1", "provider": "p"},
					{"name": "unpriced-2", "provider": "p"}
				]}"#,
			&HashMap::new(),
			crate::net::Egress::Public,
		)
		.unwrap();
		let models = Catalogue::build(&spec).unwrap().models().to_vec();
		let t0 = UNIX_EPOCH + Duration::from_secs(1_800_000_000);
		let mut first_a = 0;
		for _ in 0..1000 {
			let order = health.order(models.clone(), Some(Sort::Price), t0);
			let order = names(&order);
			assert_eq!(order[2..], ["c", "unpriced-1", "unpriced-2"]);
			first_a += u32::from(order[0] == "a");
		}
		assert!((350..=650).contains(&first_a), "{first_a} of 1000");
	}

	#[test]
	fn a_conversation_stays_on_its_cache_while_it_is_healthy_and_has_room() {
		let health = Health::default();
		let models = spread_models();
		let t0 = UNIX_EPOCH + Duration::from_secs(1_800_000_000 / 60 * 60);
		let plan = Plan {
			sort: Some(Sort::Balanced),
			sticky: Some(7),
			..Plan::default()
		};
		// An answer proves nothing until a cache hit is seen or the caller named the conversation.
		health.stick(7, "acct-a", "near", false, t0);
		assert!(health.arrange(models.clone(), &plan, t0).stuck.is_none());
		health.stick(7, "acct-a", "near", true, t0);
		for _ in 0..200 {
			let arranged = health.arrange(models.clone(), &plan, t0);
			assert_eq!(arranged.targets[0].name, "near");
			assert_eq!(arranged.stuck.as_deref(), Some("near"));
		}

		// Cooling, it gives way, and comes back when the cooldown has ended and the target answered.
		for _ in 0..3 {
			health.seen("acct-a", "near", failed(Class::Upstream), t0);
		}
		assert!(health.arrange(models.clone(), &plan, t0).stuck.is_none());
		health.seen("acct-a", "near", answered(100, 1, 200), t0);
		assert_eq!(
			health.arrange(models.clone(), &plan, t0).stuck.as_deref(),
			Some("near")
		);

		// Full, it gives way too.
		for _ in 0..100 {
			health.begin(&models[0], t0);
		}
		let arranged = health.arrange(models.clone(), &plan, t0);
		assert_eq!(arranged.targets[0].name, "far");
		assert!(arranged.stuck.is_none());

		// An answer from its own target keeps it for another ten minutes; after that it is gone.
		let healthy = Health::default();
		healthy.stick(7, "acct-b", "far", true, t0);
		healthy.stick(
			7,
			"acct-b",
			"far",
			false,
			t0 + STICKY_IDLE - Duration::from_secs(1),
		);
		let plan = Plan {
			sticky: Some(7),
			..Plan::default()
		};
		assert!(
			healthy
				.arrange(
					models.clone(),
					&plan,
					t0 + STICKY_IDLE + Duration::from_secs(1)
				)
				.stuck
				.is_some()
		);
		assert!(
			healthy
				.arrange(models.clone(), &plan, t0 + STICKY_IDLE * 2)
				.stuck
				.is_none()
		);
	}

	#[test]
	fn a_conversation_is_known_by_its_first_messages_or_its_name() {
		let key = |body: Value| {
			conversation(body.as_object().unwrap(), "route", None).map(|(key, named)| {
				assert!(!named);
				key
			})
		};
		let first = key(json!({"messages": [
			{"role": "system", "content": "be brief"},
			{"role": "user", "content": "hello", "cache_control": {"type": "ephemeral"}},
		]}));
		// A later turn of the same conversation, its breakpoint moved along.
		let later = key(json!({"messages": [
			{"role": "system", "content": "be brief"},
			{"role": "user", "content": "hello"},
			{"role": "assistant", "content": "hi"},
			{"role": "user", "content": "more", "cache_control": {"type": "ephemeral"}},
		]}));
		assert_eq!(first, later);
		assert!(first.is_some());
		assert_ne!(
			first,
			key(json!({"messages": [
				{"role": "system", "content": "be brief"},
				{"role": "user", "content": "goodbye"},
			]}))
		);
		assert_eq!(key(json!({"prompt": "no messages"})), None);

		let body = json!({"messages": [{"role": "user", "content": "hello"}]});
		let body = body.as_object().unwrap();
		let session = conversation(body, "route", Some("s-1")).unwrap();
		assert!(session.1);
		assert_ne!(
			session.0,
			conversation(body, "route", Some("s-2")).unwrap().0
		);
		assert_ne!(
			session.0,
			conversation(body, "other", Some("s-1")).unwrap().0
		);
		let by_cache_key = json!({"prompt_cache_key": "k", "messages": []});
		assert!(
			conversation(by_cache_key.as_object().unwrap(), "route", None)
				.unwrap()
				.1
		);
	}

	#[test]
	fn names_sent_keep_their_order_when_sorted_within_each() {
		let health = Health::default();
		let all = models();
		let t0 = UNIX_EPOCH + Duration::from_secs(1_800_000_000);
		health.seen("p", "a", answered(900, 100, 1900), t0);
		health.seen("p", "b", answered(500, 100, 1500), t0);
		health.seen("p", "c", answered(100, 100, 1100), t0);
		let groups = HashMap::from([
			("a".to_string(), 0),
			("b".to_string(), 0),
			("c".to_string(), 1),
		]);
		let within = Plan {
			sort: Some(Sort::Latency),
			groups: Some(&groups),
			..Plan::default()
		};
		assert_eq!(
			names(&health.arrange(all.clone(), &within, t0).targets),
			["b", "a", "c"]
		);
		let as_one = Plan {
			sort: Some(Sort::Latency),
			..Plan::default()
		};
		assert_eq!(
			names(&health.arrange(all, &as_one, t0).targets),
			["c", "b", "a"]
		);
	}

	#[test]
	fn a_preference_moves_a_target_that_misses_it_behind_the_rest_and_never_removes_it() {
		let health = Health::default();
		let all = models();
		let t0 = UNIX_EPOCH + Duration::from_secs(1_800_000_000);
		let seen = |provider: &str, name: &str, ms: u64, tokens: u64| {
			health.seen(
				provider,
				name,
				Seen {
					outcome: Outcome::Completed,
					first_byte: Duration::from_millis(ms),
					total: Duration::from_millis(ms + 1000),
					streamed: true,
					output_tokens: Some(tokens),
					tokens: Some(tokens),
					retry_after: None,
					remaining: Remaining::default(),
				},
				t0,
			)
		};
		let (a, b, c) = (&all[0], &all[1], &all[2]);
		seen(&a.provider.name, &a.name, 5000, 500); // slow
		seen(&b.provider.name, &b.name, 100, 500); // fast
		let preferred = |value: Value| crate::prefs::Prefs::parse(value).unwrap().preferred();
		let plan = Plan {
			preferred: preferred(json!({"preferred_max_latency": 1})),
			..Plan::default()
		};
		let arranged = health.arrange(all.clone(), &plan, t0);
		assert_eq!(
			names(&arranged.targets),
			[b.name.as_str(), c.name.as_str(), a.name.as_str()],
			"the one measured slow goes last, the unmeasured keeps its place among the rest"
		);
		assert_eq!(arranged.behind, vec![a.name.clone()]);
		// Nobody misses: nothing moves.
		let easy = Plan {
			preferred: preferred(json!({"preferred_max_latency": 60})),
			..Plan::default()
		};
		assert_eq!(
			names(&health.arrange(all.clone(), &easy, t0).targets),
			[a.name.as_str(), b.name.as_str(), c.name.as_str()]
		);
		// Tokens a second, the other measurement.
		let throughput = Plan {
			preferred: preferred(json!({"preferred_min_throughput": 1000})),
			..Plan::default()
		};
		let arranged = health.arrange(all.clone(), &throughput, t0);
		assert_eq!(
			names(&arranged.targets).len(),
			3,
			"a miss never removes a target"
		);
		assert_eq!(arranged.targets.last().unwrap().name, b.name);
		assert_eq!(arranged.behind, [a.name.clone(), b.name.clone()]);

		// Within each name sent: a miss in the first stays ahead of the second name's.
		let groups = HashMap::from([
			(a.name.clone(), 0),
			(b.name.clone(), 0),
			(c.name.clone(), 1),
		]);
		let within = Plan {
			groups: Some(&groups),
			preferred: preferred(json!({"preferred_max_latency": 1})),
			..Plan::default()
		};
		assert_eq!(
			names(&health.arrange(all.clone(), &within, t0).targets),
			[b.name.as_str(), a.name.as_str(), c.name.as_str()]
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
				tokens: Some(80000),
				..Remaining::default()
			}
		);
		assert_eq!(
			Remaining::read(&reqwest::header::HeaderMap::new()),
			Remaining::default()
		);
	}
}
