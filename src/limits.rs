//! Rate limits: requests and tokens a minute, for the project, one key, each signed-in user, one
//! user or a group (`who.rs`), as token buckets in memory (PS7: one process per project, so memory
//! is the whole truth). A limit for every user keeps a bucket per user, and a bucket that has
//! filled up again is dropped, since a missing bucket starts full.
//!
//! A limit holds two buckets, each as full as its per-minute number and refilled continuously. A
//! call takes one request and its token bound (the same worst case a budget reserves: the prompt's
//! bytes and the reply's cap) from every limit in its scope, or is refused with how long to wait;
//! when it ends, the tokens it did not use go back. A call whose bound is more than a limit's whole
//! minute could never fit, so it is refused with that sentence rather than told to wait.
//!
//! Every response to a call a limit applies to carries OpenAI's headers for the tightest one:
//! `x-ratelimit-limit-requests`, `-remaining-requests`, `-reset-requests` and the same for tokens.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::keys::KeyId;
use crate::who::{Scope, Subject, UserId, Who};

#[derive(Debug, Clone, PartialEq)]
pub struct Limit {
	pub id: KeyId,
	pub name: String,
	pub scope: Scope,
	pub requests_per_minute: Option<u32>,
	pub tokens_per_minute: Option<u32>,
}

#[derive(Debug, Clone, Copy)]
struct Bucket {
	requests: f64,
	tokens: f64,
	at: Instant,
}

impl Bucket {
	fn full(limit: &Limit, now: Instant) -> Self {
		Self {
			requests: limit.requests_per_minute.map_or(0.0, f64::from),
			tokens: limit.tokens_per_minute.map_or(0.0, f64::from),
			at: now,
		}
	}

	fn refill(&mut self, limit: &Limit, now: Instant) {
		let minutes = now.saturating_duration_since(self.at).as_secs_f64() / 60.0;
		self.at = now;
		if let Some(rpm) = limit.requests_per_minute {
			self.requests = (self.requests + minutes * f64::from(rpm)).min(f64::from(rpm));
		}
		if let Some(tpm) = limit.tokens_per_minute {
			self.tokens = (self.tokens + minutes * f64::from(tpm)).min(f64::from(tpm));
		}
	}
}

/// How often buckets that have filled up again are dropped.
const SWEEP_EVERY: Duration = Duration::from_secs(60);

#[derive(Default)]
struct State {
	limits: Vec<Limit>,
	/// The users a limit names one by one, whom the every-user limits leave alone.
	overridden: HashSet<UserId>,
	/// Per limit and per what it holds: one for most, one per user for an every-user limit.
	buckets: HashMap<(KeyId, Subject), Bucket>,
	swept: Option<Instant>,
}

impl State {
	/// Drops the buckets that are full again: a missing bucket starts full, so nothing is lost.
	fn sweep(&mut self, now: Instant) {
		if self
			.swept
			.is_some_and(|at| now.saturating_duration_since(at) < SWEEP_EVERY)
		{
			return;
		}
		self.swept = Some(now);
		let State {
			limits, buckets, ..
		} = self;
		buckets.retain(|(id, _), bucket| {
			let Some(limit) = limits.iter().find(|l| l.id == *id) else {
				return false;
			};
			let mut later = *bucket;
			later.refill(limit, now);
			let full = |have: f64, per_minute: Option<u32>| {
				per_minute.is_none_or(|n| have >= f64::from(n))
			};
			!(full(later.requests, limit.requests_per_minute)
				&& full(later.tokens, limit.tokens_per_minute))
		});
	}
}

/// Why a call was refused, and when it could be tried again.
#[derive(Debug, Clone, PartialEq)]
pub struct Refused {
	pub message: String,
	/// `None` when waiting would not help.
	pub retry_after: Option<Duration>,
}

/// What the response tells the caller: the tightest limit, as OpenAI words it.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Headers(pub Vec<(&'static str, String)>);

#[derive(Clone, Default)]
pub struct Limits(Arc<Mutex<State>>);

impl std::fmt::Debug for Limits {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.write_str("Limits(..)")
	}
}

/// How long until a bucket short of `need` has it, at `per_minute`.
fn wait(have: f64, need: f64, per_minute: u32) -> Duration {
	Duration::from_secs_f64(((need - have).max(0.0) * 60.0 / f64::from(per_minute)).max(0.0))
}

/// OpenAI's way of writing a reset: `1s`, `6m0s`, `0.5s`.
fn reset(d: Duration) -> String {
	let secs = d.as_secs_f64();
	if secs < 60.0 {
		let rounded = (secs * 10.0).ceil() / 10.0;
		if rounded.fract() == 0.0 {
			return format!("{}s", rounded as u64);
		}
		return format!("{rounded:.1}s");
	}
	let whole = secs.ceil() as u64;
	format!("{}m{}s", whole / 60, whole % 60)
}

impl Limits {
	fn state(&self) -> std::sync::MutexGuard<'_, State> {
		self.0.lock().unwrap_or_else(|e| e.into_inner())
	}

	/// The limits from `psyche.limits`, keeping the buckets of those that did not change (a changed
	/// limit's buckets go, and start full).
	pub fn replace(&self, limits: Vec<Limit>) {
		let mut state = self.state();
		let unchanged: HashSet<KeyId> = limits
			.iter()
			.filter(|limit| state.limits.iter().any(|old| old == *limit))
			.map(|limit| limit.id)
			.collect();
		state.buckets.retain(|(id, _), _| unchanged.contains(id));
		state.overridden = crate::who::overridden(limits.iter().map(|l| l.scope));
		state.limits = limits;
	}

	pub fn applies(&self, who: &Who) -> bool {
		let state = self.state();
		state
			.limits
			.iter()
			.any(|l| l.scope.subject(who, &state.overridden).is_some())
	}

	/// Takes one request and `tokens` from every limit on a call by `who`, or none of them.
	pub fn take(&self, who: &Who, tokens: u64, now: Instant) -> Result<(Taken, Headers), Refused> {
		let mut state = self.state();
		state.sweep(now);
		let State {
			limits,
			buckets,
			overridden,
			..
		} = &mut *state;
		let scope: Vec<(&Limit, (KeyId, Subject))> = limits
			.iter()
			.filter_map(|l| Some((l, (l.id, l.scope.subject(who, overridden)?))))
			.collect();
		let need = tokens as f64;
		let mut longest: Option<(Duration, &Limit, &'static str)> = None;
		for (limit, at) in &scope {
			let bucket = buckets
				.entry(*at)
				.or_insert_with(|| Bucket::full(limit, now));
			bucket.refill(limit, now);
			if let Some(tpm) = limit.tokens_per_minute
				&& tokens > u64::from(tpm)
			{
				return Err(Refused {
					message: format!(
						"This call could use up to {tokens} tokens, more than the limit {} allows in a minute ({tpm}). Ask for fewer with max_tokens, or send less.",
						limit.name
					),
					retry_after: None,
				});
			}
			let short = [
				limit
					.requests_per_minute
					.filter(|_| bucket.requests < 1.0)
					.map(|rpm| (wait(bucket.requests, 1.0, rpm), "requests")),
				limit
					.tokens_per_minute
					.filter(|_| bucket.tokens < need)
					.map(|tpm| (wait(bucket.tokens, need, tpm), "tokens")),
			];
			for (after, what) in short.into_iter().flatten() {
				if longest.is_none_or(|(d, _, _)| after > d) {
					longest = Some((after, limit, what));
				}
			}
		}
		if let Some((after, limit, what)) = longest {
			let per_minute = match what {
				"requests" => limit.requests_per_minute,
				_ => limit.tokens_per_minute,
			}
			.unwrap_or(0);
			return Err(Refused {
				message: format!(
					"The limit {} ({per_minute} {what} a minute) is used up; try again in {}.",
					limit.name,
					reset(after)
				),
				retry_after: Some(after),
			});
		}
		let mut headers = Headers::default();
		let mut tightest_requests: Option<(f64, u32)> = None;
		let mut tightest_tokens: Option<(f64, u32)> = None;
		for (limit, at) in &scope {
			let bucket = buckets.get_mut(at).expect("refilled above");
			if let Some(rpm) = limit.requests_per_minute {
				bucket.requests -= 1.0;
				if tightest_requests.is_none_or(|(left, _)| bucket.requests < left) {
					tightest_requests = Some((bucket.requests, rpm));
				}
			}
			if let Some(tpm) = limit.tokens_per_minute {
				bucket.tokens -= need;
				if tightest_tokens.is_none_or(|(left, _)| bucket.tokens < left) {
					tightest_tokens = Some((bucket.tokens, tpm));
				}
			}
		}
		for (which, tightest) in [("requests", tightest_requests), ("tokens", tightest_tokens)] {
			if let Some((left, per_minute)) = tightest {
				let (limit, remaining, reset_name) = match which {
					"requests" => (
						"x-ratelimit-limit-requests",
						"x-ratelimit-remaining-requests",
						"x-ratelimit-reset-requests",
					),
					_ => (
						"x-ratelimit-limit-tokens",
						"x-ratelimit-remaining-tokens",
						"x-ratelimit-reset-tokens",
					),
				};
				headers.0.push((limit, per_minute.to_string()));
				headers
					.0
					.push((remaining, (left.max(0.0).floor() as u64).to_string()));
				headers.0.push((
					reset_name,
					reset(wait(left, f64::from(per_minute), per_minute)),
				));
			}
		}
		Ok((
			Taken {
				limits: self.clone(),
				held: scope.iter().map(|(_, at)| *at).collect(),
				tokens,
			},
			headers,
		))
	}
}

/// What a call took, so the tokens it did not use can be given back.
#[derive(Debug)]
pub struct Taken {
	limits: Limits,
	held: Vec<(KeyId, Subject)>,
	tokens: u64,
}

impl Taken {
	/// Gives back what the call took beyond the `used` tokens it reported. A call that reported
	/// nothing keeps what it took.
	pub fn settle(self, used: Option<u64>) {
		let Some(used) = used else {
			return;
		};
		let back = self.tokens.saturating_sub(used) as f64;
		if back == 0.0 {
			return;
		}
		let mut state = self.limits.state();
		let State {
			limits, buckets, ..
		} = &mut *state;
		for at in &self.held {
			let Some(tpm) = limits
				.iter()
				.find(|l| l.id == at.0)
				.and_then(|l| l.tokens_per_minute)
			else {
				continue;
			};
			if let Some(bucket) = buckets.get_mut(at) {
				bucket.tokens = (bucket.tokens + back).min(f64::from(tpm));
			}
		}
	}
}

/// Every limit, from `psyche.limits`.
pub async fn load(client: &tokio_postgres::Client) -> Result<Vec<Limit>, tokio_postgres::Error> {
	let rows = client
		.query(
			"SELECT id::text, name, key_id::text, requests_per_minute, tokens_per_minute, every_user, \
			 user_id::text, group_id::text FROM psyche.limits",
			&[],
		)
		.await?;
	Ok(rows
		.iter()
		.filter_map(|row| {
			Some(Limit {
				id: KeyId::parse(row.get(0))?,
				name: row.get(1),
				scope: Scope::from_row(row.get(2), row.get(5), row.get(6), row.get(7))?,
				requests_per_minute: row
					.get::<_, Option<i32>>(3)
					.and_then(|n| u32::try_from(n).ok()),
				tokens_per_minute: row
					.get::<_, Option<i32>>(4)
					.and_then(|n| u32::try_from(n).ok()),
			})
		})
		.collect())
}

#[cfg(test)]
mod tests {
	use super::*;

	fn by(key: KeyId) -> Who {
		Who {
			key: Some(key),
			..Who::default()
		}
	}

	fn limit(n: u8, key: Option<KeyId>, rpm: Option<u32>, tpm: Option<u32>) -> Limit {
		Limit {
			id: KeyId([n; 16]),
			name: format!("l{n}"),
			scope: key.map_or(Scope::Project, Scope::Key),
			requests_per_minute: rpm,
			tokens_per_minute: tpm,
		}
	}

	#[test]
	fn requests_run_out_and_come_back_with_time() {
		let limits = Limits::default();
		limits.replace(vec![limit(1, None, Some(60), None)]);
		let start = Instant::now();
		for _ in 0..60 {
			limits.take(&Who::default(), 0, start).unwrap();
		}
		let refused = limits.take(&Who::default(), 0, start).unwrap_err();
		assert_eq!(refused.retry_after, Some(Duration::from_secs(1)));
		assert!(
			refused.message.contains("60 requests a minute"),
			"{}",
			refused.message
		);
		let (_, headers) = limits
			.take(&Who::default(), 0, start + Duration::from_secs(1))
			.unwrap();
		assert!(
			headers
				.0
				.contains(&("x-ratelimit-remaining-requests", "0".into()))
		);
		assert!(
			headers
				.0
				.contains(&("x-ratelimit-limit-requests", "60".into()))
		);
	}

	#[test]
	fn tokens_are_taken_at_their_bound_and_the_unused_given_back() {
		let limits = Limits::default();
		limits.replace(vec![limit(1, None, None, Some(1000))]);
		let now = Instant::now();
		let (taken, headers) = limits.take(&Who::default(), 800, now).unwrap();
		assert!(
			headers
				.0
				.contains(&("x-ratelimit-remaining-tokens", "200".into()))
		);
		assert!(limits.take(&Who::default(), 300, now).is_err());
		taken.settle(Some(100));
		assert!(
			limits.take(&Who::default(), 300, now).is_ok(),
			"700 came back"
		);
		let never = limits.take(&Who::default(), 1001, now).unwrap_err();
		assert_eq!(never.retry_after, None, "waiting would not help");
	}

	#[test]
	fn a_keys_limit_holds_only_that_key_and_a_refusal_takes_nothing() {
		let key = KeyId([9; 16]);
		let limits = Limits::default();
		limits.replace(vec![
			limit(1, Some(key), Some(1), None),
			limit(2, None, Some(100), None),
		]);
		let now = Instant::now();
		limits.take(&by(key), 0, now).unwrap();
		assert!(limits.take(&by(key), 0, now).is_err());
		let (_, headers) = limits.take(&Who::default(), 0, now).unwrap();
		// The project's bucket lost two: the key's call and this one; the refused call took nothing.
		assert!(
			headers
				.0
				.contains(&("x-ratelimit-remaining-requests", "98".into())),
			"{headers:?}"
		);
	}

	#[test]
	fn every_user_has_a_bucket_of_their_own_and_full_ones_are_dropped() {
		let limits = Limits::default();
		let (alice, bob) = (KeyId([1; 16]), KeyId([2; 16]));
		let user = |id: KeyId| Who {
			user: Some(id),
			..Who::default()
		};
		limits.replace(vec![
			Limit {
				scope: Scope::EveryUser,
				..limit(1, None, Some(1), None)
			},
			Limit {
				scope: Scope::User(alice),
				..limit(2, None, Some(2), None)
			},
		]);
		let now = Instant::now();
		assert!(!limits.applies(&Who::default()));
		limits.take(&user(bob), 0, now).unwrap();
		assert!(limits.take(&user(bob), 0, now).is_err(), "one a minute");
		// Alice has two of her own, and the every-user limit leaves her alone.
		limits.take(&user(alice), 0, now).unwrap();
		limits.take(&user(alice), 0, now).unwrap();
		assert!(limits.take(&user(alice), 0, now).is_err());
		assert!(limits.take(&user(KeyId([3; 16])), 0, now).is_ok());
		assert_eq!(limits.state().buckets.len(), 3);
		// A minute on, every bucket is full again and the next take sweeps them away.
		let later = now + Duration::from_secs(61);
		limits.take(&user(bob), 0, later).unwrap();
		assert_eq!(limits.state().buckets.len(), 1, "only Bob's, just used");
	}

	#[test]
	fn a_reload_keeps_the_buckets_of_unchanged_limits() {
		let limits = Limits::default();
		let l = limit(1, None, Some(2), None);
		limits.replace(vec![l.clone()]);
		let now = Instant::now();
		limits.take(&Who::default(), 0, now).unwrap();
		limits.take(&Who::default(), 0, now).unwrap();
		limits.replace(vec![l.clone()]);
		assert!(
			limits.take(&Who::default(), 0, now).is_err(),
			"still used up"
		);
		limits.replace(vec![limit(1, None, Some(3), None)]);
		assert!(
			limits.take(&Who::default(), 0, now).is_ok(),
			"a changed limit starts full"
		);
	}

	#[test]
	fn resets_read_as_openai_writes_them() {
		assert_eq!(reset(Duration::from_secs(1)), "1s");
		assert_eq!(reset(Duration::from_millis(450)), "0.5s");
		assert_eq!(reset(Duration::from_secs(360)), "6m0s");
	}
}
