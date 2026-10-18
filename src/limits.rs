//! Rate limits: requests and tokens a minute, for the project or one key, as token buckets in
//! memory (PS7: one process per project, so memory is the whole truth).
//!
//! A limit holds two buckets, each as full as its per-minute number and refilled continuously. A
//! call takes one request and its token bound (the same worst case a budget reserves: the prompt's
//! bytes and the reply's cap) from every limit in its scope, or is refused with how long to wait;
//! when it ends, the tokens it did not use go back. A call whose bound is more than a limit's whole
//! minute could never fit, so it is refused with that sentence rather than told to wait.
//!
//! Every response to a call a limit applies to carries OpenAI's headers for the tightest one:
//! `x-ratelimit-limit-requests`, `-remaining-requests`, `-reset-requests` and the same for tokens.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::keys::KeyId;

#[derive(Debug, Clone, PartialEq)]
pub struct Limit {
	pub id: KeyId,
	pub name: String,
	/// `None` for the project's own limit, which every call counts against.
	pub key: Option<KeyId>,
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

#[derive(Default)]
struct State {
	limits: Vec<Limit>,
	buckets: HashMap<KeyId, Bucket>,
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

	/// The limits from `psyche.limits`, keeping the buckets of those that did not change.
	pub fn replace(&self, limits: Vec<Limit>) {
		let mut state = self.state();
		let now = Instant::now();
		let mut buckets = HashMap::new();
		for limit in &limits {
			let unchanged = state.limits.iter().any(|old| old == limit);
			let bucket = match state.buckets.get(&limit.id) {
				Some(bucket) if unchanged => *bucket,
				_ => Bucket::full(limit, now),
			};
			buckets.insert(limit.id, bucket);
		}
		state.limits = limits;
		state.buckets = buckets;
	}

	pub fn applies(&self, key: Option<KeyId>) -> bool {
		self.state()
			.limits
			.iter()
			.any(|l| l.key.is_none() || l.key == key)
	}

	/// Takes one request and `tokens` from every limit on a call by `key`, or none of them.
	pub fn take(
		&self,
		key: Option<KeyId>,
		tokens: u64,
		now: Instant,
	) -> Result<(Taken, Headers), Refused> {
		let mut state = self.state();
		let State { limits, buckets } = &mut *state;
		let scope: Vec<&Limit> = limits
			.iter()
			.filter(|l| l.key.is_none() || l.key == key)
			.collect();
		let need = tokens as f64;
		let mut longest: Option<(Duration, &Limit, &'static str)> = None;
		for limit in &scope {
			let bucket = buckets
				.entry(limit.id)
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
		for limit in &scope {
			let bucket = buckets.get_mut(&limit.id).expect("refilled above");
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
				ids: scope.iter().map(|l| l.id).collect(),
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
	ids: Vec<KeyId>,
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
		let State { limits, buckets } = &mut *state;
		for id in &self.ids {
			let Some(tpm) = limits
				.iter()
				.find(|l| l.id == *id)
				.and_then(|l| l.tokens_per_minute)
			else {
				continue;
			};
			if let Some(bucket) = buckets.get_mut(id) {
				bucket.tokens = (bucket.tokens + back).min(f64::from(tpm));
			}
		}
	}
}

/// Every limit, from `psyche.limits`.
pub async fn load(client: &tokio_postgres::Client) -> Result<Vec<Limit>, tokio_postgres::Error> {
	let rows = client
		.query(
			"SELECT id::text, name, key_id::text, requests_per_minute, tokens_per_minute FROM psyche.limits",
			&[],
		)
		.await?;
	Ok(rows
		.iter()
		.filter_map(|row| {
			Some(Limit {
				id: KeyId::parse(row.get(0))?,
				name: row.get(1),
				key: match row.get::<_, Option<&str>>(2) {
					None => None,
					Some(key) => Some(KeyId::parse(key)?),
				},
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

	fn limit(n: u8, key: Option<KeyId>, rpm: Option<u32>, tpm: Option<u32>) -> Limit {
		Limit {
			id: KeyId([n; 16]),
			name: format!("l{n}"),
			key,
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
			limits.take(None, 0, start).unwrap();
		}
		let refused = limits.take(None, 0, start).unwrap_err();
		assert_eq!(refused.retry_after, Some(Duration::from_secs(1)));
		assert!(
			refused.message.contains("60 requests a minute"),
			"{}",
			refused.message
		);
		let (_, headers) = limits
			.take(None, 0, start + Duration::from_secs(1))
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
		let (taken, headers) = limits.take(None, 800, now).unwrap();
		assert!(
			headers
				.0
				.contains(&("x-ratelimit-remaining-tokens", "200".into()))
		);
		assert!(limits.take(None, 300, now).is_err());
		taken.settle(Some(100));
		assert!(limits.take(None, 300, now).is_ok(), "700 came back");
		let never = limits.take(None, 1001, now).unwrap_err();
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
		limits.take(Some(key), 0, now).unwrap();
		assert!(limits.take(Some(key), 0, now).is_err());
		let (_, headers) = limits.take(None, 0, now).unwrap();
		// The project's bucket lost two: the key's call and this one; the refused call took nothing.
		assert!(
			headers
				.0
				.contains(&("x-ratelimit-remaining-requests", "98".into())),
			"{headers:?}"
		);
	}

	#[test]
	fn a_reload_keeps_the_buckets_of_unchanged_limits() {
		let limits = Limits::default();
		let l = limit(1, None, Some(2), None);
		limits.replace(vec![l.clone()]);
		let now = Instant::now();
		limits.take(None, 0, now).unwrap();
		limits.take(None, 0, now).unwrap();
		limits.replace(vec![l.clone()]);
		assert!(limits.take(None, 0, now).is_err(), "still used up");
		limits.replace(vec![limit(1, None, Some(3), None)]);
		assert!(
			limits.take(None, 0, now).is_ok(),
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
