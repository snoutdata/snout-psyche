//! The server's configuration, from the environment. Every setting is listed here with its default
//! and whether it is secret; the README's configuration reference is this list.
//!
//! There are no demo secrets: without its JWT secret and its providers file the server refuses to
//! start, and says which one is missing.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::Duration;

use crate::net::Egress;

#[derive(Clone)]
pub struct Config {
	/// `PSYCHE_ADDR`, default `0.0.0.0:5300`: the `/ai/v1` API (and `/v1`).
	pub addr: SocketAddr,
	/// `PSYCHE_JWT_SECRET` (secret, required): the project's JWT secret, which callers' tokens are
	/// verified with. At least 32 characters.
	pub jwt_secret: String,
	/// `PSYCHE_PROVIDERS_FILE` (required): the catalogue (`providers.rs`). The file holds no keys;
	/// it names the environment variables that do, and those are secret.
	pub providers_file: String,
	/// `PSYCHE_EGRESS`, `public` (default) or `private`: where Psyche may send (`net.rs`).
	pub egress: Egress,
	pub timeouts: Timeouts,
	/// `PSYCHE_MAX_REQUEST_MB`, default 32: the largest request body accepted (images travel inline).
	pub max_request_bytes: usize,
	/// `DATABASE_URL` (secret, optional): the Postgres the request log is written to, in the
	/// `psyche` schema, which Psyche makes and migrates at start. Without it no call is stored; each
	/// is still logged to stderr.
	pub log: Option<crate::log::Settings>,
}

/// How long Psyche waits on a provider. A reasoning model can think for minutes before its first
/// token, so the first-byte wait is long; once a stream flows, a long silence means it has died.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timeouts {
	/// `PSYCHE_FIRST_BYTE_TIMEOUT_S`, default 300: from sending the request to the provider's
	/// response headers.
	pub first_byte: Duration,
	/// `PSYCHE_IDLE_TIMEOUT_S`, default 120: the longest gap between two pieces of a stream.
	pub idle: Duration,
	/// `PSYCHE_TOTAL_TIMEOUT_S`, default 600: a whole non-streamed reply.
	pub total: Duration,
}

impl Default for Timeouts {
	fn default() -> Self {
		Self {
			first_byte: Duration::from_secs(300),
			idle: Duration::from_secs(120),
			total: Duration::from_secs(600),
		}
	}
}

/// Never the secrets, not even in a debug print.
impl std::fmt::Debug for Config {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("Config")
			.field("addr", &self.addr)
			.field("providers_file", &self.providers_file)
			.field("egress", &self.egress)
			.field("timeouts", &self.timeouts)
			.field("max_request_bytes", &self.max_request_bytes)
			.finish_non_exhaustive()
	}
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct ConfigError(pub String);

impl Config {
	pub fn from_env() -> Result<Self, ConfigError> {
		Self::from_map(&std::env::vars().collect())
	}

	pub fn from_map(env: &HashMap<String, String>) -> Result<Self, ConfigError> {
		let get = |name: &str| {
			env.get(name)
				.map(|v| v.trim().to_string())
				.filter(|v| !v.is_empty())
		};
		let require = |name: &str, what: &str| {
			get(name).ok_or_else(|| {
				ConfigError(format!(
					"{name} is not set: {what}. Psyche has no default for it."
				))
			})
		};
		let seconds = |name: &str, default: Duration| -> Result<Duration, ConfigError> {
			match get(name) {
				None => Ok(default),
				Some(text) => match text.parse::<u64>() {
					Ok(n) if n > 0 => Ok(Duration::from_secs(n)),
					_ => Err(ConfigError(format!(
						"{name} is a whole number of seconds above zero, not {text}"
					))),
				},
			}
		};

		let jwt_secret = require(
			"PSYCHE_JWT_SECRET",
			"the project's JWT secret, which callers' tokens are verified with",
		)?;
		if jwt_secret.len() < 32 {
			return Err(ConfigError(
				"PSYCHE_JWT_SECRET is at least 32 characters.".into(),
			));
		}
		let providers_file = require(
			"PSYCHE_PROVIDERS_FILE",
			"the file that lists the providers and models Psyche serves",
		)?;
		let addr = get("PSYCHE_ADDR").unwrap_or_else(|| "0.0.0.0:5300".into());
		let addr = addr
			.parse::<SocketAddr>()
			.map_err(|_| ConfigError(format!("PSYCHE_ADDR is not an address and port: {addr}")))?;
		let egress = match get("PSYCHE_EGRESS").as_deref() {
			None | Some("public") => Egress::Public,
			Some("private") => Egress::Private,
			Some(other) => {
				return Err(ConfigError(format!(
					"PSYCHE_EGRESS is public or private, not {other}"
				)));
			}
		};
		let defaults = Timeouts::default();
		let timeouts = Timeouts {
			first_byte: seconds("PSYCHE_FIRST_BYTE_TIMEOUT_S", defaults.first_byte)?,
			idle: seconds("PSYCHE_IDLE_TIMEOUT_S", defaults.idle)?,
			total: seconds("PSYCHE_TOTAL_TIMEOUT_S", defaults.total)?,
		};
		let max_request_bytes = match get("PSYCHE_MAX_REQUEST_MB") {
			None => 32 * 1024 * 1024,
			Some(text) => match text.parse::<usize>() {
				Ok(mb) if (1..=256).contains(&mb) => mb * 1024 * 1024,
				_ => {
					return Err(ConfigError(format!(
						"PSYCHE_MAX_REQUEST_MB is a number of megabytes from 1 to 256, not {text}"
					)));
				}
			},
		};
		// `PSYCHE_LOG_RETENTION_DAYS`, default 30: days of request rows kept (the daily totals stay).
		let retention_days = match get("PSYCHE_LOG_RETENTION_DAYS") {
			None => 30,
			Some(text) => match text.parse::<u32>() {
				Ok(days) if (1..=3650).contains(&days) => days,
				_ => {
					return Err(ConfigError(format!(
						"PSYCHE_LOG_RETENTION_DAYS is a number of days from 1 to 3650, not {text}"
					)));
				}
			},
		};
		let log = get("DATABASE_URL").map(|database_url| crate::log::Settings {
			database_url,
			retention_days,
		});
		Ok(Self {
			addr,
			jwt_secret,
			providers_file,
			egress,
			timeouts,
			max_request_bytes,
			log,
		})
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn base() -> HashMap<String, String> {
		[
			("PSYCHE_JWT_SECRET", "a-project-secret-of-at-least-32-chars"),
			("PSYCHE_PROVIDERS_FILE", "/etc/psyche/providers.json"),
		]
		.into_iter()
		.map(|(k, v)| (k.to_string(), v.to_string()))
		.collect()
	}

	#[test]
	fn it_starts_and_the_defaults_hold() {
		let config = Config::from_map(&base()).unwrap();
		assert_eq!(config.addr.to_string(), "0.0.0.0:5300");
		assert_eq!(config.egress, Egress::Public);
		assert_eq!(config.timeouts, Timeouts::default());
		assert_eq!(config.max_request_bytes, 32 * 1024 * 1024);
		assert!(config.log.is_none());
		let mut env = base();
		env.insert(
			"DATABASE_URL".into(),
			"postgres://psyche:a-db-password@db/postgres".into(),
		);
		let config = Config::from_map(&env).unwrap();
		assert_eq!(config.log.unwrap().retention_days, 30);
	}

	#[test]
	fn nothing_secret_has_a_default() {
		for missing in ["PSYCHE_JWT_SECRET", "PSYCHE_PROVIDERS_FILE"] {
			let mut env = base();
			env.remove(missing);
			let error = Config::from_map(&env).unwrap_err();
			assert!(error.0.contains(missing), "{error}");
		}
		let mut short = base();
		short.insert("PSYCHE_JWT_SECRET".into(), "short".into());
		assert!(Config::from_map(&short).is_err());
	}

	#[test]
	fn settings_are_checked() {
		for (name, bad) in [
			("PSYCHE_EGRESS", "anywhere"),
			("PSYCHE_IDLE_TIMEOUT_S", "0"),
			("PSYCHE_FIRST_BYTE_TIMEOUT_S", "soon"),
			("PSYCHE_MAX_REQUEST_MB", "1000"),
			("PSYCHE_ADDR", "5300"),
			("PSYCHE_LOG_RETENTION_DAYS", "0"),
		] {
			let mut env = base();
			env.insert(name.into(), bad.into());
			let error = Config::from_map(&env).unwrap_err();
			assert!(error.0.contains(name), "{error}");
		}
		let mut env = base();
		env.insert("PSYCHE_EGRESS".into(), "private".into());
		assert_eq!(Config::from_map(&env).unwrap().egress, Egress::Private);
	}

	#[test]
	fn debug_never_prints_a_secret() {
		let mut env = base();
		env.insert(
			"DATABASE_URL".into(),
			"postgres://psyche:a-db-password@db/postgres".into(),
		);
		let printed = format!("{:?}", Config::from_map(&env).unwrap());
		assert!(!printed.contains("a-project-secret"), "{printed}");
		assert!(!printed.contains("a-db-password"), "{printed}");
	}
}
