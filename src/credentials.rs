//! Stored provider credentials: written once, never read back (Phase 3, the credentials box).
//!
//! A provider in the providers file may name a credential (`"credential": "openai"`) instead of the
//! environment variable its key is in. The credential is a row of `psyche.credentials`, set with the
//! service key through `PUT /ai/v1/admin/credentials/<name>`. Before it is stored it is **proven**:
//! the provider that names it is sent an empty request on the endpoint calls use, which a provider
//! answers with 401 or 403 when it does not take the credential, and with a validation error when
//! it does, so the proof spends no tokens and checks the base URL too. A credential that is refused,
//! or that cannot be checked, is not stored. Lists show a credential's kind, a fingerprint and its
//! last four characters, never the secret.
//!
//! Only Psyche's own role reads the table (migration 7). Encryption on top of that was weighed and
//! not added, as for push's credentials: its key would sit in Psyche's environment, beside the
//! database URL, so it would protect only a copy of the database taken without the environment
//! (a backup), and the project's backups are the project's own.

use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use serde_json::{Value, json};
use tokio_postgres::Client;

use crate::providers::{Catalogue, Kind, Model, Provider, Secret, Stored, credential_name_ok};
use crate::sigv4::{self, Credentials};

/// How long a proof waits for the provider.
const PROVE_WITHIN: Duration = Duration::from_secs(15);

/// What the admin API takes: an API key, or AWS credentials for a Bedrock provider.
#[derive(Debug, Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NewCredential {
	#[serde(default)]
	pub api_key: Option<String>,
	#[serde(default)]
	pub access_key_id: Option<String>,
	#[serde(default)]
	pub secret_access_key: Option<String>,
	#[serde(default)]
	pub session_token: Option<String>,
}

impl NewCredential {
	pub fn stored(self) -> Result<Stored, String> {
		let given = |v: Option<String>| v.map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
		match (
			given(self.api_key),
			given(self.access_key_id),
			given(self.secret_access_key),
			given(self.session_token),
		) {
			(Some(key), None, None, None) => Ok(Stored::ApiKey(Secret::new(key))),
			(None, Some(id), Some(secret), token) => Ok(Stored::Aws(Credentials {
				access_key_id: id,
				secret_access_key: Secret::new(secret),
				session_token: token.map(Secret::new),
			})),
			_ => Err("A credential is {\"api_key\"}, or {\"access_key_id\", \"secret_access_key\"} and an optional \"session_token\" for Bedrock.".into()),
		}
	}
}

/// The secret part of a credential, which the fingerprint and the last four are taken from.
fn secret_of(stored: &Stored) -> &str {
	match stored {
		Stored::ApiKey(key) => key.expose(),
		Stored::Aws(aws) => aws.secret_access_key.expose(),
	}
}

pub fn fingerprint(secret: &str) -> String {
	let digest = ring::digest::digest(&ring::digest::SHA256, secret.as_bytes());
	let hex: String = digest.as_ref()[..6]
		.iter()
		.map(|b| format!("{b:02x}"))
		.collect();
	format!("sha256:{hex}")
}

pub fn last_four(secret: &str) -> String {
	let chars: Vec<char> = secret.chars().collect();
	chars[chars.len().saturating_sub(4)..].iter().collect()
}

/// Sends `provider`, holding `stored`, an empty request on the endpoint `model`'s calls use, and
/// says whether it took the credential. A refusal is the provider's 401 or 403; a 404 means
/// nothing at that address serves these calls (a base URL missing its `/v1`, say), so it proves
/// nothing either way and is refused too; anything else it answers with below 500, a validation
/// error included, means the credential was accepted; a 5xx or no answer means it could not be
/// checked. The credential is scrubbed from what the provider says back.
pub async fn prove(
	http: &reqwest::Client,
	provider: &Provider,
	model: &Model,
	stored: &Stored,
) -> Result<(), String> {
	let mut provider = provider.clone();
	match (stored, provider.kind, &provider.region) {
		(Stored::ApiKey(key), _, _) => provider.api_key = Some(key.clone()),
		(Stored::Aws(credentials), Kind::Bedrock, Some(region)) => {
			provider.aws = Some(crate::providers::Aws {
				region: region.clone(),
				credentials: credentials.clone(),
			});
		}
		(Stored::Aws(_), _, _) => {
			return Err(format!(
				"{} is not a Bedrock provider, so it takes an api_key, not AWS credentials.",
				provider.name
			));
		}
	}
	let path = match provider.kind {
		Kind::Openai => "chat/completions".to_string(),
		Kind::Anthropic => "messages".to_string(),
		Kind::Bedrock => format!("model/{}/converse", sigv4::encode(&model.upstream)),
	};
	let url = url::Url::parse(&provider.endpoint(&path))
		.map_err(|_| format!("{}'s endpoint is not a URL.", provider.name))?;
	let body = b"{}";
	let unix = SystemTime::now()
		.duration_since(SystemTime::UNIX_EPOCH)
		.map_or(0, |d| d.as_secs());
	let mut request = http
		.post(url.clone())
		.header("content-type", "application/json");
	request = provider.authorize(request, &url, body, unix);
	if provider.kind == Kind::Anthropic {
		request = request.header("anthropic-version", crate::anthropic::VERSION);
	}
	let response = tokio::time::timeout(PROVE_WITHIN, request.body(body.to_vec()).send())
		.await
		.map_err(|_| {
			format!(
				"{} did not answer within {} seconds, so the credential could not be checked.",
				provider.name,
				PROVE_WITHIN.as_secs()
			)
		})?
		.map_err(|e| {
			format!(
				"{} could not be reached, so the credential could not be checked: {}",
				provider.name,
				crate::error::scrub(&e.to_string(), &[secret_of(stored)])
			)
		})?;
	let status = response.status().as_u16();
	if matches!(status, 401 | 403) {
		let text = response.text().await.unwrap_or_default();
		let said = crate::error::scrub(&text, &[secret_of(stored)]);
		return Err(format!(
			"{} refused this credential ({status}): {}",
			provider.name,
			said.chars().take(300).collect::<String>()
		));
	}
	if status == 404 {
		let text = response.text().await.unwrap_or_default();
		let said = crate::error::scrub(&text, &[secret_of(stored)]);
		let check = if provider.kind == Kind::Bedrock {
			"its region, and that the model is offered there"
		} else {
			"its base_url, which usually ends in /v1"
		};
		return Err(format!(
			"{} answered 404 at {}, so nothing there serves these calls and the credential could not be checked; check {check}.{}",
			provider.name,
			crate::error::scrub(url.as_str(), &[secret_of(stored)]),
			quoted(&said)
		));
	}
	if status >= 500 {
		return Err(format!(
			"{} answered {status}, so the credential could not be checked; try again.",
			provider.name
		));
	}
	Ok(())
}

/// What a provider said, as the end of a sentence: nothing when it said nothing (Anthropic's 404
/// has an empty body), else its first 300 characters after a space.
fn quoted(said: &str) -> String {
	let said = said.trim();
	if said.is_empty() {
		String::new()
	} else {
		format!(" It said: {}", said.chars().take(300).collect::<String>())
	}
}

/// Stores `stored` as `name`, replacing what was there, and returns its row as lists show it.
pub async fn save(
	client: &Client,
	name: &str,
	stored: &Stored,
) -> Result<Value, tokio_postgres::Error> {
	let secret = secret_of(stored);
	let (kind, api_key, id, secret_key, token) = match stored {
		Stored::ApiKey(key) => ("api_key", Some(key.expose()), None, None, None),
		Stored::Aws(aws) => (
			"aws",
			None,
			Some(aws.access_key_id.as_str()),
			Some(aws.secret_access_key.expose()),
			aws.session_token.as_ref().map(Secret::expose),
		),
	};
	let (print, four) = (fingerprint(secret), last_four(secret));
	client
		.execute(
			"INSERT INTO psyche.credentials (name, kind, api_key, access_key_id, secret_access_key, \
			 session_token, fingerprint, last_four) VALUES ($1, $2, $3, $4, $5, $6, $7, $8) \
			 ON CONFLICT (name) DO UPDATE SET kind = excluded.kind, api_key = excluded.api_key, \
			 access_key_id = excluded.access_key_id, secret_access_key = excluded.secret_access_key, \
			 session_token = excluded.session_token, fingerprint = excluded.fingerprint, \
			 last_four = excluded.last_four, updated_at = now()",
			&[
				&name,
				&kind,
				&api_key,
				&id,
				&secret_key,
				&token,
				&print,
				&four,
			],
		)
		.await?;
	Ok(json!({"name": name, "kind": kind, "fingerprint": print, "last_four": four}))
}

/// Every credential, from `psyche.credentials`.
pub async fn load(client: &Client) -> Result<HashMap<String, Stored>, tokio_postgres::Error> {
	let rows = client
		.query(
			"SELECT name, kind, api_key, access_key_id, secret_access_key, session_token \
			 FROM psyche.credentials",
			&[],
		)
		.await?;
	let mut out = HashMap::with_capacity(rows.len());
	for row in rows {
		let name: String = row.get(0);
		let stored = match (
			row.get::<_, &str>(1),
			row.get::<_, Option<String>>(2),
			row.get::<_, Option<String>>(3),
			row.get::<_, Option<String>>(4),
		) {
			("api_key", Some(key), _, _) => Stored::ApiKey(Secret::new(key)),
			("aws", _, Some(id), Some(secret)) => Stored::Aws(Credentials {
				access_key_id: id,
				secret_access_key: Secret::new(secret),
				session_token: row.get::<_, Option<String>>(5).map(Secret::new),
			}),
			_ => continue,
		};
		out.insert(name, stored);
	}
	Ok(out)
}

/// Every credential as lists show it, with the providers that name it; and every name a provider
/// asks for that is not stored, so a missing one is plain to see.
pub async fn list(
	client: &Client,
	catalogue: &Catalogue,
) -> Result<Vec<Value>, tokio_postgres::Error> {
	let rows = client
		.query(
			"SELECT name, kind, fingerprint, last_four, extract(epoch FROM updated_at)::bigint \
			 FROM psyche.credentials ORDER BY name",
			&[],
		)
		.await?;
	let used_by = |name: &str| -> Vec<String> {
		catalogue
			.credentialed()
			.iter()
			.filter(|p| p.credential.as_deref() == Some(name))
			.map(|p| p.name.clone())
			.collect()
	};
	let mut out: Vec<Value> = rows
		.iter()
		.map(|row| {
			let name: String = row.get(0);
			json!({"name": name, "kind": row.get::<_, String>(1), "fingerprint": row.get::<_, String>(2),
				"last_four": row.get::<_, String>(3), "updated_at": row.get::<_, i64>(4),
				"used_by": used_by(&name), "stored": true})
		})
		.collect();
	for provider in catalogue.credentialed() {
		let name = provider.credential.as_deref().unwrap_or("");
		if !out.iter().any(|row| row["name"] == name) {
			out.push(json!({"name": name, "used_by": used_by(name), "stored": false}));
		}
	}
	Ok(out)
}

/// Checks a name from a URL path.
pub fn name(text: &str) -> Result<&str, String> {
	if credential_name_ok(text) {
		Ok(text)
	} else {
		Err("A credential's name is lowercase letters, digits, - and _, up to 63.".into())
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn a_credential_is_one_kind_and_shown_only_by_its_print() {
		let key = NewCredential {
			api_key: Some(" sk-live-0123456789abcd ".into()),
			..Default::default()
		};
		let Ok(Stored::ApiKey(key)) = key.stored() else {
			panic!("an api key");
		};
		assert_eq!(key.expose(), "sk-live-0123456789abcd", "trimmed");
		assert_eq!(last_four(key.expose()), "abcd");
		assert!(fingerprint(key.expose()).starts_with("sha256:"));
		assert_eq!(fingerprint(key.expose()).len(), "sha256:".len() + 12);
		let both = NewCredential {
			api_key: Some("k".into()),
			access_key_id: Some("AKID".into()),
			..Default::default()
		};
		assert!(both.stored().is_err());
		let half = NewCredential {
			access_key_id: Some("AKID".into()),
			..Default::default()
		};
		assert!(half.stored().is_err());
		assert!(NewCredential::default().stored().is_err());
		assert!(name("openai-prod").is_ok() && name("Open AI").is_err() && name("").is_err());
	}

	#[tokio::test]
	async fn an_address_that_serves_nothing_proves_nothing() {
		let fake = crate::fake::Fake::start().await;
		let spec = crate::providers::Spec::from_json(
			&format!(
				r#"{{"providers": [
					{{"name": "here", "kind": "openai", "base_url": "{0}/v1", "credential": "c"}},
					{{"name": "lost", "kind": "openai", "base_url": "{0}/missing/v1", "credential": "c"}}
				],
				"models": [
					{{"name": "ok", "provider": "here", "upstream": "fake-ok"}},
					{{"name": "lost-ok", "provider": "lost", "upstream": "fake-ok"}}
				]}}"#,
				fake.origin
			),
			&std::collections::HashMap::new(),
			crate::net::Egress::Private,
		)
		.unwrap();
		let catalogue = crate::providers::Catalogue::build(&spec).unwrap();
		let http = crate::net::client(crate::net::Egress::Private).unwrap();
		let key = Stored::ApiKey(Secret::new(crate::fake::Fake::KEY.to_string()));
		let prove_on = |provider: &str, model: &str| {
			prove(
				&http,
				spec.provider(provider).unwrap(),
				catalogue.model(model).unwrap(),
				&key,
			)
		};
		assert_eq!(prove_on("here", "ok").await, Ok(()));
		let refused = prove_on("lost", "lost-ok").await.unwrap_err();
		assert!(refused.contains("answered 404"), "{refused}");
		assert!(
			refused.contains("/missing/v1/chat/completions"),
			"{refused}"
		);
		assert!(!refused.contains(crate::fake::Fake::KEY), "{refused}");
		assert!(refused.contains("ends in /v1. It said: "), "{refused}");
		assert_eq!(quoted("  "), "", "an empty body ends the sentence");
	}
}
