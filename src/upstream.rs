//! A provider's own list of models, for a screen that adds one (`GET
//! /ai/v1/admin/providers/<name>/upstream-models`, and `POST /ai/v1/admin/upstream-models` for a
//! connection not yet saved).
//!
//! The provider is asked with the credential its calls use, and the answer is normalised to one
//! shape: `{id, name?, owned_by?, created?, kind?, context_window?}`, chat models first, then
//! embeddings, then the rest, newest first within each. `kind` is the provider's own word where it
//! gives one, else a guess from the id (`embed` is an embedding model, `whisper` is not a chat
//! model), and absent when Psyche cannot tell. Nothing a provider says back carries the credential:
//! it is scrubbed from every error, and the list holds ids and names only.
//!
//! - OpenAI's API (OpenAI, OpenRouter, Groq, Together, Mistral, a local Ollama's `/v1`):
//!   `GET {base}/models`.
//! - Anthropic: `GET {base}/models`, with its version header, a page of 1000 at a time.
//! - Bedrock: the control plane beside the runtime (`bedrock.<region>.amazonaws.com`): the
//!   inference profiles (`us.anthropic.claude-sonnet-5`, which is what most models need) first,
//!   then the foundation models that take on-demand calls by their own id.

use std::cmp::Reverse;
use std::time::{Duration, SystemTime};

use serde::Serialize;
use serde_json::Value;

use crate::error::{ApiError, Class, scrub};
use crate::providers::{Kind, Provider};
use crate::sigv4;

/// How long the provider has to answer each page.
pub const LIST_WITHIN: Duration = Duration::from_secs(10);
/// How long the whole list may take, every page together.
const WHOLE_LIST_WITHIN: Duration = Duration::from_secs(25);
/// Pages followed at most, so a provider that keeps paging cannot hold the call.
const MAX_PAGES: usize = 5;
/// Models returned at most.
const MAX_MODELS: usize = 2000;
/// How much of an answer is read.
const MAX_BODY: usize = 16 * 1024 * 1024;

/// One model, as the list gives it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Listed {
	pub id: String,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub name: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub owned_by: Option<String>,
	/// Seconds since the epoch, when the provider says.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub created: Option<i64>,
	/// `chat`, `embedding` or `completion`; absent when it cannot be told.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub kind: Option<&'static str>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub context_window: Option<u64>,
	/// Within a kind, lower comes first (a Bedrock inference profile before a foundation model).
	#[serde(skip)]
	rank: u8,
}

/// Asks `provider` for its models, with the credential it holds.
pub async fn list(http: &reqwest::Client, provider: &Provider) -> Result<Vec<Listed>, ApiError> {
	let asked = async {
		match provider.kind {
			Kind::Openai => openai(http, provider).await,
			Kind::Anthropic => anthropic(http, provider).await,
			Kind::Bedrock => bedrock(http, provider).await,
		}
	};
	// Each page has its own time; the whole list has this, well inside a relay's own wait.
	let mut out = tokio::time::timeout(WHOLE_LIST_WITHIN, asked)
		.await
		.map_err(|_| {
			ApiError::new(
				Class::Timeout,
				format!(
					"{} did not finish listing its models within {} seconds. Type the model's id instead.",
					provider.name,
					WHOLE_LIST_WITHIN.as_secs()
				),
			)
		})??;
	sort(&mut out);
	out.truncate(MAX_MODELS);
	Ok(out)
}

fn sort(models: &mut Vec<Listed>) {
	let mut seen = std::collections::HashSet::new();
	models.retain(|m| !m.id.is_empty() && seen.insert(m.id.clone()));
	let kind_rank = |kind: Option<&str>| match kind {
		Some("chat") => 0,
		Some("embedding") => 1,
		Some("completion") => 2,
		_ => 3,
	};
	models.sort_by(|a, b| {
		(
			kind_rank(a.kind),
			a.rank,
			Reverse(a.created.unwrap_or(i64::MIN)),
			&a.id,
		)
			.cmp(&(
				kind_rank(b.kind),
				b.rank,
				Reverse(b.created.unwrap_or(i64::MIN)),
				&b.id,
			))
	})
}

/// A guess from an OpenAI-style id, for a provider that does not say what a model is.
pub fn guess_kind(id: &str) -> Option<&'static str> {
	let id = id.to_ascii_lowercase();
	let any = |words: &[&str]| words.iter().any(|w| id.contains(w));
	if any(&["embed", "bge-", "nomic-", "e5-"]) {
		return Some("embedding");
	}
	if any(&[
		"whisper",
		"tts",
		"dall-e",
		"gpt-image",
		"image",
		"moderation",
		"audio",
		"realtime",
		"transcribe",
		"sora",
		"rerank",
		"guard",
		"diffusion",
		"flux",
	]) {
		return None;
	}
	if any(&["instruct-", "davinci", "babbage"])
		|| id.ends_with("-instruct") && id.starts_with("gpt-3.5")
	{
		return Some("completion");
	}
	Some("chat")
}

/// Together's and OpenRouter's own word for a model, when they give one.
fn stated_kind(entry: &Value) -> Option<Option<&'static str>> {
	if let Some(kind) = entry["type"].as_str() {
		return Some(match kind {
			"chat" => Some("chat"),
			"embedding" | "embeddings" => Some("embedding"),
			"language" | "code" => Some("completion"),
			"model" => return None,
			_ => None,
		});
	}
	let outputs = entry
		.pointer("/architecture/output_modalities")?
		.as_array()?;
	let has = |word: &str| outputs.iter().any(|o| o.as_str() == Some(word));
	Some(if has("embeddings") {
		Some("embedding")
	} else if has("text") {
		Some("chat")
	} else {
		None
	})
}

fn number(entry: &Value, fields: &[&str]) -> Option<u64> {
	fields
		.iter()
		.find_map(|f| entry.get(*f).and_then(Value::as_u64))
}

fn text(entry: &Value, fields: &[&str]) -> Option<String> {
	fields
		.iter()
		.find_map(|f| entry.get(*f).and_then(Value::as_str))
		.map(str::to_string)
		.filter(|s| !s.is_empty())
}

async fn openai(http: &reqwest::Client, provider: &Provider) -> Result<Vec<Listed>, ApiError> {
	let url = parse(provider, &provider.endpoint("models"))?;
	let body = get(http, provider, url, false).await?;
	// OpenAI's `{"data": [...]}`, or Together's bare array.
	let entries = body
		.get("data")
		.and_then(Value::as_array)
		.or_else(|| body.as_array())
		.ok_or_else(|| not_a_list(provider))?;
	Ok(entries
		.iter()
		.filter_map(|entry| {
			let id = entry["id"].as_str()?.to_string();
			let kind = stated_kind(entry).unwrap_or_else(|| guess_kind(&id));
			Some(Listed {
				name: text(entry, &["name", "display_name"]).filter(|n| *n != id),
				owned_by: text(entry, &["owned_by", "organization"]),
				created: entry["created"].as_i64().filter(|c| *c > 0),
				kind,
				context_window: number(
					entry,
					&["context_window", "context_length", "max_context_length"],
				),
				rank: 0,
				id,
			})
		})
		.collect())
}

async fn anthropic(http: &reqwest::Client, provider: &Provider) -> Result<Vec<Listed>, ApiError> {
	let mut out = Vec::new();
	let mut after: Option<String> = None;
	for _ in 0..MAX_PAGES {
		let mut url = parse(provider, &provider.endpoint("models"))?;
		let mut query = "limit=1000".to_string();
		if let Some(after) = &after {
			query.push_str(&format!("&after_id={}", sigv4::encode(after)));
		}
		url.set_query(Some(&query));
		let body = get(http, provider, url, true).await?;
		let entries = body["data"]
			.as_array()
			.ok_or_else(|| not_a_list(provider))?;
		for entry in entries {
			let Some(id) = entry["id"].as_str() else {
				continue;
			};
			out.push(Listed {
				id: id.to_string(),
				name: text(entry, &["display_name"]),
				owned_by: Some("anthropic".into()),
				created: entry["created_at"].as_str().and_then(unix_from_rfc3339),
				kind: Some("chat"),
				context_window: None,
				rank: 0,
			});
		}
		match (body["has_more"].as_bool(), body["last_id"].as_str()) {
			(Some(true), Some(last)) => after = Some(last.to_string()),
			_ => break,
		}
	}
	Ok(out)
}

/// Bedrock's control plane, beside the runtime calls go to: `bedrock-runtime.<region>…` becomes
/// `bedrock.<region>…`; any other address (a VPC endpoint, a test's) is asked as it is.
fn control_plane(provider: &Provider) -> String {
	let base = provider.base_url.as_str().trim_end_matches('/');
	match provider.base_url.host_str() {
		Some(host) if host.starts_with("bedrock-runtime.") => {
			base.replacen("bedrock-runtime.", "bedrock.", 1)
		}
		_ => base.to_string(),
	}
}

async fn bedrock(http: &reqwest::Client, provider: &Provider) -> Result<Vec<Listed>, ApiError> {
	let base = control_plane(provider);
	let url = parse(provider, &format!("{base}/foundation-models"))?;
	let body = get(http, provider, url, false).await?;
	let summaries = body["modelSummaries"]
		.as_array()
		.ok_or_else(|| not_a_list(provider))?;
	// Every foundation model's kind and maker, by id, for the profiles that point at one.
	let mut about = std::collections::HashMap::new();
	let mut out = Vec::new();
	for summary in summaries {
		let Some(id) = summary["modelId"].as_str() else {
			continue;
		};
		let outputs = summary["outputModalities"].as_array();
		let has = |word: &str| outputs.is_some_and(|o| o.iter().any(|v| v.as_str() == Some(word)));
		let kind = if has("EMBEDDING") {
			Some("embedding")
		} else if has("TEXT") {
			Some("chat")
		} else {
			None
		};
		let maker = text(summary, &["providerName"]);
		about.insert(id.to_string(), (kind, maker.clone()));
		let on_demand = summary["inferenceTypesSupported"]
			.as_array()
			.is_some_and(|t| t.iter().any(|v| v.as_str() == Some("ON_DEMAND")));
		let legacy = summary
			.pointer("/modelLifecycle/status")
			.and_then(Value::as_str)
			== Some("LEGACY");
		// A model that takes no on-demand call by its own id is reached through a profile below.
		if on_demand {
			out.push(Listed {
				id: id.to_string(),
				name: text(summary, &["modelName"]),
				owned_by: maker,
				created: None,
				kind,
				context_window: None,
				rank: if legacy { 2 } else { 1 },
			});
		}
	}
	let mut next: Option<String> = None;
	for _ in 0..MAX_PAGES {
		let mut url = parse(provider, &format!("{base}/inference-profiles"))?;
		let mut query = "maxResults=1000".to_string();
		if let Some(token) = &next {
			query.push_str(&format!("&nextToken={}", sigv4::encode(token)));
		}
		url.set_query(Some(&query));
		let body = get(http, provider, url, false).await?;
		let Some(profiles) = body["inferenceProfileSummaries"].as_array() else {
			break;
		};
		for profile in profiles {
			let Some(id) = profile["inferenceProfileId"].as_str() else {
				continue;
			};
			if profile["status"].as_str().is_some_and(|s| s != "ACTIVE") {
				continue;
			}
			// The model a profile routes to is the last part of its first model's ARN.
			let model = profile
				.pointer("/models/0/modelArn")
				.and_then(Value::as_str)
				.and_then(|arn| arn.rsplit('/').next())
				.unwrap_or("");
			let (kind, maker) = about.get(model).cloned().unwrap_or((Some("chat"), None));
			out.push(Listed {
				id: id.to_string(),
				name: text(profile, &["inferenceProfileName"]),
				owned_by: maker,
				created: profile["createdAt"].as_str().and_then(unix_from_rfc3339),
				kind,
				context_window: None,
				rank: 0,
			});
		}
		match body["nextToken"].as_str() {
			Some(token) if !token.is_empty() => next = Some(token.to_string()),
			_ => break,
		}
	}
	Ok(out)
}

fn parse(provider: &Provider, url: &str) -> Result<url::Url, ApiError> {
	url::Url::parse(url).map_err(|_| {
		ApiError::new(
			Class::BadRequest,
			format!("{}'s address is not a URL.", provider.name),
		)
	})
}

fn not_a_list(provider: &Provider) -> ApiError {
	ApiError::new(
		Class::Upstream,
		format!(
			"{} answered, but not with a list of models. Type the model's id instead.",
			provider.name
		),
	)
}

/// One GET, signed or keyed as the provider's calls are, read as JSON.
async fn get(
	http: &reqwest::Client,
	provider: &Provider,
	url: url::Url,
	anthropic: bool,
) -> Result<Value, ApiError> {
	let secrets = provider.secrets();
	let clean = |text: &str| scrub(text, &secrets);
	let unix = SystemTime::now()
		.duration_since(SystemTime::UNIX_EPOCH)
		.map_or(0, |d| d.as_secs());
	let mut request = http
		.get(url.clone())
		.header("content-type", "application/json")
		.header("accept", "application/json");
	request = provider.authorize_as("GET", request, &url, b"", unix);
	if anthropic {
		request = request.header("anthropic-version", crate::anthropic::VERSION);
	}
	let response = tokio::time::timeout(LIST_WITHIN, request.send())
		.await
		.map_err(|_| {
			ApiError::new(
				Class::Timeout,
				format!(
					"{} did not list its models within {} seconds. Type the model's id instead.",
					provider.name,
					LIST_WITHIN.as_secs()
				),
			)
		})?
		.map_err(|e| {
			ApiError::new(
				Class::Transport,
				format!(
					"{} could not be reached to list its models: {}",
					provider.name,
					clean(&e.to_string())
				),
			)
		})?;
	let status = response.status().as_u16();
	let bytes = tokio::time::timeout(LIST_WITHIN, read_capped(response))
		.await
		.map_err(|_| {
			ApiError::new(
				Class::Timeout,
				format!(
					"{} did not finish listing its models in time.",
					provider.name
				),
			)
		})?;
	let said = || {
		let text = String::from_utf8_lossy(&bytes);
		let text = clean(text.trim());
		if text.is_empty() {
			String::new()
		} else {
			format!(" It said: {}", text.chars().take(300).collect::<String>())
		}
	};
	match status {
		200..=299 => serde_json::from_slice(&bytes).map_err(|_| not_a_list(provider)),
		401 | 403 => Err(ApiError::new(
			Class::ProviderAuth,
			format!(
				"{} refused to list its models with this credential ({status}).{}",
				provider.name,
				said()
			),
		)),
		404 => Err(ApiError::new(
			Class::Upstream,
			format!(
				"{} has no model list at {} (404). Type the model's id instead.",
				provider.name,
				clean(url.as_str())
			),
		)),
		429 => Err(ApiError::new(
			Class::Upstream,
			format!(
				"{} is rate-limiting, so it did not list its models; try again.",
				provider.name
			),
		)),
		_ => Err(ApiError::new(
			Class::Upstream,
			format!(
				"{} answered {status} when asked for its models.{}",
				provider.name,
				said()
			),
		)),
	}
}

async fn read_capped(response: reqwest::Response) -> Vec<u8> {
	use futures_util::StreamExt;
	let mut out = Vec::new();
	let mut stream = response.bytes_stream();
	while let Some(Ok(chunk)) = stream.next().await {
		out.extend_from_slice(&chunk);
		if out.len() > MAX_BODY {
			break;
		}
	}
	out
}

/// `2025-02-19T00:00:00Z` (fractions and an offset allowed) as seconds since the epoch.
pub fn unix_from_rfc3339(text: &str) -> Option<i64> {
	let b = text.as_bytes();
	if b.len() < 19 || b[4] != b'-' || b[7] != b'-' || !matches!(b[10], b'T' | b't' | b' ') {
		return None;
	}
	let n = |from: usize, to: usize| text.get(from..to)?.parse::<i64>().ok();
	let (year, month, day) = (n(0, 4)?, n(5, 7)?, n(8, 10)?);
	let (hour, minute, second) = (n(11, 13)?, n(14, 16)?, n(17, 19)?);
	if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
		return None;
	}
	// The offset, after any fraction.
	let rest = &text[19..];
	let rest = rest.trim_start_matches(|c: char| c == '.' || c.is_ascii_digit());
	let offset = match rest.as_bytes().first() {
		None | Some(b'Z' | b'z') => 0,
		Some(sign @ (b'+' | b'-')) => {
			let h: i64 = rest.get(1..3)?.parse().ok()?;
			let m: i64 = rest.get(4..6)?.parse().ok()?;
			let total = h * 3600 + m * 60;
			if *sign == b'+' { total } else { -total }
		}
		_ => return None,
	};
	// Howard Hinnant's days-from-civil.
	let y = if month <= 2 { year - 1 } else { year };
	let era = y.div_euclid(400);
	let yoe = y - era * 400;
	let mp = (month + 9) % 12;
	let doy = (153 * mp + 2) / 5 + day - 1;
	let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
	let days = era * 146_097 + doe - 719_468;
	Some(days * 86_400 + hour * 3600 + minute * 60 + second - offset)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::fake::Fake;
	use crate::net::Egress;
	use crate::providers::{Secret, Stored};

	#[test]
	fn dates_and_guesses() {
		assert_eq!(unix_from_rfc3339("1970-01-01T00:00:00Z"), Some(0));
		assert_eq!(
			unix_from_rfc3339("2025-02-19T00:00:00Z"),
			Some(1_739_923_200)
		);
		assert_eq!(
			unix_from_rfc3339("2025-02-19T01:00:00.123+01:00"),
			Some(1_739_923_200)
		);
		assert_eq!(unix_from_rfc3339("yesterday"), None);
		assert_eq!(guess_kind("text-embedding-3-small"), Some("embedding"));
		assert_eq!(guess_kind("whisper-1"), None);
		assert_eq!(guess_kind("gpt-5-mini"), Some("chat"));
		assert_eq!(guess_kind("davinci-002"), Some("completion"));
	}

	fn provider(kind: Kind, base: &str, region: Option<&str>, key: Option<Stored>) -> Provider {
		let made = Provider::from_row(
			"here",
			kind,
			Some(base),
			region,
			Some("c"),
			None,
			Egress::Private,
		)
		.unwrap();
		let mut stored = std::collections::HashMap::new();
		if let Some(key) = key {
			stored.insert("c".to_string(), key);
		}
		made.with_stored(&stored)
	}

	fn key(text: &str) -> Option<Stored> {
		Some(Stored::ApiKey(Secret::new(text)))
	}

	#[tokio::test]
	async fn each_kind_lists_its_models_newest_and_chat_first() {
		let fake = Fake::start().await;
		let http = crate::net::client(Egress::Private).unwrap();
		let base = format!("{}/v1", fake.origin);

		let openai = list(&http, &provider(Kind::Openai, &base, None, key(Fake::KEY)))
			.await
			.unwrap();
		let ids: Vec<&str> = openai.iter().map(|m| m.id.as_str()).collect();
		assert_eq!(
			ids,
			["fake-new", "fake-ok", "fake-embed", "whisper-fake"],
			"{openai:?}"
		);
		assert_eq!(openai[0].kind, Some("chat"));
		assert_eq!(openai[0].context_window, Some(128_000));
		assert_eq!(openai[2].kind, Some("embedding"));
		assert_eq!(openai[3].kind, None);

		let refused = list(
			&http,
			&provider(Kind::Openai, &base, None, key("sk-wrong-key-0123456789")),
		)
		.await
		.unwrap_err();
		assert!(matches!(refused.class, Class::ProviderAuth));
		assert!(
			!refused.message.contains("sk-wrong-key-0123456789"),
			"{}",
			refused.message
		);
		let missing = list(
			&http,
			&provider(
				Kind::Openai,
				&format!("{}/missing/v1", fake.origin),
				None,
				key(Fake::KEY),
			),
		)
		.await
		.unwrap_err();
		assert!(missing.message.contains("404"), "{}", missing.message);

		let anthropic = list(
			&http,
			&provider(Kind::Anthropic, &base, None, key(Fake::KEY)),
		)
		.await
		.unwrap();
		let ids: Vec<&str> = anthropic.iter().map(|m| m.id.as_str()).collect();
		assert_eq!(
			ids,
			["claude-fake-5", "claude-fake-4", "claude-fake-3"],
			"two pages, newest first"
		);
		assert_eq!(anthropic[0].name.as_deref(), Some("Claude Fake 5"));
		assert!(fake.last().anthropic_version.is_some());

		for credential in [
			key(Fake::KEY),
			Some(Stored::Aws(sigv4::Credentials {
				access_key_id: Fake::AWS_ACCESS_KEY_ID.into(),
				secret_access_key: Secret::new(Fake::AWS_SECRET),
				session_token: None,
			})),
		] {
			let bedrock = list(
				&http,
				&provider(Kind::Bedrock, &fake.origin, Some("us-west-2"), credential),
			)
			.await
			.unwrap();
			let ids: Vec<&str> = bedrock.iter().map(|m| m.id.as_str()).collect();
			assert_eq!(
				ids,
				[
					"us.openai.fake-luna",
					"us.anthropic.fake-sonnet",
					"qwen.fake-coder-v1:0",
					"amazon.fake-embed-v2:0",
					"stability.fake-image-v1:0"
				],
				"profiles first (both pages), then on-demand models; the profile-only model only through its profile"
			);
			assert_eq!(bedrock[1].owned_by.as_deref(), Some("Anthropic"));
			assert_eq!(bedrock[3].kind, Some("embedding"));
		}
		let wrong = list(
			&http,
			&provider(
				Kind::Bedrock,
				&fake.origin,
				Some("us-west-2"),
				key("not-the-bedrock-key"),
			),
		)
		.await
		.unwrap_err();
		assert!(matches!(wrong.class, Class::ProviderAuth));
	}

	#[test]
	fn the_control_plane_is_beside_the_runtime() {
		let p = provider(
			Kind::Bedrock,
			"https://bedrock-runtime.us-west-2.amazonaws.com",
			Some("us-west-2"),
			None,
		);
		assert_eq!(control_plane(&p), "https://bedrock.us-west-2.amazonaws.com");
	}
}
