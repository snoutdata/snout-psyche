//! The catalogue: which providers Psyche may call, with which credential, and which model names it
//! answers to.
//!
//! In Phase 1 it is a JSON file (`PSYCHE_PROVIDERS_FILE`), the standalone case. From Phase 3 the same
//! shape is rows in the project's `psyche` schema, and the file only seeds them. A provider's key is
//! never in the file: the file names the environment variable that holds it (`api_key_env`), so the
//! file can be committed and shared.
//!
//! ```json
//! {
//!   "providers": [
//!     {"name": "openai", "kind": "openai", "base_url": "https://api.openai.com/v1",
//!      "api_key_env": "OPENAI_API_KEY"}
//!   ],
//!   "models": [
//!     {"name": "gpt-4o-mini", "provider": "openai", "upstream": "gpt-4o-mini"}
//!   ]
//! }
//! ```
//!
//! A Bedrock provider names its region instead of a base URL, and takes either a Bedrock API key
//! (`api_key_env`, sent as a bearer token) or AWS credentials (`aws`, each part named by the
//! variable that holds it), which sign every request with SigV4 (`sigv4.rs`):
//!
//! ```json
//! {"name": "bedrock", "kind": "bedrock", "region": "us-west-2",
//!  "aws": {"access_key_id_env": "AWS_ACCESS_KEY_ID", "secret_access_key_env": "AWS_SECRET_ACCESS_KEY"}}
//! ```
//!
//! A model may say what it can do. Today that is `prompt_caching`, which is **off unless written**:
//! a `cache_control` breakpoint reaches a provider only for a model that says it takes one, because
//! a model that does not can answer the whole request with an error (and guessing it from a price
//! list gets it wrong). And `max_output_tokens`, the reply
//! length asked for when the caller names none, which Anthropic requires on every request. And
//! `reasoning` (`budget`, `adaptive` or `effort`, `Reasoning` below), how the model takes
//! `reasoning_effort`; without it a request that asks for reasoning is refused, not guessed at.
//! And, for a Bedrock embedding model, `embeddings` (`titan` or `cohere`), the body its InvokeModel
//! takes (`embeddings.rs`); a Bedrock model without it is not called for embeddings.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use serde::Deserialize;

use crate::config::ConfigError;
use crate::net::{Egress, check_base_url};
use crate::sigv4::{self, Credentials};

/// How Psyche speaks to a provider. Phase 2 adds Bedrock, Gemini, Ollama and the rest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
	/// OpenAI's API, or any server that speaks it (vLLM, Groq, Together, DeepSeek, LM Studio...).
	Openai,
	/// Anthropic's Messages API (`anthropic.rs` translates to and from it).
	Anthropic,
	/// Amazon Bedrock's Converse API (`bedrock.rs`).
	Bedrock,
}

/// How a model takes `reasoning_effort`, which differs by model and cannot be guessed from the
/// family (2026-10-18, against Bedrock: Claude Sonnet 5 refuses the budget form older Claude models
/// take, and gpt-5.6-luna refuses `reasoning_effort` by that name).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Reasoning {
	/// `thinking: {type: enabled, budget_tokens}`: Claude before Sonnet 5.
	Budget,
	/// `thinking: {type: adaptive}` with `output_config: {effort}`: Claude Sonnet 5 and after.
	Adaptive,
	/// `reasoning: {effort}`: OpenAI's models on Bedrock.
	Effort,
}

/// The InvokeModel body a Bedrock embedding model takes (`embeddings.rs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Embeddings {
	/// `{inputText, dimensions}`, one text per call: Amazon Titan.
	Titan,
	/// `{texts, input_type, embedding_types, output_dimension}`, up to 96 texts: Cohere v3 and v4.
	Cohere,
}

/// A credential, which never prints.
#[derive(Clone, PartialEq, Eq)]
pub struct Secret(String);

impl Secret {
	pub fn new(value: impl Into<String>) -> Self {
		Self(value.into())
	}

	pub fn expose(&self) -> &str {
		&self.0
	}
}

impl std::fmt::Debug for Secret {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.write_str("Secret(..)")
	}
}

#[derive(Debug, Clone)]
pub struct Provider {
	pub name: String,
	pub kind: Kind,
	/// The API's base, ending before the operation (`…/v1`, never `…/v1/chat/completions`).
	pub base_url: url::Url,
	/// `None` for a server that takes no key (a local Ollama or vLLM).
	pub api_key: Option<Secret>,
	/// Bedrock signed with AWS credentials rather than a Bedrock API key.
	pub aws: Option<Aws>,
}

/// What a Bedrock provider signs with, and where.
#[derive(Debug, Clone)]
pub struct Aws {
	pub region: String,
	pub credentials: Credentials,
}

impl Provider {
	/// The URL of one operation, such as `chat/completions`.
	pub fn endpoint(&self, operation: &str) -> String {
		format!(
			"{}/{operation}",
			self.base_url.as_str().trim_end_matches('/')
		)
	}

	/// The credential, presented the way this provider takes it. `url` and `body` are what is
	/// sent, which a SigV4 signature covers.
	pub fn authorize(
		&self,
		request: reqwest::RequestBuilder,
		url: &url::Url,
		body: &[u8],
		unix: u64,
	) -> reqwest::RequestBuilder {
		if let Some(aws) = &self.aws {
			let host = match url.port() {
				Some(port) => format!("{}:{port}", url.host_str().unwrap_or("")),
				None => url.host_str().unwrap_or("").to_string(),
			};
			let signed = sigv4::sign(
				&aws.credentials,
				&aws.region,
				"bedrock",
				&sigv4::Request {
					method: "POST",
					host: &host,
					path: url.path(),
					content_type: "application/json",
					body,
				},
				&sigv4::amz_date(unix),
			);
			let mut request = request
				.header("authorization", signed.authorization)
				.header("x-amz-date", signed.date);
			if let Some(token) = signed.security_token {
				request = request.header("x-amz-security-token", token);
			}
			return request;
		}
		match (self.kind, &self.api_key) {
			(_, None) => request,
			(Kind::Openai | Kind::Bedrock, Some(key)) => request.bearer_auth(key.expose()),
			(Kind::Anthropic, Some(key)) => request.header("x-api-key", key.expose()),
		}
	}
}

#[derive(Debug, Clone)]
pub struct Model {
	/// What callers ask for.
	pub name: String,
	pub provider: Arc<Provider>,
	/// What the provider calls it.
	pub upstream: String,
	/// Whether the model takes `cache_control` breakpoints. Off unless the catalogue says so.
	pub prompt_caching: bool,
	/// The reply length to ask for when the caller names none.
	pub max_output_tokens: Option<u64>,
	/// How the model takes `reasoning_effort`; `None` refuses it rather than guess.
	pub reasoning: Option<Reasoning>,
	/// For a Bedrock embedding model, the body it takes.
	pub embeddings: Option<Embeddings>,
}

#[derive(Debug, Clone, Default)]
pub struct Catalogue {
	models: Vec<Arc<Model>>,
	by_name: HashMap<String, Arc<Model>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
	providers: Vec<ProviderEntry>,
	models: Vec<ModelEntry>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderEntry {
	name: String,
	kind: Kind,
	/// Required, except for Bedrock, where it defaults to the region's endpoint.
	#[serde(default)]
	base_url: Option<String>,
	#[serde(default)]
	api_key_env: Option<String>,
	/// Bedrock's region.
	#[serde(default)]
	region: Option<String>,
	#[serde(default)]
	aws: Option<AwsEntry>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AwsEntry {
	access_key_id_env: String,
	secret_access_key_env: String,
	#[serde(default)]
	session_token_env: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelEntry {
	name: String,
	provider: String,
	#[serde(default)]
	upstream: Option<String>,
	#[serde(default)]
	prompt_caching: bool,
	#[serde(default)]
	max_output_tokens: Option<u64>,
	#[serde(default)]
	reasoning: Option<Reasoning>,
	#[serde(default)]
	embeddings: Option<Embeddings>,
}

impl Catalogue {
	/// Reads a catalogue file. `env` is where each `api_key_env` is looked up.
	pub fn from_json(
		text: &str,
		env: &HashMap<String, String>,
		egress: Egress,
	) -> Result<Self, ConfigError> {
		let file: File = serde_json::from_str(text)
			.map_err(|e| ConfigError(format!("the providers file is not valid: {e}")))?;
		let mut providers: HashMap<String, Arc<Provider>> = HashMap::new();
		for entry in file.providers {
			if providers.contains_key(&entry.name) {
				return Err(ConfigError(format!(
					"the providers file names the provider {} twice",
					entry.name
				)));
			}
			let name = entry.name.clone();
			let fail = |what: String| ConfigError(format!("provider {name}: {what}"));
			let region = match (&entry.kind, &entry.region) {
				(Kind::Bedrock, None) => {
					return Err(fail("a Bedrock provider names its region".into()));
				}
				(Kind::Bedrock, Some(region))
					if region.is_empty()
						|| !region
							.bytes()
							.all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-') =>
				{
					return Err(fail(format!("{region} is not an AWS region")));
				}
				(Kind::Bedrock, Some(region)) => Some(region.clone()),
				(_, Some(_)) => return Err(fail("only a Bedrock provider takes a region".into())),
				(_, None) => None,
			};
			let base_url = match (&entry.base_url, &region) {
				(Some(base), _) => base.clone(),
				(None, Some(region)) => format!("https://bedrock-runtime.{region}.amazonaws.com"),
				(None, None) => return Err(fail("base_url is missing".into())),
			};
			let base_url = url::Url::parse(&base_url)
				.map_err(|_| fail(format!("base_url is not a URL: {base_url}")))?;
			check_base_url(&base_url, egress).map_err(&fail)?;
			let read = |var: &str| {
				env.get(var)
					.map(|v| v.trim())
					.filter(|v| !v.is_empty())
					.map(Secret::new)
					.ok_or_else(|| fail(format!("takes a credential from {var}, which is not set")))
			};
			let api_key = entry.api_key_env.as_deref().map(read).transpose()?;
			let aws = match (entry.aws, &region) {
				(None, _) => None,
				(Some(_), None) => {
					return Err(fail("only a Bedrock provider takes aws credentials".into()));
				}
				(Some(aws), Some(region)) => {
					if api_key.is_some() {
						return Err(fail("takes either api_key_env or aws, not both".into()));
					}
					Some(Aws {
						region: region.clone(),
						credentials: Credentials {
							access_key_id: read(&aws.access_key_id_env)?.expose().to_string(),
							secret_access_key: read(&aws.secret_access_key_env)?,
							session_token: aws
								.session_token_env
								.as_deref()
								.map(read)
								.transpose()?,
						},
					})
				}
			};
			if entry.kind == Kind::Bedrock && api_key.is_none() && aws.is_none() {
				return Err(fail(
					"a Bedrock provider takes api_key_env (a Bedrock API key) or aws credentials"
						.into(),
				));
			}
			providers.insert(
				entry.name.clone(),
				Arc::new(Provider {
					name: entry.name,
					kind: entry.kind,
					base_url,
					api_key,
					aws,
				}),
			);
		}
		let mut catalogue = Catalogue::default();
		let mut seen = HashSet::new();
		for entry in file.models {
			if !seen.insert(entry.name.clone()) {
				return Err(ConfigError(format!(
					"the providers file names the model {} twice",
					entry.name
				)));
			}
			let provider = providers.get(&entry.provider).cloned().ok_or_else(|| {
				ConfigError(format!(
					"model {} names the provider {}, which the file does not define",
					entry.name, entry.provider
				))
			})?;
			if entry.embeddings.is_some() && provider.kind != Kind::Bedrock {
				return Err(ConfigError(format!(
					"model {}: only a Bedrock model names its embeddings form; other providers take OpenAI's request as it is",
					entry.name
				)));
			}
			let model = Arc::new(Model {
				upstream: entry.upstream.unwrap_or_else(|| entry.name.clone()),
				name: entry.name,
				provider,
				prompt_caching: entry.prompt_caching,
				max_output_tokens: entry.max_output_tokens.filter(|n| *n > 0),
				reasoning: entry.reasoning,
				embeddings: entry.embeddings,
			});
			catalogue.by_name.insert(model.name.clone(), model.clone());
			catalogue.models.push(model);
		}
		Ok(catalogue)
	}

	pub fn model(&self, name: &str) -> Option<&Arc<Model>> {
		self.by_name.get(name)
	}

	/// Every model, in the order the catalogue lists them.
	pub fn models(&self) -> &[Arc<Model>] {
		&self.models
	}

	/// Every credential the catalogue holds, for scrubbing provider errors (`error::scrub`).
	pub fn secrets(&self) -> Vec<&str> {
		let mut secrets: Vec<&str> = Vec::new();
		for model in &self.models {
			let provider = &model.provider;
			secrets.extend(provider.api_key.as_ref().map(Secret::expose));
			if let Some(aws) = &provider.aws {
				secrets.push(aws.credentials.secret_access_key.expose());
				secrets.extend(aws.credentials.session_token.as_ref().map(Secret::expose));
			}
		}
		secrets.sort_unstable();
		secrets.dedup();
		secrets
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn env() -> HashMap<String, String> {
		[(
			"OPENAI_API_KEY".to_string(),
			"sk-test-0123456789".to_string(),
		)]
		.into()
	}

	const FILE: &str = r#"{
		"providers": [
			{"name": "openai", "kind": "openai", "base_url": "https://api.openai.com/v1/", "api_key_env": "OPENAI_API_KEY"}
		],
		"models": [
			{"name": "fast", "provider": "openai", "upstream": "gpt-4o-mini"},
			{"name": "gpt-4o", "provider": "openai"}
		]
	}"#;

	#[test]
	fn a_catalogue_resolves_names_to_upstream_models() {
		let catalogue = Catalogue::from_json(FILE, &env(), Egress::Public).unwrap();
		let fast = catalogue.model("fast").unwrap();
		assert_eq!(fast.upstream, "gpt-4o-mini");
		assert_eq!(
			fast.provider.endpoint("chat/completions"),
			"https://api.openai.com/v1/chat/completions"
		);
		assert_eq!(catalogue.model("gpt-4o").unwrap().upstream, "gpt-4o");
		assert!(
			!fast.prompt_caching,
			"caching is off unless the catalogue says so"
		);
		assert_eq!(catalogue.models().len(), 2);
		assert_eq!(catalogue.secrets(), vec!["sk-test-0123456789"]);
		assert!(!format!("{fast:?}").contains("sk-test"));
	}

	#[test]
	fn mistakes_stop_the_server_with_their_name() {
		let error = Catalogue::from_json(FILE, &HashMap::new(), Egress::Public).unwrap_err();
		assert!(error.0.contains("OPENAI_API_KEY"), "{error}");

		let plain_http = FILE.replace("https://api.openai.com", "http://api.openai.com");
		let error = Catalogue::from_json(&plain_http, &env(), Egress::Public).unwrap_err();
		assert!(error.0.contains("not https"), "{error}");
		assert!(Catalogue::from_json(&plain_http, &env(), Egress::Private).is_ok());

		let orphan = FILE.replace(
			r#""provider": "openai", "upstream""#,
			r#""provider": "nobody", "upstream""#,
		);
		let error = Catalogue::from_json(&orphan, &env(), Egress::Public).unwrap_err();
		assert!(error.0.contains("nobody"), "{error}");

		let twice = FILE.replace(r#""name": "gpt-4o""#, r#""name": "fast""#);
		assert!(
			Catalogue::from_json(&twice, &env(), Egress::Public)
				.unwrap_err()
				.0
				.contains("twice")
		);

		let bedrock = |extra: &str| {
			format!(
				r#"{{"providers": [{{"name": "b", "kind": "bedrock"{extra}}}], "models": [{{"name": "m", "provider": "b"}}]}}"#
			)
		};
		let keys: HashMap<String, String> = [
			("AK", "AKIDEXAMPLE"),
			("SK", "a-secret-access-key-0123"),
			("BK", "a-bedrock-api-key-0123"),
		]
		.into_iter()
		.map(|(k, v)| (k.to_string(), v.to_string()))
		.collect();
		let signed = Catalogue::from_json(
			&bedrock(
				r#", "region": "us-west-2", "aws": {"access_key_id_env": "AK", "secret_access_key_env": "SK"}"#,
			),
			&keys,
			Egress::Public,
		)
		.unwrap();
		let provider = &signed.model("m").unwrap().provider;
		assert_eq!(
			provider.base_url.as_str(),
			"https://bedrock-runtime.us-west-2.amazonaws.com/"
		);
		assert_eq!(provider.aws.as_ref().unwrap().region, "us-west-2");
		assert_eq!(signed.secrets(), vec!["a-secret-access-key-0123"]);
		assert!(!format!("{provider:?}").contains("a-secret-access-key"));
		for (extra, said) in [
			("", "region"),
			(r#", "region": "us-west-2""#, "api_key_env"),
			(r#", "region": "US WEST""#, "not an AWS region"),
			(
				r#", "region": "us-west-2", "api_key_env": "BK", "aws": {"access_key_id_env": "AK", "secret_access_key_env": "SK"}"#,
				"not both",
			),
			(
				r#", "region": "us-west-2", "aws": {"access_key_id_env": "AK", "secret_access_key_env": "NOPE"}"#,
				"NOPE",
			),
		] {
			let error = Catalogue::from_json(&bedrock(extra), &keys, Egress::Public).unwrap_err();
			assert!(error.0.contains(said), "{extra}: {error}");
		}
		assert!(
			Catalogue::from_json(
				&bedrock(r#", "region": "us-west-2", "api_key_env": "BK""#),
				&keys,
				Egress::Public
			)
			.is_ok()
		);
		let openai_with_region = FILE.replace(
			r#""kind": "openai","#,
			r#""kind": "openai", "region": "us-west-2","#,
		);
		assert!(Catalogue::from_json(&openai_with_region, &env(), Egress::Public).is_err());

		let unknown_field = FILE.replace(
			r#""kind": "openai""#,
			r#""kind": "openai", "api_key": "sk-in-a-file""#,
		);
		assert!(Catalogue::from_json(&unknown_field, &env(), Egress::Public).is_err());
	}
}
