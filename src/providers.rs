//! The catalogue: which providers Psyche may call, with which credential, and which model names it
//! answers to.
//!
//! Without a database it is a JSON file (`PSYCHE_PROVIDERS_FILE`), the standalone case. With one,
//! the same shape is rows of `psyche.providers` and `psyche.models`, which the file seeds and the
//! admin API edits (`models.rs`); both are read into a [`Spec`] and built the same way. A provider's
//! key is never in the file: the file names the environment variable that holds it (`api_key_env`), so the
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
//! With a database, a provider may instead name a **stored credential** (`"credential": "openai"`),
//! a row of `psyche.credentials` set through the admin API and never read back (`credentials.rs`).
//! The catalogue as the file describes it is then the base, and the one calls use is the base with
//! the stored credentials filled in ([`Catalogue::with_credentials`], held by [`Live`]), rebuilt
//! whenever a credential changes, so a new key restarts nothing.
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
//! And `price` (`price.rs`), which overrides the price catalogue shipped in the binary; a model
//! that has neither is served and logged with no cost.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use serde::Deserialize;

use crate::config::ConfigError;
use crate::net::{Egress, check_base_url};
use crate::price::{self, Place, Price, PriceEntry};
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

impl Kind {
	pub fn as_str(self) -> &'static str {
		match self {
			Kind::Openai => "openai",
			Kind::Anthropic => "anthropic",
			Kind::Bedrock => "bedrock",
		}
	}
}

/// Reads one of the enums below from the word a row holds for it.
pub fn word<T: serde::de::DeserializeOwned>(text: &str) -> Option<T> {
	serde_json::from_value(serde_json::Value::String(text.to_string())).ok()
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

impl Reasoning {
	pub fn as_str(self) -> &'static str {
		match self {
			Reasoning::Budget => "budget",
			Reasoning::Adaptive => "adaptive",
			Reasoning::Effort => "effort",
		}
	}
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

impl Embeddings {
	pub fn as_str(self) -> &'static str {
		match self {
			Embeddings::Titan => "titan",
			Embeddings::Cohere => "cohere",
		}
	}
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
	/// A Bedrock provider's region.
	pub region: Option<String>,
	/// The stored credential it takes (`psyche.credentials`), by name. Its key or AWS credentials
	/// are filled in from there, and are `None` until one is stored.
	pub credential: Option<String>,
	/// The header the key is sent in, for a server behind a gateway that checks one, instead of the
	/// provider's own way (a Bearer token, or Anthropic's `x-api-key`).
	pub auth_header: Option<String>,
}

/// A stored credential, as `psyche.credentials` holds it.
#[derive(Clone)]
pub enum Stored {
	ApiKey(Secret),
	Aws(Credentials),
}

/// The catalogue calls are made with: the base the file describes, with the stored credentials
/// filled in, swapped whole when one changes. A call holds its own copy of its model, so a swap
/// never changes a call already sent.
#[derive(Clone)]
pub struct Live(Arc<RwLock<Arc<Catalogue>>>);

impl Live {
	pub fn new(catalogue: Catalogue) -> Self {
		Self(Arc::new(RwLock::new(Arc::new(catalogue))))
	}

	pub fn get(&self) -> Arc<Catalogue> {
		self.0.read().unwrap_or_else(|e| e.into_inner()).clone()
	}

	pub fn set(&self, catalogue: Catalogue) {
		*self.0.write().unwrap_or_else(|e| e.into_inner()) = Arc::new(catalogue);
	}
}

impl std::fmt::Debug for Live {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.write_str("Live(..)")
	}
}

/// A stored credential's name: what a URL path and a log line can carry.
pub fn credential_name_ok(name: &str) -> bool {
	!name.is_empty()
		&& name.len() <= 63
		&& name
			.bytes()
			.all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_')
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
		if let (Some(header), Some(key)) = (&self.auth_header, &self.api_key) {
			return request.header(header.as_str(), key.expose());
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
	/// What it costs; `None` when neither the file nor the shipped catalogue prices it.
	pub price: Option<Price>,
}

#[derive(Debug, Clone, Default)]
pub struct Catalogue {
	models: Vec<Arc<Model>>,
	by_name: HashMap<String, Arc<Model>>,
	routes: Vec<Arc<Route>>,
}

/// A name a caller may ask for that stands for several models (PS15): tried in turn, each next one
/// only when the one before failed for a reason that was not the caller's (`proxy.rs`).
#[derive(Debug, Clone, PartialEq)]
pub struct Route {
	pub name: String,
	/// The models, as written; those not in the catalogue are skipped and named by the admin API.
	pub targets: Vec<String>,
	pub sort: Sort,
}

/// The order a route tries its models in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Sort {
	/// As written.
	#[default]
	Order,
	/// The cheapest first, by input plus output price per token; a model with no price last, and
	/// a tie kept in written order.
	Price,
}

impl Sort {
	pub fn as_str(self) -> &'static str {
		match self {
			Sort::Order => "order",
			Sort::Price => "price",
		}
	}
}

/// A route as the file or a row writes it.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouteSpec {
	pub name: String,
	pub targets: Vec<String>,
	#[serde(default)]
	pub sort: Sort,
}

/// A catalogue before its models are built: the providers, resolved (with what the environment
/// holds for them), and the models as written. The providers file is read into one, and so are the
/// `psyche.providers` and `psyche.models` rows (`models.rs`), so both build the same way.
#[derive(Debug, Clone, Default)]
pub struct Spec {
	pub providers: Vec<Arc<Provider>>,
	pub models: Vec<ModelSpec>,
	pub routes: Vec<RouteSpec>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
	providers: Vec<ProviderEntry>,
	models: Vec<ModelSpec>,
	#[serde(default)]
	routes: Vec<RouteSpec>,
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
	/// A stored credential, by name, instead of `api_key_env` or `aws`.
	#[serde(default)]
	credential: Option<String>,
	#[serde(default)]
	auth_header: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AwsEntry {
	access_key_id_env: String,
	secret_access_key_env: String,
	#[serde(default)]
	session_token_env: Option<String>,
}

/// A model as the file or a row writes it.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSpec {
	pub name: String,
	pub provider: String,
	#[serde(default)]
	pub upstream: Option<String>,
	#[serde(default)]
	pub prompt_caching: bool,
	#[serde(default)]
	pub max_output_tokens: Option<u64>,
	#[serde(default)]
	pub reasoning: Option<Reasoning>,
	#[serde(default)]
	pub embeddings: Option<Embeddings>,
	/// As written, in dollars per million tokens (`price.rs`); checked when the model is built.
	#[serde(default)]
	pub price: Option<serde_json::Value>,
}

impl Provider {
	/// A provider as a row describes it: no secret from the environment, only a stored credential.
	pub fn from_row(
		name: &str,
		kind: Kind,
		base_url: Option<&str>,
		region: Option<&str>,
		credential: Option<&str>,
		auth_header: Option<&str>,
		egress: Egress,
	) -> Result<Provider, String> {
		let entry = ProviderEntry {
			name: name.to_string(),
			kind,
			base_url: base_url.map(str::to_string),
			api_key_env: None,
			region: region.map(str::to_string),
			aws: None,
			credential: credential.map(str::to_string),
			auth_header: auth_header.map(str::to_string),
		};
		resolve(entry, &HashMap::new(), egress).map_err(|e| e.0)
	}
}

/// Checks a provider as the file or a row writes it, and reads its secrets from `env`.
fn resolve(
	entry: ProviderEntry,
	env: &HashMap<String, String>,
	egress: Egress,
) -> Result<Provider, ConfigError> {
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
					session_token: aws.session_token_env.as_deref().map(read).transpose()?,
				},
			})
		}
	};
	if let Some(credential) = &entry.credential {
		if api_key.is_some() || aws.is_some() {
			return Err(fail(
				"takes a stored credential or api_key_env or aws, only one".into(),
			));
		}
		if !credential_name_ok(credential) {
			return Err(fail(format!(
				"{credential} is not a credential's name: lowercase letters, digits, - and _, up to 63"
			)));
		}
	}
	if let Some(header) = &entry.auth_header
		&& !auth_header_ok(header)
	{
		return Err(fail(format!(
			"{header} is not a header a key can be sent in: lowercase letters, digits and -, up to 64, and not one HTTP itself uses"
		)));
	}
	if entry.auth_header.is_some() && entry.kind == Kind::Bedrock {
		return Err(fail(
			"a Bedrock provider signs its own requests, so it takes no auth_header".into(),
		));
	}
	if entry.kind == Kind::Bedrock
		&& api_key.is_none()
		&& aws.is_none()
		&& entry.credential.is_none()
	{
		return Err(fail(
			"a Bedrock provider takes api_key_env (a Bedrock API key), aws credentials or a stored credential"
				.into(),
		));
	}
	Ok(Provider {
		name: entry.name,
		kind: entry.kind,
		base_url,
		api_key,
		aws,
		region,
		credential: entry.credential,
		auth_header: entry.auth_header,
	})
}

/// A header a key may be sent in.
pub fn auth_header_ok(name: &str) -> bool {
	(1..=64).contains(&name.len())
		&& name
			.bytes()
			.all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
		&& !matches!(
			name,
			"host" | "content-type" | "content-length" | "transfer-encoding" | "connection"
		)
}

impl Spec {
	/// Reads a providers file. `env` is where each `api_key_env` is looked up.
	pub fn from_json(
		text: &str,
		env: &HashMap<String, String>,
		egress: Egress,
	) -> Result<Self, ConfigError> {
		let file: File = serde_json::from_str(text)
			.map_err(|e| ConfigError(format!("the providers file is not valid: {e}")))?;
		let mut spec = Spec::default();
		for entry in file.providers {
			if spec.provider(&entry.name).is_some() {
				return Err(ConfigError(format!(
					"the providers file names the provider {} twice",
					entry.name
				)));
			}
			spec.providers.push(Arc::new(resolve(entry, env, egress)?));
		}
		spec.models = file.models;
		spec.routes = file.routes;
		Ok(spec)
	}

	pub fn provider(&self, name: &str) -> Option<&Arc<Provider>> {
		self.providers.iter().find(|p| p.name == name)
	}
}

/// Builds one model of `spec`, against its provider.
pub fn build_model(
	spec: &ModelSpec,
	provider: &Arc<Provider>,
	prices: &price::Catalogue,
) -> Result<Model, String> {
	if spec.embeddings.is_some() && provider.kind != Kind::Bedrock {
		return Err(
			"only a Bedrock model names its embeddings form; other providers take OpenAI's request as it is"
				.into(),
		);
	}
	let upstream = spec
		.upstream
		.clone()
		.filter(|u| !u.is_empty())
		.unwrap_or_else(|| spec.name.clone());
	let price = match &spec.price {
		Some(written) => Some(
			serde_json::from_value::<PriceEntry>(written.clone())
				.map_err(|e| format!("price: {e}"))?
				.price()?,
		),
		None => prices.find(&Place {
			kind: provider.kind,
			region: provider.region.as_deref(),
			host: provider.base_url.host_str(),
			upstream: &upstream,
		}),
	};
	Ok(Model {
		upstream,
		name: spec.name.clone(),
		provider: provider.clone(),
		prompt_caching: spec.prompt_caching,
		max_output_tokens: spec.max_output_tokens.filter(|n| *n > 0),
		reasoning: spec.reasoning,
		embeddings: spec.embeddings,
		price,
	})
}

impl Catalogue {
	/// Reads a catalogue file. `env` is where each `api_key_env` is looked up.
	pub fn from_json(
		text: &str,
		env: &HashMap<String, String>,
		egress: Egress,
	) -> Result<Self, ConfigError> {
		Self::build(&Spec::from_json(text, env, egress)?)
	}

	/// Builds every model of `spec`, stopping at the first mistake: a file that is wrong stops the
	/// start.
	pub fn build(spec: &Spec) -> Result<Self, ConfigError> {
		let (catalogue, skipped) = Self::assemble(spec, true)?;
		debug_assert!(skipped.is_empty());
		Ok(catalogue)
	}

	/// Builds the models of `spec` it can, and says why it left out each of the others: rows are
	/// edited one at a time, and one that is wrong must not take the others down.
	pub fn build_lenient(spec: &Spec) -> Result<(Self, Vec<(String, String)>), ConfigError> {
		Self::assemble(spec, false)
	}

	fn assemble(spec: &Spec, strict: bool) -> Result<(Self, Vec<(String, String)>), ConfigError> {
		let prices = price::Catalogue::shipped()?;
		let mut catalogue = Catalogue::default();
		let mut skipped = Vec::new();
		let mut seen = HashSet::new();
		for entry in &spec.models {
			let built = if !seen.insert(entry.name.clone()) {
				Err(format!(
					"the providers file names the model {} twice",
					entry.name
				))
			} else {
				match spec.provider(&entry.provider) {
					None => Err(format!(
						"model {} names the provider {}, which is not among the providers",
						entry.name, entry.provider
					)),
					Some(provider) => build_model(entry, provider, &prices)
						.map_err(|e| format!("model {}: {e}", entry.name)),
				}
			};
			match built {
				Ok(model) => {
					let model = Arc::new(model);
					catalogue.by_name.insert(model.name.clone(), model.clone());
					catalogue.models.push(model);
				}
				Err(why) if strict => return Err(ConfigError(why)),
				Err(why) => skipped.push((entry.name.clone(), why)),
			}
		}
		for entry in &spec.routes {
			let why = if !seen.insert(entry.name.clone()) {
				Some(format!(
					"the route {} has the name of a model or another route",
					entry.name
				))
			} else if entry.targets.is_empty() {
				Some(format!("the route {} names no models", entry.name))
			} else if strict
				&& let Some(missing) = entry
					.targets
					.iter()
					.find(|t| !catalogue.by_name.contains_key(*t))
			{
				Some(format!(
					"the route {} names the model {missing}, which is not among the models",
					entry.name
				))
			} else {
				None
			};
			match why {
				Some(why) if strict => return Err(ConfigError(why)),
				Some(why) => skipped.push((entry.name.clone(), why)),
				None => catalogue.routes.push(Arc::new(Route {
					name: entry.name.clone(),
					targets: entry.targets.clone(),
					sort: entry.sort,
				})),
			}
		}
		Ok((catalogue, skipped))
	}

	pub fn route(&self, name: &str) -> Option<&Arc<Route>> {
		self.routes.iter().find(|r| r.name == name)
	}

	/// Every route, in the order the catalogue lists them.
	pub fn routes(&self) -> &[Arc<Route>] {
		&self.routes
	}

	/// The models a route tries, in the order it tries them: those in the catalogue, sorted as it
	/// says.
	pub fn targets(&self, route: &Route) -> Vec<Arc<Model>> {
		let mut models: Vec<Arc<Model>> = route
			.targets
			.iter()
			.filter_map(|t| self.by_name.get(t).cloned())
			.collect();
		if route.sort == Sort::Price {
			// A stable sort, so equal prices keep the written order.
			models.sort_by_key(|m| {
				m.price
					.map_or(u64::MAX, |p| p.rates.input.saturating_add(p.rates.output))
			});
		}
		models
	}

	pub fn model(&self, name: &str) -> Option<&Arc<Model>> {
		self.by_name.get(name)
	}

	/// Every model, in the order the catalogue lists them.
	pub fn models(&self) -> &[Arc<Model>] {
		&self.models
	}

	/// The models nothing prices, whose calls are logged with no cost.
	pub fn unpriced(&self) -> Vec<&str> {
		self.models
			.iter()
			.filter(|m| m.price.is_none())
			.map(|m| m.name.as_str())
			.collect()
	}

	/// This catalogue with each named credential filled in from `stored`; a provider whose
	/// credential is not there has none. AWS credentials go only to a Bedrock provider.
	pub fn with_credentials(&self, stored: &HashMap<String, Stored>) -> Catalogue {
		let mut providers: HashMap<String, Arc<Provider>> = HashMap::new();
		let mut out = Catalogue::default();
		for model in &self.models {
			let provider = providers
				.entry(model.provider.name.clone())
				.or_insert_with(|| {
					let mut provider = (*model.provider).clone();
					if let Some(name) = &provider.credential {
						provider.api_key = None;
						provider.aws = None;
						match (stored.get(name), &provider.region) {
							(Some(Stored::ApiKey(key)), _) => provider.api_key = Some(key.clone()),
							(Some(Stored::Aws(credentials)), Some(region))
								if provider.kind == Kind::Bedrock =>
							{
								provider.aws = Some(Aws {
									region: region.clone(),
									credentials: credentials.clone(),
								});
							}
							_ => {}
						}
					}
					Arc::new(provider)
				})
				.clone();
			let model = Arc::new(Model {
				provider,
				..(**model).clone()
			});
			out.by_name.insert(model.name.clone(), model.clone());
			out.models.push(model);
		}
		out.routes = self.routes.clone();
		out
	}

	/// The providers that take a stored credential, with its name.
	pub fn credentialed(&self) -> Vec<&Provider> {
		let mut out: Vec<&Provider> = Vec::new();
		for model in &self.models {
			if model.provider.credential.is_some()
				&& !out.iter().any(|p| p.name == model.provider.name)
			{
				out.push(&model.provider);
			}
		}
		out
	}

	/// The first model a provider serves, by provider name, for a request that proves a credential.
	pub fn first_model_of(&self, provider: &str) -> Option<&Arc<Model>> {
		self.models.iter().find(|m| m.provider.name == provider)
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
	fn a_stored_credential_is_filled_in_and_scrubbed_like_any_other() {
		let file = r#"{"providers": [
			{"name": "openai", "kind": "openai", "base_url": "https://api.openai.com/v1", "credential": "openai"},
			{"name": "bedrock", "kind": "bedrock", "region": "us-west-2", "credential": "aws-prod"}
		], "models": [
			{"name": "fast", "provider": "openai", "upstream": "gpt-4o-mini"},
			{"name": "sonnet", "provider": "bedrock", "upstream": "us.anthropic.claude-sonnet-5"}
		]}"#;
		let base = Catalogue::from_json(file, &HashMap::new(), Egress::Public).unwrap();
		assert!(base.secrets().is_empty());
		assert_eq!(base.credentialed().len(), 2);
		let stored = HashMap::from([
			(
				"openai".to_string(),
				Stored::ApiKey(Secret::new("sk-stored-0123")),
			),
			(
				"aws-prod".to_string(),
				Stored::Aws(Credentials {
					access_key_id: "AKID".into(),
					secret_access_key: Secret::new("aws-secret-0123"),
					session_token: None,
				}),
			),
		]);
		let live = base.with_credentials(&stored);
		assert_eq!(
			live.model("fast")
				.unwrap()
				.provider
				.api_key
				.as_ref()
				.map(Secret::expose),
			Some("sk-stored-0123")
		);
		let bedrock = &live.model("sonnet").unwrap().provider;
		assert_eq!(bedrock.aws.as_ref().unwrap().region, "us-west-2");
		assert_eq!(live.secrets(), vec!["aws-secret-0123", "sk-stored-0123"]);
		// Taken away, it is gone again.
		assert!(base.with_credentials(&HashMap::new()).secrets().is_empty());

		let both = file.replace(
			r#""credential": "openai""#,
			r#""credential": "openai", "api_key_env": "OPENAI_API_KEY""#,
		);
		let error = Catalogue::from_json(&both, &env(), Egress::Public).unwrap_err();
		assert!(error.0.contains("only one"), "{error}");
		let badly_named = file.replace(r#""credential": "openai""#, r#""credential": "Open AI""#);
		let error = Catalogue::from_json(&badly_named, &env(), Egress::Public).unwrap_err();
		assert!(error.0.contains("not a credential's name"), "{error}");
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
