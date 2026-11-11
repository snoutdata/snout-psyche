//! The OpenAI-compatible API: `/ai/v1/chat/completions`, `/ai/v1/completions`, `/ai/v1/embeddings`
//! and `/ai/v1/models` (each also at `/v1/…`, for clients whose base URL ends in `/v1`), and
//! Anthropic's Messages API, `/ai/v1/messages` and its `count_tokens` (`messages.rs`), and OpenAI's
//! Responses API, `/ai/v1/responses` (`responses.rs`), which are the same call in other shapes.
//!
//! A call is checked, resolved to a model in the catalogue, and sent to that model's provider with
//! Psyche's credential. To a provider that speaks OpenAI's API the request goes as the caller wrote
//! it, with three changes only: `model` becomes the provider's name for it, a streamed call asks for
//! usage (`stream_options.include_usage`) so every call is counted, and `cache_control` breakpoints
//! are removed for a model the catalogue does not say caches. A caller that did not ask for the
//! usage chunk does not get it. To Anthropic and to Bedrock it is translated, both ways
//! (`anthropic.rs`, `bedrock.rs`), and their streams are turned into OpenAI's as they arrive
//! (`translate.rs`). Embeddings from Bedrock are InvokeModel calls, one per text or per batch of
//! texts, sent a few at a time and put back in order (`embeddings.rs`).
//!
//! The reply carries the model name the caller asked for, never the provider's own name for it:
//! the catalogue's names are the API, and which upstream model serves one is
//! Psyche's business (the `x-psyche-provider` header says which provider did).
//!
//! The rules this file keeps:
//!  - **Bytes stream through as they arrive** (PS9): a provider's events are forwarded as the bytes
//!    it sent, one network read at a time, never buffered into a whole reply.
//!  - **A caller that goes away cancels the provider's call** (PS9): the stream is dropped, and with
//!    it the provider's connection, so nobody pays for tokens no one reads.
//!  - **After the first byte, a failure is reported in the stream**, as OpenAI reports one (a `data:`
//!    event holding an `error` object), and the stream ends. Before it, the caller gets a status.
//!  - **A provider's error never carries Psyche's credential back out** (`error::scrub`).
//!  - **A refused credential is Psyche's problem, not the caller's**: a provider's 401 or 403 is
//!    answered 503 `provider_auth`, never passed on as if the caller's own key were wrong.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::rejection::BytesRejection;
use axum::extract::{DefaultBodyLimit, Path, RawQuery, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, patch, post, put};
use futures_util::{Stream, StreamExt, TryStreamExt};
use serde_json::{Map, Value, json};

use crate::budgets::{self, Budgets, Reservation};
use crate::config::Timeouts;
use crate::credentials;
use crate::error::{ApiError, Class, scrub};
use crate::fits::{self, Needs, WebWay};
use crate::health::{self, Health, Remaining};
use crate::jwt::{self, Role};
use crate::keys::{self, Key, KeyId, Refusal};
use crate::limits::{self, Limits, Taken};
use crate::models;
use crate::net;
use crate::prefs::Prefs;
use crate::price::{Dollars, Price};
use crate::providers::{self, Catalogue, Internet, Kind, Live, Model, Provider, Sort};
use crate::sse::{Event, Parser};
use crate::translate::{Out, Translator};
use crate::usage::Usage;
use crate::who::{Members, UserId, Who};
use crate::{anthropic, bedrock, embeddings, messages, responses, sigv4};

/// How much of a provider's error body is read. Errors are a few hundred bytes.
const MAX_ERROR_BODY: usize = 64 * 1024;
/// How much of a non-streamed reply is read. A long reply is well under a megabyte.
const MAX_REPLY_BODY: usize = 64 * 1024 * 1024;
/// How many of one request's embedding calls are in flight at once.
const EMBED_CONCURRENCY: usize = 8;
/// How many times a route goes through its targets. A target that was failing when the call reached
/// it may be answering a moment later, while the one that answered before has started to fail
/// (the chaos run's excused failures).
const PASSES: usize = 2;

/// The server: the catalogue, the one outbound client, and the rules for callers.
pub struct Proxy {
	/// The catalogue calls use: the providers file's, or with a database the providers and models
	/// tables', with any stored credentials filled in.
	pub catalogue: Live,
	/// The tables behind it (`models.rs`), when there is a database.
	pub tables: Option<models::Held>,
	pub http: reqwest::Client,
	/// The client for a model at the internet level `none`, which connects only to private
	/// addresses (`net.rs`); there is none under public egress, so such a model cannot be sent.
	pub sandbox: Option<reqwest::Client>,
	/// The egress rules `http` was built with, for an address the admin API is handed.
	pub egress: net::Egress,
	pub jwt_secret: String,
	pub timeouts: Timeouts,
	pub max_request_bytes: usize,
	/// Where each finished call is sent: the request log (`log.rs`) when there is a database, and in
	/// tests whoever listens. Every call is also logged (`psyche::call`).
	pub calls: Option<tokio::sync::mpsc::UnboundedSender<Call>>,
	/// Keys (`keys.rs`), when there is a database to keep them in.
	pub keys: Option<Keys>,
	/// Budgets (`budgets.rs`), likewise.
	pub budgets: Option<Budgets>,
	/// Rate limits (`limits.rs`), likewise.
	pub limits: Option<Limits>,
	/// Each target's health (`health.rs`), which orders a route's targets.
	pub health: Health,
}

/// The keys and memberships in memory, and the database they are kept in, which the admin API
/// writes to.
#[derive(Clone)]
pub struct Keys {
	pub store: keys::Store,
	pub members: Members,
	pub database_url: String,
}

/// Who is calling.
#[derive(Debug, Clone)]
pub enum Caller {
	/// The project's service key: every model, and the admin API.
	Service,
	Key(Arc<Key>),
	/// A signed-in user of the project's app, with their own token (PS10b).
	User(UserId),
}

impl Caller {
	/// Who the call is from, for budgets, limits and the log: a key's owner and a user's group
	/// included.
	fn who(&self, members: Option<&Members>) -> Who {
		let (key, user) = match self {
			Caller::Service => (None, None),
			Caller::Key(key) => (Some(key.id), key.owner),
			Caller::User(user) => (None, Some(*user)),
		};
		match members {
			Some(members) => members.who(key, user),
			None => Who {
				key,
				user,
				group: None,
			},
		}
	}

	fn may_use(&self, model: &str) -> bool {
		match self {
			Caller::Service | Caller::User(_) => true,
			Caller::Key(key) => key.allows(model),
		}
	}
}

/// One call, as the request log will hold it.
#[derive(Debug, Clone, PartialEq)]
pub struct Call {
	/// When the call reached Psyche.
	pub at: SystemTime,
	pub operation: Operation,
	pub model: String,
	pub provider: String,
	pub streamed: bool,
	pub outcome: Outcome,
	pub usage: Option<Usage>,
	/// The model's price, so the log can cost the call; `None` when nothing prices it.
	pub price: Option<Price>,
	/// Who made the call: its key, its user and the user's group; none of them for the service key.
	pub who: Who,
	/// From sending the request to the provider's response headers.
	pub first_byte: Duration,
	pub total: Duration,
	/// Paid by the user's credits rather than their budgets (`budgets.rs`).
	pub credits: bool,
	/// The soft budgets the call went over, as the response's header names them.
	pub budget_warning: Option<String>,
	/// The route the caller named, or `models` for OpenRouter's fallback list; `None` for a model.
	pub route: Option<String>,
	/// Which attempt of the call this was, from 1.
	pub attempt: u16,
	/// For a call made for `auto`, a route or a fallback list: how its models were ordered and which
	/// were left out, and why (`fits.rs`).
	pub routing: Option<String>,
}

impl Call {
	/// What the call cost in dollars, when it was counted and its model is priced.
	pub fn cost(&self) -> Option<Dollars> {
		Some(Dollars(self.price?.cost(&self.usage?)))
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
	Completed,
	/// The caller went away before the reply ended.
	Cancelled,
	Failed(Class),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
	Chat,
	Completions,
	Embeddings,
	/// Anthropic's Messages API. To a provider that speaks OpenAI's API it is a chat completion.
	Messages,
	/// OpenAI's Responses API. To every provider it is a chat completion.
	Responses,
}

impl Operation {
	fn path(self) -> &'static str {
		match self {
			Operation::Chat | Operation::Messages | Operation::Responses => "chat/completions",
			Operation::Completions => "completions",
			Operation::Embeddings => "embeddings",
		}
	}

	pub fn name(self) -> &'static str {
		match self {
			Operation::Chat => "chat",
			Operation::Completions => "completions",
			Operation::Embeddings => "embeddings",
			Operation::Messages => "messages",
			Operation::Responses => "responses",
		}
	}
}

/// What a caller sent, read only as far as Psyche needs.
#[derive(Debug, Clone)]
pub struct Inbound {
	pub body: Map<String, Value>,
	pub model: String,
	/// OpenRouter's `models`: more to try, in order, after `model`. Taken out of the body.
	pub fallbacks: Vec<String>,
	pub stream: bool,
	/// Whether the caller asked for the usage chunk itself.
	pub wants_usage: bool,
	/// OpenRouter's `provider` object (`prefs.rs`). Taken out of the body.
	pub prefs: Option<Prefs>,
	/// OpenRouter's `session_id`: what the caller calls this conversation. Taken out of the body.
	pub session: Option<String>,
	/// Whether a name was sent with the `:online` suffix. One call, one capability: it applies to
	/// every target, since one that answered without the web would look like one that used it.
	pub online: bool,
	/// For `/messages`: the caller's `anthropic-beta` and `anthropic-version`, which an Anthropic
	/// provider gets as the caller sent them.
	pub beta: Option<String>,
	pub version: Option<String>,
}

impl Inbound {
	pub fn parse(bytes: &[u8]) -> Result<Self, ApiError> {
		let value: Value = serde_json::from_slice(bytes).map_err(|e| {
			ApiError::new(
				Class::BadRequest,
				format!("The request is not valid JSON: {e}"),
			)
		})?;
		let Value::Object(mut body) = value else {
			return Err(ApiError::new(
				Class::BadRequest,
				"The request is a JSON object.",
			));
		};
		// OpenRouter's fallback list, and its `route: "fallback"`, which says what a list already
		// does; neither goes to a provider.
		body.remove("route");
		let prefs = match body.remove("provider") {
			None | Some(Value::Null) => None,
			Some(value) => Some(Prefs::parse(value)?),
		};
		let session = match body.remove("session_id") {
			None | Some(Value::Null) => None,
			Some(Value::String(session)) => Some(session_checked(session)?),
			Some(_) => return Err(ApiError::new(Class::BadRequest, "session_id is a string.")),
		};
		let mut fallbacks = match body.remove("models") {
			None | Some(Value::Null) => Vec::new(),
			Some(Value::Array(names))
				if names.len() <= models::MOST_TARGETS
					&& names
						.iter()
						.all(|n| n.as_str().is_some_and(|n| !n.is_empty())) =>
			{
				names
					.iter()
					.filter_map(|n| n.as_str().map(String::from))
					.collect()
			}
			Some(_) => {
				return Err(ApiError::new(
					Class::BadRequest,
					format!(
						"models is a list of up to {} model names, tried in order after model.",
						models::MOST_TARGETS
					),
				));
			}
		};
		let model = match body.get("model") {
			Some(Value::String(model)) if !model.is_empty() => model.clone(),
			None | Some(Value::Null) if !fallbacks.is_empty() => fallbacks.remove(0),
			_ => {
				return Err(ApiError::new(
					Class::BadRequest,
					"The request names no model. GET /ai/v1/models lists them.",
				));
			}
		};
		let stream = match body.get("stream") {
			None | Some(Value::Null) => false,
			Some(Value::Bool(stream)) => *stream,
			Some(_) => {
				return Err(ApiError::new(Class::BadRequest, "stream is true or false."));
			}
		};
		let wants_usage = body
			.get("stream_options")
			.and_then(|o| o.get("include_usage"))
			.and_then(Value::as_bool)
			== Some(true);
		Ok(Self {
			body,
			model,
			fallbacks,
			stream,
			wants_usage,
			prefs,
			session,
			online: false,
			beta: None,
			version: None,
		})
	}

	/// The body the provider gets, for this operation.
	fn upstream(&self, model: &Model, operation: Operation) -> Result<Vec<u8>, ApiError> {
		if operation == Operation::Messages {
			return self.messages_upstream(model);
		}
		if operation == Operation::Responses {
			return self.responses_upstream(model);
		}
		match model.provider.kind {
			Kind::Openai => return Ok(self.openai(model)),
			Kind::Anthropic | Kind::Bedrock if operation != Operation::Chat => {
				return Err(ApiError::new(
					Class::BadRequest,
					format!(
						"{}'s provider has no text-completion API. Use /chat/completions.",
						model.name
					),
				));
			}
			_ => {}
		}
		let body = match model.provider.kind {
			Kind::Bedrock => bedrock::request(&self.body, model)?,
			_ => anthropic::request(&self.anthropic_body(), model, self.stream)?,
		};
		Ok(serde_json::to_vec(&body).unwrap_or_default())
	}

	/// The body the provider gets for `/messages`: the caller's, to an Anthropic provider, and
	/// otherwise the translation `messages.rs` makes, carried as a chat completion is.
	fn messages_upstream(&self, model: &Model) -> Result<Vec<u8>, ApiError> {
		let provider = &model.provider;
		if provider.kind == Kind::Anthropic {
			let mut body = self.anthropic_body().into_owned();
			body.insert("model".into(), Value::String(model.upstream.clone()));
			if !model.prompt_caching {
				strip_cache_control(&mut body);
			}
			return Ok(serde_json::to_vec(&Value::Object(body)).unwrap_or_default());
		}
		let mut chat = messages::to_chat(&self.body, model)?;
		if provider.kind == Kind::Bedrock {
			let converse = bedrock::request(&chat, model)?;
			return Ok(serde_json::to_vec(&converse).unwrap_or_default());
		}
		let field = cap_field(provider, Operation::Chat);
		if field != "max_tokens"
			&& let Some(cap) = chat.remove("max_tokens")
		{
			chat.insert(field.into(), cap);
		}
		let translated = Self {
			body: chat,
			model: self.model.clone(),
			fallbacks: Vec::new(),
			stream: self.stream,
			wants_usage: true,
			prefs: None,
			session: None,
			online: self.online,
			beta: None,
			version: None,
		};
		Ok(translated.openai(model))
	}

	/// The body the provider gets for `/responses`: the translation `responses.rs` makes, carried as
	/// a chat completion is, whichever the provider.
	fn responses_upstream(&self, model: &Model) -> Result<Vec<u8>, ApiError> {
		let mut chat = responses::to_chat(&self.body, model)?;
		let field = cap_field(&model.provider, Operation::Chat);
		if field != "max_tokens"
			&& let Some(cap) = chat.remove("max_tokens")
		{
			chat.insert(field.into(), cap);
		}
		let translated = Self {
			body: chat,
			model: self.model.clone(),
			fallbacks: Vec::new(),
			stream: self.stream,
			wants_usage: true,
			prefs: None,
			session: None,
			online: self.online,
			beta: None,
			version: None,
		};
		translated.upstream(model, Operation::Chat)
	}

	/// Writes out what `:online` and a plain `web` plugin ask for, gives a search tool that names no
	/// `max_uses` the bound (`anthropic::SEARCHES_BOUND`), and says the most web searches this call
	/// can be billed for. Only Anthropic bills them apart from the tokens, as far as Psyche knows.
	fn bound_searches(&mut self, model: &Model) -> u64 {
		if model.provider.kind != Kind::Anthropic {
			return 0;
		}
		let written = match self.anthropic_body() {
			Cow::Owned(body) => Some(body),
			Cow::Borrowed(_) => None,
		};
		if let Some(body) = written {
			self.body = body;
		}
		anthropic::bound_searches(&mut self.body)
	}

	/// The caller's body for Anthropic: `:online` and a plain `web` plugin become its own web search
	/// tool, the form it takes. The needs check has already kept out any other form.
	fn anthropic_body(&self) -> Cow<'_, Map<String, Value>> {
		if !self.online && !self.body.contains_key("plugins") {
			return Cow::Borrowed(&self.body);
		}
		let mut body = self.body.clone();
		if fits::take_plain_web_plugin(&mut body) || self.online {
			let tools = body
				.entry("tools")
				.or_insert_with(|| Value::Array(Vec::new()));
			if !tools.is_array() {
				*tools = Value::Array(Vec::new());
			}
			if let Value::Array(tools) = tools
				&& !tools.iter().any(|t| {
					t["type"]
						.as_str()
						.is_some_and(|k| k.starts_with("web_search_"))
				}) {
				tools.push(json!({"type": anthropic::WEB_SEARCH, "name": "web_search"}));
			}
		}
		Cow::Owned(body)
	}

	/// To an OpenAI-style provider: the provider's name for the model, usage asked for, and no
	/// breakpoint for a model that does not cache.
	fn openai(&self, model: &Model) -> Vec<u8> {
		let mut body = self.body.clone();
		body.insert("model".into(), Value::String(model.upstream.clone()));
		if !model.prompt_caching {
			strip_cache_control(&mut body);
		}
		openai_web(&mut body, self.online, model);
		if self.stream {
			let options = body
				.entry("stream_options")
				.or_insert_with(|| Value::Object(Map::new()));
			if !options.is_object() {
				*options = Value::Object(Map::new());
			}
			if let Value::Object(options) = options {
				options.insert("include_usage".into(), Value::Bool(true));
			}
		}
		serde_json::to_vec(&Value::Object(body)).unwrap_or_default()
	}
}

/// Gives an OpenAI-style provider web search in the form it takes, for `:online` (and a plain `web`
/// plugin, for a provider that has no plugins): OpenRouter's plugin, or OpenAI's field. A provider
/// whose upstream name already says `:online` needs nothing added.
fn openai_web(body: &mut Map<String, Value>, online: bool, model: &Model) {
	if model.upstream.ends_with(":online") {
		return;
	}
	match WebWay::of(&model.provider) {
		WebWay::OpenRouter if online => {
			let plugins = body
				.entry("plugins")
				.or_insert_with(|| Value::Array(Vec::new()));
			if !plugins.is_array() {
				*plugins = Value::Array(Vec::new());
			}
			if let Value::Array(plugins) = plugins
				&& !plugins.iter().any(|p| p["id"] == "web")
			{
				plugins.push(json!({"id": "web"}));
			}
		}
		WebWay::OpenAi | WebWay::Other => {
			let converted =
				WebWay::of(&model.provider) == WebWay::OpenAi && fits::take_plain_web_plugin(body);
			if online || converted {
				body.entry("web_search_options")
					.or_insert_with(|| json!({}));
			}
		}
		_ => {}
	}
}

/// A conversation's name, as long as OpenRouter takes one.
fn session_checked(session: String) -> Result<String, ApiError> {
	if session.len() > health::LONGEST_SESSION {
		return Err(ApiError::new(
			Class::BadRequest,
			format!(
				"session_id is at most {} characters.",
				health::LONGEST_SESSION
			),
		));
	}
	Ok(session)
}

/// What a name's suffixes asked for (`prefs.rs`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Suffixes {
	floor: bool,
	nitro: bool,
	online: bool,
}

/// A name without OpenRouter's suffixes, and what they asked for. A name the catalogue has as it
/// stands is left alone, which is how a model's own `:0` or `auto:floor` keeps its meaning, and a
/// name that is unknown without its suffixes too is left for the caller to be told about whole.
fn without_suffixes<'a>(catalogue: &Catalogue, name: &'a str) -> (&'a str, Suffixes) {
	let known =
		|n: &str| fits::is_auto(n) || catalogue.route(n).is_some() || catalogue.model(n).is_some();
	let mut suffixes = Suffixes::default();
	let mut base = name;
	while !known(base) {
		let Some((rest, suffix)) = base.rsplit_once(':') else {
			return (name, Suffixes::default());
		};
		match suffix {
			"online" if !suffixes.online => suffixes.online = true,
			"floor" if !suffixes.floor => suffixes.floor = true,
			"nitro" if !suffixes.nitro => suffixes.nitro = true,
			_ => return (name, Suffixes::default()),
		}
		base = rest;
	}
	(base, suffixes)
}

/// Why `provider.require_parameters` leaves `model` out: the translation for it would refuse
/// something the request sends, or it cannot be shown to take what the request needs.
fn parameters_refused(
	inbound: &Inbound,
	model: &Model,
	operation: Operation,
	needs: &Needs,
) -> Option<String> {
	if let Err(error) = inbound.upstream(model, operation) {
		return Some(error.message);
	}
	if model.fits.is_none() {
		let wanted = [
			(needs.tools, "tools"),
			(needs.images, "images"),
			(needs.json_schema, "a JSON schema"),
		]
		.into_iter()
		.find_map(|(asked, what)| asked.then_some(what));
		if let Some(what) = wanted {
			return Some(format!(
				"provider.require_parameters is set and it does not say it takes {what}"
			));
		}
	}
	needs.rules_out(model)
}

/// The most a reply can be, in tokens: what is sent as its cap (Anthropic always gets one, Bedrock
/// when one is named), else the caller's own, else the model's entry, else [`budgets::UNCAPPED_OUTPUT`].
fn output_cap(inbound: &Inbound, upstream: &[u8], model: &Model) -> u64 {
	let sent =
		match model.provider.kind {
			Kind::Openai => None,
			Kind::Anthropic | Kind::Bedrock => serde_json::from_slice::<Value>(upstream)
				.ok()
				.and_then(|v| {
					v.get("max_tokens")
						.or_else(|| v.pointer("/inferenceConfig/maxTokens"))
						.and_then(Value::as_u64)
				}),
		};
	sent.or_else(|| {
		["max_completion_tokens", "max_tokens", "max_output_tokens"]
			.iter()
			.find_map(|f| inbound.body.get(*f).and_then(Value::as_u64))
	})
	.or(model.max_output_tokens)
	.unwrap_or(budgets::UNCAPPED_OUTPUT)
}

/// A call's worst-case cost: every prompt token at the dearer of the plain and cache-write prices,
/// `output` reply tokens and `searches` web searches.
fn worst(price: &Price, prompt: u64, output: u64, searches: u64) -> u64 {
	let at = |write: bool| {
		price.cost(&Usage {
			prompt_tokens: prompt,
			completion_tokens: output,
			cache_write_tokens: write.then_some(prompt),
			..Usage::default()
		})
	};
	at(false)
		.max(at(true))
		.saturating_add(price.searches(searches))
}

/// Where a reply cap goes in what the caller sent, so each translation carries it on: OpenAI's own
/// API (and Azure's) refuse `max_tokens` for reasoning models and take `max_completion_tokens`;
/// the legacy completions API and the other OpenAI-style servers take `max_tokens`.
fn cap_field(provider: &Provider, operation: Operation) -> &'static str {
	// `responses.rs` turns it into the provider's own on the way out.
	if operation == Operation::Responses {
		return "max_output_tokens";
	}
	let host = provider.base_url.host_str().unwrap_or("");
	let openai = host == "api.openai.com" || host.ends_with(".openai.azure.com");
	if provider.kind == Kind::Openai && operation == Operation::Chat && openai {
		"max_completion_tokens"
	} else {
		"max_tokens"
	}
}

/// The soft budgets a reservation goes over, for the response's header.
fn warned(reservation: &Option<Reservation>) -> Option<HeaderValue> {
	let names = &reservation.as_ref()?.warnings;
	if names.is_empty() {
		return None;
	}
	HeaderValue::from_str(&names.join(", ")).ok()
}

/// The rate-limit headers, on a call a limit applies to.
fn headed(mut response: Response, rate: limits::Headers) -> Response {
	for (name, value) in rate.0 {
		if let Ok(value) = HeaderValue::from_str(&value) {
			response.headers_mut().insert(name, value);
		}
	}
	response
}

fn warn(mut response: Response, warned: Option<HeaderValue>) -> Response {
	if let Some(names) = warned {
		response
			.headers_mut()
			.insert("x-psyche-budget-warning", names);
	}
	response
}

/// Removes every `cache_control` breakpoint from a request: on a message, on a content part, inside
/// a tool result, on a tool, on a system block or on the request as a whole (a model that does not
/// cache can answer a request that carries one with an error, whatever the gateway's settings say).
fn strip_cache_control(body: &mut Map<String, Value>) {
	body.remove("cache_control");
	for list in ["messages", "tools", "system"] {
		let Some(Value::Array(items)) = body.get_mut(list) else {
			continue;
		};
		for item in items.iter_mut().filter_map(Value::as_object_mut) {
			strip_cache_control_of(item);
		}
	}
}

/// Removes `cache_control` from one item and whatever content it holds.
fn strip_cache_control_of(item: &mut Map<String, Value>) {
	item.remove("cache_control");
	if let Some(Value::Array(parts)) = item.get_mut("content") {
		for part in parts.iter_mut().filter_map(Value::as_object_mut) {
			strip_cache_control_of(part);
		}
	}
}

pub fn router(proxy: Arc<Proxy>) -> Router {
	let api = Router::new()
		.route("/chat/completions", post(chat))
		.route("/completions", post(completions))
		.route("/messages", post(messages))
		.route("/messages/count_tokens", post(count_tokens))
		.route("/responses", post(responses))
		.route("/embeddings", post(embed))
		.route("/models", get(models))
		.route("/admin/keys", get(list_keys).post(create_key))
		.route("/admin/keys/{id}", delete(revoke_key))
		.route("/admin/budgets", get(list_budgets).post(create_budget))
		.route(
			"/admin/budgets/{id}",
			patch(edit_budget).delete(delete_budget),
		)
		.route("/admin/limits", get(list_limits).post(create_limit))
		.route("/admin/limits/{id}", delete(delete_limit))
		.route("/admin/groups", get(list_groups).post(create_group))
		.route("/admin/groups/{id}", delete(delete_group))
		.route(
			"/admin/members/{user_id}",
			put(set_member).delete(remove_member),
		)
		.route("/admin/providers", get(list_providers))
		.route(
			"/admin/providers/{name}",
			put(set_provider).delete(delete_provider),
		)
		.route(
			"/admin/providers/{name}/upstream-models",
			get(upstream_models_of),
		)
		.route("/admin/upstream-models", post(upstream_models_for))
		.route("/admin/grants", get(list_grants).post(create_grant))
		.route("/admin/grants/{id}", delete(delete_grant))
		.route("/admin/models", get(list_models))
		.route("/admin/models/{*name}", put(set_model).delete(delete_model))
		.route("/admin/routes", get(list_routes))
		.route("/admin/routes/{*name}", put(set_route).delete(delete_route))
		.route("/admin/credentials", get(list_credentials))
		.route("/admin/usage", get(usage))
		.route("/admin/health", get(health_report))
		.route("/admin/requests", get(requests))
		.route(
			"/admin/credentials/{name}",
			put(set_credential).delete(delete_credential),
		);
	let limit = proxy.max_request_bytes;
	Router::new()
		.nest("/ai/v1", api.clone())
		.nest("/v1", api)
		.route("/health", get(|| async { "ok" }))
		.layer(DefaultBodyLimit::max(limit))
		.with_state(proxy)
}

async fn chat(
	State(proxy): State<Arc<Proxy>>,
	headers: HeaderMap,
	body: Result<Bytes, BytesRejection>,
) -> Response {
	match proxy.forward(&headers, body, Operation::Chat).await {
		Ok(response) => response,
		Err(error) => error.into_response(),
	}
}

async fn completions(
	State(proxy): State<Arc<Proxy>>,
	headers: HeaderMap,
	body: Result<Bytes, BytesRejection>,
) -> Response {
	match proxy.forward(&headers, body, Operation::Completions).await {
		Ok(response) => response,
		Err(error) => error.into_response(),
	}
}

async fn embed(
	State(proxy): State<Arc<Proxy>>,
	headers: HeaderMap,
	body: Result<Bytes, BytesRejection>,
) -> Response {
	match proxy.forward(&headers, body, Operation::Embeddings).await {
		Ok(response) => response,
		Err(error) => error.into_response(),
	}
}

async fn messages(
	State(proxy): State<Arc<Proxy>>,
	headers: HeaderMap,
	body: Result<Bytes, BytesRejection>,
) -> Response {
	match proxy.forward(&headers, body, Operation::Messages).await {
		Ok(response) => response,
		Err(error) => messages::error_response(error),
	}
}

async fn responses(
	State(proxy): State<Arc<Proxy>>,
	headers: HeaderMap,
	body: Result<Bytes, BytesRejection>,
) -> Response {
	match proxy.forward(&headers, body, Operation::Responses).await {
		Ok(response) => response,
		Err(error) => error.into_response(),
	}
}

/// Claude Code's `count_tokens`: the same estimate the needs check makes (`fits::estimated_tokens`),
/// a third of the request's bytes, not a provider's count. Nothing is sent to a provider, so it
/// costs nothing and names no target.
async fn count_tokens(
	State(proxy): State<Arc<Proxy>>,
	headers: HeaderMap,
	body: Result<Bytes, BytesRejection>,
) -> Response {
	let counted = proxy.authorize(&headers).and_then(|_| {
		let body = read_body(body)?;
		Inbound::parse(&body)?;
		Ok(json!({"input_tokens": fits::estimated_tokens(budgets::prompt_bound(&body))}))
	});
	match counted {
		Ok(count) => axum::Json(count).into_response(),
		Err(error) => messages::error_response(error),
	}
}

/// A request's bytes, or why they could not be read.
fn read_body(body: Result<Bytes, BytesRejection>) -> Result<Bytes, ApiError> {
	body.map_err(|rejection| {
		let mut error = ApiError::new(
			Class::BadRequest,
			format!("The request could not be read: {}", rejection.body_text()),
		);
		error.status = Some(rejection.status());
		error
	})
}

/// One entry of the model list, which both API shapes read: OpenAI's fields, and Anthropic's `type`,
/// `display_name` and `created_at`, for a client written for either.
fn listed(id: &str, owned_by: &str) -> Value {
	json!({"id": id, "object": "model", "created": 0, "owned_by": owned_by,
		"type": "model", "display_name": id, "created_at": "1970-01-01T00:00:00Z"})
}

async fn models(State(proxy): State<Arc<Proxy>>, headers: HeaderMap) -> Response {
	let caller = match proxy.authorize(&headers) {
		Ok(caller) => caller,
		Err(error) => return error.into_response(),
	};
	let who = caller.who(proxy.keys.as_ref().map(|k| &k.members));
	let catalogue = proxy.catalogue.get();
	let mut data: Vec<Value> = catalogue
		.models()
		.iter()
		.filter(|m| caller.may_use(&m.name) && proxy.granted(&m.name, &who))
		.map(|m| {
			// The price each model is charged at, as the admin API writes it (dollars per million
			// tokens), so a picker can say what a model costs; null when nothing prices it.
			let mut model = listed(&m.name, &m.provider.name);
			model["pricing"] = json!(m.price.map(|p| p.to_json()));
			if let Some(context) = m.fits.and_then(|f| f.context) {
				model["max_input_tokens"] = json!(context);
			}
			if let Some(output) = m.max_output_tokens {
				model["max_tokens"] = json!(output);
			}
			model
		})
		.collect();
	// A route, with the models it tries that this caller may use; one with none is not listed.
	for route in catalogue.routes() {
		let targets: Vec<String> = catalogue
			.targets(route)
			.iter()
			.filter(|m| proxy.granted(&m.name, &who))
			.map(|m| m.name.clone())
			.collect();
		if caller.may_use(&route.name) && !targets.is_empty() {
			let mut model = listed(&route.name, "route");
			model["targets"] = json!(targets);
			data.push(model);
		}
	}
	// Automatic choice, once a model this caller may use says what it can take; a project's own
	// `auto` route is listed above as the route it is.
	let pool: Vec<&str> = catalogue
		.models()
		.iter()
		.filter(|m| m.fits.is_some() && m.embeddings.is_none() && proxy.granted(&m.name, &who))
		.map(|m| m.name.as_str())
		.collect();
	if !pool.is_empty() {
		for name in [fits::AUTO, fits::AUTO_FLOOR, fits::AUTO_NITRO] {
			if caller.may_use(name) && (name != fits::AUTO || catalogue.route(name).is_none()) {
				let mut model = listed(name, "auto");
				model["targets"] = json!(pool);
				data.push(model);
			}
		}
	}
	// Anthropic's list says where it ends; this one is whole.
	let edge = |at: Option<&Value>| at.map_or(Value::Null, |m| m["id"].clone());
	let (first, last) = (edge(data.first()), edge(data.last()));
	axum::Json(json!({"object": "list", "data": data, "has_more": false,
		"first_id": first, "last_id": last}))
	.into_response()
}

fn now() -> u64 {
	SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.map(|d| d.as_secs())
		.unwrap_or(0)
}

/// The admin API's database connection, for the service key only.
async fn admin(
	proxy: &Proxy,
	headers: &HeaderMap,
) -> Result<(Keys, tokio_postgres::Client), ApiError> {
	if !matches!(proxy.authorize(headers)?, Caller::Service) {
		return Err(ApiError::new(
			Class::Forbidden,
			"Only the project's service key may use the admin API.",
		));
	}
	let keys = proxy.keys.clone().ok_or_else(|| {
		ApiError::new(
			Class::BadRequest,
			"This Psyche has no database (DATABASE_URL), so it keeps no keys.",
		)
	})?;
	let client = crate::log::connect(&keys.database_url)
		.await
		.map_err(|e| database_error(&e))?;
	Ok((keys, client))
}

fn database_error(error: &tokio_postgres::Error) -> ApiError {
	tracing::warn!(error = %crate::db_message(error), "an admin call's database work failed");
	let mut out = ApiError::new(
		Class::Upstream,
		"Psyche's database could not do that just now.",
	);
	out.status = Some(StatusCode::SERVICE_UNAVAILABLE);
	out
}

async fn list_keys(State(proxy): State<Arc<Proxy>>, headers: HeaderMap) -> Response {
	let result = async {
		let (_, client) = admin(&proxy, &headers).await?;
		keys::list(&client).await.map_err(|e| database_error(&e))
	}
	.await;
	match result {
		Ok(data) => axum::Json(json!({"object": "list", "data": data})).into_response(),
		Err(error) => error.into_response(),
	}
}

async fn create_key(
	State(proxy): State<Arc<Proxy>>,
	headers: HeaderMap,
	body: Result<Bytes, BytesRejection>,
) -> Response {
	let result = async {
		let (keys, client) = admin(&proxy, &headers).await?;
		let body = body.map_err(|r| ApiError::new(Class::BadRequest, r.body_text()))?;
		let new: keys::NewKey = serde_json::from_slice(&body).map_err(|e| {
			ApiError::new(
				Class::BadRequest,
				format!(
					"A key is made from {{\"name\", \"models\", \"expires_in\", \"owner\"}}: {e}"
				),
			)
		})?;
		// A key may name a route as it names a model.
		let catalogue = proxy.catalogue.get();
		if let Some(unknown) = new
			.models
			.iter()
			.flatten()
			.find(|m| catalogue.model(m).is_none() && catalogue.route(m).is_none())
		{
			return Err(ApiError::new(
				Class::BadRequest,
				format!(
					"There is no model or route named {unknown} here. GET /ai/v1/models lists them."
				),
			));
		}
		keys::create(&client, &keys.store, new)
			.await
			.map_err(|e| match e {
				keys::CreateError::Invalid(message) => ApiError::new(Class::BadRequest, message),
				keys::CreateError::Database(e) => database_error(&e),
			})
	}
	.await;
	match result {
		Ok(made) => (StatusCode::CREATED, axum::Json(made)).into_response(),
		Err(error) => error.into_response(),
	}
}

async fn revoke_key(
	State(proxy): State<Arc<Proxy>>,
	headers: HeaderMap,
	Path(id): Path<String>,
) -> Response {
	let result = async {
		let (keys, client) = admin(&proxy, &headers).await?;
		let id = KeyId::parse(&id)
			.ok_or_else(|| ApiError::new(Class::BadRequest, "A key's id is a uuid."))?;
		match keys::revoke(&client, &keys.store, id).await {
			Ok(true) => Ok(()),
			Ok(false) => {
				let mut error = ApiError::new(
					Class::BadRequest,
					"There is no key with that id that is not already revoked.",
				);
				error.status = Some(StatusCode::NOT_FOUND);
				Err(error)
			}
			Err(e) => Err(database_error(&e)),
		}
	}
	.await;
	match result {
		Ok(()) => StatusCode::NO_CONTENT.into_response(),
		Err(error) => error.into_response(),
	}
}

/// What the admin API makes a budget from.
#[derive(serde::Deserialize)]
struct NewBudget {
	name: String,
	/// US dollars.
	amount: f64,
	period: String,
	#[serde(default)]
	days: Option<i32>,
	/// The first day of a `since` budget, `YYYY-MM-DD`.
	#[serde(default)]
	starts_on: Option<String>,
	#[serde(flatten)]
	whom: Whom,
	#[serde(default = "yes")]
	hard: bool,
}

/// What the admin API changes on a budget; whom it is for stays. A new `period` takes `days` and
/// `starts_on` as given (absent is none); without one, each is changed only when it is given.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct BudgetEdit {
	#[serde(default)]
	name: Option<String>,
	#[serde(default)]
	amount: Option<f64>,
	#[serde(default)]
	period: Option<String>,
	#[serde(default)]
	days: Option<i32>,
	#[serde(default)]
	starts_on: Option<String>,
	#[serde(default)]
	hard: Option<bool>,
}

fn yes() -> bool {
	true
}

/// Whom a new budget or limit is for, as the admin API takes it: at most one of these (none is the
/// project).
#[derive(serde::Deserialize, Default)]
struct Whom {
	#[serde(default)]
	key_id: Option<String>,
	#[serde(default)]
	every_user: bool,
	#[serde(default)]
	user_id: Option<String>,
	#[serde(default)]
	group_id: Option<String>,
}

/// The four columns that say whom a row is for.
type WhomColumns = (Option<KeyId>, bool, Option<KeyId>, Option<KeyId>);

impl Whom {
	fn columns(&self) -> Result<WhomColumns, ApiError> {
		let id = |text: &Option<String>, what: &str| match text {
			None => Ok(None),
			Some(text) => KeyId::parse(text)
				.map(Some)
				.ok_or_else(|| ApiError::new(Class::BadRequest, format!("{what} is a uuid."))),
		};
		let columns = (
			id(&self.key_id, "key_id")?,
			self.every_user,
			id(&self.user_id, "user_id")?,
			id(&self.group_id, "group_id")?,
		);
		let named = [
			columns.0.is_some(),
			columns.1,
			columns.2.is_some(),
			columns.3.is_some(),
		];
		if named.iter().filter(|n| **n).count() > 1 {
			return Err(ApiError::new(
				Class::BadRequest,
				"Name one of key_id, every_user, user_id and group_id, or none for the project.",
			));
		}
		Ok(columns)
	}

	fn json(&self) -> Value {
		json!({"key_id": self.key_id, "every_user": self.every_user, "user_id": self.user_id,
			"group_id": self.group_id})
	}
}

/// The `whom` columns of a listed row, from `key_id::text, every_user, user_id::text,
/// group_id::text` starting at `at`.
fn whom_json(row: &tokio_postgres::Row, at: usize) -> Value {
	json!({
		"key_id": row.get::<_, Option<String>>(at),
		"every_user": row.get::<_, bool>(at + 1),
		"user_id": row.get::<_, Option<String>>(at + 2),
		"group_id": row.get::<_, Option<String>>(at + 3),
	})
}

fn merge(mut into: Value, from: Value) -> Value {
	if let (Value::Object(into), Value::Object(from)) = (&mut into, from) {
		into.extend(from);
	}
	into
}

/// What the admin API makes a rate limit from.
#[derive(serde::Deserialize)]
struct NewLimit {
	name: String,
	#[serde(flatten)]
	whom: Whom,
	#[serde(default)]
	requests_per_minute: Option<i32>,
	#[serde(default)]
	tokens_per_minute: Option<i32>,
}

async fn reload_limits(proxy: &Proxy, client: &tokio_postgres::Client) {
	if let Some(limits) = &proxy.limits {
		match limits::load(client).await {
			Ok(list) => limits.replace(list),
			Err(error) => {
				tracing::warn!(error = %crate::db_message(&error), "could not reload the limits")
			}
		}
	}
}

const WHOM_FIELDS: [&str; 4] = ["key_id", "every_user", "user_id", "group_id"];
const BUDGET_FIELDS: &[&str] = &["name", "amount", "period", "days", "starts_on", "hard"];
const LIMIT_FIELDS: &[&str] = &["name", "requests_per_minute", "tokens_per_minute"];

/// An admin body, refusing a field it does not know (which `deny_unknown_fields` cannot do next to
/// a flattened struct).
fn admin_body<T: serde::de::DeserializeOwned>(
	body: &[u8],
	fields: &[&str],
	what: &str,
) -> Result<T, ApiError> {
	let known = || {
		let all: Vec<&str> = fields.iter().copied().chain(WHOM_FIELDS).collect();
		format!("{what} is made from {}", all.join(", "))
	};
	let value: Value = serde_json::from_slice(body)
		.map_err(|e| ApiError::new(Class::BadRequest, format!("{}: {e}", known())))?;
	if let Some(unknown) = value.as_object().and_then(|o| {
		o.keys()
			.find(|k| !fields.contains(&k.as_str()) && !WHOM_FIELDS.contains(&k.as_str()))
	}) {
		return Err(ApiError::new(
			Class::BadRequest,
			format!("{}, not {unknown}.", known()),
		));
	}
	serde_json::from_value(value)
		.map_err(|e| ApiError::new(Class::BadRequest, format!("{}: {e}", known())))
}

/// The providers, models and credentials tables, for the admin API.
fn held_tables(proxy: &Proxy) -> Result<&models::Held, ApiError> {
	proxy.tables.as_ref().ok_or_else(|| {
		ApiError::new(
			Class::BadRequest,
			"This Psyche has no database (DATABASE_URL), so its providers and models are its providers file and nothing else.",
		)
	})
}

/// Reads the catalogue again after the admin API changed a credential, so the next call uses it.
async fn reload_credentials(held: &models::Held, client: &tokio_postgres::Client) {
	if let Err(error) = held.reload(client).await {
		tracing::warn!(error = %crate::db_message(&error), "could not reload the credentials")
	}
}

fn refused(refusal: models::Refused) -> ApiError {
	match refusal {
		models::Refused::Invalid(message) => ApiError::new(Class::BadRequest, message),
		models::Refused::Conflict(message) => {
			let mut error = ApiError::new(Class::BadRequest, message);
			error.status = Some(StatusCode::CONFLICT);
			error
		}
		models::Refused::NotFound(message) => not_found(&message),
		models::Refused::Database(error) => database_error(&error),
	}
}

async fn list_providers(State(proxy): State<Arc<Proxy>>, headers: HeaderMap) -> Response {
	let result = async {
		let (_, client) = admin(&proxy, &headers).await?;
		let held = held_tables(&proxy)?;
		models::list_providers(&client, held).await.map_err(refused)
	}
	.await;
	match result {
		Ok(data) => axum::Json(json!({"object": "list", "data": data})).into_response(),
		Err(error) => error.into_response(),
	}
}

async fn set_provider(
	State(proxy): State<Arc<Proxy>>,
	headers: HeaderMap,
	Path(name): Path<String>,
	body: Result<Bytes, BytesRejection>,
) -> Response {
	let result = async {
		let (_, client) = admin(&proxy, &headers).await?;
		let held = held_tables(&proxy)?;
		let body = body.map_err(|r| ApiError::new(Class::BadRequest, r.body_text()))?;
		let new: models::NewProvider = serde_json::from_slice(&body).map_err(|e| {
			ApiError::new(
				Class::BadRequest,
				format!("A provider is made from kind, base_url, region and credential: {e}"),
			)
		})?;
		models::put_provider(&client, held, &name, new)
			.await
			.map_err(refused)
	}
	.await;
	match result {
		Ok(made) => axum::Json(made).into_response(),
		Err(error) => error.into_response(),
	}
}

async fn delete_provider(
	State(proxy): State<Arc<Proxy>>,
	headers: HeaderMap,
	Path(name): Path<String>,
) -> Response {
	let result = async {
		let (_, client) = admin(&proxy, &headers).await?;
		let held = held_tables(&proxy)?;
		models::delete_provider(&client, held, &name)
			.await
			.map_err(refused)
	}
	.await;
	match result {
		Ok(()) => StatusCode::NO_CONTENT.into_response(),
		Err(error) => error.into_response(),
	}
}

async fn list_models(State(proxy): State<Arc<Proxy>>, headers: HeaderMap) -> Response {
	let result = async {
		let (_, client) = admin(&proxy, &headers).await?;
		let held = held_tables(&proxy)?;
		models::list_models(&client, held).await.map_err(refused)
	}
	.await;
	match result {
		Ok(data) => axum::Json(json!({"object": "list", "data": data})).into_response(),
		Err(error) => error.into_response(),
	}
}

async fn set_model(
	State(proxy): State<Arc<Proxy>>,
	headers: HeaderMap,
	Path(name): Path<String>,
	body: Result<Bytes, BytesRejection>,
) -> Response {
	let result = async {
		let (_, client) = admin(&proxy, &headers).await?;
		let held = held_tables(&proxy)?;
		let body = body.map_err(|r| ApiError::new(Class::BadRequest, r.body_text()))?;
		let new: models::NewModel = serde_json::from_slice(&body).map_err(|e| {
			ApiError::new(
				Class::BadRequest,
				format!(
					"A model is made from provider, upstream, prompt_caching, max_output_tokens, reasoning, embeddings, price, fits, internet (none, provider or web), quota and enabled: {e}"
				),
			)
		})?;
		models::put_model(&client, held, &name, new)
			.await
			.map_err(refused)
	}
	.await;
	match result {
		Ok(made) => axum::Json(made).into_response(),
		Err(error) => error.into_response(),
	}
}

async fn delete_model(
	State(proxy): State<Arc<Proxy>>,
	headers: HeaderMap,
	Path(name): Path<String>,
) -> Response {
	let result = async {
		let (_, client) = admin(&proxy, &headers).await?;
		let held = held_tables(&proxy)?;
		models::delete_model(&client, held, &name)
			.await
			.map_err(refused)
	}
	.await;
	match result {
		Ok(done) => axum::Json(done).into_response(),
		Err(error) => error.into_response(),
	}
}

/// The models a saved provider offers, asked of the provider with the credential its calls use
/// (`upstream.rs`): the stored one, or the providers file's environment key.
async fn upstream_models_of(
	State(proxy): State<Arc<Proxy>>,
	headers: HeaderMap,
	Path(name): Path<String>,
) -> Response {
	let result = async {
		let provider = match &proxy.tables {
			Some(held) => {
				let (_, client) = admin(&proxy, &headers).await?;
				let (spec, left_out) = models::load(&client, &held.file, held.egress)
					.await
					.map_err(|e| database_error(&e))?;
				let Some(provider) = spec.provider(&name) else {
					return Err(match left_out.iter().find(|(n, _)| *n == name) {
						Some((_, why)) => ApiError::new(Class::BadRequest, format!("{name} is not served: {why}")),
						None => not_found(&format!("There is no provider {name}.")),
					});
				};
				let stored = credentials::load(&client)
					.await
					.map_err(|e| database_error(&e))?;
				if let Some(credential) = &provider.credential
					&& !stored.contains_key(credential)
				{
					return Err(ApiError::new(
						Class::ProviderAuth,
						format!("{name} takes the stored credential {credential}, which is not set, so it cannot be asked for its models."),
					));
				}
				provider.with_stored(&stored)
			}
			None => {
				if !matches!(proxy.authorize(&headers)?, Caller::Service) {
					return Err(ApiError::new(
						Class::Forbidden,
						"Only the project's service key may use the admin API.",
					));
				}
				let catalogue = proxy.catalogue.get();
				let model = catalogue
					.models()
					.iter()
					.find(|m| m.provider.name == name)
					.ok_or_else(|| not_found(&format!("There is no provider {name}.")))?;
				(*model.provider).clone()
			}
		};
		crate::upstream::list(&proxy.http, &provider).await
	}
	.await;
	match result {
		Ok(data) => axum::Json(json!({"object": "list", "data": data})).into_response(),
		Err(error) => error.into_response(),
	}
}

/// What `POST /admin/upstream-models` takes: a provider as `PUT /admin/providers/<name>` would
/// make it, and the credential `PUT /admin/credentials/<name>` would store, neither of them kept.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct UnsavedProvider {
	provider: models::NewProvider,
	#[serde(default)]
	credential: Option<credentials::NewCredential>,
}

/// The models a provider not saved yet offers, for a screen that adds a provider and its first
/// model together: nothing is written, and the credential is used for this one list only.
async fn upstream_models_for(
	State(proxy): State<Arc<Proxy>>,
	headers: HeaderMap,
	body: Result<Bytes, BytesRejection>,
) -> Response {
	let result = async {
		if !matches!(proxy.authorize(&headers)?, Caller::Service) {
			return Err(ApiError::new(
				Class::Forbidden,
				"Only the project's service key may use the admin API.",
			));
		}
		let body = body.map_err(|r| ApiError::new(Class::BadRequest, r.body_text()))?;
		let unsaved: UnsavedProvider = serde_json::from_slice(&body).map_err(|e| {
			ApiError::new(
				Class::BadRequest,
				format!("This takes {{\"provider\": {{kind, base_url, region, credential, auth_header}}, \"credential\": {{api_key}} or AWS credentials}}: {e}"),
			)
		})?;
		let stored = unsaved
			.credential
			.map(credentials::NewCredential::stored)
			.transpose()
			.map_err(|m| ApiError::new(Class::BadRequest, m))?;
		let egress = proxy.egress;
		let new = unsaved.provider;
		// The credential given is used under a name of its own, so a stored one by the provider's
		// name is never picked up in its place.
		let credential = match (&stored, &new.credential) {
			(Some(_), _) => Some("unsaved"),
			(None, named) => named.as_deref(),
		};
		let provider = Provider::from_row(
			"The provider",
			new.kind,
			new.base_url.as_deref(),
			new.region.as_deref(),
			credential,
			new.auth_header.as_deref(),
			egress,
		)
		.map_err(|m| ApiError::new(Class::BadRequest, m))?;
		let mut held = std::collections::HashMap::new();
		match (stored, credential, &proxy.tables) {
			(Some(stored), _, _) => {
				held.insert("unsaved".to_string(), stored);
			}
			(None, Some(_), Some(_)) => {
				let (_, client) = admin(&proxy, &headers).await?;
				held = credentials::load(&client)
					.await
					.map_err(|e| database_error(&e))?;
			}
			_ => {}
		}
		crate::upstream::list(&proxy.http, &provider.with_stored(&held)).await
	}
	.await;
	match result {
		Ok(data) => axum::Json(json!({"object": "list", "data": data})).into_response(),
		Err(error) => error.into_response(),
	}
}

/// Each target's health (`health.rs`), as Psyche holds it now; the service key only, and no
/// database needed.
async fn health_report(State(proxy): State<Arc<Proxy>>, headers: HeaderMap) -> Response {
	match proxy.authorize(&headers) {
		Ok(Caller::Service) => axum::Json(json!({
			"object": "list",
			"data": proxy.health.report(SystemTime::now()),
		}))
		.into_response(),
		Ok(_) => ApiError::new(
			Class::Forbidden,
			"Only the project's service key may use the admin API.",
		)
		.into_response(),
		Err(error) => error.into_response(),
	}
}

async fn list_routes(State(proxy): State<Arc<Proxy>>, headers: HeaderMap) -> Response {
	let result = async {
		let (_, client) = admin(&proxy, &headers).await?;
		let held = held_tables(&proxy)?;
		models::list_routes(&client, held).await.map_err(refused)
	}
	.await;
	match result {
		Ok(data) => axum::Json(json!({"object": "list", "data": data})).into_response(),
		Err(error) => error.into_response(),
	}
}

async fn set_route(
	State(proxy): State<Arc<Proxy>>,
	headers: HeaderMap,
	Path(name): Path<String>,
	body: Result<Bytes, BytesRejection>,
) -> Response {
	let result = async {
		let (_, client) = admin(&proxy, &headers).await?;
		let held = held_tables(&proxy)?;
		let body = body.map_err(|r| ApiError::new(Class::BadRequest, r.body_text()))?;
		let new: models::NewRoute = serde_json::from_slice(&body).map_err(|e| {
			ApiError::new(
				Class::BadRequest,
				format!(
					"A route is made from targets (the models, in order), sort (order, price, latency, throughput, weighted or balanced), weights and constraints: {e}"
				),
			)
		})?;
		models::put_route(&client, held, &name, new)
			.await
			.map_err(refused)
	}
	.await;
	match result {
		Ok(made) => axum::Json(made).into_response(),
		Err(error) => error.into_response(),
	}
}

async fn delete_route(
	State(proxy): State<Arc<Proxy>>,
	headers: HeaderMap,
	Path(name): Path<String>,
) -> Response {
	let result = async {
		let (_, client) = admin(&proxy, &headers).await?;
		let held = held_tables(&proxy)?;
		models::delete_route(&client, held, &name)
			.await
			.map_err(refused)
	}
	.await;
	match result {
		Ok(()) => StatusCode::NO_CONTENT.into_response(),
		Err(error) => error.into_response(),
	}
}

async fn list_credentials(State(proxy): State<Arc<Proxy>>, headers: HeaderMap) -> Response {
	let result = async {
		let (_, client) = admin(&proxy, &headers).await?;
		let held = held_tables(&proxy)?;
		credentials::list(&client, &held.live.get())
			.await
			.map_err(|e| database_error(&e))
	}
	.await;
	match result {
		Ok(data) => axum::Json(json!({"object": "list", "data": data})).into_response(),
		Err(error) => error.into_response(),
	}
}

/// Proves a credential against every provider that names it, then stores it. Nothing that a
/// provider refuses, or that cannot be checked, is stored.
async fn set_credential(
	State(proxy): State<Arc<Proxy>>,
	headers: HeaderMap,
	Path(name): Path<String>,
	body: Result<Bytes, BytesRejection>,
) -> Response {
	let result = async {
		let (_, client) = admin(&proxy, &headers).await?;
		let held = held_tables(&proxy)?;
		let served = held.live.get();
		let name = credentials::name(&name).map_err(|m| ApiError::new(Class::BadRequest, m))?;
		let body = body.map_err(|r| ApiError::new(Class::BadRequest, r.body_text()))?;
		let new: credentials::NewCredential = serde_json::from_slice(&body)
			.map_err(|e| ApiError::new(Class::BadRequest, format!("A credential: {e}")))?;
		let stored = new
			.stored()
			.map_err(|m| ApiError::new(Class::BadRequest, m))?;
		let providers: Vec<&Provider> = served
			.credentialed()
			.into_iter()
			.filter(|p| p.credential.as_deref() == Some(name))
			.collect();
		if providers.is_empty() {
			return Err(ApiError::new(
				Class::BadRequest,
				format!(
					"No provider that serves a model takes the credential {name}, so it cannot be checked. Name it with \"credential\": \"{name}\" on a provider with a model first."
				),
			));
		}
		for provider in providers {
			let model = served.first_model_of(&provider.name).ok_or_else(|| {
				ApiError::new(
					Class::BadRequest,
					format!("{} serves no model, so the credential cannot be checked.", provider.name),
				)
			})?;
			credentials::prove(proxy.client_for(model)?, provider, model, &stored)
				.await
				.map_err(|m| ApiError::new(Class::BadRequest, m))?;
		}
		let made = credentials::save(&client, name, &stored)
			.await
			.map_err(|e| write_error(&e))?;
		reload_credentials(held, &client).await;
		Ok::<_, ApiError>(made)
	}
	.await;
	match result {
		Ok(made) => axum::Json(made).into_response(),
		Err(error) => error.into_response(),
	}
}

async fn delete_credential(
	State(proxy): State<Arc<Proxy>>,
	headers: HeaderMap,
	Path(name): Path<String>,
) -> Response {
	let result = async {
		let (_, client) = admin(&proxy, &headers).await?;
		let held = held_tables(&proxy)?;
		let name = credentials::name(&name).map_err(|m| ApiError::new(Class::BadRequest, m))?;
		let deleted = client
			.execute("DELETE FROM psyche.credentials WHERE name = $1", &[&name])
			.await
			.map_err(|e| database_error(&e))?;
		reload_credentials(held, &client).await;
		if deleted == 0 {
			return Err(not_found("There is no credential by that name."));
		}
		Ok(())
	}
	.await;
	match result {
		Ok(()) => StatusCode::NO_CONTENT.into_response(),
		Err(error) => error.into_response(),
	}
}

/// Reads the memberships again after the admin API changed them, so the next call sees it.
async fn reload_members(keys: &Keys, client: &tokio_postgres::Client) {
	match crate::who::load(client).await {
		Ok(members) => keys.members.replace(members),
		Err(error) => {
			tracing::warn!(error = %crate::db_message(&error), "could not reload the members")
		}
	}
}

fn not_found(message: &str) -> ApiError {
	let mut error = ApiError::new(Class::BadRequest, message);
	error.status = Some(StatusCode::NOT_FOUND);
	error
}

async fn list_groups(State(proxy): State<Arc<Proxy>>, headers: HeaderMap) -> Response {
	let result = async {
		let (_, client) = admin(&proxy, &headers).await?;
		let rows = client
			.query(
				"SELECT g.id::text, g.name, count(m.user_id) FROM psyche.groups g \
				 LEFT JOIN psyche.members m ON m.group_id = g.id GROUP BY g.id ORDER BY g.created_at, g.id",
				&[],
			)
			.await
			.map_err(|e| database_error(&e))?;
		Ok::<_, ApiError>(
			rows.iter()
				.map(|row| {
					json!({"id": row.get::<_, String>(0), "name": row.get::<_, String>(1),
						"members": row.get::<_, i64>(2)})
				})
				.collect::<Vec<_>>(),
		)
	}
	.await;
	match result {
		Ok(data) => axum::Json(json!({"object": "list", "data": data})).into_response(),
		Err(error) => error.into_response(),
	}
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct NewGroup {
	name: String,
}

async fn create_group(
	State(proxy): State<Arc<Proxy>>,
	headers: HeaderMap,
	body: Result<Bytes, BytesRejection>,
) -> Response {
	let result = async {
		let (_, client) = admin(&proxy, &headers).await?;
		let body = body.map_err(|r| ApiError::new(Class::BadRequest, r.body_text()))?;
		let new: NewGroup = serde_json::from_slice(&body).map_err(|e| {
			ApiError::new(
				Class::BadRequest,
				format!("A group is made from {{\"name\"}}: {e}"),
			)
		})?;
		let row = client
			.query_one(
				"INSERT INTO psyche.groups (name) VALUES ($1) RETURNING id::text",
				&[&new.name.trim()],
			)
			.await
			.map_err(|e| write_error(&e))?;
		Ok::<_, ApiError>(json!({"id": row.get::<_, String>(0), "name": new.name.trim()}))
	}
	.await;
	match result {
		Ok(made) => (StatusCode::CREATED, axum::Json(made)).into_response(),
		Err(error) => error.into_response(),
	}
}

async fn delete_group(
	State(proxy): State<Arc<Proxy>>,
	headers: HeaderMap,
	Path(id): Path<String>,
) -> Response {
	let result = async {
		let (keys, client) = admin(&proxy, &headers).await?;
		let id = KeyId::parse(&id)
			.ok_or_else(|| ApiError::new(Class::BadRequest, "A group's id is a uuid."))?;
		let deleted = client
			.execute("DELETE FROM psyche.groups WHERE id = $1", &[&id])
			.await
			.map_err(|e| database_error(&e))?;
		// Its members, budgets and limits went with it.
		reload_members(&keys, &client).await;
		reload_budgets(&proxy, &client).await;
		reload_limits(&proxy, &client).await;
		if deleted == 0 {
			return Err(not_found("There is no group with that id."));
		}
		Ok(())
	}
	.await;
	match result {
		Ok(()) => StatusCode::NO_CONTENT.into_response(),
		Err(error) => error.into_response(),
	}
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Membership {
	group_id: String,
}

/// Puts a user in a group, moving them out of any other.
async fn set_member(
	State(proxy): State<Arc<Proxy>>,
	headers: HeaderMap,
	Path(user): Path<String>,
	body: Result<Bytes, BytesRejection>,
) -> Response {
	let result = async {
		let (keys, client) = admin(&proxy, &headers).await?;
		let user = KeyId::parse(&user)
			.ok_or_else(|| ApiError::new(Class::BadRequest, "A user's id is a uuid."))?;
		let body = body.map_err(|r| ApiError::new(Class::BadRequest, r.body_text()))?;
		let membership: Membership = serde_json::from_slice(&body).map_err(|e| {
			ApiError::new(
				Class::BadRequest,
				format!("A membership is {{\"group_id\"}}: {e}"),
			)
		})?;
		let group = KeyId::parse(&membership.group_id)
			.ok_or_else(|| ApiError::new(Class::BadRequest, "group_id is a uuid."))?;
		client
			.execute(
				"INSERT INTO psyche.members (user_id, group_id) VALUES ($1, $2) \
				 ON CONFLICT (user_id) DO UPDATE SET group_id = excluded.group_id",
				&[&user, &group],
			)
			.await
			.map_err(|e| write_error(&e))?;
		reload_members(&keys, &client).await;
		Ok::<_, ApiError>(json!({"user_id": user.to_string(), "group_id": group.to_string()}))
	}
	.await;
	match result {
		Ok(made) => axum::Json(made).into_response(),
		Err(error) => error.into_response(),
	}
}

async fn remove_member(
	State(proxy): State<Arc<Proxy>>,
	headers: HeaderMap,
	Path(user): Path<String>,
) -> Response {
	let result = async {
		let (keys, client) = admin(&proxy, &headers).await?;
		let user = KeyId::parse(&user)
			.ok_or_else(|| ApiError::new(Class::BadRequest, "A user's id is a uuid."))?;
		let deleted = client
			.execute("DELETE FROM psyche.members WHERE user_id = $1", &[&user])
			.await
			.map_err(|e| database_error(&e))?;
		reload_members(&keys, &client).await;
		if deleted == 0 {
			return Err(not_found("That user is in no group."));
		}
		Ok(())
	}
	.await;
	match result {
		Ok(()) => StatusCode::NO_CONTENT.into_response(),
		Err(error) => error.into_response(),
	}
}

/// Who a model that is `granted` is open to (`psyche.model_grants`): one user or one group a row.
/// `?model=` reads one model's.
async fn list_grants(
	State(proxy): State<Arc<Proxy>>,
	headers: HeaderMap,
	RawQuery(query): RawQuery,
) -> Response {
	let result = async {
		let (_, client) = admin(&proxy, &headers).await?;
		let pairs = query_pairs(query.as_deref());
		let model = pairs.get("model").filter(|m| !m.is_empty());
		let rows = client
			.query(
				"SELECT g.id::text, g.model, g.user_id::text, g.group_id::text, gr.name, \
				 (extract(epoch FROM g.created_at) * 1000)::bigint FROM psyche.model_grants g \
				 LEFT JOIN psyche.groups gr ON gr.id = g.group_id \
				 WHERE $1::text IS NULL OR g.model = $1 ORDER BY g.created_at, g.id",
				&[&model],
			)
			.await
			.map_err(|e| database_error(&e))?;
		Ok::<_, ApiError>(rows.iter().map(grant_json).collect::<Vec<_>>())
	}
	.await;
	match result {
		Ok(data) => axum::Json(json!({"object": "list", "data": data})).into_response(),
		Err(error) => error.into_response(),
	}
}

fn grant_json(row: &tokio_postgres::Row) -> Value {
	json!({
		"id": row.get::<_, String>(0),
		"model": row.get::<_, String>(1),
		"user_id": row.get::<_, Option<String>>(2),
		"group_id": row.get::<_, Option<String>>(3),
		"group_name": row.get::<_, Option<String>>(4),
		"created_at": row.get::<_, i64>(5),
	})
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct NewGrant {
	model: String,
	user_id: Option<String>,
	group_id: Option<String>,
}

/// Grants a model to one user or one group. Granting it again changes nothing and answers the
/// grant that is there (200, not 201).
async fn create_grant(
	State(proxy): State<Arc<Proxy>>,
	headers: HeaderMap,
	body: Result<Bytes, BytesRejection>,
) -> Response {
	let result = async {
		let (_, client) = admin(&proxy, &headers).await?;
		let held = held_tables(&proxy)?;
		let body = body.map_err(|r| ApiError::new(Class::BadRequest, r.body_text()))?;
		let new: NewGrant = serde_json::from_slice(&body).map_err(|e| {
			ApiError::new(
				Class::BadRequest,
				format!("A grant is {{\"model\"}} and one of user_id or group_id: {e}"),
			)
		})?;
		let id = |text: &Option<String>, what: &str| -> Result<Option<KeyId>, ApiError> {
			text.as_deref()
				.map(|t| {
					KeyId::parse(t.trim()).ok_or_else(|| {
						ApiError::new(Class::BadRequest, format!("{what} is a uuid."))
					})
				})
				.transpose()
		};
		let (user, group) = (id(&new.user_id, "user_id")?, id(&new.group_id, "group_id")?);
		if user.is_some() == group.is_some() {
			return Err(ApiError::new(
				Class::BadRequest,
				"A grant names one user (user_id) or one group (group_id), not both.",
			));
		}
		let model = new.model.trim();
		if client
			.query_opt("SELECT 1 FROM psyche.models WHERE name = $1", &[&model])
			.await
			.map_err(|e| database_error(&e))?
			.is_none()
		{
			return Err(not_found(&format!("There is no model named {model}.")));
		}
		if let Some(group) = group
			&& client
				.query_opt("SELECT 1 FROM psyche.groups WHERE id = $1", &[&group])
				.await
				.map_err(|e| database_error(&e))?
				.is_none()
		{
			return Err(not_found("There is no group with that id."));
		}
		let made = client
			.query_opt(
				"WITH made AS (INSERT INTO psyche.model_grants (model, user_id, group_id) \
				 VALUES ($1, $2, $3) ON CONFLICT DO NOTHING RETURNING *) \
				 SELECT m.id::text, m.model, m.user_id::text, m.group_id::text, gr.name, \
				 (extract(epoch FROM m.created_at) * 1000)::bigint FROM made m \
				 LEFT JOIN psyche.groups gr ON gr.id = m.group_id",
				&[&model, &user, &group],
			)
			.await
			.map_err(|e| write_error(&e))?;
		let (status, row) = match made {
			Some(row) => (StatusCode::CREATED, row),
			None => (
				StatusCode::OK,
				client
					.query_one(
						"SELECT g.id::text, g.model, g.user_id::text, g.group_id::text, gr.name, \
						 (extract(epoch FROM g.created_at) * 1000)::bigint FROM psyche.model_grants g \
						 LEFT JOIN psyche.groups gr ON gr.id = g.group_id WHERE g.model = $1 \
						 AND g.user_id IS NOT DISTINCT FROM $2 AND g.group_id IS NOT DISTINCT FROM $3",
						&[&model, &user, &group],
					)
					.await
					.map_err(|e| database_error(&e))?,
			),
		};
		reload_credentials(held, &client).await;
		Ok::<_, ApiError>((status, grant_json(&row)))
	}
	.await;
	match result {
		Ok((status, made)) => (status, axum::Json(made)).into_response(),
		Err(error) => error.into_response(),
	}
}

async fn delete_grant(
	State(proxy): State<Arc<Proxy>>,
	headers: HeaderMap,
	Path(id): Path<String>,
) -> Response {
	let result = async {
		let (_, client) = admin(&proxy, &headers).await?;
		let held = held_tables(&proxy)?;
		let id = KeyId::parse(&id)
			.ok_or_else(|| ApiError::new(Class::BadRequest, "A grant's id is a uuid."))?;
		let deleted = client
			.execute("DELETE FROM psyche.model_grants WHERE id = $1", &[&id])
			.await
			.map_err(|e| database_error(&e))?;
		reload_credentials(held, &client).await;
		if deleted == 0 {
			return Err(not_found("There is no grant with that id."));
		}
		Ok(())
	}
	.await;
	match result {
		Ok(()) => StatusCode::NO_CONTENT.into_response(),
		Err(error) => error.into_response(),
	}
}

/// A broken rule of a table (a value it does not take, a key that does not exist, a name used
/// twice) is the caller's to fix, in Postgres's words; anything else is ours.
fn write_error(e: &tokio_postgres::Error) -> ApiError {
	match e.as_db_error() {
		Some(db) if db.code().code().starts_with("23") => {
			ApiError::new(Class::BadRequest, db.message().to_string())
		}
		_ => database_error(e),
	}
}

async fn list_limits(State(proxy): State<Arc<Proxy>>, headers: HeaderMap) -> Response {
	let result = async {
		let (_, client) = admin(&proxy, &headers).await?;
		let rows = client
			.query(
				"SELECT id::text, name, requests_per_minute, tokens_per_minute, \
				 key_id::text, every_user, user_id::text, group_id::text \
				 FROM psyche.limits ORDER BY created_at, id",
				&[],
			)
			.await
			.map_err(|e| database_error(&e))?;
		Ok::<_, ApiError>(
			rows.iter()
				.map(|row| {
					merge(
						json!({
							"id": row.get::<_, String>(0),
							"name": row.get::<_, String>(1),
							"requests_per_minute": row.get::<_, Option<i32>>(2),
							"tokens_per_minute": row.get::<_, Option<i32>>(3),
						}),
						whom_json(row, 4),
					)
				})
				.collect::<Vec<_>>(),
		)
	}
	.await;
	match result {
		Ok(data) => axum::Json(json!({"object": "list", "data": data})).into_response(),
		Err(error) => error.into_response(),
	}
}

async fn create_limit(
	State(proxy): State<Arc<Proxy>>,
	headers: HeaderMap,
	body: Result<Bytes, BytesRejection>,
) -> Response {
	let result = async {
		let (_, client) = admin(&proxy, &headers).await?;
		let body = body.map_err(|r| ApiError::new(Class::BadRequest, r.body_text()))?;
		let new: NewLimit = admin_body(&body, LIMIT_FIELDS, "A limit")?;
		let (key, every_user, user, group) = new.whom.columns()?;
		let row = client
			.query_one(
				"INSERT INTO psyche.limits (name, key_id, every_user, user_id, group_id, \
				 requests_per_minute, tokens_per_minute) VALUES ($1, $2, $3, $4, $5, $6, $7) RETURNING id::text",
				&[&new.name.trim(), &key, &every_user, &user, &group, &new.requests_per_minute, &new.tokens_per_minute],
			)
			.await
			.map_err(|e| write_error(&e))?;
		reload_limits(&proxy, &client).await;
		Ok::<_, ApiError>(merge(
			json!({"id": row.get::<_, String>(0), "name": new.name.trim(),
				"requests_per_minute": new.requests_per_minute, "tokens_per_minute": new.tokens_per_minute}),
			new.whom.json(),
		))
	}
	.await;
	match result {
		Ok(made) => (StatusCode::CREATED, axum::Json(made)).into_response(),
		Err(error) => error.into_response(),
	}
}

async fn delete_limit(
	State(proxy): State<Arc<Proxy>>,
	headers: HeaderMap,
	Path(id): Path<String>,
) -> Response {
	let result = async {
		let (_, client) = admin(&proxy, &headers).await?;
		let id = KeyId::parse(&id)
			.ok_or_else(|| ApiError::new(Class::BadRequest, "A limit's id is a uuid."))?;
		let deleted = client
			.execute("DELETE FROM psyche.limits WHERE id = $1", &[&id])
			.await
			.map_err(|e| database_error(&e))?;
		reload_limits(&proxy, &client).await;
		if deleted == 0 {
			let mut error = ApiError::new(Class::BadRequest, "There is no limit with that id.");
			error.status = Some(StatusCode::NOT_FOUND);
			return Err(error);
		}
		Ok(())
	}
	.await;
	match result {
		Ok(()) => StatusCode::NO_CONTENT.into_response(),
		Err(error) => error.into_response(),
	}
}

/// Reads the budgets again after the admin API changed them, so the change holds from the next
/// call rather than from the notification.
async fn reload_budgets(proxy: &Proxy, client: &tokio_postgres::Client) {
	if let Some(budgets) = &proxy.budgets {
		match budgets::load(client).await {
			Ok(list) => budgets.replace(list),
			Err(error) => {
				tracing::warn!(error = %crate::db_message(&error), "could not reload the budgets")
			}
		}
	}
}

async fn list_budgets(State(proxy): State<Arc<Proxy>>, headers: HeaderMap) -> Response {
	let result = async {
		let (_, client) = admin(&proxy, &headers).await?;
		let rows = client
			.query(
				"SELECT id::text, name, amount::text, period, days, hard, created_at, \
				 key_id::text, every_user, user_id::text, group_id::text, starts_on::text \
				 FROM psyche.budgets ORDER BY created_at, id",
				&[],
			)
			.await
			.map_err(|e| database_error(&e))?;
		Ok::<_, ApiError>(
			rows.iter()
				.map(|row| {
					merge(
						json!({
							"id": row.get::<_, String>(0),
							"name": row.get::<_, String>(1),
							"amount": row.get::<_, String>(2),
							"period": row.get::<_, String>(3),
							"days": row.get::<_, Option<i32>>(4),
							"hard": row.get::<_, bool>(5),
							"created_at": row.get::<_, SystemTime>(6).duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs()),
							"starts_on": row.get::<_, Option<String>>(11),
						}),
						whom_json(row, 7),
					)
				})
				.collect::<Vec<_>>(),
		)
	}
	.await;
	match result {
		Ok(data) => axum::Json(json!({"object": "list", "data": data})).into_response(),
		Err(error) => error.into_response(),
	}
}

async fn create_budget(
	State(proxy): State<Arc<Proxy>>,
	headers: HeaderMap,
	body: Result<Bytes, BytesRejection>,
) -> Response {
	let result = async {
		let (_, client) = admin(&proxy, &headers).await?;
		let body = body.map_err(|r| ApiError::new(Class::BadRequest, r.body_text()))?;
		let new: NewBudget = admin_body(&body, BUDGET_FIELDS, "A budget")?;
		budget_amount(new.amount)?;
		let (key, every_user, user, group) = new.whom.columns()?;
		// Six decimals is a millionth of a dollar, finer than any budget needs; numeric takes it
		// exactly from text.
		let amount = format!("{:.6}", new.amount);
		let row = client
			.query_one(
				"INSERT INTO psyche.budgets (name, amount, period, days, hard, key_id, every_user, \
				 user_id, group_id, starts_on) VALUES ($1, $2::text::numeric, $3, $4, $5, $6, $7, $8, $9, \
				 $10::text::date) RETURNING id::text",
				&[
					&new.name.trim(),
					&amount,
					&new.period,
					&new.days,
					&new.hard,
					&key,
					&every_user,
					&user,
					&group,
					&new.starts_on,
				],
			)
			.await
			.map_err(|e| write_error(&e))?;
		reload_budgets(&proxy, &client).await;
		Ok::<_, ApiError>(merge(
			json!({"id": row.get::<_, String>(0), "name": new.name.trim(), "amount": amount,
				"period": new.period, "days": new.days, "starts_on": new.starts_on, "hard": new.hard}),
			new.whom.json(),
		))
	}
	.await;
	match result {
		Ok(made) => (StatusCode::CREATED, axum::Json(made)).into_response(),
		Err(error) => error.into_response(),
	}
}

/// A budget's amount as the admin API takes it.
fn budget_amount(amount: f64) -> Result<(), ApiError> {
	if !amount.is_finite() || !(0.0..1_000_000.0).contains(&amount) {
		return Err(ApiError::new(
			Class::BadRequest,
			"amount is US dollars, from 0 to under 1,000,000.",
		));
	}
	Ok(())
}

async fn edit_budget(
	State(proxy): State<Arc<Proxy>>,
	headers: HeaderMap,
	Path(id): Path<String>,
	body: Result<Bytes, BytesRejection>,
) -> Response {
	let result = async {
		let (_, client) = admin(&proxy, &headers).await?;
		let id = KeyId::parse(&id)
			.ok_or_else(|| ApiError::new(Class::BadRequest, "A budget's id is a uuid."))?;
		let body = body.map_err(|r| ApiError::new(Class::BadRequest, r.body_text()))?;
		let edit: BudgetEdit = serde_json::from_slice(&body).map_err(|e| {
			ApiError::new(
				Class::BadRequest,
				format!(
					"A budget is changed with name, amount, period, days, starts_on and hard: {e}"
				),
			)
		})?;
		if let Some(amount) = edit.amount {
			budget_amount(amount)?;
		}
		let amount = edit.amount.map(|a| format!("{a:.6}"));
		let name = edit.name.as_deref().map(str::trim);
		let row = client
			.query_opt(
				"UPDATE psyche.budgets SET name = coalesce($2, name), \
				 amount = coalesce($3::text::numeric, amount), hard = coalesce($4, hard), \
				 period = coalesce($5, period), \
				 days = CASE WHEN $5::text IS NULL THEN coalesce($6, days) ELSE $6 END, \
				 starts_on = CASE WHEN $5::text IS NULL THEN coalesce($7::text::date, starts_on) \
				 ELSE $7::text::date END \
				 WHERE id = $1 RETURNING name, amount::text, period, days, hard, starts_on::text",
				&[
					&id,
					&name,
					&amount,
					&edit.hard,
					&edit.period,
					&edit.days,
					&edit.starts_on,
				],
			)
			.await
			.map_err(|e| write_error(&e))?
			.ok_or_else(|| not_found("There is no budget with that id."))?;
		reload_budgets(&proxy, &client).await;
		Ok::<_, ApiError>(json!({
			"id": id.to_string(),
			"name": row.get::<_, String>(0),
			"amount": row.get::<_, String>(1),
			"period": row.get::<_, String>(2),
			"days": row.get::<_, Option<i32>>(3),
			"hard": row.get::<_, bool>(4),
			"starts_on": row.get::<_, Option<String>>(5),
		}))
	}
	.await;
	match result {
		Ok(edited) => axum::Json(edited).into_response(),
		Err(error) => error.into_response(),
	}
}

/// A query string's pairs, the last of each name winning.
fn query_pairs(query: Option<&str>) -> std::collections::HashMap<String, String> {
	url::form_urlencoded::parse(query.unwrap_or("").as_bytes())
		.into_owned()
		.collect()
}

/// A uuid filter from the query string, refused when it is not one.
fn uuid_filter(
	pairs: &std::collections::HashMap<String, String>,
	name: &str,
) -> Result<Option<KeyId>, ApiError> {
	match pairs.get(name).filter(|v| !v.is_empty()) {
		None => Ok(None),
		Some(text) => KeyId::parse(text)
			.map(Some)
			.ok_or_else(|| ApiError::new(Class::BadRequest, format!("{name} is a uuid."))),
	}
}

/// What `GET /admin/usage` groups the daily totals by, and the column each one is.
const USAGE_BY: [(&str, &str); 7] = [
	("day", "day::text"),
	("model", "model"),
	("provider", "provider"),
	("operation", "operation"),
	("key", "key_id::text"),
	("user", "user_id::text"),
	("group", "group_id::text"),
];

/// The daily totals, summed over `from`..=`to` (UTC days, the last 30 by default) and grouped by
/// one of `USAGE_BY`, optionally for one model, key, user or group.
async fn usage(
	State(proxy): State<Arc<Proxy>>,
	headers: HeaderMap,
	RawQuery(query): RawQuery,
) -> Response {
	let result = async {
		let (_, client) = admin(&proxy, &headers).await?;
		let pairs = query_pairs(query.as_deref());
		let by = pairs.get("by").map_or("day", String::as_str);
		let column = USAGE_BY
			.iter()
			.find(|(name, _)| *name == by)
			.map(|(_, column)| *column)
			.ok_or_else(|| {
				let names: Vec<&str> = USAGE_BY.iter().map(|(name, _)| *name).collect();
				ApiError::new(
					Class::BadRequest,
					format!("by is one of {}.", names.join(", ")),
				)
			})?;
		let day = |name: &str| -> Result<Option<String>, ApiError> {
			match pairs.get(name).filter(|v| !v.is_empty()) {
				None => Ok(None),
				Some(text)
					if text.len() == 10
						&& text.as_bytes()[4] == b'-'
						&& text.as_bytes()[7] == b'-' =>
				{
					Ok(Some(text.clone()))
				}
				Some(_) => Err(ApiError::new(
					Class::BadRequest,
					format!("{name} is a day, YYYY-MM-DD."),
				)),
			}
		};
		let (from, to) = (day("from")?, day("to")?);
		let model = pairs.get("model").filter(|v| !v.is_empty()).cloned();
		let (key, user, group) = (
			uuid_filter(&pairs, "key_id")?,
			uuid_filter(&pairs, "user_id")?,
			uuid_filter(&pairs, "group_id")?,
		);
		let span = client
			.query_one(
				"SELECT lo::text, hi::text, hi - lo FROM (SELECT \
				 coalesce($1::text::date, coalesce($2::text::date, current_date) - 29) AS lo, \
				 coalesce($2::text::date, current_date) AS hi) span",
				&[&from, &to],
			)
			.await
			.map_err(|e| write_error(&e))?;
		let (lo, hi, days) = (
			span.get::<_, String>(0),
			span.get::<_, String>(1),
			span.get::<_, i32>(2),
		);
		if !(0..=366).contains(&days) {
			return Err(ApiError::new(
				Class::BadRequest,
				"from is on or before to, and at most 366 days before it.",
			));
		}
		let sql = format!(
			"SELECT {column}, sum(requests)::bigint, sum(failed)::bigint, sum(cancelled)::bigint, \
			 sum(prompt_tokens)::bigint, sum(completion_tokens)::bigint, sum(cached_tokens)::bigint, \
			 sum(reasoning_tokens)::bigint, sum(cost)::text, sum(credits_cost)::text, sum(unpriced)::bigint \
			 FROM psyche.usage_daily WHERE day BETWEEN $1::text::date AND $2::text::date \
			 AND ($3::text IS NULL OR model = $3) AND ($4::uuid IS NULL OR key_id = $4) \
			 AND ($5::uuid IS NULL OR user_id = $5) AND ($6::uuid IS NULL OR group_id = $6) \
			 GROUP BY 1 ORDER BY {order}",
			order = if by == "day" {
				"1"
			} else {
				"sum(cost) DESC NULLS LAST, 1"
			},
		);
		let rows = client
			.query(&sql, &[&lo, &hi, &model, &key, &user, &group])
			.await
			.map_err(|e| write_error(&e))?;
		let data: Vec<Value> = rows
			.iter()
			.map(|row| {
				json!({
					by: row.get::<_, Option<String>>(0),
					"requests": row.get::<_, i64>(1),
					"failed": row.get::<_, i64>(2),
					"cancelled": row.get::<_, i64>(3),
					"prompt_tokens": row.get::<_, i64>(4),
					"completion_tokens": row.get::<_, i64>(5),
					"cached_tokens": row.get::<_, i64>(6),
					"reasoning_tokens": row.get::<_, i64>(7),
					"cost": row.get::<_, Option<String>>(8),
					"credits_cost": row.get::<_, Option<String>>(9),
					"unpriced": row.get::<_, i64>(10),
				})
			})
			.collect();
		Ok::<_, ApiError>(json!({"object": "list", "by": by, "from": lo, "to": hi, "data": data}))
	}
	.await;
	match result {
		Ok(listed) => axum::Json(listed).into_response(),
		Err(error) => error.into_response(),
	}
}

/// The request log, newest first: `limit` rows (50, at most 500) before `before` (microseconds
/// since the epoch, from the last page's `next_before`), optionally for one model, key, user or
/// group, or one outcome. The cursor is the last row's own time to the microsecond the log keeps,
/// never a rounded one: a millisecond rounded up would bring that row back on the next page.
async fn requests(
	State(proxy): State<Arc<Proxy>>,
	headers: HeaderMap,
	RawQuery(query): RawQuery,
) -> Response {
	let result = async {
		let (_, client) = admin(&proxy, &headers).await?;
		let pairs = query_pairs(query.as_deref());
		let number = |name: &str| -> Result<Option<i64>, ApiError> {
			pairs
				.get(name)
				.filter(|v| !v.is_empty())
				.map(|v| {
					v.parse::<i64>().map_err(|_| {
						ApiError::new(Class::BadRequest, format!("{name} is a number."))
					})
				})
				.transpose()
		};
		let limit = number("limit")?.unwrap_or(50).clamp(1, 500);
		let before = number("before")?;
		let model = pairs.get("model").filter(|v| !v.is_empty()).cloned();
		let outcome = pairs.get("outcome").filter(|v| !v.is_empty()).cloned();
		let (key, user, group) = (
			uuid_filter(&pairs, "key_id")?,
			uuid_filter(&pairs, "user_id")?,
			uuid_filter(&pairs, "group_id")?,
		);
		let rows = client
			.query(
				"SELECT (extract(epoch FROM at) * 1000)::bigint, operation, model, provider, streamed, \
				 outcome, prompt_tokens, completion_tokens, cached_tokens, reasoning_tokens, first_byte_ms, \
				 total_ms, cost::text, key_id::text, user_id::text, group_id::text, credits, budget_warning, \
				 (extract(epoch FROM at) * 1000000)::bigint, route, attempt, routing \
				 FROM psyche.requests \
				 WHERE ($1::bigint IS NULL OR at < timestamptz 'epoch' + $1::bigint * interval '1 microsecond') \
				 AND ($2::text IS NULL OR model = $2) AND ($3::text IS NULL OR outcome = $3) \
				 AND ($4::uuid IS NULL OR key_id = $4) AND ($5::uuid IS NULL OR user_id = $5) \
				 AND ($6::uuid IS NULL OR group_id = $6) \
				 ORDER BY at DESC LIMIT $7",
				&[&before, &model, &outcome, &key, &user, &group, &limit],
			)
			.await
			.map_err(|e| write_error(&e))?;
		let data: Vec<Value> = rows
			.iter()
			.map(|row| {
				json!({
					"at": row.get::<_, i64>(0),
					"operation": row.get::<_, String>(1),
					"model": row.get::<_, String>(2),
					"provider": row.get::<_, String>(3),
					"streamed": row.get::<_, bool>(4),
					"outcome": row.get::<_, String>(5),
					"prompt_tokens": row.get::<_, Option<i32>>(6),
					"completion_tokens": row.get::<_, Option<i32>>(7),
					"cached_tokens": row.get::<_, Option<i32>>(8),
					"reasoning_tokens": row.get::<_, Option<i32>>(9),
					"first_byte_ms": row.get::<_, i32>(10),
					"total_ms": row.get::<_, i32>(11),
					"cost": row.get::<_, Option<String>>(12),
					"key_id": row.get::<_, Option<String>>(13),
					"user_id": row.get::<_, Option<String>>(14),
					"group_id": row.get::<_, Option<String>>(15),
					"credits": row.get::<_, bool>(16),
					"budget_warning": row.get::<_, Option<String>>(17),
					"route": row.get::<_, Option<String>>(19),
					"attempt": row.get::<_, Option<i16>>(20),
					"routing": row.get::<_, Option<String>>(21),
				})
			})
			.collect();
		let next = (data.len() as i64 == limit)
			.then(|| rows.last().map(|row| row.get::<_, i64>(18)))
			.flatten();
		Ok::<_, ApiError>(json!({"object": "list", "data": data, "next_before": next}))
	}
	.await;
	match result {
		Ok(listed) => axum::Json(listed).into_response(),
		Err(error) => error.into_response(),
	}
}

async fn delete_budget(
	State(proxy): State<Arc<Proxy>>,
	headers: HeaderMap,
	Path(id): Path<String>,
) -> Response {
	let result = async {
		let (_, client) = admin(&proxy, &headers).await?;
		let id = KeyId::parse(&id)
			.ok_or_else(|| ApiError::new(Class::BadRequest, "A budget's id is a uuid."))?;
		let deleted = client
			.execute("DELETE FROM psyche.budgets WHERE id = $1", &[&id])
			.await
			.map_err(|e| database_error(&e))?;
		reload_budgets(&proxy, &client).await;
		if deleted == 0 {
			let mut error = ApiError::new(Class::BadRequest, "There is no budget with that id.");
			error.status = Some(StatusCode::NOT_FOUND);
			return Err(error);
		}
		Ok(())
	}
	.await;
	match result {
		Ok(()) => StatusCode::NO_CONTENT.into_response(),
		Err(error) => error.into_response(),
	}
}

impl Proxy {
	/// Whether `who` may use `model` by its grants (`models.rs`); every model without a database.
	fn granted(&self, model: &str, who: &Who) -> bool {
		self.tables.as_ref().is_none_or(|t| t.allows(model, who))
	}

	/// Takes a call's request and tokens from every rate limit on it, or refuses it with when to
	/// try again. `prompt_bound` is `None` when no limit applies.
	fn take(
		&self,
		who: &Who,
		prompt_bound: Option<u64>,
		output: u64,
	) -> Result<(Option<Taken>, limits::Headers), ApiError> {
		let (Some(limits), Some(prompt)) = (&self.limits, prompt_bound) else {
			return Ok((None, limits::Headers::default()));
		};
		match limits.take(who, prompt.saturating_add(output), Instant::now()) {
			Ok((taken, headers)) => Ok((Some(taken), headers)),
			Err(refused) => {
				let mut error =
					ApiError::new(Class::Limited, refused.message).scoped(refused.scope);
				error.retry_after = refused
					.retry_after
					.map(|d| d.as_secs_f64().ceil().max(1.0).to_string());
				Err(error)
			}
		}
	}

	/// Holds room for a call in every budget that applies to it, or refuses it. `prompt_bound` is
	/// `None` when no budget applies.
	fn reserve(
		&self,
		who: &Who,
		model: &Model,
		prompt_bound: Option<u64>,
		output: u64,
		searches: u64,
	) -> Result<Option<Reservation>, ApiError> {
		let (Some(budgets), Some(prompt)) = (&self.budgets, prompt_bound) else {
			return Ok(None);
		};
		let Some(price) = model.price else {
			return Err(ApiError::new(
				Class::Budget,
				format!(
					"{} has no price, so a budget cannot be kept for it. Give it a price in the providers file.",
					model.name
				),
			));
		};
		let amount = worst(&price, prompt, output, searches);
		budgets
			.reserve(who, amount, SystemTime::now())
			.map(Some)
			.map_err(|over| ApiError::new(Class::Budget, over.message).scoped(over.scope))
	}

	/// The reply cap to send for a call that names none, when a hard budget applies: the default, or
	/// what the tightest budget still has room for after the prompt if that is less. A cap below
	/// [`budgets::MIN_SENT_OUTPUT`] is not shrunk to, so a call with less room is refused instead.
	fn sent_cap(
		&self,
		who: &Who,
		inbound: &Inbound,
		model: &Model,
		prompt_bound: Option<u64>,
		searches: u64,
	) -> Option<u64> {
		let named = ["max_tokens", "max_completion_tokens", "max_output_tokens"]
			.iter()
			.any(|f| inbound.body.get(*f).is_some_and(|v| !v.is_null()));
		if named || model.max_output_tokens.is_some() {
			return None;
		}
		let room = self.budgets.as_ref()?.room(who, SystemTime::now())?;
		let (price, prompt) = (model.price?, prompt_bound?);
		let base = worst(&price, prompt, 0, searches);
		let per_token = worst(&price, prompt, 1, searches).saturating_sub(base);
		let fits = room
			.saturating_sub(base)
			.checked_div(per_token)
			.unwrap_or(budgets::UNCAPPED_OUTPUT);
		Some(fits.clamp(budgets::MIN_SENT_OUTPUT, budgets::UNCAPPED_OUTPUT))
	}

	/// The caller, from `Authorization: Bearer`, the `apikey` header (the front door sends both) or
	/// `x-api-key`, where a client written for Anthropic sends it: a Psyche key, the project's service key, or a signed-in user's token, which is taken only
	/// once the project has budgets that hold every user (PS10b). The anon key alone is not a
	/// caller.
	fn authorize(&self, headers: &HeaderMap) -> Result<Caller, ApiError> {
		let bearer = headers
			.get(header::AUTHORIZATION)
			.and_then(|v| v.to_str().ok())
			.and_then(|v| {
				v.strip_prefix("Bearer ")
					.or_else(|| v.strip_prefix("bearer "))
			});
		let token = bearer
			.or_else(|| headers.get("apikey").and_then(|v| v.to_str().ok()))
			.or_else(|| headers.get("x-api-key").and_then(|v| v.to_str().ok()))
			.map(str::trim)
			.filter(|t| !t.is_empty())
			.ok_or_else(|| {
				ApiError::new(
					Class::Unauthorized,
					"Send a Psyche key (psy_…), a signed-in user's token or the project's service key as a Bearer token.",
				)
			})?;
		if token.starts_with(keys::PREFIX) {
			let Some(keys) = &self.keys else {
				return Err(ApiError::new(
					Class::Unauthorized,
					"This Psyche has no database, so it has no keys. Use the project's service key.",
				));
			};
			return match keys.store.find(token, SystemTime::now()) {
				Ok(key) => Ok(Caller::Key(key)),
				Err(Refusal::Unknown) => Err(ApiError::new(
					Class::Unauthorized,
					"This key is not one Psyche knows: it was never made here, or it was revoked.",
				)),
				Err(Refusal::Expired) => {
					Err(ApiError::new(Class::Unauthorized, "This key has expired."))
				}
			};
		}
		let caller = jwt::verify(token, &self.jwt_secret, now())
			.map_err(|e| ApiError::new(Class::Unauthorized, e.0))?;
		match caller.role {
			Role::Service => Ok(Caller::Service),
			Role::Anon => Err(ApiError::new(
				Class::Forbidden,
				"The anon key alone cannot call a model: sign the user in and send their token as the Bearer token.",
			)),
			Role::Authenticated => {
				let user = caller
					.claims
					.get("sub")
					.and_then(Value::as_str)
					.and_then(KeyId::parse)
					.ok_or_else(|| {
						ApiError::new(
							Class::Unauthorized,
							"The token names no user: its sub is not a user's id.",
						)
					})?;
				let budgets = self.budgets.as_ref().ok_or_else(|| {
					ApiError::new(
						Class::Forbidden,
						"This Psyche has no database, so it cannot budget signed-in users. Use a Psyche key or the service key.",
					)
				})?;
				budgets
					.admits(user)
					.map_err(|why| ApiError::new(Class::Forbidden, why))?;
				Ok(Caller::User(user))
			}
		}
	}

	/// A call: the caller, what it named and what that stands for (a model, a route's models, and
	/// OpenRouter's `models` after them), then each in turn until one answers. A target is tried
	/// again once on a failure a moment may fix, and the next is tried on any failure that was not
	/// the caller's (`Class::falls_over`); never once the reply has started. Each attempt is its own
	/// row in the request log.
	async fn forward(
		&self,
		headers: &HeaderMap,
		body: Result<Bytes, BytesRejection>,
		operation: Operation,
	) -> Result<Response, ApiError> {
		let caller = self.authorize(headers)?;
		let who = caller.who(self.keys.as_ref().map(|k| &k.members));
		let body = read_body(body)?;
		let mut inbound = Inbound::parse(&body)?;
		if operation == Operation::Messages {
			messages::check(&inbound.body)?;
			let text = |name: &str| {
				headers
					.get(name)
					.and_then(|v| v.to_str().ok())
					.map(String::from)
			};
			inbound.beta = text("anthropic-beta");
			inbound.version = text("anthropic-version");
		}
		if operation == Operation::Responses {
			responses::check(&inbound.body)?;
		}
		let session = match (&inbound.session, headers.get("x-session-id")) {
			(Some(session), _) => Some(session.clone()),
			(None, Some(header)) => Some(session_checked(
				header
					.to_str()
					.map_err(|_| ApiError::new(Class::BadRequest, "x-session-id is plain text."))?
					.to_string(),
			)?),
			(None, None) => None,
		};
		let catalogue = self.catalogue.get();
		let mut targets: Vec<Arc<Model>> = Vec::new();
		// Which of the names sent each target came from.
		let mut groups: HashMap<String, usize> = HashMap::new();
		let mut route = None;
		let mut sort = None;
		let mut weights: Option<HashMap<String, u32>> = None;
		let mut constraints: Option<fits::Constraints> = None;
		let written: Vec<String> = std::iter::once(&inbound.model)
			.chain(&inbound.fallbacks)
			.cloned()
			.collect();
		for (i, written) in written.iter().enumerate() {
			let (named, suffixes) = without_suffixes(&catalogue, written);
			inbound.online |= suffixes.online;
			if !caller.may_use(named) {
				return Err(ApiError::new(
					Class::Forbidden,
					format!("This key may not use {named}."),
				));
			}
			let models = match (catalogue.route(named), catalogue.model(named)) {
				// Automatic choice (PS3): among the models that say what they can take, those in the
				// project's own `auto` route when it has one, in the order its name asks for.
				(own, None) if fits::is_auto(named) => {
					let own = own.filter(|_| named == fits::AUTO);
					let mut pool: Vec<Arc<Model>> = match own {
						Some(r) => r
							.targets
							.iter()
							.filter_map(|t| catalogue.model(t))
							.filter(|m| m.fits.is_some())
							.cloned()
							.collect(),
						None => catalogue
							.models()
							.iter()
							.filter(|m| m.fits.is_some() && m.embeddings.is_none())
							.cloned()
							.collect(),
					};
					let order = match named {
						fits::AUTO_FLOOR => Sort::Price,
						fits::AUTO_NITRO => Sort::Throughput,
						_ => own.map_or(Sort::Price, |r| r.sort),
					};
					if order == Sort::Price {
						providers::by_price(&mut pool);
					}
					if pool.is_empty() {
						return Err(ApiError::new(
							Class::ModelNotFound,
							format!(
								"{named} chooses among the models that say what they can take (their fits), and none here does yet. GET /ai/v1/admin/models lists them."
							),
						));
					}
					if i == 0 {
						route = Some(named.to_string());
						sort = Some(order);
						weights = own.map(|r| r.weights.clone());
						constraints = own.and_then(|r| r.constraints.clone());
					}
					pool
				}
				(Some(r), _) => {
					if i == 0 {
						route = Some(r.name.clone());
						sort = Some(r.sort);
						weights = Some(r.weights.clone());
						constraints = r.constraints.clone();
					}
					catalogue.targets(r)
				}
				(None, Some(model)) => vec![model.clone()],
				(None, None) => {
					return Err(ApiError::new(
						Class::ModelNotFound,
						format!(
							"There is no model named {named} here. GET /ai/v1/models lists them."
						),
					));
				}
			};
			if i == 0 && suffixes.floor {
				sort = Some(Sort::Price);
			} else if i == 0 && suffixes.nitro {
				sort = Some(Sort::Throughput);
			}
			for model in models {
				groups.entry(model.name.clone()).or_insert(i);
				if !targets.iter().any(|t| t.name == model.name) {
					targets.push(model);
				}
			}
		}
		if !inbound.fallbacks.is_empty() && route.is_none() {
			route = Some("models".to_string());
		}
		let inbound = inbound;
		let prefs = inbound.prefs.as_ref();
		let open: Vec<Arc<Model>> = targets
			.into_iter()
			.filter(|m| self.granted(&m.name, &who))
			.collect();
		// Before anything is sent, every model that may not or cannot take this request is left out
		// (`fits.rs`). A route, `auto` or a fallback list holds a model to its own facts as well; a
		// model named alone is the caller's choice, and is held only to its internet level and to
		// what the caller's own controls ask.
		let mut left_out: Vec<(String, String)> = Vec::new();
		// Whether only the caller's controls left anything out, which is a 404 as OpenRouter's is.
		let mut only_controls = true;
		let open: Vec<Arc<Model>> = if open.is_empty() {
			open
		} else {
			let web = fits::Web::of(&inbound.body, inbound.online);
			let web = if operation == Operation::Responses {
				web.offered_search_is_no_need()
			} else {
				web
			};
			let alone = route.is_none();
			let needs = (!alone || prefs.is_some())
				.then(|| Needs::of(&inbound.body, budgets::prompt_bound(&body), web.clone()));
			let prompt = needs.as_ref().map_or(0, |n| n.prompt);
			let mut fit = Vec::with_capacity(open.len());
			for model in open {
				let mut why = self.sandbox_refused(&model).await;
				if why.is_none() {
					why = match &needs {
						Some(needs) if !alone => needs.rules_out(&model),
						_ => web.refused(&model),
					};
				}
				if why.is_none() {
					why = constraints
						.as_ref()
						.and_then(|c| c.rules_out(&model, prompt));
				}
				// Among several, a target whose translation would refuse this request is left out, so
				// a document or a server tool's blocks in the history still reach one that can take
				// them.
				let translated = match operation {
					Operation::Messages => model.provider.kind != Kind::Anthropic,
					Operation::Responses => true,
					_ => false,
				};
				if why.is_none() && translated && !alone {
					why = inbound.upstream(&model, operation).err().map(|e| e.message);
				}
				let before_controls = why.is_some();
				if why.is_none()
					&& let Some(prefs) = prefs
				{
					why = prefs.rules_out(&model, prompt);
					if why.is_none()
						&& prefs.require_parameters()
						&& let Some(needs) = &needs
					{
						why = parameters_refused(&inbound, &model, operation, needs);
					}
				}
				match why {
					Some(why) => {
						only_controls &= !before_controls;
						left_out.push((model.name.clone(), why));
					}
					None => fit.push(model),
				}
			}
			if fit.is_empty() {
				let reasons = left_out
					.iter()
					.map(|(name, why)| format!("{name} {why}"))
					.collect::<Vec<_>>()
					.join("; ");
				return Err(ApiError::new(
					if only_controls {
						Class::ModelNotFound
					} else {
						Class::BadRequest
					},
					match (&route, left_out.first()) {
						(None, Some((name, why))) => {
							format!("{name} cannot take this request: {why}.")
						}
						_ => format!(
							"None of the models {} stands for can take this request: {reasons}.",
							inbound.model
						),
					},
				));
			}
			fit
		};
		// The caller's `order` goes first and turns off the route's balancing and stickiness; its
		// `sort`, like a suffix, replaces the route's.
		let by_order = prefs.is_some_and(Prefs::has_order);
		let mut open = open;
		if let Some(prefs) = prefs.filter(|p| p.has_order()) {
			open.sort_by_key(|m| prefs.rank(m).unwrap_or(usize::MAX));
		}
		let sort = prefs.and_then(Prefs::sort).or(sort);
		let sticky = (!by_order && open.len() > 1 && open.iter().any(|m| m.cache_pays()))
			.then(|| health::conversation(&inbound.body, &inbound.model, session.as_deref()))
			.flatten();
		let within_names =
			!inbound.fallbacks.is_empty() && !prefs.is_some_and(Prefs::sorted_as_one);
		let plan = health::Plan {
			sort: if by_order { Some(Sort::Order) } else { sort },
			weights: weights.as_ref().filter(|_| !by_order),
			groups: within_names.then_some(&groups),
			sticky: sticky.map(|(key, _)| key),
			// An explicit `order` is the caller's own choice, and a preference does not move it.
			preferred: prefs.filter(|_| !by_order).and_then(Prefs::preferred),
		};
		// A target cooling down or out of quota goes after the others, a route draws or sorts its
		// order here, and a conversation goes back to its own cache.
		let health::Arranged {
			targets: mut open,
			stuck,
			behind,
		} = self.health.arrange(open, &plan, SystemTime::now());
		// No fallbacks and no `order`: the first target of each name sent, and no other.
		if prefs.is_some_and(|p| !p.allow_fallbacks() && !p.has_order()) {
			let mut seen = HashSet::new();
			open.retain(|m| {
				seen.insert(
					plan.groups
						.and_then(|g| g.get(&m.name).copied())
						.unwrap_or(0),
				)
			});
		}
		// What the log says about the order, for anything but a model named alone.
		let routing = route.as_ref().map(|_| {
			let mut text = format!(
				"{}: {}",
				match sort {
					_ if by_order => "in the caller's provider.order",
					Some(Sort::Price) => "cheapest first",
					Some(Sort::Latency) => "fastest first byte first",
					Some(Sort::Throughput) => "most tokens a second first",
					Some(Sort::Weighted) => "drawn by weight, health and room",
					Some(Sort::Balanced) => "drawn by price, health, speed and room",
					Some(Sort::Order) | None => "in the order written",
				},
				open.iter()
					.map(|m| m.name.as_str())
					.collect::<Vec<_>>()
					.join(", ")
			);
			if let Some(name) = &stuck {
				text.push_str(&format!("; stayed on {name} for its cache"));
			}
			if !behind.is_empty() {
				text.push_str(&format!(
					"; behind the rest for provider.preferred_*: {}",
					behind.join(", ")
				));
			}
			if !left_out.is_empty() {
				text.push_str("; left out ");
				text.push_str(
					&left_out
						.iter()
						.map(|(name, why)| format!("{name} ({why})"))
						.collect::<Vec<_>>()
						.join(", "),
				);
			}
			text
		});
		if open.is_empty() {
			return Err(ApiError::new(
				Class::Forbidden,
				match route {
					None => format!("{} is not open to this account.", inbound.model),
					Some(_) => format!(
						"None of the models {} stands for is open to this account.",
						inbound.model
					),
				},
			));
		}
		// The prompt's upper bound, from what the caller sent, for a call a budget or a limit applies
		// to.
		let budgeted = self.budgets.as_ref().is_some_and(|b| b.applies(&who));
		let limited = self.limits.as_ref().is_some_and(|l| l.applies(&who));
		let shared = Shared {
			budgeted,
			limited,
			prompt_bound: (budgeted || limited).then(|| budgets::prompt_bound(&body)),
			route,
			routing,
			sticky,
			taken: Arc::default(),
			rate: std::sync::Mutex::default(),
		};
		let started = Instant::now();
		let mut number = 0u16;
		let mut last = None;
		let mut round = open;
		for pass in 0..PASSES {
			let mut again_later: Vec<Arc<Model>> = Vec::new();
			for (i, model) in round.iter().enumerate() {
				let mut tries = 0;
				loop {
					number += 1;
					tries += 1;
					let result = self
						.attempt(
							inbound.clone(),
							operation,
							model.clone(),
							who,
							&shared,
							number,
						)
						.await;
					let error = match result {
						Ok(out) => return Ok(shared.finish(out)),
						Err(error) => error,
					};
					if let Some(wait) = again(&error, tries) {
						tokio::time::sleep(wait).await;
						continue;
					}
					if !error.class.falls_over() {
						return Err(error);
					}
					if worth_another_pass(&error) {
						again_later.push(model.clone());
					}
					let last_of_round = i + 1 == round.len();
					// A route goes round again while time remains, and only if every target it has
					// is one that a moment may fix; a single model is the caller's and is told at once.
					let another = last_of_round
						&& pass + 1 < PASSES
						&& shared.route.is_some()
						&& !again_later.is_empty()
						&& started.elapsed() < self.timeouts.total / 2;
					if last_of_round && !another {
						return Err(error);
					}
					tracing::info!(target: "psyche::route", model = %model.name, class = error.class.code(), "falling over to the next target");
					last = Some(error);
					break;
				}
			}
			tokio::time::sleep(jitter()).await;
			round = again_later;
		}
		Err(last.unwrap_or_else(|| ApiError::new(Class::Upstream, "No model answered.")))
	}

	/// The client a call of this model goes through.
	fn client_for(&self, model: &Model) -> Result<&reqwest::Client, ApiError> {
		if model.internet != Internet::None {
			return Ok(&self.http);
		}
		self.sandbox.as_ref().ok_or_else(|| {
			ApiError::new(
				Class::EgressRefused,
				format!(
					"{} has the internet level none, and this Psyche only sends to public addresses.",
					model.name
				),
			)
		})
	}

	/// Why a model at the internet level `none` may not be sent this call; `None` for any other
	/// model. A hosted provider is public by what it is; for any other the address is read, and a
	/// name resolved, here so the caller hears before anything is sent. `net::sandbox_client` holds
	/// it at the connection, whatever the answer was here.
	async fn sandbox_refused(&self, model: &Model) -> Option<String> {
		if model.internet != Internet::None {
			return None;
		}
		let provider = &model.provider;
		let why = match provider.kind {
			Kind::Openai => net::sandbox_refused(&provider.base_url, self.egress).await?,
			Kind::Anthropic | Kind::Bedrock => {
				format!("{} is a public service", provider.kind.as_str())
			}
		};
		Some(format!("its internet level is none, and {why}"))
	}

	/// The rate limits a call takes, once whichever attempt gets there first; refused, the call is.
	fn take_once(
		&self,
		who: &Who,
		prompt_bound: Option<u64>,
		output: u64,
		shared: &Shared,
	) -> Result<(), ApiError> {
		let mut rate = shared.rate.lock().unwrap_or_else(|p| p.into_inner());
		if rate.is_some() {
			return Ok(());
		}
		let (taken, headers) = self.take(who, prompt_bound, output)?;
		*shared.taken.lock().unwrap_or_else(|p| p.into_inner()) = taken;
		*rate = Some(headers);
		Ok(())
	}

	/// One target, once.
	async fn attempt(
		&self,
		mut inbound: Inbound,
		operation: Operation,
		model: Arc<Model>,
		who: Who,
		shared: &Shared,
		number: u16,
	) -> Result<Response, ApiError> {
		if let Some(name) = &model.provider.credential
			&& model.provider.api_key.is_none()
			&& model.provider.aws.is_none()
		{
			return Err(ApiError::new(
				Class::ProviderAuth,
				format!(
					"{} takes the stored credential {name}, which is not set. Set it with PUT /ai/v1/admin/credentials/{name} and the service key.",
					model.provider.name
				),
			)
			.from_provider(&model.provider.name));
		}
		// The prompt's upper bound, from what the caller sent, for a call a budget or a limit applies
		// to.
		let Shared {
			budgeted,
			limited,
			prompt_bound,
			..
		} = *shared;
		let provider = &model.provider;
		if operation == Operation::Embeddings {
			if inbound.stream {
				return Err(ApiError::new(
					Class::BadRequest,
					"Embeddings are not streamed: leave stream out.",
				));
			}
			match provider.kind {
				Kind::Openai => {}
				Kind::Anthropic => {
					return Err(ApiError::new(
						Class::BadRequest,
						format!("{}'s provider has no embeddings API.", model.name),
					));
				}
				Kind::Bedrock => {
					let reservation =
						self.reserve(&who, &model, prompt_bound.filter(|_| budgeted), 0, 0)?;
					self.take_once(&who, prompt_bound.filter(|_| limited), 0, shared)?;
					let warned = warned(&reservation);
					let out = self
						.embed(&inbound, &model, who, reservation, shared, number)
						.await?;
					return Ok(warn(labelled(out, &model), warned));
				}
			}
		}
		// Under a hard budget a reply is never left uncapped: the provider is told the most it may
		// write, which is what is reserved.
		// Likewise its web searches: what a search tool can spend is bounded before it is reserved.
		let searches = if budgeted {
			inbound.bound_searches(&model)
		} else {
			0
		};
		if budgeted && let Some(cap) = self.sent_cap(&who, &inbound, &model, prompt_bound, searches)
		{
			inbound
				.body
				.insert(cap_field(provider, operation).into(), cap.into());
		}
		let body = inbound.upstream(&model, operation)?;
		let output = match operation {
			Operation::Embeddings => 0,
			_ if prompt_bound.is_some() => output_cap(&inbound, &body, &model),
			_ => 0,
		};
		let reservation = self.reserve(
			&who,
			&model,
			prompt_bound.filter(|_| budgeted),
			output,
			searches,
		)?;
		self.take_once(&who, prompt_bound.filter(|_| limited), output, shared)?;
		let warned = warned(&reservation);
		let (path, accept) = match provider.kind {
			Kind::Openai => (operation.path().to_string(), "text/event-stream"),
			Kind::Anthropic => ("messages".to_string(), "text/event-stream"),
			// The model id goes in the path, percent-encoded: `qwen.qwen3-coder-30b-a3b-v1:0` and
			// inference-profile ARNs hold characters a path segment may not.
			Kind::Bedrock => (
				format!(
					"model/{}/{}",
					sigv4::encode(&model.upstream),
					if inbound.stream {
						"converse-stream"
					} else {
						"converse"
					}
				),
				"application/vnd.amazon.eventstream",
			),
		};
		let url = url::Url::parse(&provider.endpoint(&path)).map_err(|_| {
			ApiError::new(
				Class::Upstream,
				format!("{}'s endpoint is not a URL.", provider.name),
			)
		})?;

		let started = Instant::now();
		let mut request = self
			.client_for(&model)?
			.post(url.clone())
			.header(header::CONTENT_TYPE, "application/json");
		request = provider.authorize(request, &url, &body, now());
		request = request.body(body);
		if provider.kind == Kind::Anthropic {
			request = request.header(
				"anthropic-version",
				inbound.version.as_deref().unwrap_or(anthropic::VERSION),
			);
			if let Some(beta) = &inbound.beta {
				request = request.header("anthropic-beta", beta);
			}
		}
		if inbound.stream {
			request = request.header(header::ACCEPT, accept);
		}
		let record = Record {
			sink: self.calls.clone(),
			call: Call {
				at: SystemTime::now(),
				operation,
				model: model.name.clone(),
				provider: provider.name.clone(),
				streamed: inbound.stream,
				outcome: Outcome::Cancelled,
				usage: None,
				price: model.price,
				who,
				first_byte: Duration::ZERO,
				total: Duration::ZERO,
				credits: false,
				budget_warning: None,
				route: shared.route.clone(),
				attempt: number,
				routing: shared.routing.clone(),
			},
			started,
			reservation,
			taken: shared.taken.clone(),
			billable: false,
			health: self.health.clone(),
			retry_after: None,
			remaining: Remaining::default(),
			sticky: shared.sticky.filter(|_| model.cache_pays()),
		};
		self.health.begin(&model, SystemTime::now());
		let response = match tokio::time::timeout(self.timeouts.first_byte, request.send()).await {
			Err(_) => {
				let error = ApiError::new(
					Class::Timeout,
					format!(
						"{} did not answer within {} seconds.",
						provider.name,
						self.timeouts.first_byte.as_secs()
					),
				)
				.from_provider(&provider.name);
				return Err(record.fail(error));
			}
			Ok(Err(e)) => return Err(record.fail(send_error(&e, &provider.name))),
			Ok(Ok(response)) => response,
		};
		let mut record = record;
		record.call.first_byte = started.elapsed();
		record.remaining = Remaining::read(response.headers());

		if !response.status().is_success() {
			let error = self.provider_error(response, &model).await;
			return Err(record.fail(error));
		}
		// From here the provider may be producing tokens, so a call that ends with no usage counted
		// is charged its whole reservation.
		record.billable = true;
		let out = if inbound.stream {
			self.stream(response, &model, &inbound, operation, record)
		} else {
			self.whole(response, &model, &inbound, operation, record)
				.await?
		};
		Ok(warn(labelled(out, &model), warned))
	}

	/// Embeddings from a Bedrock model, in the InvokeModel body the catalogue names for it: every
	/// call made (a few at a time), the vectors put back in the order of the texts, and the input
	/// counted from what each call reported. Any call failing fails the request; the rest are
	/// dropped, and with them their connections.
	async fn embed(
		&self,
		inbound: &Inbound,
		model: &Model,
		who: Who,
		reservation: Option<Reservation>,
		shared: &Shared,
		number: u16,
	) -> Result<Response, ApiError> {
		let provider = &model.provider;
		let family = model.embeddings.ok_or_else(|| {
			ApiError::new(
				Class::BadRequest,
				format!(
					"{} is not an embedding model. GET /ai/v1/models lists the models.",
					model.name
				),
			)
		})?;
		let request = embeddings::request(&inbound.body, family)?;
		let url = url::Url::parse(
			&provider.endpoint(&format!("model/{}/invoke", sigv4::encode(&model.upstream))),
		)
		.map_err(|_| {
			ApiError::new(
				Class::Upstream,
				format!("{}'s endpoint is not a URL.", provider.name),
			)
		})?;
		let mut record = Record {
			sink: self.calls.clone(),
			call: Call {
				at: SystemTime::now(),
				operation: Operation::Embeddings,
				model: model.name.clone(),
				provider: provider.name.clone(),
				streamed: false,
				outcome: Outcome::Cancelled,
				usage: None,
				price: model.price,
				who,
				first_byte: Duration::ZERO,
				total: Duration::ZERO,
				credits: false,
				budget_warning: None,
				route: shared.route.clone(),
				attempt: number,
				routing: shared.routing.clone(),
			},
			started: Instant::now(),
			reservation,
			taken: shared.taken.clone(),
			// Each text is its own call, and some may be billed before another fails.
			billable: true,
			health: self.health.clone(),
			retry_after: None,
			remaining: Remaining::default(),
			sticky: None,
		};
		self.health.begin(model, SystemTime::now());
		let calls = request
			.bodies(family)
			.into_iter()
			.map(|body| self.invoke(&url, model, body));
		let replies = tokio::time::timeout(
			self.timeouts.total,
			futures_util::stream::iter(calls)
				.buffered(EMBED_CONCURRENCY)
				.try_collect::<Vec<_>>(),
		)
		.await;
		let replies = match replies {
			Err(_) => {
				return Err(record.fail(
					ApiError::new(
						Class::Timeout,
						format!("{} did not finish the embeddings in time.", provider.name),
					)
					.from_provider(&provider.name),
				));
			}
			Ok(Err(error)) => return Err(record.fail(error)),
			Ok(Ok(replies)) => replies,
		};
		let mut vectors = Vec::with_capacity(request.inputs.len());
		let mut tokens = Some(0u64);
		let mut broken = None;
		for invoked in &replies {
			record.call.first_byte = record.call.first_byte.max(invoked.first_byte);
			tokens = tokens.zip(invoked.tokens).map(|(a, b)| a + b);
			match embeddings::vectors(family, &invoked.reply) {
				Ok(more) => vectors.extend(more),
				Err(why) => {
					broken = Some(why);
					break;
				}
			}
		}
		if broken.is_none() && vectors.len() != request.inputs.len() {
			broken = Some(format!(
				"{} embeddings for {} texts",
				vectors.len(),
				request.inputs.len()
			));
		}
		let body = match broken {
			None => embeddings::reply(vectors, request.base64, &model.name, tokens),
			Some(why) => Err(why),
		};
		let body = match body {
			Ok(body) => body,
			Err(why) => {
				return Err(record.fail(
					ApiError::new(
						Class::Upstream,
						format!(
							"{} answered with something that is not its API: {why}.",
							provider.name
						),
					)
					.from_provider(&provider.name),
				));
			}
		};
		record.call.usage = tokens.map(|prompt_tokens| Usage {
			prompt_tokens,
			..Usage::default()
		});
		record.finish(Outcome::Completed);
		Ok((
			StatusCode::OK,
			[(header::CONTENT_TYPE, "application/json")],
			serde_json::to_vec(&body).unwrap_or_default(),
		)
			.into_response())
	}

	/// One InvokeModel call.
	async fn invoke(
		&self,
		url: &url::Url,
		model: &Model,
		body: Value,
	) -> Result<Invoked, ApiError> {
		let provider = &model.provider;
		let body = serde_json::to_vec(&body).unwrap_or_default();
		let started = Instant::now();
		let request = self
			.client_for(model)?
			.post(url.clone())
			.header(header::CONTENT_TYPE, "application/json");
		let request = provider.authorize(request, url, &body, now()).body(body);
		let response = match tokio::time::timeout(self.timeouts.first_byte, request.send()).await {
			Err(_) => {
				return Err(ApiError::new(
					Class::Timeout,
					format!(
						"{} did not answer within {} seconds.",
						provider.name,
						self.timeouts.first_byte.as_secs()
					),
				)
				.from_provider(&provider.name));
			}
			Ok(Err(e)) => return Err(send_error(&e, &provider.name)),
			Ok(Ok(response)) => response,
		};
		let first_byte = started.elapsed();
		if !response.status().is_success() {
			return Err(self.provider_error(response, model).await);
		}
		let counted = response
			.headers()
			.get("x-amzn-bedrock-input-token-count")
			.and_then(|v| v.to_str().ok())
			.and_then(|v| v.trim().parse::<u64>().ok());
		let body = read(response, MAX_REPLY_BODY).await.map_err(|e| {
			ApiError::new(
				Class::Transport,
				format!("{}'s reply broke off: {e}", provider.name),
			)
			.from_provider(&provider.name)
		})?;
		let reply: Value = serde_json::from_slice(&body).map_err(|_| {
			ApiError::new(
				Class::Upstream,
				format!(
					"{} answered with something that is not JSON.",
					provider.name
				),
			)
			.from_provider(&provider.name)
		})?;
		let tokens = counted.or_else(|| reply.get("inputTextTokenCount").and_then(Value::as_u64));
		Ok(Invoked {
			reply,
			tokens,
			first_byte,
		})
	}

	/// A non-streamed reply: read whole (it is bounded), counted, passed on as the provider wrote it.
	async fn whole(
		&self,
		response: reqwest::Response,
		model: &Model,
		inbound: &Inbound,
		operation: Operation,
		mut record: Record,
	) -> Result<Response, ApiError> {
		let provider = record.call.provider.clone();
		let deadline = self.timeouts.total.saturating_sub(record.started.elapsed());
		let body = match tokio::time::timeout(deadline, read(response, MAX_REPLY_BODY)).await {
			Err(_) => {
				return Err(record.fail(
					ApiError::new(
						Class::Timeout,
						format!("{provider} did not finish its reply in time."),
					)
					.from_provider(&provider),
				));
			}
			Ok(Err(e)) => {
				return Err(record.fail(
					ApiError::new(
						Class::Transport,
						format!("{provider}'s reply broke off: {e}"),
					)
					.from_provider(&provider),
				));
			}
			Ok(Ok(body)) => body,
		};
		let Ok(value) = serde_json::from_slice::<Value>(&body) else {
			return Err(record.fail(
				ApiError::new(
					Class::Upstream,
					format!("{provider} answered with something that is not JSON."),
				)
				.from_provider(&provider),
			));
		};
		if operation == Operation::Messages {
			let reply = match model.provider.kind {
				Kind::Anthropic => {
					record.call.usage = value.get("usage").and_then(Usage::from_anthropic);
					let mut value = value;
					rename(&mut value, &model.name);
					Ok(value)
				}
				Kind::Openai => {
					record.call.usage = Usage::from_openai(&value);
					messages::reply(&value, record.call.usage, &model.name)
				}
				Kind::Bedrock => {
					record.call.usage = value.get("usage").and_then(Usage::from_bedrock);
					bedrock::reply(&value, &model.name, &bedrock::new_id(), now())
						.and_then(|chat| messages::reply(&chat, record.call.usage, &model.name))
				}
			};
			let reply = match reply {
				Ok(reply) => reply,
				Err(why) => {
					record.call.usage = None;
					return Err(record.fail(
						ApiError::new(
							Class::Upstream,
							format!(
								"{provider} answered with something that is not its API: {why}."
							),
						)
						.from_provider(&provider),
					));
				}
			};
			record.finish(Outcome::Completed);
			return Ok((
				StatusCode::OK,
				[(header::CONTENT_TYPE, "application/json")],
				serde_json::to_vec(&reply).unwrap_or_default(),
			)
				.into_response());
		}
		if operation == Operation::Responses {
			let chat = match model.provider.kind {
				Kind::Openai => {
					record.call.usage = Usage::from_openai(&value);
					Ok(value)
				}
				Kind::Anthropic => {
					record.call.usage = value.get("usage").and_then(Usage::from_anthropic);
					anthropic::reply(&value, &model.name, now())
				}
				Kind::Bedrock => {
					record.call.usage = value.get("usage").and_then(Usage::from_bedrock);
					bedrock::reply(&value, &model.name, &bedrock::new_id(), now())
				}
			};
			let custom = responses::custom_tools(&inbound.body);
			let reply = chat.and_then(|chat| {
				responses::reply(&chat, record.call.usage, &model.name, &custom, now())
			});
			let reply = match reply {
				Ok(reply) => reply,
				Err(why) => {
					record.call.usage = None;
					return Err(record.fail(
						ApiError::new(
							Class::Upstream,
							format!(
								"{provider} answered with something that is not its API: {why}."
							),
						)
						.from_provider(&provider),
					));
				}
			};
			record.finish(Outcome::Completed);
			return Ok((
				StatusCode::OK,
				[(header::CONTENT_TYPE, "application/json")],
				serde_json::to_vec(&reply).unwrap_or_default(),
			)
				.into_response());
		}
		let body = match model.provider.kind {
			Kind::Openai => {
				record.call.usage = Usage::from_openai(&value);
				let mut value = value;
				if rename(&mut value, &model.name) {
					serde_json::to_vec(&value).unwrap_or_default()
				} else {
					body
				}
			}
			Kind::Anthropic | Kind::Bedrock => match match model.provider.kind {
				Kind::Bedrock => bedrock::reply(&value, &model.name, &bedrock::new_id(), now()),
				_ => anthropic::reply(&value, &model.name, now()),
			} {
				Ok(reply) => {
					record.call.usage = value.get("usage").and_then(match model.provider.kind {
						Kind::Bedrock => Usage::from_bedrock,
						_ => Usage::from_anthropic,
					});
					serde_json::to_vec(&reply).unwrap_or_default()
				}
				Err(why) => {
					return Err(record.fail(
						ApiError::new(
							Class::Upstream,
							format!(
								"{provider} answered with something that is not its API: {why}."
							),
						)
						.from_provider(&provider),
					));
				}
			},
		};
		record.finish(Outcome::Completed);
		Ok((
			StatusCode::OK,
			[(header::CONTENT_TYPE, "application/json")],
			body,
		)
			.into_response())
	}

	/// A streamed reply, forwarded read by read.
	fn stream(
		&self,
		response: reqwest::Response,
		model: &Model,
		inbound: &Inbound,
		operation: Operation,
		record: Record,
	) -> Response {
		let wants_usage = inbound.wants_usage;
		let inbound_messages = operation == Operation::Messages;
		// Written in another shape, so it needs the usage whether or not the caller asked.
		let written = inbound_messages || operation == Operation::Responses;
		let messages = match operation {
			Operation::Messages => Some(match model.provider.kind {
				Kind::Anthropic => MessagesStream::Pass(anthropic::Meter::default()),
				kind => {
					let mut events = messages::Events::new(&model.name);
					if kind == Kind::Bedrock {
						events.signed();
					}
					MessagesStream::Translate(Box::new(events))
				}
			}),
			Operation::Responses => Some(MessagesStream::Translate(Box::new(
				responses::Events::new(&model.name, responses::custom_tools(&inbound.body), now()),
			))),
			_ => None,
		};
		let state = Streaming {
			upstream: Box::pin(response.bytes_stream()),
			parser: Parser::new(),
			translate: match model.provider.kind {
				Kind::Openai => None,
				Kind::Anthropic if inbound_messages => None,
				Kind::Anthropic => Some(Translator::anthropic(
					wants_usage || written,
					&model.name,
					now(),
				)),
				Kind::Bedrock => Some(Translator::bedrock(
					wants_usage || written,
					&model.name,
					&bedrock::new_id(),
					now(),
				)),
			},
			messages,
			wants_usage,
			idle: self.timeouts.idle,
			secrets: self
				.catalogue
				.get()
				.secrets()
				.into_iter()
				.map(String::from)
				.collect(),
			done: false,
			record,
		};
		let body = futures_util::stream::unfold(state, |mut state| async move {
			let out = state.next().await?;
			Some((Ok::<Bytes, std::io::Error>(out), state))
		});
		(
			StatusCode::OK,
			[
				(header::CONTENT_TYPE, "text/event-stream"),
				(header::CACHE_CONTROL, "no-cache"),
				// Anything between here and the caller (the front door, nginx) must not hold the
				// stream back to buffer it.
				(header::HeaderName::from_static("x-accel-buffering"), "no"),
			],
			Body::from_stream(body),
		)
			.into_response()
	}

	/// A provider's error status, as the caller and the log should see it.
	async fn provider_error(&self, response: reqwest::Response, model: &Model) -> ApiError {
		let provider = &model.provider.name;
		let status = response.status();
		let retry_after = response
			.headers()
			.get(header::RETRY_AFTER)
			.and_then(|v| v.to_str().ok())
			.map(String::from);
		let body = tokio::time::timeout(self.timeouts.idle, read(response, MAX_ERROR_BODY))
			.await
			.ok()
			.and_then(Result::ok)
			.unwrap_or_default();
		let catalogue = self.catalogue.get();
		let text = scrub(&String::from_utf8_lossy(&body), &catalogue.secrets());
		let parsed = serde_json::from_str::<Value>(&text).ok();
		let said = parsed
			.as_ref()
			.and_then(|v| {
				v.pointer("/error/message")
					.or_else(|| v.get("message"))
					.or_else(|| v.get("Message"))
					.or_else(|| v.get("error"))
					.and_then(Value::as_str)
					.map(String::from)
			})
			.unwrap_or_else(|| text.chars().take(500).collect());
		// Out of credit or quota: OpenAI says so in a 429 (`insufficient_quota`), OpenRouter with a
		// 402, Anthropic in a 400 about the credit balance. A retry cannot succeed, so none of it is
		// `rate_limited` or the caller's `rejected_by_provider`.
		let names = |name: &str| {
			parsed.as_ref().is_some_and(|v| {
				["/error/code", "/error/type"]
					.iter()
					.any(|at| v.pointer(at).and_then(Value::as_str) == Some(name))
			})
		};
		let billing = match status.as_u16() {
			402 => true,
			429 => names("insufficient_quota"),
			400 if model.provider.kind == Kind::Anthropic => {
				said.contains("credit balance is too low")
			}
			_ => false,
		};
		let (class, message) = match status.as_u16() {
			_ if billing => (
				Class::ProviderBilling,
				format!(
					"{provider} refused the call because the account behind Psyche's credential has no credit or quota left ({status}): {said}"
				),
			),
			// Bedrock's ServiceUnavailableException: the model has no capacity for this account just
			// now, which is a limit to route around, not a failure.
			503 if model.provider.kind == Kind::Bedrock => (
				Class::RateLimited,
				format!("{provider} has no capacity just now: {said}"),
			),
			401 | 403 => (
				Class::ProviderAuth,
				format!("{provider} refused Psyche's credential ({status}): {said}"),
			),
			429 => (
				Class::RateLimited,
				format!("{provider} is rate-limiting: {said}"),
			),
			408 => (Class::Timeout, format!("{provider} timed out: {said}")),
			400..=499 => (
				Class::Rejected,
				format!("{provider} refused the request: {said}"),
			),
			_ => (
				Class::Upstream,
				format!("{provider} failed ({status}): {said}"),
			),
		};
		let mut error = ApiError::new(class, message).from_provider(provider);
		if class == Class::Rejected {
			// The caller's own request was at fault: they get the provider's status.
			error.status =
				Some(StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_REQUEST));
		}
		// OpenAI sends `retry-after` with `insufficient_quota` too, which would invite a retry.
		if class != Class::ProviderBilling {
			error.retry_after = retry_after;
		}
		error
	}
}

/// One embedding call's reply.
struct Invoked {
	reply: Value,
	tokens: Option<u64>,
	first_byte: Duration,
}

/// Says which catalogue model and which provider answered.
fn labelled(mut out: Response, model: &Model) -> Response {
	for (name, value) in [
		("x-psyche-model", &model.name),
		("x-psyche-provider", &model.provider.name),
	] {
		if let Ok(value) = HeaderValue::from_str(value) {
			out.headers_mut().insert(name, value);
		}
	}
	out
}

/// Gives a reply (or a chunk of one) the model name the caller asked for. Whether it changed
/// anything, so an unchanged event is forwarded as the bytes the provider sent.
fn rename(value: &mut Value, name: &str) -> bool {
	match value.get_mut("model") {
		Some(model) if model.as_str() != Some(name) && model.is_string() => {
			*model = Value::String(name.to_string());
			true
		}
		_ => false,
	}
}

/// Why a request never got a response.
fn send_error(error: &reqwest::Error, provider: &str) -> ApiError {
	let chain = std::iter::successors(std::error::Error::source(error), |s| s.source())
		.map(ToString::to_string)
		.collect::<Vec<_>>()
		.join(": ");
	let class = if chain.contains(net::REFUSED) {
		Class::EgressRefused
	} else if error.is_timeout() {
		Class::Timeout
	} else {
		Class::Transport
	};
	let detail = if chain.is_empty() {
		error.to_string()
	} else {
		chain
	};
	ApiError::new(class, format!("Could not reach {provider}: {detail}")).from_provider(provider)
}

/// Reads a body up to `limit` bytes, refusing one that is longer.
async fn read(mut response: reqwest::Response, limit: usize) -> Result<Vec<u8>, String> {
	let mut body = Vec::new();
	while let Some(chunk) = response.chunk().await.map_err(|e| e.to_string())? {
		if body.len() + chunk.len() > limit {
			return Err(format!("the reply is longer than {limit} bytes"));
		}
		body.extend_from_slice(&chunk);
	}
	Ok(body)
}

/// A call in flight. Whatever ends it (completion, failure, or the caller dropping the stream),
/// it is recorded exactly once, when this is dropped.
/// What the attempts of one call share.
struct Shared {
	budgeted: bool,
	limited: bool,
	prompt_bound: Option<u64>,
	route: Option<String>,
	routing: Option<String>,
	/// The conversation, and whether the caller named it (`health::conversation`).
	sticky: Option<(u64, bool)>,
	taken: Arc<std::sync::Mutex<Option<Taken>>>,
	/// The rate-limit headers, once a limit was taken.
	rate: std::sync::Mutex<Option<limits::Headers>>,
}

impl Shared {
	/// The answer, with the call's rate-limit headers and, for a route, which it was.
	fn finish(&self, out: Response) -> Response {
		let rate = self
			.rate
			.lock()
			.unwrap_or_else(|p| p.into_inner())
			.take()
			.unwrap_or_default();
		let mut out = headed(out, rate);
		if let Some(route) = &self.route
			&& let Ok(value) = HeaderValue::from_str(route)
		{
			out.headers_mut().insert("x-psyche-route", value);
		}
		out
	}
}

/// The longest a provider's `retry-after` is waited out on the same target; a longer one moves to
/// the next target, or comes back to the caller.
const LONGEST_RETRY_WAIT: Duration = Duration::from_secs(2);

/// Whether to try the same target again after `error`, on its `tries`-th try, and after how long:
/// once, on a rate limit with a short or no `retry-after`, a 5xx or no connection. Never on a
/// timeout, which already took its whole wait.
fn again(error: &ApiError, tries: u32) -> Option<Duration> {
	if tries > 1 {
		return None;
	}
	match error.class {
		Class::RateLimited => {
			let wait = match error.retry_after.as_deref().map(str::parse::<f64>) {
				None => jitter(),
				Some(Ok(seconds)) if seconds >= 0.0 => Duration::from_secs_f64(seconds),
				Some(_) => return None,
			};
			(wait <= LONGEST_RETRY_WAIT).then_some(wait)
		}
		Class::Upstream | Class::Transport => Some(jitter()),
		_ => None,
	}
}

/// Whether a failed target is worth asking again on a route's second pass: what a moment may fix.
/// A rate limit that named a longer wait is not, and neither is a refused or empty account.
fn worth_another_pass(error: &ApiError) -> bool {
	error.class == Class::Timeout || again(error, 1).is_some()
}

/// 100 to 300 ms, so callers that failed together do not come back together.
fn jitter() -> Duration {
	let nanos = SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.map_or(0, |d| d.subsec_nanos());
	Duration::from_millis(100 + u64::from(nanos % 200))
}

struct Record {
	sink: Option<tokio::sync::mpsc::UnboundedSender<Call>>,
	call: Call,
	started: Instant,
	/// The room held in the call's budgets, settled when this is dropped.
	reservation: Option<Reservation>,
	/// What the call took from its rate limits, shared by its attempts; the tokens it did not use go
	/// back when the attempt that answered ends, and a failed attempt leaves it for the next.
	taken: Arc<std::sync::Mutex<Option<Taken>>>,
	/// Whether the provider may have produced anything it bills for: false until a successful
	/// response starts, so a refused or failed request releases its reservation and is charged
	/// nothing.
	billable: bool,
	/// Told how the attempt went when this is dropped.
	health: Health,
	/// A provider's `retry-after` on a 429, for the target's cooldown.
	retry_after: Option<Duration>,
	/// What the provider said is left on Psyche's credential.
	remaining: Remaining,
	/// The conversation this call is part of, and whether the caller named it, for a target whose
	/// cache is worth coming back to.
	sticky: Option<(u64, bool)>,
}

impl Record {
	fn finish(&mut self, outcome: Outcome) {
		self.call.outcome = outcome;
	}

	fn fail(mut self, error: ApiError) -> ApiError {
		self.finish(Outcome::Failed(error.class));
		self.retry_after = error
			.retry_after
			.as_deref()
			.and_then(|s| s.trim().parse::<f64>().ok())
			.filter(|s| s.is_finite() && *s >= 0.0)
			.map(Duration::from_secs_f64);
		error
	}
}

impl Drop for Record {
	fn drop(&mut self) {
		self.call.total = self.started.elapsed();
		let call = &self.call;
		let usage = call.usage.unwrap_or_default();
		let cost = call.cost().map(|c| c.to_string());
		tracing::info!(
			target: "psyche::call",
			operation = call.operation.name(),
			model = %call.model,
			provider = %call.provider,
			streamed = call.streamed,
			outcome = ?call.outcome,
			prompt_tokens = usage.prompt_tokens,
			completion_tokens = usage.completion_tokens,
			counted = call.usage.is_some(),
			cost = cost.as_deref(),
			first_byte_ms = call.first_byte.as_millis() as u64,
			total_ms = call.total.as_millis() as u64,
		);
		if self.billable
			&& let Some(taken) = self.taken.lock().unwrap_or_else(|p| p.into_inner()).take()
		{
			taken.settle(self.call.usage.map(|u| u.total()));
		}
		if let Some(reservation) = self.reservation.take() {
			self.call.credits = reservation.credits.is_some();
			if !reservation.warnings.is_empty() {
				self.call.budget_warning = Some(reservation.warnings.join(", "));
			}
			let charge = match (self.billable, self.call.cost()) {
				(false, _) => Some(0),
				(true, cost) => cost.map(|c| c.0),
			};
			reservation.settle(charge, SystemTime::now());
		}
		self.health.seen(
			&self.call.provider,
			&self.call.model,
			health::Seen::of(&self.call, self.retry_after, self.remaining),
			SystemTime::now(),
		);
		if self.call.outcome == Outcome::Completed
			&& let Some((conversation, named)) = self.sticky
		{
			let hit = self
				.call
				.usage
				.and_then(|u| u.cached_tokens)
				.is_some_and(|read| read > 0);
			self.health.stick(
				conversation,
				&self.call.provider,
				&self.call.model,
				named || hit,
				SystemTime::now(),
			);
		}
		if let Some(sink) = &self.sink {
			let _ = sink.send(self.call.clone());
		}
	}
}

type Upstream = Pin<Box<dyn Stream<Item = reqwest::Result<Bytes>> + Send>>;

/// What a stream on `/messages` or `/responses` is doing.
enum MessagesStream {
	/// An Anthropic provider's events, forwarded and counted.
	Pass(anthropic::Meter),
	/// OpenAI's chunks, written in the shape the caller asked for.
	Translate(Box<dyn Writer>),
}

impl MessagesStream {
	fn keep_alive(&self) -> &'static [u8] {
		match self {
			Self::Pass(_) => messages::PING,
			Self::Translate(writer) => writer.keep_alive(),
		}
	}
}

/// What turns a stream of OpenAI's chunks into another shape's events.
trait Writer: Send {
	fn chunk(&mut self, chunk: &Value) -> Vec<u8>;
	fn end(&mut self) -> Vec<u8>;
	fn finished(&self) -> bool;
	fn keep_alive(&self) -> &'static [u8];
	/// A failure once the reply has started, the way this shape ends one.
	fn failure(&mut self, error: &ApiError) -> Vec<u8>;
}

impl Writer for messages::Events {
	fn chunk(&mut self, chunk: &Value) -> Vec<u8> {
		messages::Events::chunk(self, chunk)
	}

	fn end(&mut self) -> Vec<u8> {
		messages::Events::end(self)
	}

	fn finished(&self) -> bool {
		messages::Events::finished(self)
	}

	fn keep_alive(&self) -> &'static [u8] {
		messages::PING
	}

	fn failure(&mut self, error: &ApiError) -> Vec<u8> {
		messages::error_event(error)
	}
}

impl Writer for responses::Events {
	fn chunk(&mut self, chunk: &Value) -> Vec<u8> {
		responses::Events::chunk(self, chunk)
	}

	fn end(&mut self) -> Vec<u8> {
		responses::Events::end(self)
	}

	fn finished(&self) -> bool {
		responses::Events::finished(self)
	}

	fn keep_alive(&self) -> &'static [u8] {
		responses::KEEP_ALIVE
	}

	fn failure(&mut self, error: &ApiError) -> Vec<u8> {
		self.failed(error)
	}
}

/// The state of one streamed reply.
struct Streaming {
	upstream: Upstream,
	parser: Parser,
	/// For a provider whose stream is not OpenAI's: what turns it into OpenAI's chunks.
	translate: Option<Translator>,
	/// For `/messages` and `/responses`: the shape the caller gets.
	messages: Option<MessagesStream>,
	wants_usage: bool,
	idle: Duration,
	secrets: Vec<String>,
	done: bool,
	record: Record,
}

impl Streaming {
	/// The next bytes for the caller, or `None` once the stream is over.
	async fn next(&mut self) -> Option<Bytes> {
		loop {
			if self.done {
				return None;
			}
			let provider = self.record.call.provider.clone();
			match tokio::time::timeout(self.idle, self.upstream.next()).await {
				Err(_) => {
					return Some(self.fail(
						Class::Timeout,
						format!(
							"{provider} went quiet for {} seconds in the middle of its reply.",
							self.idle.as_secs()
						),
					));
				}
				Ok(Some(Err(e))) => {
					return Some(self.fail(
						Class::Transport,
						format!("{provider}'s stream broke off: {e}"),
					));
				}
				Ok(None) => {
					let out = match self.translate.as_mut() {
						// The provider closed its stream: a cut reply is an error event.
						Some(translate) => {
							let pieces = translate.end();
							self.render(pieces)
						}
						None if self.messages.is_some() => self.close_messages(),
						None => {
							let rest = std::mem::take(&mut self.parser).finish();
							self.pass(rest)
						}
					};
					if !self.done {
						self.done = true;
						if self.record.call.outcome == Outcome::Cancelled {
							self.record.finish(Outcome::Completed);
						}
					}
					return (!out.is_empty()).then_some(out);
				}
				Ok(Some(Ok(chunk))) if self.translate.is_some() => {
					let pieces = self
						.translate
						.as_mut()
						.map(|t| t.feed(&chunk))
						.unwrap_or_default();
					let out = self.render(pieces);
					if !out.is_empty() {
						return Some(out);
					}
				}
				Ok(Some(Ok(chunk))) => match self.parser.feed(&chunk) {
					Err(e) => return Some(self.fail(Class::Upstream, format!("{provider}: {e}"))),
					Ok(events) => {
						let out = if self.messages.is_some() {
							self.pass_messages(events)
						} else {
							self.pass(events)
						};
						if !out.is_empty() {
							return Some(out);
						}
					}
				},
			}
		}
	}

	/// The bytes of the events the caller gets: usage counted, the usage chunk kept back from a
	/// caller that did not ask for it, an error event noted and scrubbed.
	fn pass(&mut self, events: Vec<Event>) -> Bytes {
		let mut out = Vec::new();
		for event in events {
			let mut parsed = event
				.data
				.as_deref()
				.filter(|d| *d != "[DONE]")
				.and_then(|d| serde_json::from_str::<Value>(d).ok());
			if let Some(value) = &mut parsed {
				if value.get("error").is_some() {
					self.record.finish(Outcome::Failed(Class::Upstream));
					let secrets: Vec<&str> = self.secrets.iter().map(String::as_str).collect();
					let clean = scrub(&String::from_utf8_lossy(&event.raw), &secrets);
					out.extend_from_slice(clean.as_bytes());
					continue;
				}
				if let Some(usage) = Usage::from_openai(value) {
					self.record.call.usage = Some(usage);
					let no_choices = value
						.get("choices")
						.and_then(Value::as_array)
						.is_none_or(Vec::is_empty);
					if !self.wants_usage && no_choices {
						continue;
					}
				}
				if rename(value, &self.record.call.model) {
					out.extend_from_slice(format!("data: {value}\n\n").as_bytes());
					continue;
				}
			}
			out.extend_from_slice(&event.raw);
		}
		Bytes::from(out)
	}

	/// A `/messages` or `/responses` stream's next events, from an Anthropic provider or an
	/// OpenAI-style one. A failure the provider reports ends the reply as that API ends one.
	fn pass_messages(&mut self, events: Vec<Event>) -> Bytes {
		let mut out = Vec::new();
		let mut failed = None;
		for event in events {
			let parsed = event
				.data
				.as_deref()
				.filter(|d| *d != "[DONE]")
				.and_then(|d| serde_json::from_str::<Value>(d).ok());
			match self.messages.as_mut() {
				Some(MessagesStream::Pass(meter)) => {
					let Some(mut value) = parsed else {
						out.extend_from_slice(&event.raw);
						continue;
					};
					meter.read(&value);
					self.record.call.usage = meter.usage();
					if value["type"] == "error" {
						let kind = value.pointer("/error/type").and_then(Value::as_str);
						let class = anthropic::error_class(kind);
						self.record.finish(Outcome::Failed(class));
						let secrets: Vec<&str> = self.secrets.iter().map(String::as_str).collect();
						let clean = scrub(&String::from_utf8_lossy(&event.raw), &secrets);
						out.extend_from_slice(clean.as_bytes());
						continue;
					}
					if value["type"] == "message_start"
						&& rename(&mut value["message"], &self.record.call.model)
					{
						out.extend_from_slice(
							format!("event: message_start\ndata: {value}\n\n").as_bytes(),
						);
						continue;
					}
					out.extend_from_slice(&event.raw);
				}
				Some(MessagesStream::Translate(writer)) => {
					if event.data.as_deref() == Some("[DONE]") {
						out.extend(writer.end());
						continue;
					}
					let Some(value) = parsed else {
						continue;
					};
					if let Some(error) = value.get("error") {
						let said = error
							.get("message")
							.and_then(Value::as_str)
							.unwrap_or("an error with no message");
						failed = Some(said.to_string());
						break;
					}
					if let Some(usage) = Usage::from_openai(&value) {
						self.record.call.usage = Some(usage);
					}
					out.extend(writer.chunk(&value));
				}
				None => {}
			}
		}
		if let Some(message) = failed {
			let secrets: Vec<&str> = self.secrets.iter().map(String::as_str).collect();
			let provider = self.record.call.provider.clone();
			let message = format!("{provider}: {}", scrub(&message, &secrets));
			out.extend_from_slice(&self.fail(Class::Upstream, message));
		}
		Bytes::from(out)
	}

	/// The provider closed a `/messages` stream: what is still owed, or an error event if the reply
	/// was cut.
	fn close_messages(&mut self) -> Bytes {
		let rest = std::mem::take(&mut self.parser).finish();
		let mut out = self.pass_messages(rest).to_vec();
		if self.done {
			return Bytes::from(out);
		}
		let cut = match self.messages.as_mut() {
			// A failure the provider reported has been passed on already.
			Some(MessagesStream::Pass(meter)) => {
				!meter.ended() && self.record.call.outcome == Outcome::Cancelled
			}
			Some(MessagesStream::Translate(writer)) if writer.finished() => {
				out.extend(writer.end());
				false
			}
			Some(MessagesStream::Translate(_)) => true,
			None => false,
		};
		if cut {
			let provider = self.record.call.provider.clone();
			out.extend_from_slice(&self.fail(
				Class::Upstream,
				format!("{provider}: the stream ended before the reply was complete"),
			));
		}
		Bytes::from(out)
	}

	/// The bytes of a translated stream's pieces. The first error ends the stream.
	fn render(&mut self, pieces: Vec<Out>) -> Bytes {
		let mut out = Vec::new();
		let mut failed = None;
		for piece in pieces {
			match (piece, self.messages.as_mut()) {
				(Out::Chunk(chunk), Some(MessagesStream::Translate(writer))) => {
					out.extend(writer.chunk(&chunk));
				}
				(Out::Done, Some(MessagesStream::Translate(writer))) => out.extend(writer.end()),
				(Out::KeepAlive, Some(stream)) => out.extend_from_slice(stream.keep_alive()),
				(Out::Chunk(chunk), _) => {
					out.extend_from_slice(format!("data: {chunk}\n\n").as_bytes());
				}
				(Out::Done, _) => out.extend_from_slice(b"data: [DONE]\n\n"),
				(Out::KeepAlive, _) => out.extend_from_slice(b": ping\n\n"),
				(Out::Error(class, message), _) => {
					failed = Some((class, message));
					break;
				}
			}
		}
		self.record.call.usage = self.translate.as_ref().and_then(Translator::usage);
		if let Some((class, message)) = failed {
			let secrets: Vec<&str> = self.secrets.iter().map(String::as_str).collect();
			let provider = self.record.call.provider.clone();
			let message = format!("{provider}: {}", scrub(&message, &secrets));
			out.extend_from_slice(&self.fail(class, message));
		}
		Bytes::from(out)
	}

	/// Ends the stream with an error event, in the caller's protocol.
	fn fail(&mut self, class: Class, message: String) -> Bytes {
		self.done = true;
		self.record.finish(Outcome::Failed(class));
		let error = ApiError::new(class, message).from_provider(&self.record.call.provider);
		match self.messages.as_mut() {
			Some(MessagesStream::Translate(writer)) => return Bytes::from(writer.failure(&error)),
			Some(MessagesStream::Pass(_)) => return Bytes::from(messages::error_event(&error)),
			None => {}
		}
		Bytes::from(format!("data: {}\n\n", error.body()))
	}
}

#[cfg(test)]
mod tests {
	use std::collections::HashMap;

	use super::*;
	use crate::fake::Fake;
	use crate::net::Egress;

	const SECRET: &str = "a-project-secret-of-at-least-32-chars";

	struct Setup {
		base: String,
		fake: Fake,
		/// The file's catalogue and the live one, for a watcher a test starts.
		tables: models::Held,
		calls: tokio::sync::mpsc::UnboundedReceiver<Call>,
		client: reqwest::Client,
		/// What routing knows, for a test that measures it.
		health: Health,
	}

	impl Setup {
		async fn new() -> Self {
			Self::with(Fake::KEY, Timeouts::default()).await
		}

		/// Psyche in front of a fake provider, presenting `key` to it.
		async fn with(key: &str, timeouts: Timeouts) -> Self {
			Self::build(key, timeouts, None, None).await
		}

		async fn build(
			key: &str,
			timeouts: Timeouts,
			keys: Option<Keys>,
			budgets: Option<Budgets>,
		) -> Self {
			let limits = budgets.as_ref().map(|_| Limits::default());
			Self::build_with(key, timeouts, keys, budgets, limits).await
		}

		async fn build_with(
			key: &str,
			mut timeouts: Timeouts,
			keys: Option<Keys>,
			budgets: Option<Budgets>,
			limits: Option<Limits>,
		) -> Self {
			let fake = Fake::start().await;
			let file = crate::providers::Spec::from_json(
				&format!(
					r#"{{"providers": [
						{{"name": "fake", "kind": "openai", "base_url": "{0}/v1", "api_key_env": "FAKE_KEY"}},
						{{"name": "anthropic", "kind": "anthropic", "base_url": "{0}/v1", "api_key_env": "FAKE_KEY"}},
						{{"name": "bedrock", "kind": "bedrock", "region": "us-west-2", "base_url": "{0}",
							"aws": {{"access_key_id_env": "AWS_AK", "secret_access_key_env": "AWS_SK"}}}},
						{{"name": "bedrock-key", "kind": "bedrock", "region": "us-west-2", "base_url": "{0}", "api_key_env": "FAKE_KEY"}},
						{{"name": "stored", "kind": "openai", "base_url": "{0}/v1", "credential": "stored"}},
						{{"name": "stored-aws", "kind": "bedrock", "region": "us-west-2", "base_url": "{0}", "credential": "stored-aws"}},
						{{"name": "bedrock-wrong", "kind": "bedrock", "region": "us-west-2", "base_url": "{0}",
							"aws": {{"access_key_id_env": "AWS_AK", "secret_access_key_env": "AWS_SK_WRONG"}}}},
						{{"name": "public", "kind": "openai", "base_url": "http://8.8.8.8/v1", "api_key_env": "FAKE_KEY"}},
						{{"name": "cache-a", "kind": "anthropic", "base_url": "{0}/a/v1", "api_key_env": "FAKE_KEY"}},
						{{"name": "cache-b", "kind": "anthropic", "base_url": "{0}/b/v1", "api_key_env": "FAKE_KEY"}},
						{{"name": "acct-near", "kind": "bedrock", "region": "us-west-2", "base_url": "{0}", "api_key_env": "FAKE_KEY"}},
						{{"name": "acct-far", "kind": "bedrock", "region": "us-east-1", "base_url": "{0}", "api_key_env": "FAKE_KEY"}}
					],
					"models": [
						{{"name": "ok", "provider": "fake", "upstream": "fake-ok"}},
						{{"name": "ok-priced", "provider": "fake", "upstream": "fake-ok", "price": {{"input": 1, "output": 2.5}}}},
						{{"name": "cached", "provider": "fake", "upstream": "fake-ok", "prompt_caching": true}},
						{{"name": "claude", "provider": "anthropic", "upstream": "fake-ok", "prompt_caching": true}},
						{{"name": "claude-tool", "provider": "anthropic", "upstream": "fake-tool"}},
						{{"name": "claude-search", "provider": "anthropic", "upstream": "fake-search", "internet": "web",
							"price": {{"input": 0, "output": 0}}}},
						{{"name": "claude-529", "provider": "anthropic", "upstream": "fake-529"}},
						{{"name": "claude-overloaded", "provider": "anthropic", "upstream": "fake-overloaded"}},
						{{"name": "claude-truncated", "provider": "anthropic", "upstream": "fake-truncated"}},
						{{"name": "qwen", "provider": "bedrock", "upstream": "fake.qwen:0"}},
						{{"name": "bedrock-tool", "provider": "bedrock-key", "upstream": "fake-tool", "prompt_caching": true}},
						{{"name": "luna", "provider": "bedrock", "upstream": "fake-luna"}},
						{{"name": "luna-marked-caching", "provider": "bedrock", "upstream": "fake-luna", "prompt_caching": true}},
						{{"name": "bedrock-throttle", "provider": "bedrock", "upstream": "fake-throttle"}},
						{{"name": "bedrock-unavailable", "provider": "bedrock", "upstream": "fake-unavailable"}},
						{{"name": "bedrock-stream-throttle", "provider": "bedrock", "upstream": "fake-stream-throttle"}},
						{{"name": "bedrock-truncated", "provider": "bedrock", "upstream": "fake-truncated"}},
						{{"name": "bedrock-wrong", "provider": "bedrock-wrong", "upstream": "fake-ok"}},
						{{"name": "via-store", "provider": "stored", "upstream": "fake-ok"}},
						{{"name": "via-store-aws", "provider": "stored-aws", "upstream": "fake-ok"}},
						{{"name": "slow", "provider": "fake", "upstream": "fake-slow"}},
						{{"name": "fails", "provider": "fake", "upstream": "fake-500"}},
						{{"name": "limited", "provider": "fake", "upstream": "fake-429"}},
						{{"name": "picky", "provider": "fake", "upstream": "fake-400"}},
						{{"name": "no-credit", "provider": "fake", "upstream": "fake-no-credit"}},
						{{"name": "no-credit-402", "provider": "fake", "upstream": "fake-402"}},
						{{"name": "claude-no-credit", "provider": "anthropic", "upstream": "fake-no-credit"}},
						{{"name": "hangs", "provider": "fake", "upstream": "fake-hang"}},
						{{"name": "cut", "provider": "fake", "upstream": "fake-cut"}},
						{{"name": "embed", "provider": "fake", "upstream": "fake-embed"}},
						{{"name": "titan", "provider": "bedrock", "upstream": "amazon.titan-embed-text-v2:0", "embeddings": "titan"}},
						{{"name": "cohere", "provider": "bedrock-key", "upstream": "cohere.embed-english-v3", "embeddings": "cohere"}},
						{{"name": "cohere-throttle", "provider": "bedrock", "upstream": "cohere.embed-fake-throttle", "embeddings": "cohere"}},
						{{"name": "ok-cheap", "provider": "fake", "upstream": "fake-ok", "price": {{"input": 0.1, "output": 0.2}}}},
						{{"name": "small-text", "provider": "fake", "upstream": "fake-ok", "price": {{"input": 0.05, "output": 0.1}},
							"fits": {{"context": 1000}}}},
						{{"name": "mid-images", "provider": "fake", "upstream": "fake-ok", "price": {{"input": 0.5, "output": 1}},
							"fits": {{"context": 100000, "images": true}}}},
						{{"name": "big-tools", "provider": "fake", "upstream": "fake-ok", "price": {{"input": 1, "output": 2}},
							"fits": {{"context": 200000, "tools": true, "images": true, "json_schema": true}}}},
						{{"name": "local", "provider": "fake", "upstream": "fake-ok", "internet": "none"}},
						{{"name": "none-public", "provider": "public", "upstream": "fake-ok", "internet": "none"}},
						{{"name": "none-claude", "provider": "anthropic", "upstream": "fake-ok", "internet": "none"}},
						{{"name": "web-ok", "provider": "fake", "upstream": "fake-ok", "internet": "web"}},
						{{"name": "web-claude", "provider": "anthropic", "upstream": "fake-ok", "internet": "web"}},
						{{"name": "web-bedrock", "provider": "bedrock-key", "upstream": "fake-ok", "internet": "web"}},
						{{"name": "cache-on-a", "provider": "cache-a", "upstream": "fake-cache", "prompt_caching": true,
							"price": {{"input": 3, "output": 15, "cache_read": 0.3}}}},
						{{"name": "cache-on-b", "provider": "cache-b", "upstream": "fake-cache", "prompt_caching": true,
							"price": {{"input": 3, "output": 15, "cache_read": 0.3}}}},
						{{"name": "near", "provider": "acct-near", "upstream": "fake-quota-near", "quota": {{"requests": 8}}}},
						{{"name": "far", "provider": "acct-far", "upstream": "fake-far", "quota": {{"requests": 1000}}}},
						{{"name": "flaky-a", "provider": "fake", "upstream": "fake-flaky-a"}},
						{{"name": "flaky-b", "provider": "fake", "upstream": "fake-flaky-b"}},
						{{"name": "weighted-a", "provider": "fake", "upstream": "fake-ok"}},
						{{"name": "weighted-b", "provider": "fake", "upstream": "fake-ok"}}
					],
					"routes": [
						{{"name": "web-first", "targets": ["ok", "web-ok"]}},
						{{"name": "web-bedrock-first", "targets": ["web-bedrock", "web-claude"]}},
						{{"name": "sandbox-first", "targets": ["none-public", "ok"]}},
						{{"name": "sandbox-only", "targets": ["none-public", "none-claude"]}},
						{{"name": "two-caches", "targets": ["cache-on-a", "cache-on-b"], "sort": "balanced"}},
						{{"name": "spread", "targets": ["near", "far"], "sort": "balanced"}},
						{{"name": "flaky", "targets": ["flaky-a", "flaky-b"]}},
						{{"name": "only-three-to-one", "targets": ["weighted-a", "weighted-b"], "sort": "weighted",
							"weights": {{"weighted-a": 3}}}},
						{{"name": "must-have-tools", "targets": ["ok", "big-tools"], "constraints": {{"tools": true}}}},
						{{"name": "falls-over", "targets": ["fails", "ok"]}},
						{{"name": "cheapest", "targets": ["ok-priced", "ok-cheap"], "sort": "price"}},
						{{"name": "picky-first", "targets": ["picky", "ok"]}},
						{{"name": "small-then-big", "targets": ["small-text", "big-tools"]}}
					]}}"#,
					fake.origin
				),
				&HashMap::from([
					("FAKE_KEY".to_string(), key.to_string()),
					("AWS_AK".to_string(), Fake::AWS_ACCESS_KEY_ID.to_string()),
					("AWS_SK".to_string(), Fake::AWS_SECRET.to_string()),
					("AWS_SK_WRONG".to_string(), "a-secret-the-fake-does-not-know".to_string()),
				]),
				Egress::Private,
			)
			.unwrap();
			if timeouts == Timeouts::default() {
				timeouts.idle = Duration::from_secs(5);
			}
			let (tx, calls) = tokio::sync::mpsc::unbounded_channel();
			let live = Live::new(crate::providers::Catalogue::build(&file).unwrap());
			let held = models::Held::new(file, live.clone(), Egress::Private);
			let health = Health::default();
			let proxy = Arc::new(Proxy {
				catalogue: live,
				tables: keys.as_ref().map(|_| held.clone()),
				http: net::client(Egress::Private).unwrap(),
				sandbox: net::sandbox_client(Egress::Private).unwrap(),
				egress: Egress::Private,
				jwt_secret: SECRET.into(),
				timeouts,
				max_request_bytes: 1024 * 1024,
				calls: Some(tx),
				keys,
				budgets,
				limits,
				health: health.clone(),
			});
			let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
			let base = format!("http://{}", listener.local_addr().unwrap());
			tokio::spawn(async move { axum::serve(listener, router(proxy)).await.unwrap() });
			Self {
				base,
				fake,
				tables: held,
				calls,
				client: reqwest::Client::new(),
				health,
			}
		}

		fn post(&self, path: &str, body: Value) -> reqwest::RequestBuilder {
			self.client
				.post(format!("{}{path}", self.base))
				.bearer_auth(jwt::sign(&json!({"role": "service_role"}), SECRET))
				.json(&body)
		}

		fn get(&self, path: &str) -> reqwest::RequestBuilder {
			self.client
				.get(format!("{}{path}", self.base))
				.bearer_auth(jwt::sign(&json!({"role": "service_role"}), SECRET))
		}

		fn delete(&self, path: &str) -> reqwest::RequestBuilder {
			self.client
				.delete(format!("{}{path}", self.base))
				.bearer_auth(jwt::sign(&json!({"role": "service_role"}), SECRET))
		}

		async fn call(&mut self) -> Call {
			tokio::time::timeout(Duration::from_secs(5), self.calls.recv())
				.await
				.expect("a call is recorded")
				.unwrap()
		}
	}

	fn chat(model: &str, stream: bool) -> Value {
		json!({"model": model, "stream": stream, "messages": [{"role": "user", "content": "hello there"}]})
	}

	/// The `data:` payloads of an SSE body, in order.
	fn data_of(body: &str) -> Vec<String> {
		body.split("\n\n")
			.filter_map(|e| e.strip_prefix("data: "))
			.map(String::from)
			.collect()
	}

	#[tokio::test]
	async fn a_provider_lists_its_own_models_and_never_the_key() {
		let s = Setup::new().await;
		let listed: Value = s
			.get("/ai/v1/admin/providers/fake/upstream-models")
			.send()
			.await
			.unwrap()
			.json()
			.await
			.unwrap();
		assert_eq!(listed["data"][0]["id"], "fake-new", "{listed}");
		assert_eq!(listed["data"][0]["kind"], "chat");
		let anthropic: Value = s
			.get("/ai/v1/admin/providers/anthropic/upstream-models")
			.send()
			.await
			.unwrap()
			.json()
			.await
			.unwrap();
		assert_eq!(anthropic["data"][0]["id"], "claude-fake-5", "{anthropic}");
		let none = s
			.get("/ai/v1/admin/providers/nobody/upstream-models")
			.send()
			.await
			.unwrap();
		assert_eq!(none.status(), 404);
		let user = s
			.client
			.get(format!(
				"{}/ai/v1/admin/providers/fake/upstream-models",
				s.base
			))
			.bearer_auth(jwt::sign(&json!({"role": "anon"}), SECRET))
			.send()
			.await
			.unwrap();
		assert_eq!(user.status(), 403, "the service key only");

		// A provider not saved yet, with the key it would be saved with.
		let base = format!("{}/v1", s.fake.origin);
		let unsaved = s
			.post(
				"/ai/v1/admin/upstream-models",
				json!({"provider": {"kind": "openai", "base_url": base}, "credential": {"api_key": Fake::KEY}}),
			)
			.send()
			.await
			.unwrap();
		assert_eq!(unsaved.status(), 200);
		let text = unsaved.text().await.unwrap();
		assert!(
			text.contains("fake-new") && !text.contains(Fake::KEY),
			"{text}"
		);
		let wrong = s
			.post(
				"/ai/v1/admin/upstream-models",
				json!({"provider": {"kind": "openai", "base_url": base}, "credential": {"api_key": "sk-not-the-fake-key-42"}}),
			)
			.send()
			.await
			.unwrap();
		assert_eq!(wrong.status(), 503);
		let text = wrong.text().await.unwrap();
		assert!(
			text.contains("provider_auth") && !text.contains("sk-not-the-fake-key-42"),
			"{text}"
		);
	}

	#[tokio::test]
	async fn a_whole_reply_passes_through_with_the_model_renamed() {
		let mut s = Setup::new().await;
		let response = s
			.post("/ai/v1/chat/completions", chat("ok", false))
			.send()
			.await
			.unwrap();
		assert_eq!(response.status(), 200);
		assert_eq!(response.headers()["x-psyche-provider"], "fake");
		let body: Value = response.json().await.unwrap();
		assert_eq!(
			body["choices"][0]["message"]["content"],
			"Hello from the fake."
		);
		assert_eq!(body["model"], "ok", "the reply carries the caller's name");
		let seen = s.fake.last();
		assert_eq!(seen.path, "/v1/chat/completions");
		assert_eq!(
			seen.body["model"], "fake-ok",
			"the provider gets its own name for the model"
		);
		assert_eq!(seen.body["messages"][0]["content"], "hello there");
		assert_eq!(
			seen.authorization.as_deref(),
			Some(format!("Bearer {}", Fake::KEY).as_str())
		);
		assert!(
			seen.body.get("stream_options").is_none(),
			"a whole reply is not asked for usage"
		);
		let call = s.call().await;
		assert_eq!(call.outcome, Outcome::Completed);
		assert_eq!(call.usage.unwrap().completion_tokens, 4);
		assert!(!call.streamed);
	}

	#[tokio::test]
	async fn a_stream_is_counted_without_handing_the_caller_a_chunk_it_did_not_ask_for() {
		let mut s = Setup::new().await;
		let response = s
			.post("/v1/chat/completions", chat("ok", true))
			.send()
			.await
			.unwrap();
		assert_eq!(response.status(), 200);
		assert_eq!(response.headers()["content-type"], "text/event-stream");
		let body = response.text().await.unwrap();
		let data = data_of(&body);
		assert_eq!(data.last().unwrap(), "[DONE]");
		let text: String = data
			.iter()
			.filter_map(|d| serde_json::from_str::<Value>(d).ok())
			.filter_map(|v| {
				v.pointer("/choices/0/delta/content")
					.and_then(Value::as_str)
					.map(String::from)
			})
			.collect();
		assert_eq!(text, "Hello from the fake.");
		assert!(
			!body.contains("fake-ok"),
			"every chunk carries the caller's name: {body}"
		);
		assert!(!body.contains("\"usage\""), "{body}");
		assert_eq!(s.fake.last().body["stream_options"]["include_usage"], true);
		let call = s.call().await;
		assert_eq!(call.outcome, Outcome::Completed);
		assert_eq!(call.usage.unwrap().completion_tokens, 4);
	}

	#[tokio::test]
	async fn a_call_is_costed_at_the_providers_file_price_and_an_unpriced_one_is_not() {
		let mut s = Setup::new().await;
		let response = s
			.post("/ai/v1/chat/completions", chat("ok-priced", true))
			.send()
			.await
			.unwrap();
		response.text().await.unwrap();
		let call = s.call().await;
		let usage = call.usage.unwrap();
		assert!(usage.prompt_tokens > 0 && usage.completion_tokens > 0);
		assert_eq!(
			call.cost().unwrap().0,
			usage.prompt_tokens * 1_000_000 + usage.completion_tokens * 2_500_000
		);

		s.post("/ai/v1/chat/completions", chat("ok", false))
			.send()
			.await
			.unwrap()
			.text()
			.await
			.unwrap();
		let call = s.call().await;
		assert!(call.usage.is_some());
		assert_eq!(call.cost(), None, "nothing prices fake-ok");
	}

	#[tokio::test]
	async fn a_key_is_made_used_within_its_models_and_revoked() {
		let Some((db, url)) = crate::log::tests::fresh_database("keys").await else {
			return;
		};
		let store = keys::Store::default();
		let (ready, loaded) = tokio::sync::oneshot::channel();
		tokio::spawn(keys::watch(
			url.clone(),
			keys::Held {
				keys: store.clone(),
				members: Members::default(),
				budgets: None,
				limits: None,
				catalogue: None,
			},
			Some(ready),
		));
		loaded.await.unwrap();
		let mut s = Setup::build(
			Fake::KEY,
			Timeouts::default(),
			Some(Keys {
				store,
				members: Members::default(),
				database_url: url,
			}),
			None,
		)
		.await;

		let made = s
			.post("/ai/v1/admin/keys", json!({"name": "ci", "models": ["ok"]}))
			.send()
			.await
			.unwrap();
		assert_eq!(made.status(), 201);
		let made: Value = made.json().await.unwrap();
		let key = made["key"].as_str().unwrap().to_string();
		let id = made["id"].as_str().unwrap().to_string();
		assert!(key.starts_with("psy_"));
		let stored: Vec<u8> = db
			.query_one("SELECT hash FROM psyche.keys", &[])
			.await
			.unwrap()
			.get(0);
		assert_eq!(stored, keys::hash(&key), "only the hash is kept");

		let (client, base) = (s.client.clone(), s.base.clone());
		let with_key = |path: &str, body: Value| {
			client
				.post(format!("{base}{path}"))
				.bearer_auth(&key)
				.json(&body)
		};
		let ok = with_key("/ai/v1/chat/completions", chat("ok", false))
			.send()
			.await
			.unwrap();
		assert_eq!(ok.status(), 200);
		let refused = with_key("/ai/v1/chat/completions", chat("claude", false))
			.send()
			.await
			.unwrap();
		assert_eq!(refused.status(), 403);
		assert!(refused.text().await.unwrap().contains("may not use claude"));
		let call = s.call().await;
		assert_eq!(call.who.key.map(|k| k.to_string()), Some(id.clone()));

		let listed: Value = s
			.client
			.get(format!("{}/ai/v1/models", s.base))
			.bearer_auth(&key)
			.send()
			.await
			.unwrap()
			.json()
			.await
			.unwrap();
		assert_eq!(listed["data"].as_array().unwrap().len(), 1, "{listed}");
		assert_eq!(listed["data"][0]["id"], "ok");

		let admin_with_key = s
			.client
			.get(format!("{}/ai/v1/admin/keys", s.base))
			.bearer_auth(&key)
			.send()
			.await
			.unwrap();
		assert_eq!(admin_with_key.status(), 403, "a key cannot manage keys");
		let unknown_model = s
			.post(
				"/ai/v1/admin/keys",
				json!({"name": "x", "models": ["nope"]}),
			)
			.send()
			.await
			.unwrap();
		assert_eq!(unknown_model.status(), 400);

		// Revoked in SQL, by anyone: the notification reaches Psyche and the key stops.
		db.execute("UPDATE psyche.keys SET revoked_at = now()", &[])
			.await
			.unwrap();
		let mut status = 200;
		for _ in 0..50 {
			status = with_key("/ai/v1/chat/completions", chat("ok", false))
				.send()
				.await
				.unwrap()
				.status()
				.as_u16();
			if status == 401 {
				break;
			}
			tokio::time::sleep(Duration::from_millis(20)).await;
		}
		assert_eq!(
			status, 401,
			"a revoke reaches Psyche without waiting for the reload"
		);

		let listed: Value = s
			.client
			.get(format!("{}/ai/v1/admin/keys", s.base))
			.bearer_auth(jwt::sign(&json!({"role": "service_role"}), SECRET))
			.send()
			.await
			.unwrap()
			.json()
			.await
			.unwrap();
		let row = &listed["data"][0];
		assert_eq!(row["id"], id.as_str());
		assert!(row["revoked_at"].is_u64());
		assert!(
			!listed.to_string().contains(&key),
			"a list never shows a key"
		);
		let again = s
			.client
			.delete(format!("{}/ai/v1/admin/keys/{id}", s.base))
			.bearer_auth(jwt::sign(&json!({"role": "service_role"}), SECRET))
			.send()
			.await
			.unwrap();
		assert_eq!(again.status(), 404, "already revoked");
	}

	/// Psyche with keys and budgets, on a fresh database, and a key of its own.
	async fn with_budgets(name: &str) -> Option<(Setup, tokio_postgres::Client, String, String)> {
		let (mut db, url) = crate::log::tests::fresh_database(name).await?;
		let store = keys::Store::default();
		let members = Members::default();
		let budgets = Budgets::default();
		let limits = Limits::default();
		let s = Setup::build_with(
			Fake::KEY,
			Timeouts::default(),
			Some(Keys {
				store: store.clone(),
				members: members.clone(),
				database_url: url.clone(),
			}),
			Some(budgets.clone()),
			Some(limits.clone()),
		)
		.await;
		// As at start: the file written into the tables, which are then what is served.
		assert!(
			models::seed(&mut db, &s.tables.file)
				.await
				.unwrap()
				.is_empty()
		);
		let (ready, loaded) = tokio::sync::oneshot::channel();
		tokio::spawn(keys::watch(
			url,
			keys::Held {
				keys: store,
				budgets: Some(budgets),
				limits: Some(limits),
				members,
				catalogue: Some(s.tables.clone()),
			},
			Some(ready),
		));
		loaded.await.unwrap();
		let made: Value = s
			.post("/ai/v1/admin/keys", json!({"name": "app"}))
			.send()
			.await
			.unwrap()
			.json()
			.await
			.unwrap();
		let key = made["key"].as_str().unwrap().to_string();
		let id = made["id"].as_str().unwrap().to_string();
		Some((s, db, key, id))
	}

	#[tokio::test]
	async fn the_admin_api_edits_a_budget_and_reads_usage_and_the_log() {
		let Some((s, db, key, id)) = with_budgets("admin_reads").await else {
			return;
		};
		let service = jwt::sign(&json!({"role": "service_role"}), SECRET);
		let send = |method: reqwest::Method, path: &str, body: Option<Value>| {
			let mut request = s
				.client
				.request(method, format!("{}{path}", s.base))
				.bearer_auth(&service);
			if let Some(body) = body {
				request = request.json(&body);
			}
			request.send()
		};

		// A budget from a billing period's first day, made and listed with it, then edited in place.
		let made: Value = send(
			reqwest::Method::POST,
			"/ai/v1/admin/budgets",
			Some(
				json!({"name": "team", "amount": 10, "period": "since", "starts_on": "2026-10-01"}),
			),
		)
		.await
		.unwrap()
		.json()
		.await
		.unwrap();
		assert_eq!(made["starts_on"], "2026-10-01");
		let budget = made["id"].as_str().unwrap().to_string();
		let path = format!("/ai/v1/admin/budgets/{budget}");
		let edited: Value = send(
			reqwest::Method::PATCH,
			&path,
			Some(json!({"amount": 25, "hard": false})),
		)
		.await
		.unwrap()
		.json()
		.await
		.unwrap();
		assert_eq!(edited["amount"], "25.000000");
		assert_eq!(edited["hard"], false);
		assert_eq!(
			edited["starts_on"], "2026-10-01",
			"kept when the period is not changed"
		);
		let moved: Value = send(
			reqwest::Method::PATCH,
			&path,
			Some(json!({"period": "rolling", "days": 30})),
		)
		.await
		.unwrap()
		.json()
		.await
		.unwrap();
		assert_eq!(moved["days"], 30);
		assert_eq!(
			moved["starts_on"],
			Value::Null,
			"a new period takes only what it is given"
		);
		let listed: Value = send(reqwest::Method::GET, "/ai/v1/admin/budgets", None)
			.await
			.unwrap()
			.json()
			.await
			.unwrap();
		let row = listed["data"]
			.as_array()
			.unwrap()
			.iter()
			.find(|b| b["id"] == budget.as_str())
			.unwrap();
		assert_eq!(
			(row["period"].as_str(), row["days"].as_i64()),
			(Some("rolling"), Some(30))
		);
		for (body, status) in [
			(json!({"period": "since"}), 400),
			(json!({"amount": -1}), 400),
			(json!({"user_id": id}), 400),
		] {
			assert_eq!(
				send(reqwest::Method::PATCH, &path, Some(body.clone()))
					.await
					.unwrap()
					.status(),
				status,
				"{body}"
			);
		}
		assert_eq!(
			send(
				reqwest::Method::PATCH,
				"/ai/v1/admin/budgets/00000000-0000-0000-0000-000000000001",
				Some(json!({"amount": 1}))
			)
			.await
			.unwrap()
			.status(),
			404
		);

		// Usage: the daily totals summed and grouped, and filtered to one key.
		db.execute(
			"INSERT INTO psyche.usage_daily (day, operation, model, provider, requests, prompt_tokens, \
			 completion_tokens, cost, key_id) VALUES \
			 (current_date, 'chat', 'ok-priced', 'fake', 3, 30, 12, 0.003, $1::text::uuid), \
			 (current_date - 1, 'chat', 'ok-priced', 'fake', 1, 10, 4, 0.001, NULL), \
			 (current_date - 1, 'embeddings', 'embed', 'fake', 5, 50, 0, 0.0005, NULL), \
			 (current_date - 40, 'chat', 'ok-priced', 'fake', 9, 90, 9, 0.009, NULL)",
			&[&id],
		)
		.await
		.unwrap();
		let usage = |query: &str| {
			let query = query.to_string();
			let request = send(
				reqwest::Method::GET,
				&format!("/ai/v1/admin/usage{query}"),
				None,
			);
			async move { request.await.unwrap() }
		};
		let by_model: Value = usage("?by=model").await.json().await.unwrap();
		let models = by_model["data"].as_array().unwrap();
		assert_eq!(models.len(), 2, "{by_model}");
		assert_eq!(models[0]["model"], "ok-priced", "the most spent first");
		assert_eq!(
			models[0]["requests"], 4,
			"the day forty back is outside the last thirty"
		);
		assert_eq!(models[0]["cost"], "0.004");
		let by_day: Value = usage("?by=day").await.json().await.unwrap();
		assert_eq!(by_day["data"].as_array().unwrap().len(), 2);
		let mine: Value = usage(&format!("?by=key&key_id={id}"))
			.await
			.json()
			.await
			.unwrap();
		assert_eq!(mine["data"][0]["key"], id.as_str());
		assert_eq!(mine["data"][0]["requests"], 3);
		let wide: Value = usage("?by=model&from=2000-01-01&to=2000-12-31")
			.await
			.json()
			.await
			.unwrap();
		assert_eq!(wide["data"].as_array().unwrap().len(), 0);
		for bad in [
			"?by=colour",
			"?from=yesterday",
			"?from=2020-01-01&to=2026-01-01",
			"?key_id=nope",
		] {
			assert_eq!(usage(bad).await.status(), 400, "{bad}");
		}

		// The log, newest first and paged by `next_before`.
		db.batch_execute(
			"CREATE TABLE IF NOT EXISTS psyche.requests_admin_reads PARTITION OF psyche.requests DEFAULT; \
			 INSERT INTO psyche.requests (at, operation, model, provider, streamed, outcome, first_byte_ms, \
			 total_ms, cost) VALUES \
			 (now() - interval '3 seconds', 'chat', 'ok-priced', 'fake', true, 'completed', 5, 9, 0.001), \
			 (now() - interval '2 seconds', 'chat', 'ok-priced', 'fake', false, 'budget_exceeded', 0, 0, NULL), \
			 (now() - interval '1 second', 'embeddings', 'embed', 'fake', false, 'completed', 3, 3, 0.0001)",
		)
		.await
		.unwrap();
		let page: Value = send(reqwest::Method::GET, "/ai/v1/admin/requests?limit=2", None)
			.await
			.unwrap()
			.json()
			.await
			.unwrap();
		let rows = page["data"].as_array().unwrap();
		assert_eq!((rows.len(), rows[0]["model"].as_str()), (2, Some("embed")));
		let next = page["next_before"].as_i64().unwrap();
		let rest: Value = send(
			reqwest::Method::GET,
			&format!("/ai/v1/admin/requests?limit=2&before={next}"),
			None,
		)
		.await
		.unwrap()
		.json()
		.await
		.unwrap();
		assert_eq!(rest["data"].as_array().unwrap().len(), 1);
		assert_eq!(rest["data"][0]["outcome"], "completed");
		assert_eq!(rest["next_before"], Value::Null);
		let refused: Value = send(
			reqwest::Method::GET,
			"/ai/v1/admin/requests?outcome=budget_exceeded",
			None,
		)
		.await
		.unwrap()
		.json()
		.await
		.unwrap();
		assert_eq!(refused["data"].as_array().unwrap().len(), 1);

		// None of it is a caller's: a key is refused.
		assert_eq!(
			s.client
				.get(format!("{}/ai/v1/admin/usage", s.base))
				.bearer_auth(&key)
				.send()
				.await
				.unwrap()
				.status(),
			403
		);
	}

	#[tokio::test]
	async fn budgets_hold_under_concurrency_and_a_long_reply_is_refused_before_it_is_sent() {
		let Some((mut s, _db, key, id)) = with_budgets("budget_race").await else {
			return;
		};
		// ok-priced: $1 in and $2.50 out per million. A call below reserves its body's bytes as
		// prompt tokens and 8 out, about $0.0001; it costs 9 in and 4 out, $0.000019. So $0.001
		// fits about fifty calls, and only a few can be in flight at once.
		let made = s
			.post(
				"/ai/v1/admin/budgets",
				json!({"name": "app daily", "amount": 0.001, "period": "day", "key_id": id}),
			)
			.send()
			.await
			.unwrap();
		assert_eq!(made.status(), 201, "{}", made.text().await.unwrap());
		let limit = crate::price::PICO / 1000;

		let request = json!({"model": "ok-priced", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]});
		let sends = (0..300).map(|_| {
			s.client
				.post(format!("{}/ai/v1/chat/completions", s.base))
				.bearer_auth(&key)
				.json(&request)
				.send()
		});
		let replies = futures_util::future::join_all(sends).await;
		let mut ok = 0;
		let mut refused = 0;
		for reply in replies {
			let reply = reply.unwrap();
			match reply.status().as_u16() {
				200 => ok += 1,
				429 => {
					let body: Value = reply.json().await.unwrap();
					assert_eq!(body["error"]["code"], "budget_exceeded");
					// Whose budget refused, so a client can tell its own from the project's.
					assert_eq!(body["error"]["scope"], "key");
					assert!(
						body["error"]["message"]
							.as_str()
							.unwrap()
							.contains("app daily")
					);
					refused += 1;
				}
				other => panic!("{other}"),
			}
		}
		assert_eq!(ok + refused, 300);
		assert!(ok > 10 && refused > 0, "{ok} answered, {refused} refused");
		let mut spent = 0;
		for _ in 0..ok {
			spent += s.call().await.cost().unwrap().0;
		}
		assert!(spent <= limit, "spent {spent} of {limit} picodollars");
		eprintln!(
			"300 at once: {ok} answered, {refused} refused, {spent} of {limit} picodollars spent"
		);
		assert_eq!(
			s.fake.count(),
			ok,
			"a refused call never reached the provider"
		);

		// A reply that could be longer than the whole budget is refused before it is sent.
		let sent = s.fake.count();
		let long = s
			.client
			.post(format!("{}/ai/v1/chat/completions", s.base))
			.bearer_auth(&key)
			.json(&json!({"model": "ok-priced", "max_tokens": 1_000_000, "messages": [{"role": "user", "content": "hi"}]}))
			.send()
			.await
			.unwrap();
		assert_eq!(long.status(), 429);
		assert!(long.text().await.unwrap().contains("could cost up to $2.5"));
		assert_eq!(s.fake.count(), sent);

		// The service key is not held by the key's budget; an unpriced model cannot be budgeted.
		assert_eq!(
			s.post("/ai/v1/chat/completions", chat("ok-priced", false))
				.send()
				.await
				.unwrap()
				.status(),
			200
		);
		let unpriced = s
			.client
			.post(format!("{}/ai/v1/chat/completions", s.base))
			.bearer_auth(&key)
			.json(&chat("ok", false))
			.send()
			.await
			.unwrap();
		assert_eq!(unpriced.status(), 429);
		assert!(unpriced.text().await.unwrap().contains("has no price"));
	}

	#[tokio::test]
	async fn an_uncapped_reply_under_a_hard_budget_is_sent_a_cap_that_fits() {
		let Some((s, _db, key, id)) = with_budgets("budget_cap").await else {
			return;
		};
		let call = |key: String, body: Value| {
			let s = &s;
			async move {
				let reply = s
					.client
					.post(format!("{}/ai/v1/chat/completions", s.base))
					.bearer_auth(key)
					.json(&body)
					.send()
					.await
					.unwrap();
				let status = reply.status().as_u16();
				reply.text().await.unwrap();
				(status, s.fake.last().body)
			}
		};
		let budget = |name: &str, amount: f64, hard: bool, key_id: &str| {
			s.post(
				"/ai/v1/admin/budgets",
				json!({"name": name, "amount": amount, "period": "day", "hard": hard, "key_id": key_id}),
			)
			.send()
		};

		// No budget: nothing is added.
		let (status, seen) = call(key.clone(), chat("ok-priced", false)).await;
		assert_eq!(status, 200);
		assert!(seen.get("max_tokens").is_none());

		// A soft budget never refuses, so it adds nothing either.
		assert_eq!(
			budget("soft", 0.001, false, &id).await.unwrap().status(),
			201
		);
		let (_, seen) = call(key.clone(), chat("ok-priced", false)).await;
		assert!(seen.get("max_tokens").is_none());

		// A roomy hard budget: the default cap, the one Anthropic is always sent.
		assert_eq!(
			budget("roomy", 10.0, true, &id).await.unwrap().status(),
			201
		);
		let (status, seen) = call(key.clone(), chat("ok-priced", false)).await;
		assert_eq!(status, 200);
		assert_eq!(seen["max_tokens"], budgets::UNCAPPED_OUTPUT);
		assert!(seen.get("max_completion_tokens").is_none());
		// The caller's own cap is left alone.
		let mut capped = chat("ok-priced", false);
		capped["max_tokens"] = json!(12);
		let (_, seen) = call(key.clone(), capped).await;
		assert_eq!(seen["max_tokens"], 12);

		// A tight one: $0.001 at $1 in and $2.50 out per million leaves room for a few hundred
		// tokens after the prompt's bound, so the cap shrinks to fit and the call still runs.
		let made: Value = s
			.post("/ai/v1/admin/keys", json!({"name": "tight"}))
			.send()
			.await
			.unwrap()
			.json()
			.await
			.unwrap();
		let tight = made["key"].as_str().unwrap().to_string();
		let tight_id = made["id"].as_str().unwrap();
		assert_eq!(
			budget("tight", 0.001, true, tight_id)
				.await
				.unwrap()
				.status(),
			201
		);
		let (status, seen) = call(tight.clone(), chat("ok-priced", false)).await;
		assert_eq!(status, 200);
		let cap = seen["max_tokens"].as_u64().unwrap();
		assert!(
			(budgets::MIN_SENT_OUTPUT..budgets::UNCAPPED_OUTPUT).contains(&cap),
			"{cap}"
		);

		// Under the shortest cap worth sending, the call is refused and never sent.
		let sent = s.fake.count();
		let mut long = chat("ok-priced", false);
		long["messages"][0]["content"] = json!("x".repeat(400_000));
		let (status, _) = call(tight, long).await;
		assert_eq!(status, 429);
		assert_eq!(s.fake.count(), sent);
	}

	/// A call by a signed-in user, with their own token.
	async fn as_user(s: &Setup, sub: &str, body: &Value) -> reqwest::Response {
		let token = jwt::sign(
			&json!({"role": "authenticated", "sub": sub, "exp": now() + 3600}),
			SECRET,
		);
		s.client
			.post(format!("{}/ai/v1/chat/completions", s.base))
			.bearer_auth(token)
			.json(body)
			.send()
			.await
			.unwrap()
	}

	fn add(
		s: &Setup,
		path: &str,
		row: Value,
	) -> impl std::future::Future<Output = reqwest::Result<reqwest::Response>> + use<> {
		s.post(path, row).send()
	}

	async fn error_of(reply: reqwest::Response) -> (u16, String, String) {
		let status = reply.status().as_u16();
		let body: Value = reply.json().await.unwrap();
		(
			status,
			body["error"]["code"].as_str().unwrap_or("").to_string(),
			body["error"]["message"].as_str().unwrap_or("").to_string(),
		)
	}

	/// A change made in SQL reaches Psyche through the trigger: calls as `sub` until the answer is
	/// `status`.
	async fn until(s: &Setup, sub: &str, body: &Value, status: u16) -> reqwest::Response {
		for _ in 0..250 {
			let reply = as_user(s, sub, body).await;
			if reply.status() == status {
				return reply;
			}
			tokio::time::sleep(Duration::from_millis(20)).await;
		}
		panic!("{sub} never got {status}");
	}

	#[tokio::test]
	async fn an_account_is_held_to_its_grants_and_pays_with_credits_once_its_budget_is_spent() {
		let Some((mut s, db, _key, _id)) = with_budgets("accounts").await else {
			return;
		};
		let alice = "00000000-0000-4000-8000-00000000a11c";
		let body = json!({"model": "ok-priced", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]});
		for budget in [
			json!({"name": "project", "amount": 10, "period": "month"}),
			json!({"name": "each user", "amount": 0.0002, "period": "day", "every_user": true}),
		] {
			assert_eq!(
				add(&s, "/ai/v1/admin/budgets", budget)
					.await
					.unwrap()
					.status(),
				201
			);
		}
		as_user(&s, alice, &body).await.text().await.unwrap();

		// Granted only: refused, and not listed, until she is granted it.
		db.batch_execute("UPDATE psyche.models SET access = 'granted' WHERE name = 'ok-priced'")
			.await
			.unwrap();
		let (status, _, message) = error_of(until(&s, alice, &body, 403).await).await;
		assert_eq!(status, 403);
		assert!(message.contains("not open to this account"), "{message}");
		let token = jwt::sign(
			&json!({"role": "authenticated", "sub": alice, "exp": now() + 3600}),
			SECRET,
		);
		let listed = || async {
			let models: Value = s
				.client
				.get(format!("{}/ai/v1/models", s.base))
				.bearer_auth(&token)
				.send()
				.await
				.unwrap()
				.json()
				.await
				.unwrap();
			models.to_string().contains("\"ok-priced\"")
		};
		assert!(!listed().await);
		// A grant names one user or one group, and a model that exists.
		for (bad, status) in [
			(json!({"model": "ok-priced"}), 400),
			(
				json!({"model": "ok-priced", "user_id": alice, "group_id": alice}),
				400,
			),
			(json!({"model": "ok-priced", "user_id": "alice"}), 400),
			(json!({"model": "nothing", "user_id": alice}), 404),
			(json!({"model": "ok-priced", "group_id": alice}), 404),
		] {
			let reply = add(&s, "/ai/v1/admin/grants", bad.clone()).await.unwrap();
			assert_eq!(reply.status(), status, "{bad}");
		}
		let reply = add(
			&s,
			"/ai/v1/admin/grants",
			json!({"model": "ok-priced", "user_id": alice}),
		)
		.await
		.unwrap();
		assert_eq!(reply.status(), 201);
		let grant: Value = reply.json().await.unwrap();
		assert_eq!(grant["model"], "ok-priced");
		assert_eq!(grant["user_id"], alice);
		// Granting it again answers the same grant.
		let again = add(
			&s,
			"/ai/v1/admin/grants",
			json!({"model": "ok-priced", "user_id": alice}),
		)
		.await
		.unwrap();
		assert_eq!(again.status(), 200);
		assert_eq!(again.json::<Value>().await.unwrap()["id"], grant["id"]);
		until(&s, alice, &body, 200).await.text().await.unwrap();
		assert!(listed().await);
		let grants: Value = s
			.get("/ai/v1/admin/grants?model=ok-priced")
			.send()
			.await
			.unwrap()
			.json()
			.await
			.unwrap();
		assert_eq!(grants["data"].as_array().unwrap().len(), 1);
		let models: Value = s
			.get("/ai/v1/admin/models")
			.send()
			.await
			.unwrap()
			.json()
			.await
			.unwrap();
		let held = models["data"]
			.as_array()
			.unwrap()
			.iter()
			.find(|m| m["name"] == "ok-priced")
			.unwrap();
		assert_eq!(held["grants"], 1);
		// Taken back, she is refused again; then granted once more for what follows.
		let id = grant["id"].as_str().unwrap();
		assert_eq!(
			s.delete(&format!("/ai/v1/admin/grants/{id}"))
				.send()
				.await
				.unwrap()
				.status(),
			204
		);
		assert_eq!(
			s.delete(&format!("/ai/v1/admin/grants/{id}"))
				.send()
				.await
				.unwrap()
				.status(),
			404
		);
		until(&s, alice, &body, 403).await.text().await.unwrap();
		assert_eq!(
			add(
				&s,
				"/ai/v1/admin/grants",
				json!({"model": "ok-priced", "user_id": alice})
			)
			.await
			.unwrap()
			.status(),
			201
		);
		until(&s, alice, &body, 200).await.text().await.unwrap();
		assert!(listed().await);
		// The list says what each model costs.
		let models: Value = s
			.client
			.get(format!("{}/ai/v1/models", s.base))
			.bearer_auth(&token)
			.send()
			.await
			.unwrap()
			.json()
			.await
			.unwrap();
		let priced = models["data"]
			.as_array()
			.unwrap()
			.iter()
			.find(|m| m["id"] == "ok-priced")
			.unwrap();
		assert_eq!(priced["pricing"]["input"], 1.0);
		assert_eq!(priced["pricing"]["output"], 2.5);

		// Her own budget spent, she is refused until she has credits, which then pay.
		let reply = until(&s, alice, &body, 429).await;
		assert_eq!(error_of(reply).await.1, "budget_exceeded");
		db.execute(
			"INSERT INTO psyche.credits (user_id, granted) VALUES ($1::text::uuid, 0.001)",
			&[&alice],
		)
		.await
		.unwrap();
		until(&s, alice, &body, 200).await.text().await.unwrap();
		let mut paid = false;
		while let Ok(call) = s.calls.try_recv() {
			paid |= call.credits;
		}
		if !paid {
			paid = s.call().await.credits;
		}
		assert!(paid, "the last call was paid by credits");
		// Switched off, they pay nothing.
		db.batch_execute("UPDATE psyche.credits SET enabled = false")
			.await
			.unwrap();
		assert_eq!(
			error_of(until(&s, alice, &body, 429).await).await.1,
			"budget_exceeded"
		);

		// After a restart, what credits paid is read back: off her budget, and off her balance.
		db.batch_execute("UPDATE psyche.credits SET enabled = true")
			.await
			.unwrap();
		db.execute(
			"INSERT INTO psyche.usage_daily (day, operation, model, provider, requests, cost, \
			 credits_cost, user_id) VALUES (current_date - 1, 'chat', 'ok-priced', 'fake', 1, \
			 0.0004, 0.0003, $1::text::uuid)",
			&[&alice],
		)
		.await
		.unwrap();
		let again = Budgets::default();
		again.replace_credits(crate::budgets::load_credits(&db).await.unwrap());
		crate::budgets::load_spent(&db, &again, SystemTime::now())
			.await
			.unwrap();
		let user = KeyId::parse(alice).unwrap();
		assert_eq!(
			again.credits_left(user),
			Some(700_000_000),
			"$0.001 less $0.0003"
		);

		// A provider behind a gate takes its stored key in the gate's header, proof included.
		let service = || jwt::sign(&json!({"role": "service_role"}), SECRET);
		let put = |path: String, body: Value| {
			s.client
				.put(format!("{}/ai/v1/admin/{path}", s.base))
				.bearer_auth(service())
				.json(&body)
				.send()
		};
		let made = put(
			"providers/gated".into(),
			json!({"kind": "openai", "base_url": format!("{}/v1", s.fake.origin),
				"credential": "gated", "auth_header": Fake::GATE_HEADER}),
		);
		assert_eq!(made.await.unwrap().status(), 200);
		let made = put(
			"models/via-gate".into(),
			json!({"provider": "gated", "upstream": "fake-ok", "price": {"input": 1, "output": 1}}),
		);
		assert_eq!(made.await.unwrap().status(), 200);
		let made = put("credentials/gated".into(), json!({"api_key": Fake::KEY}));
		assert_eq!(made.await.unwrap().status(), 200, "proven through the gate");
		let reply = s
			.post("/ai/v1/chat/completions", chat("via-gate", false))
			.send()
			.await
			.unwrap();
		assert_eq!(reply.status(), 200);
		let seen = s.fake.last();
		assert_eq!(seen.gate.as_deref(), Some(Fake::KEY));
		assert_eq!(seen.authorization, None, "no Bearer token beside it");
		let bad = put(
			"providers/bad".into(),
			json!({"kind": "openai", "base_url": format!("{}/v1", s.fake.origin), "auth_header": "Host"}),
		);
		assert_eq!(bad.await.unwrap().status(), 400);
	}

	#[tokio::test]
	async fn signed_in_users_are_held_to_their_own_budget_their_group_s_and_their_limit() {
		let Some((mut s, _db, _key, _id)) = with_budgets("users").await else {
			return;
		};
		let (alice, bob, carol) = (
			"00000000-0000-4000-8000-00000000a11c",
			"00000000-0000-4000-8000-000000000b0b",
			"00000000-0000-4000-8000-0000000ca201",
		);
		let body = json!({"model": "ok-priced", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]});

		// Refused until the project has a hard budget of its own and one for each user (PS10b).
		let (status, _, message) = error_of(as_user(&s, alice, &body).await).await;
		assert_eq!(status, 403);
		assert!(message.contains("of its own"), "{message}");
		let made = add(
			&s,
			"/ai/v1/admin/budgets",
			json!({"name": "project", "amount": 10, "period": "month"}),
		);
		assert_eq!(made.await.unwrap().status(), 201);
		let (status, _, message) = error_of(as_user(&s, alice, &body).await).await;
		assert_eq!(status, 403);
		assert!(message.contains("every_user"), "{message}");
		let unsigned = jwt::sign(
			&json!({"role": "authenticated", "exp": now() + 3600}),
			SECRET,
		);
		let no_sub = s
			.client
			.post(format!("{}/ai/v1/chat/completions", s.base))
			.bearer_auth(unsigned)
			.json(&body)
			.send()
			.await
			.unwrap();
		assert_eq!(no_sub.status(), 401, "a user token names its user");

		// $0.0002 a day each: a call reserves about $0.00011 and costs $0.000019, so a few fit.
		let made = add(
			&s,
			"/ai/v1/admin/budgets",
			json!({"name": "each user", "amount": 0.0002, "period": "day", "every_user": true}),
		);
		assert_eq!(made.await.unwrap().status(), 201);
		let mut answered = 0;
		loop {
			let reply = as_user(&s, alice, &body).await;
			if reply.status() == 200 {
				reply.text().await.unwrap();
				answered += 1;
				assert!(answered < 20, "the budget holds");
				continue;
			}
			let (status, code, message) = error_of(reply).await;
			assert_eq!((status, code.as_str()), (429, "budget_exceeded"));
			assert!(message.contains("each user"), "{message}");
			break;
		}
		assert!(answered >= 2, "{answered}");
		let first = s.call().await;
		assert_eq!(
			first.who.user.map(|u| u.to_string()).as_deref(),
			Some(alice)
		);
		assert_eq!((first.who.key, first.who.group), (None, None));
		for _ in 1..answered {
			s.call().await;
		}
		// Bob's dollar is his own.
		let reply = as_user(&s, bob, &body).await;
		assert_eq!(reply.status(), 200);
		reply.text().await.unwrap();
		s.call().await;

		// A budget of Alice's own replaces the default.
		let made = add(
			&s,
			"/ai/v1/admin/budgets",
			json!({"name": "alice", "amount": 1, "period": "day", "user_id": alice}),
		);
		assert_eq!(made.await.unwrap().status(), 201);
		let reply = as_user(&s, alice, &body).await;
		assert_eq!(reply.status(), 200);
		reply.text().await.unwrap();
		s.call().await;

		// A group spends from one budget: too little for Bob's next call.
		let group: Value = add(&s, "/ai/v1/admin/groups", json!({"name": "team"}))
			.await
			.unwrap()
			.json()
			.await
			.unwrap();
		let group = group["id"].as_str().unwrap();
		let joined = s
			.client
			.put(format!("{}/ai/v1/admin/members/{bob}", s.base))
			.bearer_auth(jwt::sign(&json!({"role": "service_role"}), SECRET))
			.json(&json!({"group_id": group}))
			.send()
			.await
			.unwrap();
		assert_eq!(joined.status(), 200);
		let made = add(
			&s,
			"/ai/v1/admin/budgets",
			json!({"name": "team", "amount": 0.00005, "period": "day", "group_id": group}),
		);
		assert_eq!(made.await.unwrap().status(), 201);
		let (status, _, message) = error_of(as_user(&s, bob, &body).await).await;
		assert_eq!(status, 429);
		assert!(message.contains("team"), "{message}");
		let both = add(
			&s,
			"/ai/v1/admin/budgets",
			json!({"name": "both", "amount": 1, "period": "day", "user_id": bob, "group_id": group}),
		);
		assert_eq!(both.await.unwrap().status(), 400, "one subject a row");
		let unknown = add(
			&s,
			"/ai/v1/admin/budgets",
			json!({"name": "typo", "amount": 1, "period": "day", "evry_user": true}),
		);
		let (status, _, message) = error_of(unknown.await.unwrap()).await;
		assert_eq!(status, 400);
		assert!(message.contains("not evry_user"), "{message}");

		// A limit for every user: one call a minute each.
		let made = add(
			&s,
			"/ai/v1/admin/limits",
			json!({"name": "each user a minute", "every_user": true, "requests_per_minute": 1}),
		);
		assert_eq!(made.await.unwrap().status(), 201);
		let reply = as_user(&s, carol, &body).await;
		assert_eq!(reply.status(), 200);
		assert_eq!(reply.headers()["x-ratelimit-remaining-requests"], "0");
		reply.text().await.unwrap();
		let (status, code, _) = error_of(as_user(&s, carol, &body).await).await;
		assert_eq!((status, code.as_str()), (429, "rate_limit_exceeded"));
		let reply = as_user(&s, alice, &body).await;
		assert_eq!(reply.status(), 200, "Alice's minute is her own");
		reply.text().await.unwrap();

		// A key Carol owns counts against Carol's limit too.
		let made: Value = add(
			&s,
			"/ai/v1/admin/keys",
			json!({"name": "carol's", "owner": carol}),
		)
		.await
		.unwrap()
		.json()
		.await
		.unwrap();
		let carols = s
			.client
			.post(format!("{}/ai/v1/chat/completions", s.base))
			.bearer_auth(made["key"].as_str().unwrap())
			.json(&body)
			.send()
			.await
			.unwrap();
		let (status, _, message) = error_of(carols).await;
		assert_eq!(status, 429);
		assert!(message.contains("each user a minute"), "{message}");
	}

	#[tokio::test]
	async fn a_stored_credential_is_proven_never_shown_and_used_at_once() {
		let Some((s, db, _key, _id)) = with_budgets("credentials").await else {
			return;
		};
		let service = || jwt::sign(&json!({"role": "service_role"}), SECRET);
		let put = |name: &str, body: Value| {
			s.client
				.put(format!("{}/ai/v1/admin/credentials/{name}", s.base))
				.bearer_auth(service())
				.json(&body)
				.send()
		};
		let call = |model: &str| s.post("/ai/v1/chat/completions", chat(model, false)).send();
		let stored = || async {
			let n: i64 = db
				.query_one("SELECT count(*) FROM psyche.credentials", &[])
				.await
				.unwrap()
				.get(0);
			n
		};

		// Not set yet: refused with what to do, and nothing is sent.
		let sent = s.fake.count();
		let (status, code, message) = error_of(call("via-store").await.unwrap()).await;
		assert_eq!((status, code.as_str()), (503, "provider_auth"));
		assert!(
			message.contains("PUT /ai/v1/admin/credentials/stored"),
			"{message}"
		);
		assert_eq!(s.fake.count(), sent);
		let models_of = |provider: &str| {
			s.client
				.get(format!(
					"{}/ai/v1/admin/providers/{provider}/upstream-models",
					s.base
				))
				.bearer_auth(service())
				.send()
		};
		let (status, code, message) = error_of(models_of("stored").await.unwrap()).await;
		assert_eq!((status, code.as_str()), (503, "provider_auth"));
		assert!(message.contains("not set"), "{message}");
		let listed: Value = s
			.client
			.get(format!("{}/ai/v1/admin/credentials", s.base))
			.bearer_auth(service())
			.send()
			.await
			.unwrap()
			.json()
			.await
			.unwrap();
		assert!(
			listed["data"]
				.as_array()
				.unwrap()
				.iter()
				.any(|c| c["name"] == "stored"
					&& c["stored"] == false
					&& c["used_by"] == json!(["stored"])),
			"{listed}"
		);

		// A key the provider refuses is not stored, and its echo of the key is scrubbed.
		let wrong = "not-the-fake-providers-own";
		let (status, _, message) =
			error_of(put("stored", json!({"api_key": wrong})).await.unwrap()).await;
		assert_eq!(status, 400);
		assert!(
			message.contains("refused this credential (401)"),
			"{message}"
		);
		assert!(!message.contains(wrong), "{message}");
		assert_eq!(stored().await, 0);
		let (status, _, message) =
			error_of(put("nobody", json!({"api_key": Fake::KEY})).await.unwrap()).await;
		assert_eq!(status, 400);
		assert!(message.contains("No provider"), "{message}");

		// The right one is stored, shown only by its print, and used by the next call.
		let made = put("stored", json!({"api_key": Fake::KEY})).await.unwrap();
		assert_eq!(made.status(), 200);
		let made: Value = made.json().await.unwrap();
		assert_eq!(made["last_four"], "6789");
		assert!(made["fingerprint"].as_str().unwrap().starts_with("sha256:"));
		assert!(!made.to_string().contains(Fake::KEY));
		let reply = call("via-store").await.unwrap();
		assert_eq!(reply.status(), 200, "no restart");
		assert_eq!(
			s.fake.last().authorization.as_deref(),
			Some(format!("Bearer {}", Fake::KEY).as_str())
		);

		// AWS credentials sign a Bedrock provider's calls; a wrong secret is refused.
		let wrong = put(
			"stored-aws",
			json!({"access_key_id": Fake::AWS_ACCESS_KEY_ID, "secret_access_key": "not-the-secret"}),
		);
		let (status, _, message) = error_of(wrong.await.unwrap()).await;
		assert_eq!(status, 400);
		assert!(message.contains("(403)"), "{message}");
		let made = put(
			"stored-aws",
			json!({"access_key_id": Fake::AWS_ACCESS_KEY_ID, "secret_access_key": Fake::AWS_SECRET}),
		);
		assert_eq!(made.await.unwrap().status(), 200);
		assert_eq!(call("via-store-aws").await.unwrap().status(), 200);
		// Each lists its provider's own models with the credential it now holds.
		let listed = models_of("stored").await.unwrap().text().await.unwrap();
		assert!(listed.contains("\"fake-new\""), "{listed}");
		let listed = models_of("stored-aws").await.unwrap().text().await.unwrap();
		assert!(listed.contains("us.openai.fake-luna"), "{listed}");
		assert!(!listed.contains(Fake::AWS_SECRET));
		let listed = s
			.client
			.get(format!("{}/ai/v1/admin/credentials", s.base))
			.bearer_auth(service())
			.send()
			.await
			.unwrap()
			.text()
			.await
			.unwrap();
		assert!(!listed.contains(Fake::KEY) && !listed.contains(Fake::AWS_SECRET));

		// Removed in plain SQL, it stops at once: the trigger tells Psyche.
		db.batch_execute("DELETE FROM psyche.credentials WHERE name = 'stored'")
			.await
			.unwrap();
		let mut stopped = false;
		for _ in 0..250 {
			if call("via-store").await.unwrap().status() == 503 {
				stopped = true;
				break;
			}
			tokio::time::sleep(Duration::from_millis(20)).await;
		}
		assert!(stopped, "the deletion reached Psyche");
		let gone = s
			.client
			.delete(format!("{}/ai/v1/admin/credentials/stored-aws", s.base))
			.bearer_auth(service())
			.send()
			.await
			.unwrap();
		assert_eq!(gone.status(), 204);
		assert_eq!(call("via-store-aws").await.unwrap().status(), 503);
	}

	#[tokio::test]
	async fn providers_and_models_are_tables_the_file_seeds_and_the_admin_api_edits() {
		let Some((mut s, mut db, _key, _id)) = with_budgets("models").await else {
			return;
		};
		// The calls below borrow these rather than `s`, which the call log needs mutably.
		let (client, base) = (s.client.clone(), s.base.clone());
		let (client, base) = (&client, &base);
		let service = || jwt::sign(&json!({"role": "service_role"}), SECRET);
		let admin = |method: reqwest::Method, path: &str, body: Option<Value>| {
			let request = client
				.request(method, format!("{base}/ai/v1/admin/{path}"))
				.bearer_auth(service());
			match body {
				Some(body) => request.json(&body).send(),
				None => request.send(),
			}
		};
		let list = |what: &'static str| async move {
			let listed: Value = admin(reqwest::Method::GET, what, None)
				.await
				.unwrap()
				.json()
				.await
				.unwrap();
			listed["data"].as_array().unwrap().clone()
		};
		let row = |rows: &[Value], name: &str| {
			rows.iter()
				.find(|r| r["name"] == name)
				.unwrap_or_else(|| panic!("{name} in {rows:?}"))
				.clone()
		};
		let call = |model: &str| {
			client
				.post(format!("{base}/ai/v1/chat/completions"))
				.bearer_auth(service())
				.json(&chat(model, false))
				.send()
		};
		// A change made in SQL reaches Psyche through the trigger; waits until `model` answers
		// `status`.
		let until = |model: &'static str, status: u16| async move {
			for _ in 0..250 {
				if call(model).await.unwrap().status() == status {
					return true;
				}
				tokio::time::sleep(Duration::from_millis(20)).await;
			}
			false
		};

		// The file is in the tables, and is what is served.
		let models = list("models").await;
		let ok = row(&models, "ok");
		assert_eq!(
			(
				ok["origin"].clone(),
				ok["served"].clone(),
				ok["edited"].clone()
			),
			(json!("file"), json!(true), json!(false))
		);
		assert_eq!(
			ok["price_from"],
			Value::Null,
			"the fake is not in the catalogue"
		);
		let priced = row(&models, "ok-priced");
		assert_eq!(priced["price_from"], "model");
		assert_eq!(priced["price"]["input"], 1.0);
		assert_eq!(priced["price"]["output"], 2.5);
		let providers = list("providers").await;
		assert_eq!(row(&providers, "fake")["key"], "environment");
		let stored = row(&providers, "stored");
		assert_eq!(stored["key"], "credential");
		assert_eq!(stored["credential_stored"], false);
		assert!(!Value::Array(providers).to_string().contains(Fake::KEY));

		// The file's providers are the file's: their key may be Psyche's own.
		let (status, _, message) = error_of(
			admin(
				reqwest::Method::PUT,
				"providers/fake",
				Some(json!({"kind": "openai", "base_url": "https://elsewhere.example/v1"})),
			)
			.await
			.unwrap(),
		)
		.await;
		assert_eq!(status, 409);
		assert!(message.contains("providers file"), "{message}");
		let gone = admin(reqwest::Method::DELETE, "providers/fake", None);
		assert_eq!(gone.await.unwrap().status(), 409);

		// A provider and a model made here, then its credential, proven on that model: no restart.
		let origin = s.fake.origin.clone();
		let made = admin(
			reqwest::Method::PUT,
			"providers/mine",
			Some(
				json!({"kind": "openai", "base_url": format!("{origin}/v1"), "credential": "mine"}),
			),
		);
		assert_eq!(made.await.unwrap().status(), 200);
		let made: Value = admin(
			reqwest::Method::PUT,
			"models/mine-ok",
			Some(
				json!({"provider": "mine", "upstream": "fake-ok", "price": {"input": 3, "output": 6}}),
			),
		)
		.await
		.unwrap()
		.json()
		.await
		.unwrap();
		assert_eq!(made["price_from"], "model", "{made}");
		let (status, code, _) = error_of(call("mine-ok").await.unwrap()).await;
		assert_eq!((status, code.as_str()), (503, "provider_auth"));
		let proven = admin(
			reqwest::Method::PUT,
			"credentials/mine",
			Some(json!({"api_key": Fake::KEY})),
		);
		assert_eq!(proven.await.unwrap().status(), 200);
		while s.calls.try_recv().is_ok() {}
		assert_eq!(call("mine-ok").await.unwrap().status(), 200);
		assert_eq!(s.fake.last().body["model"], "fake-ok");
		let charged = s.call().await;
		assert_eq!(charged.model, "mine-ok");
		assert_eq!(
			charged.price.unwrap().rates.input,
			3_000_000,
			"the row's price"
		);

		// A model with a slash in its name, as OpenRouter's are.
		let made = admin(
			reqwest::Method::PUT,
			"models/org/ok",
			Some(json!({"provider": "fake", "upstream": "fake-ok"})),
		);
		assert_eq!(made.await.unwrap().status(), 200);
		assert_eq!(call("org/ok").await.unwrap().status(), 200);

		// What cannot be served is refused with the reason, and nothing is written.
		for (path, body, said) in [
			(
				"models/x",
				json!({"provider": "nobody"}),
				"no provider named nobody",
			),
			(
				"models/x",
				json!({"provider": "fake", "embeddings": "titan"}),
				"only a Bedrock model",
			),
			(
				"models/x",
				json!({"provider": "fake", "price": {"input": -1, "output": 1}}),
				"price",
			),
			(
				"models/x",
				json!({"provider": "fake", "max_output_tokens": 0}),
				"max_output_tokens",
			),
			(
				"models/x",
				json!({"provider": "fake", "api_key": "k"}),
				"unknown field",
			),
			(
				"providers/b",
				json!({"kind": "bedrock", "region": "us-west-2"}),
				"stored credential",
			),
			(
				"providers/p",
				json!({"kind": "openai", "base_url": "ftp://example.com"}),
				"",
			),
			(
				"providers/p",
				json!({"kind": "openai", "base_url": format!("{origin}/v1"), "region": "us-west-2"}),
				"region",
			),
		] {
			let (status, _, message) = error_of(
				admin(reqwest::Method::PUT, path, Some(body.clone()))
					.await
					.unwrap(),
			)
			.await;
			assert_eq!(status, 400, "{body}: {message}");
			assert!(message.contains(said), "{body}: {message}");
		}
		let x: i64 = db
			.query_one("SELECT count(*) FROM psyche.models WHERE name = 'x'", &[])
			.await
			.unwrap()
			.get(0);
		assert_eq!(x, 0);

		// A model from the file is edited in place, and the next start leaves the edit alone.
		let edited = admin(
			reqwest::Method::PUT,
			"models/ok",
			Some(
				json!({"provider": "fake", "upstream": "fake-ok", "price": {"input": 5, "output": 5}}),
			),
		);
		assert_eq!(edited.await.unwrap().status(), 200);
		assert!(
			models::seed(&mut db, &s.tables.file)
				.await
				.unwrap()
				.is_empty()
		);
		s.tables.reload(&db).await.unwrap();
		let ok = row(&list("models").await, "ok");
		assert_eq!(
			(ok["edited"].clone(), ok["price"]["input"].clone()),
			(json!(true), json!(5.0))
		);

		// Removed, it is disabled, since the next start would put it back; and it stays away.
		let removed: Value = admin(reqwest::Method::DELETE, "models/ok", None)
			.await
			.unwrap()
			.json()
			.await
			.unwrap();
		assert_eq!(
			removed,
			json!({"name": "ok", "deleted": false, "disabled": true})
		);
		assert!(
			models::seed(&mut db, &s.tables.file)
				.await
				.unwrap()
				.is_empty()
		);
		s.tables.reload(&db).await.unwrap();
		let (status, code, _) = error_of(call("ok").await.unwrap()).await;
		assert_eq!((status, code.as_str()), (404, "model_not_found"));

		// Made in plain SQL, a model is served at once.
		db.batch_execute(
			"INSERT INTO psyche.models (name, provider, upstream) VALUES ('sql-made', 'fake', 'fake-ok')",
		)
		.await
		.unwrap();
		assert!(until("sql-made", 200).await, "the insert reached Psyche");

		// A file provider changed in SQL is not served with the environment's key at an address
		// the file did not name; the next start puts the file's back.
		db.batch_execute(
			"UPDATE psyche.providers SET base_url = 'http://127.0.0.1:9/v1' WHERE name = 'fake'",
		)
		.await
		.unwrap();
		assert!(
			until("sql-made", 404).await,
			"the file's provider is left out"
		);
		let fake = row(&list("providers").await, "fake");
		assert!(
			fake["problem"].as_str().unwrap().contains("providers file"),
			"{fake}"
		);
		let sql_made = row(&list("models").await, "sql-made");
		assert_eq!(sql_made["served"], false);
		assert!(sql_made["problem"].as_str().is_some(), "{sql_made}");
		models::seed(&mut db, &s.tables.file).await.unwrap();
		assert!(until("sql-made", 200).await, "the file's provider is back");

		// A provider with models is not removed; without them it is.
		let (status, _, message) = error_of(
			admin(reqwest::Method::DELETE, "providers/mine", None)
				.await
				.unwrap(),
		)
		.await;
		assert_eq!(status, 409);
		assert!(message.contains("mine-ok"), "{message}");
		let removed: Value = admin(reqwest::Method::DELETE, "models/mine-ok", None)
			.await
			.unwrap()
			.json()
			.await
			.unwrap();
		assert_eq!(removed["deleted"], true);
		let gone = admin(reqwest::Method::DELETE, "providers/mine", None);
		assert_eq!(gone.await.unwrap().status(), 204);
		let (status, code, _) = error_of(call("mine-ok").await.unwrap()).await;
		assert_eq!((status, code.as_str()), (404, "model_not_found"));
	}

	#[test]
	fn openai_s_own_api_takes_max_completion_tokens_and_the_rest_max_tokens() {
		let provider = |kind: Kind, base: &str| Provider {
			name: "p".into(),
			kind,
			base_url: url::Url::parse(base).unwrap(),
			api_key: None,
			aws: None,
			region: None,
			credential: None,
			auth_header: None,
		};
		let openai = provider(Kind::Openai, "https://api.openai.com/v1");
		assert_eq!(cap_field(&openai, Operation::Chat), "max_completion_tokens");
		assert_eq!(cap_field(&openai, Operation::Completions), "max_tokens");
		let azure = provider(Kind::Openai, "https://acme.openai.azure.com/openai/v1");
		assert_eq!(cap_field(&azure, Operation::Chat), "max_completion_tokens");
		let vllm = provider(Kind::Openai, "http://10.0.0.2:8000/v1");
		assert_eq!(cap_field(&vllm, Operation::Chat), "max_tokens");
		let bedrock = provider(
			Kind::Bedrock,
			"https://bedrock-runtime.us-west-2.amazonaws.com",
		);
		assert_eq!(cap_field(&bedrock, Operation::Chat), "max_tokens");
	}

	#[tokio::test]
	async fn rate_limits_refuse_with_when_to_try_again_and_say_what_is_left() {
		let Some((s, _db, key, id)) = with_budgets("limits").await else {
			return;
		};
		let made = s
			.post(
				"/ai/v1/admin/limits",
				json!({"name": "app", "key_id": id, "requests_per_minute": 3}),
			)
			.send()
			.await
			.unwrap();
		assert_eq!(made.status(), 201, "{}", made.text().await.unwrap());
		let call = || {
			s.client
				.post(format!("{}/ai/v1/chat/completions", s.base))
				.bearer_auth(&key)
				.json(&chat("ok", false))
				.send()
		};
		let first = call().await.unwrap();
		assert_eq!(first.status(), 200);
		assert_eq!(first.headers()["x-ratelimit-limit-requests"], "3");
		assert_eq!(first.headers()["x-ratelimit-remaining-requests"], "2");
		call().await.unwrap();
		call().await.unwrap();
		let refused = call().await.unwrap();
		assert_eq!(refused.status(), 429);
		let wait: u64 = refused.headers()["retry-after"]
			.to_str()
			.unwrap()
			.parse()
			.unwrap();
		assert!((1..=20).contains(&wait), "{wait}");
		let body: Value = refused.json().await.unwrap();
		assert_eq!(body["error"]["code"], "rate_limit_exceeded");
		assert_eq!(body["error"]["scope"], "key");
		assert!(
			body["error"]["message"]
				.as_str()
				.unwrap()
				.contains("3 requests a minute")
		);
		// The service key is not held by the key's limit, and gets no rate-limit headers.
		let service = s
			.post("/ai/v1/chat/completions", chat("ok", false))
			.send()
			.await
			.unwrap();
		assert_eq!(service.status(), 200);
		assert!(
			service
				.headers()
				.get("x-ratelimit-limit-requests")
				.is_none()
		);

		let tokens = s
			.post(
				"/ai/v1/admin/limits",
				json!({"name": "tokens", "tokens_per_minute": 50}),
			)
			.send()
			.await
			.unwrap();
		assert_eq!(tokens.status(), 201);
		let big = s
			.post(
				"/ai/v1/chat/completions",
				json!({"model": "ok", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]}),
			)
			.send()
			.await
			.unwrap();
		assert_eq!(big.status(), 429);
		assert!(
			big.headers().get("retry-after").is_none(),
			"waiting would not help"
		);
		let none = s
			.post("/ai/v1/admin/limits", json!({"name": "empty"}))
			.send()
			.await
			.unwrap();
		assert_eq!(none.status(), 400, "a limit limits something");
	}

	#[tokio::test]
	async fn a_soft_budget_warns_and_a_deleted_budget_stops_at_once() {
		let Some((s, _db, _key, _id)) = with_budgets("budget_soft").await else {
			return;
		};
		let made: Value = s
			.post(
				"/ai/v1/admin/budgets",
				json!({"name": "watch", "amount": 0, "period": "rolling", "days": 7, "hard": false}),
			)
			.send()
			.await
			.unwrap()
			.json()
			.await
			.unwrap();
		let reply = s
			.post("/ai/v1/chat/completions", chat("ok-priced", false))
			.send()
			.await
			.unwrap();
		assert_eq!(reply.status(), 200);
		assert_eq!(reply.headers()["x-psyche-budget-warning"], "watch");

		let hard = s
			.post(
				"/ai/v1/admin/budgets",
				json!({"name": "stop", "amount": 0, "period": "month"}),
			)
			.send()
			.await
			.unwrap();
		assert_eq!(hard.status(), 201);
		let hard: Value = hard.json().await.unwrap();
		assert_eq!(
			s.post("/ai/v1/chat/completions", chat("ok-priced", false))
				.send()
				.await
				.unwrap()
				.status(),
			429
		);
		let gone = s
			.client
			.delete(format!(
				"{}/ai/v1/admin/budgets/{}",
				s.base,
				hard["id"].as_str().unwrap()
			))
			.bearer_auth(jwt::sign(&json!({"role": "service_role"}), SECRET))
			.send()
			.await
			.unwrap();
		assert_eq!(gone.status(), 204);
		assert_eq!(
			s.post("/ai/v1/chat/completions", chat("ok-priced", false))
				.send()
				.await
				.unwrap()
				.status(),
			200
		);
		let bad = s
			.post(
				"/ai/v1/admin/budgets",
				json!({"name": "x", "amount": 1, "period": "week"}),
			)
			.send()
			.await
			.unwrap();
		assert_eq!(bad.status(), 400, "a period the table does not know");
		let listed: Value = s
			.client
			.get(format!("{}/ai/v1/admin/budgets", s.base))
			.bearer_auth(jwt::sign(&json!({"role": "service_role"}), SECRET))
			.send()
			.await
			.unwrap()
			.json()
			.await
			.unwrap();
		assert_eq!(listed["data"].as_array().unwrap().len(), 1);
		assert_eq!(listed["data"][0]["id"], made["id"]);
	}

	#[tokio::test]
	async fn a_caller_that_asked_for_usage_gets_it() {
		let mut s = Setup::new().await;
		let mut request = chat("ok", true);
		request["stream_options"] = json!({"include_usage": true});
		let body = s
			.post("/ai/v1/chat/completions", request)
			.send()
			.await
			.unwrap()
			.text()
			.await
			.unwrap();
		assert!(body.contains("\"usage\""), "{body}");
		assert_eq!(s.call().await.outcome, Outcome::Completed);
	}

	#[tokio::test]
	async fn fim_keeps_its_suffix() {
		// A gateway that drops `suffix` makes every completion duplicate the code after the cursor.
		let mut s = Setup::new().await;
		let request = json!({"model": "ok", "prompt": "SELECT id, ", "suffix": " FROM users;", "max_tokens": 16});
		let response = s.post("/ai/v1/completions", request).send().await.unwrap();
		assert_eq!(response.status(), 200);
		let seen = s.fake.last();
		assert_eq!(seen.path, "/v1/completions");
		assert_eq!(seen.body["suffix"], " FROM users;");
		assert_eq!(seen.body["prompt"], "SELECT id, ");
		assert_eq!(s.call().await.operation, Operation::Completions);
	}

	#[tokio::test]
	async fn models_lists_the_catalogue() {
		let s = Setup::new().await;
		let response = s
			.client
			.get(format!("{}/ai/v1/models", s.base))
			.bearer_auth(jwt::sign(&json!({"role": "service_role"}), SECRET))
			.send()
			.await
			.unwrap();
		let body: Value = response.json().await.unwrap();
		let ids: Vec<&str> = body["data"]
			.as_array()
			.unwrap()
			.iter()
			.map(|m| m["id"].as_str().unwrap())
			.collect();
		assert_eq!(ids[0], "ok");
		assert_eq!(body["data"][0]["owned_by"], "fake");
		// Automatic choice, over the models that wrote their facts.
		let auto = body["data"]
			.as_array()
			.unwrap()
			.iter()
			.find(|m| m["id"] == "auto")
			.unwrap();
		assert_eq!(auto["owned_by"], "auto");
		assert_eq!(
			auto["targets"],
			json!(["small-text", "mid-images", "big-tools"])
		);
		assert!(
			ids.contains(&"auto:floor") && ids.contains(&"auto:nitro"),
			"{ids:?}"
		);
	}

	#[tokio::test]
	async fn callers_are_checked_before_anything_is_sent() {
		let s = Setup::new().await;
		let url = format!("{}/ai/v1/chat/completions", s.base);
		let none = s
			.client
			.post(&url)
			.json(&chat("ok", false))
			.send()
			.await
			.unwrap();
		assert_eq!(none.status(), 401);
		let anon = s
			.client
			.post(&url)
			.bearer_auth(jwt::sign(&json!({"role": "anon"}), SECRET))
			.json(&chat("ok", false))
			.send()
			.await
			.unwrap();
		assert_eq!(anon.status(), 403);
		let forged = s
			.client
			.post(&url)
			.bearer_auth(jwt::sign(
				&json!({"role": "service_role"}),
				"another-secret-of-at-least-32-chars!",
			))
			.json(&chat("ok", false))
			.send()
			.await
			.unwrap();
		assert_eq!(forged.status(), 401);
		// The apikey header works as the front door sends it.
		let apikey = s
			.client
			.post(&url)
			.header(
				"apikey",
				jwt::sign(&json!({"role": "service_role"}), SECRET),
			)
			.json(&chat("ok", false))
			.send()
			.await
			.unwrap();
		assert_eq!(apikey.status(), 200);
		assert_eq!(
			s.fake.count(),
			1,
			"only the authorised call reached the provider"
		);

		let unknown = s
			.post("/ai/v1/chat/completions", chat("nope", false))
			.send()
			.await
			.unwrap();
		assert_eq!(unknown.status(), 404);
		let body: Value = unknown.json().await.unwrap();
		assert_eq!(body["error"]["code"], "model_not_found");
		let bad = s
			.client
			.post(&url)
			.bearer_auth(jwt::sign(&json!({"role": "service_role"}), SECRET))
			.body("{not json")
			.send()
			.await
			.unwrap();
		assert_eq!(bad.status(), 400);
		let huge = s
			.post(
				"/ai/v1/chat/completions",
				json!({"model": "ok", "pad": "x".repeat(2 * 1024 * 1024)}),
			)
			.send()
			.await
			.unwrap();
		assert_eq!(huge.status(), 413);
	}

	#[tokio::test]
	async fn provider_failures_are_classified_and_never_leak_the_key() {
		// Psyche holds the wrong key; the fake echoes what it was given, as OpenAI's 401 does.
		let wrong = "sk-psyche-held-this-wrong-key-123456";
		let mut s = Setup::with(wrong, Timeouts::default()).await;
		let response = s
			.post("/ai/v1/chat/completions", chat("ok", false))
			.send()
			.await
			.unwrap();
		assert_eq!(
			response.status(),
			503,
			"a refused credential is not the caller's fault"
		);
		let body = response.text().await.unwrap();
		assert!(!body.contains(wrong), "{body}");
		assert!(body.contains("provider_auth"), "{body}");
		assert_eq!(s.call().await.outcome, Outcome::Failed(Class::ProviderAuth));

		let mut s = Setup::new().await;
		let limited = s
			.post("/ai/v1/chat/completions", chat("limited", false))
			.send()
			.await
			.unwrap();
		assert_eq!(limited.status(), 429);
		assert_eq!(limited.headers()["retry-after"], "7");
		assert_eq!(s.call().await.outcome, Outcome::Failed(Class::RateLimited));

		let picky = s
			.post("/ai/v1/chat/completions", chat("picky", false))
			.send()
			.await
			.unwrap();
		assert_eq!(
			picky.status(),
			400,
			"the caller's own mistake keeps the provider's status"
		);
		let body: Value = picky.json().await.unwrap();
		assert!(
			body["error"]["message"]
				.as_str()
				.unwrap()
				.contains("temperature"),
			"{body}"
		);
		s.call().await;

		let fails = s
			.post("/ai/v1/chat/completions", chat("fails", true))
			.send()
			.await
			.unwrap();
		assert_eq!(fails.status(), 503);
		// A 5xx is asked once more before it comes back, and each try is its own row.
		let (first, second) = (s.call().await, s.call().await);
		assert_eq!((first.attempt, second.attempt), (1, 2));
		assert_eq!(second.outcome, Outcome::Failed(Class::Upstream));

		// No credit left is the owner's billing, never a limit to wait out or the caller's request.
		for model in ["no-credit", "no-credit-402", "claude-no-credit"] {
			let broke = s
				.post("/ai/v1/chat/completions", chat(model, false))
				.send()
				.await
				.unwrap();
			assert_eq!(broke.status(), 503, "{model}");
			assert!(broke.headers().get("retry-after").is_none(), "{model}");
			let body: Value = broke.json().await.unwrap();
			assert_eq!(body["error"]["code"], "provider_billing", "{model}: {body}");
			assert!(
				body["error"]["message"]
					.as_str()
					.unwrap()
					.contains("no credit or quota left"),
				"{body}"
			);
			assert_eq!(
				s.call().await.outcome,
				Outcome::Failed(Class::ProviderBilling)
			);
		}
	}

	#[tokio::test]
	async fn a_provider_that_never_answers_times_out() {
		let timeouts = Timeouts {
			first_byte: Duration::from_millis(300),
			..Timeouts::default()
		};
		let mut s = Setup::with(Fake::KEY, timeouts).await;
		let response = s
			.post("/ai/v1/chat/completions", chat("hangs", false))
			.send()
			.await
			.unwrap();
		assert_eq!(response.status(), 503);
		assert_eq!(s.call().await.outcome, Outcome::Failed(Class::Timeout));
	}

	fn messages_request(model: &str, stream: bool) -> Value {
		json!({"model": model, "max_tokens": 16, "stream": stream,
			"messages": [{"role": "user", "content": "hello there"}]})
	}

	#[tokio::test]
	async fn messages_pass_through_to_anthropic_under_the_name_asked_for() {
		let mut s = Setup::new().await;
		let reply = s
			.post("/ai/v1/messages", messages_request("claude", false))
			.header("anthropic-beta", "some-beta-2026-01-01")
			.send()
			.await
			.unwrap();
		assert_eq!(reply.status(), 200);
		let body: Value = reply.json().await.unwrap();
		assert_eq!(body["type"], "message");
		assert_eq!(body["model"], "claude");
		assert_eq!(body["content"][0]["text"], "Hello from the fake.");
		let seen = s.fake.last();
		assert_eq!(seen.path, "/v1/messages");
		assert_eq!(seen.body["model"], "fake-ok");
		assert_eq!(seen.anthropic_version.as_deref(), Some(anthropic::VERSION));
		let call = s.call().await;
		assert_eq!(call.operation, Operation::Messages);
		assert_eq!(call.outcome, Outcome::Completed);
		assert_eq!(call.usage.unwrap().prompt_tokens, 129);

		// Streamed, with the key where a client written for Anthropic puts it.
		let text = s
			.client
			.post(format!("{}/v1/messages", s.base))
			.header(
				"x-api-key",
				jwt::sign(&json!({"role": "service_role"}), SECRET),
			)
			.json(&messages_request("claude", true))
			.send()
			.await
			.unwrap()
			.text()
			.await
			.unwrap();
		assert!(text.contains("event: message_start"), "{text}");
		assert!(text.contains("\"model\":\"claude\""), "{text}");
		assert!(!text.contains("fake-ok"), "{text}");
		assert!(text.contains("event: message_stop"), "{text}");
		let call = s.call().await;
		assert_eq!(call.outcome, Outcome::Completed);
		assert_eq!(call.usage.unwrap().completion_tokens, 4);
	}

	#[tokio::test]
	async fn a_breakpoint_does_not_reach_a_model_that_does_not_cache_on_messages() {
		let mut s = Setup::new().await;
		let mut body = messages_request("claude-tool", false);
		body["system"] =
			json!([{"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}]);
		body["messages"] = json!([{"role": "user", "content": [
			{"type": "tool_result", "tool_use_id": "t", "cache_control": {"type": "ephemeral"},
				"content": [{"type": "text", "text": "y", "cache_control": {"type": "ephemeral"}}]}
		]}]);
		let reply = s.post("/ai/v1/messages", body).send().await.unwrap();
		assert_eq!(reply.status(), 200);
		s.call().await;
		let sent = s.fake.last().body;
		assert!(!sent.to_string().contains("cache_control"), "{sent}");
	}

	#[tokio::test]
	async fn messages_are_translated_for_any_other_provider() {
		let mut s = Setup::new().await;
		for model in ["ok", "qwen"] {
			let body: Value = s
				.post("/ai/v1/messages", messages_request(model, false))
				.send()
				.await
				.unwrap()
				.json()
				.await
				.unwrap();
			assert_eq!(body["type"], "message", "{model}: {body}");
			assert_eq!(body["model"], model);
			assert_eq!(body["content"][0]["type"], "text", "{model}: {body}");
			assert!(
				body["usage"]["output_tokens"].as_u64().unwrap() > 0,
				"{model}: {body}"
			);
			assert_eq!(s.call().await.operation, Operation::Messages);

			let text = s
				.post("/ai/v1/messages", messages_request(model, true))
				.send()
				.await
				.unwrap()
				.text()
				.await
				.unwrap();
			for name in [
				"message_start",
				"content_block_delta",
				"message_delta",
				"message_stop",
			] {
				assert!(text.contains(&format!("event: {name}")), "{model}: {text}");
			}
			assert!(!text.contains("data: [DONE]"), "{model}: {text}");
			let call = s.call().await;
			assert_eq!(call.outcome, Outcome::Completed, "{model}");
			assert!(call.usage.is_some(), "{model}");
		}
		assert!(s.fake.last().path.contains("converse"));
	}

	#[tokio::test]
	async fn a_cut_or_failed_messages_stream_ends_with_anthropic_s_error_event() {
		let mut s = Setup::new().await;
		for model in [
			"claude-truncated",
			"claude-overloaded",
			"cut",
			"bedrock-truncated",
		] {
			let text = s
				.post("/ai/v1/messages", messages_request(model, true))
				.send()
				.await
				.unwrap()
				.text()
				.await
				.unwrap();
			assert!(text.contains("event: error"), "{model}: {text}");
			assert!(!text.contains("event: message_stop"), "{model}: {text}");
			assert!(
				matches!(s.call().await.outcome, Outcome::Failed(_)),
				"{model}"
			);
		}
	}

	#[tokio::test]
	async fn messages_refusals_are_in_anthropic_s_shape() {
		let s = Setup::new().await;
		let refused = |body: Value| {
			let request = s.post("/ai/v1/messages", body);
			async move {
				let reply = request.send().await.unwrap();
				let status = reply.status().as_u16();
				let body: Value = reply.json().await.unwrap();
				(status, body)
			}
		};
		let (status, body) = refused(messages_request("nobody", false)).await;
		assert_eq!(status, 404);
		assert_eq!(body["type"], "error");
		assert_eq!(body["error"]["type"], "not_found_error");
		let (status, body) =
			refused(json!({"model": "claude", "messages": [{"role": "user", "content": "hi"}]}))
				.await;
		assert_eq!(status, 400);
		assert!(
			body["error"]["message"]
				.as_str()
				.unwrap()
				.contains("max_tokens")
		);
		// The internet level, with Anthropic's server tools counted as the web.
		let mut search = messages_request("claude", false);
		search["tools"] = json!([{"type": "web_search_20250305", "name": "web_search"}]);
		let (status, body) = refused(search).await;
		assert_eq!(status, 400);
		assert!(
			body["error"]["message"]
				.as_str()
				.unwrap()
				.contains("internet level"),
			"{body}"
		);
		let unsigned = s
			.client
			.post(format!("{}/ai/v1/messages", s.base))
			.json(&messages_request("claude", false))
			.send()
			.await
			.unwrap();
		assert_eq!(unsigned.status(), 401);
		let body: Value = unsigned.json().await.unwrap();
		assert_eq!(body["error"]["type"], "authentication_error");
	}

	fn responses_request(model: &str, stream: bool) -> Value {
		json!({"model": model, "stream": stream, "store": false, "instructions": "Be brief.",
			"input": [{"type": "message", "role": "user",
				"content": [{"type": "input_text", "text": "hello there"}]}]})
	}

	/// An SSE body's events as `(name, data)`.
	fn sse_events(text: &str) -> Vec<(String, Value)> {
		text.split("\n\n")
			.filter(|event| !event.is_empty() && !event.starts_with(':'))
			.map(|event| {
				let (name, data) = event.split_once('\n').unwrap();
				(
					name.strip_prefix("event: ").unwrap().to_string(),
					serde_json::from_str(data.strip_prefix("data: ").unwrap()).unwrap(),
				)
			})
			.collect()
	}

	#[tokio::test]
	async fn responses_are_answered_by_every_provider_and_end_in_response_completed() {
		let mut s = Setup::new().await;
		for model in ["ok", "claude", "qwen"] {
			let reply = s
				.post("/ai/v1/responses", responses_request(model, false))
				.send()
				.await
				.unwrap();
			assert_eq!(reply.status(), 200, "{model}");
			let body: Value = reply.json().await.unwrap();
			assert_eq!(body["object"], "response", "{model}: {body}");
			assert_eq!(body["model"], model);
			assert_eq!(body["status"], "completed");
			assert!(body["id"].as_str().unwrap().starts_with("resp_"));
			assert_eq!(body["output"][0]["type"], "message", "{model}: {body}");
			assert_eq!(
				body["output"][0]["content"][0]["text"], "Hello from the fake.",
				"{model}: {body}"
			);
			assert!(body["usage"]["total_tokens"].as_u64().unwrap() > 0);
			let call = s.call().await;
			assert_eq!(call.operation, Operation::Responses);
			assert_eq!(call.outcome, Outcome::Completed);
			assert!(call.usage.is_some(), "{model}");

			let text = s
				.post("/v1/responses", responses_request(model, true))
				.send()
				.await
				.unwrap()
				.text()
				.await
				.unwrap();
			let events = sse_events(&text);
			for (n, (name, data)) in events.iter().enumerate() {
				assert_eq!(data["type"], name.as_str(), "{model}: {text}");
				assert_eq!(data["sequence_number"], n, "{model}: {text}");
			}
			assert_eq!(events[0].0, "response.created", "{model}: {text}");
			let names: Vec<&str> = events.iter().map(|(n, _)| n.as_str()).collect();
			for name in [
				"response.in_progress",
				"response.output_item.added",
				"response.content_part.added",
				"response.output_text.delta",
				"response.output_text.done",
				"response.content_part.done",
				"response.output_item.done",
			] {
				assert!(names.contains(&name), "{model}: {name}: {text}");
			}
			let (last, data) = events.last().unwrap();
			assert_eq!(last, "response.completed", "{model}: {text}");
			assert_eq!(
				names.iter().filter(|n| **n == "response.completed").count(),
				1
			);
			assert_eq!(data["response"]["model"], model);
			assert!(
				data["response"]["usage"]["input_tokens"].as_u64().unwrap() > 0,
				"{model}: {text}"
			);
			assert!(!text.contains("[DONE]"), "{model}: {text}");
			assert!(!text.contains("fake-ok"), "{model}: {text}");
			let call = s.call().await;
			assert_eq!(call.operation, Operation::Responses);
			assert_eq!(call.outcome, Outcome::Completed, "{model}");
			assert!(call.usage.is_some(), "{model}");
		}
		assert!(s.fake.last().path.contains("converse"));
	}

	#[tokio::test]
	async fn the_chat_completion_a_response_becomes_is_what_the_provider_gets() {
		let mut s = Setup::new().await;
		let mut body = responses_request("ok", true);
		body["max_output_tokens"] = json!(77);
		body["reasoning"] = json!({"effort": "low", "summary": "auto"});
		body["include"] = json!(["reasoning.encrypted_content"]);
		let reply = s.post("/ai/v1/responses", body).send().await.unwrap();
		assert_eq!(reply.status(), 200);
		reply.bytes().await.unwrap();
		s.call().await;
		let seen = s.fake.last();
		assert_eq!(seen.path, "/v1/chat/completions");
		assert_eq!(seen.body["model"], "fake-ok");
		assert_eq!(seen.body["max_tokens"], 77);
		assert_eq!(seen.body["stream_options"]["include_usage"], true);
		assert_eq!(seen.body["messages"][0]["content"], "Be brief.");
		assert_eq!(seen.body["messages"][1]["content"], "hello there");
		for gone in [
			"input",
			"instructions",
			"store",
			"include",
			"reasoning",
			"reasoning_effort",
		] {
			assert!(seen.body.get(gone).is_none(), "{gone}: {}", seen.body);
		}
	}

	#[tokio::test]
	async fn a_tool_turn_goes_through_the_responses_api_both_ways() {
		let mut s = Setup::new().await;
		for model in ["claude-tool", "bedrock-tool"] {
			let mut body = responses_request(model, false);
			body["tools"] = json!([{"type": "function", "name": "run_sql", "strict": false,
				"parameters": {"type": "object", "properties": {"sql": {"type": "string"}}}}]);
			body["input"].as_array_mut().unwrap().extend([
				json!({"type": "function_call", "name": "run_sql", "arguments": "{\"sql\":\"x\"}", "call_id": "c1"}),
				json!({"type": "function_call_output", "call_id": "c1", "output": "1"}),
			]);
			let reply: Value = s
				.post("/ai/v1/responses", body.clone())
				.send()
				.await
				.unwrap()
				.json()
				.await
				.unwrap();
			let output = reply["output"].as_array().unwrap();
			assert_eq!(
				output[0]["content"][0]["text"], "Let me look.",
				"{model}: {reply}"
			);
			assert_eq!(output[1]["type"], "function_call", "{model}: {reply}");
			assert_eq!(output[1]["name"], "run_sql");
			assert_eq!(output[1]["status"], "completed");
			let arguments: Value =
				serde_json::from_str(output[1]["arguments"].as_str().unwrap()).unwrap();
			assert_eq!(arguments["sql"], "select 1", "{model}: {reply}");
			assert!(output[1]["call_id"].as_str().unwrap().starts_with("tool"));
			s.call().await;

			body["stream"] = json!(true);
			let text = s
				.post("/ai/v1/responses", body.clone())
				.send()
				.await
				.unwrap()
				.text()
				.await
				.unwrap();
			let events = sse_events(&text);
			let names: Vec<&str> = events.iter().map(|(n, _)| n.as_str()).collect();
			assert!(
				names.contains(&"response.function_call_arguments.delta"),
				"{model}: {text}"
			);
			assert!(
				names.contains(&"response.function_call_arguments.done"),
				"{model}: {text}"
			);
			let (last, data) = events.last().unwrap();
			assert_eq!(last, "response.completed", "{model}: {text}");
			let items = data["response"]["output"].as_array().unwrap();
			assert_eq!(items[0]["type"], "message", "{model}: {text}");
			assert_eq!(items[1]["type"], "function_call", "{model}: {text}");
			assert_eq!(items[1]["output_index"], Value::Null);
			let arguments: Value =
				serde_json::from_str(items[1]["arguments"].as_str().unwrap()).unwrap();
			assert_eq!(arguments["sql"], "select 1", "{model}: {text}");
			s.call().await;

			// A freeform tool of the same name comes back as what the caller declared.
			body["stream"] = json!(false);
			body["tools"] = json!([{"type": "custom", "name": "run_sql", "description": "Runs."}]);
			let reply: Value = s
				.post("/ai/v1/responses", body)
				.send()
				.await
				.unwrap()
				.json()
				.await
				.unwrap();
			assert_eq!(
				reply["output"][1]["type"], "custom_tool_call",
				"{model}: {reply}"
			);
			assert!(
				reply["output"][1]["input"]
					.as_str()
					.unwrap()
					.contains("select 1"),
				"the whole arguments, for a call that did not use the string: {reply}"
			);
			s.call().await;
		}
		let sent = s.fake.last().body.to_string();
		assert!(sent.contains("run_sql"), "{sent}");
	}

	#[tokio::test]
	async fn a_cut_or_failed_responses_stream_ends_in_response_failed() {
		let mut s = Setup::new().await;
		for model in [
			"claude-truncated",
			"claude-overloaded",
			"cut",
			"bedrock-truncated",
		] {
			let text = s
				.post("/ai/v1/responses", responses_request(model, true))
				.send()
				.await
				.unwrap()
				.text()
				.await
				.unwrap();
			let events = sse_events(&text);
			let (last, data) = events.last().unwrap();
			assert_eq!(last, "response.failed", "{model}: {text}");
			assert_eq!(data["response"]["status"], "failed", "{model}: {text}");
			assert!(
				data["response"]["error"]["message"].is_string(),
				"{model}: {text}"
			);
			assert!(!text.contains("response.completed"), "{model}: {text}");
			assert!(
				matches!(s.call().await.outcome, Outcome::Failed(_)),
				"{model}"
			);
		}
	}

	#[tokio::test]
	async fn responses_refusals_are_openai_s_shape_and_send_nothing() {
		let s = Setup::new().await;
		let refused = |body: Value| {
			let request = s.post("/ai/v1/responses", body);
			async move {
				let reply = request.send().await.unwrap();
				let status = reply.status().as_u16();
				let body: Value = reply.json().await.unwrap();
				(status, body)
			}
		};
		let sent = s.fake.count();
		let message = |body: &Value| body["error"]["message"].as_str().unwrap().to_string();
		let (status, body) = refused(responses_request("nobody", false)).await;
		assert_eq!(status, 404);
		assert_eq!(body["error"]["code"], "model_not_found");
		assert!(body.get("type").is_none(), "OpenAI's shape: {body}");

		let mut named = responses_request("ok", false);
		named["previous_response_id"] = json!("resp_1");
		let (status, body) = refused(named).await;
		assert_eq!(status, 400);
		assert!(message(&body).contains("previous_response_id"), "{body}");

		let mut unknown = responses_request("ok", false);
		unknown["conversation"] = json!("c");
		let (status, body) = refused(unknown).await;
		assert_eq!(status, 400);
		assert!(message(&body).contains("conversation"), "{body}");

		let (status, body) = refused(json!({"model": "ok"})).await;
		assert_eq!(status, 400);
		assert!(message(&body).contains("input"), "{body}");

		assert_eq!(s.fake.count(), sent, "nothing reached a provider");

		let unsigned = s
			.client
			.post(format!("{}/ai/v1/responses", s.base))
			.json(&responses_request("ok", false))
			.send()
			.await
			.unwrap();
		assert_eq!(unsigned.status(), 401);
		let body: Value = unsigned.json().await.unwrap();
		assert_eq!(body["error"]["type"], "authentication_error");
	}

	#[tokio::test]
	async fn a_web_search_on_offer_is_left_off_a_model_that_may_not_use_it() {
		let mut s = Setup::new().await;
		for model in ["ok", "claude", "web-claude", "web-bedrock"] {
			let mut body = responses_request(model, false);
			body["tools"] = json!([
				{"type": "web_search", "external_web_access": false},
				{"type": "function", "name": "shell", "parameters": {"type": "object"}}
			]);
			let reply = s.post("/ai/v1/responses", body).send().await.unwrap();
			assert_eq!(reply.status(), 200, "{model}");
			reply.bytes().await.unwrap();
			s.call().await;
			let sent = s.fake.last().body.to_string();
			assert!(sent.contains("shell"), "{model}: {sent}");
			assert!(!sent.contains("web_search"), "{model}: {sent}");
		}
	}

	#[tokio::test]
	async fn a_model_at_web_level_takes_the_web_tool() {
		let mut s = Setup::new().await;
		let mut body = responses_request("web-ok", false);
		body["tools"] = json!([{"type": "web_search"}]);
		let reply = s.post("/ai/v1/responses", body).send().await.unwrap();
		assert_eq!(reply.status(), 200);
		reply.bytes().await.unwrap();
		s.call().await;
		assert_eq!(s.fake.last().body["tools"][0]["type"], "web_search");
	}

	#[tokio::test]
	async fn auto_and_routes_read_a_response_s_needs() {
		let mut s = Setup::new().await;
		// An image in a tool's output and a schema in `text`: only a model that says it takes both.
		let mut body = responses_request("auto", false);
		body["text"] =
			json!({"format": {"type": "json_schema", "name": "n", "schema": {"type": "object"}}});
		body["input"].as_array_mut().unwrap().extend([
			json!({"type": "function_call", "name": "see", "arguments": "{}", "call_id": "c"}),
			json!({"type": "function_call_output", "call_id": "c", "output": [
				{"type": "input_image", "image_url": "data:image/png;base64,AAAA"}
			]}),
		]);
		let reply = s.post("/ai/v1/responses", body).send().await.unwrap();
		assert_eq!(reply.status(), 200, "{:?}", reply.text().await);
		let call = s.call().await;
		assert_eq!(call.model, "big-tools", "{:?}", call.routing);
	}

	#[tokio::test]
	async fn a_hard_budget_caps_a_response_in_its_own_field() {
		let budgets = Budgets::default();
		budgets.replace(vec![crate::budgets::Budget {
			id: KeyId([9; 16]),
			name: "project".into(),
			scope: crate::who::Scope::Project,
			limit: crate::price::PICO,
			period: crate::budgets::Period::Day,
			hard: true,
		}]);
		let mut s = Setup::build(Fake::KEY, Timeouts::default(), None, Some(budgets)).await;
		let reply = s
			.post("/ai/v1/responses", responses_request("ok-priced", false))
			.send()
			.await
			.unwrap();
		assert_eq!(reply.status(), 200);
		reply.bytes().await.unwrap();
		s.call().await;
		assert!(s.fake.last().body["max_tokens"].as_u64().unwrap() > 0);
	}

	#[tokio::test]
	async fn count_tokens_is_the_needs_estimate_and_the_model_list_reads_both_ways() {
		let s = Setup::new().await;
		let request = messages_request("claude", false);
		let bytes = serde_json::to_vec(&request).unwrap();
		let body: Value = s
			.post("/ai/v1/messages/count_tokens", request)
			.send()
			.await
			.unwrap()
			.json()
			.await
			.unwrap();
		assert_eq!(
			body["input_tokens"],
			fits::estimated_tokens(budgets::prompt_bound(&bytes))
		);
		let list: Value = s
			.get("/v1/models")
			.send()
			.await
			.unwrap()
			.json()
			.await
			.unwrap();
		assert_eq!(list["object"], "list", "OpenAI's");
		assert_eq!(list["has_more"], false, "Anthropic's");
		assert_eq!(list["data"][0]["type"], "model");
		assert_eq!(list["data"][0]["object"], "model");
		assert_eq!(list["first_id"], list["data"][0]["id"]);
	}

	#[tokio::test]
	async fn a_web_search_is_on_the_call_s_cost() {
		let mut s = Setup::new().await;
		for stream in [false, true] {
			let mut body = messages_request("claude-search", stream);
			body["tools"] =
				json!([{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]);
			let reply = s.post("/ai/v1/messages", body).send().await.unwrap();
			assert_eq!(reply.status(), 200);
			reply.bytes().await.unwrap();
			let call = s.call().await;
			assert_eq!(
				call.usage.unwrap().web_searches,
				Some(2),
				"stream: {stream}"
			);
			assert_eq!(
				call.cost().unwrap().to_string(),
				"0.020000000000",
				"two searches at Anthropic's rate; stream: {stream}"
			);
		}
	}

	#[tokio::test]
	async fn a_budget_holds_room_for_the_searches_a_call_may_make() {
		let budgets = Budgets::default();
		budgets.replace(vec![crate::budgets::Budget {
			id: KeyId([9; 16]),
			name: "project".into(),
			scope: crate::who::Scope::Project,
			limit: crate::price::PICO / 20,
			period: crate::budgets::Period::Day,
			hard: true,
		}]);
		let mut s = Setup::build(Fake::KEY, Timeouts::default(), None, Some(budgets)).await;
		let search = |max_uses: Option<u64>| {
			let mut tool = json!({"type": "web_search_20250305", "name": "web_search"});
			if let Some(uses) = max_uses {
				tool["max_uses"] = json!(uses);
			}
			let mut body = messages_request("claude-search", false);
			body["tools"] = json!([tool]);
			body
		};
		// Ten cents of searches against five cents of budget, sent nowhere.
		let sent = s.fake.count();
		let refused = s
			.post("/ai/v1/messages", search(Some(10)))
			.send()
			.await
			.unwrap();
		assert_eq!(refused.status(), 402);
		let body: Value = refused.json().await.unwrap();
		assert_eq!(body["error"]["type"], "billing_error");
		assert_eq!(s.fake.count(), sent);
		// No max_uses: it is given the bound, which is what makes the reservation true.
		let refused = s
			.post("/ai/v1/messages", search(None))
			.send()
			.await
			.unwrap();
		assert_eq!(
			refused.status(),
			402,
			"ten searches is the bound, and it does not fit"
		);
		assert_eq!(s.fake.count(), sent);
		let fits = s
			.post("/ai/v1/messages", search(Some(3)))
			.send()
			.await
			.unwrap();
		assert_eq!(fits.status(), 200);
		fits.bytes().await.unwrap();
		assert_eq!(s.fake.last().body["tools"][0]["max_uses"], 3);
		assert_eq!(s.call().await.usage.unwrap().web_searches, Some(2));
	}

	#[test]
	fn the_worst_case_counts_the_searches_and_online_is_bounded_too() {
		let price = serde_json::from_value::<crate::price::PriceEntry>(
			json!({"input": 0, "output": 0, "web_search": 0.01}),
		)
		.unwrap()
		.price()
		.unwrap();
		assert_eq!(worst(&price, 0, 0, 3), 3 * crate::price::PICO / 100);
		let catalogue = Catalogue::from_json(
			r#"{"providers": [{"name": "a", "kind": "anthropic", "base_url": "https://api.anthropic.com/v1"}],
				"models": [{"name": "m", "provider": "a", "internet": "web"}]}"#,
			&HashMap::new(),
			Egress::Public,
		)
		.unwrap();
		let model = catalogue.model("m").unwrap().clone();
		let mut inbound = Inbound::parse(
			br#"{"model": "m", "max_tokens": 1, "messages": [{"role": "user", "content": "x"}]}"#,
		)
		.unwrap();
		assert_eq!(inbound.bound_searches(&model), 0, "no tool, no searches");
		inbound.online = true;
		assert_eq!(inbound.bound_searches(&model), anthropic::SEARCHES_BOUND);
		let body = inbound.upstream(&model, Operation::Messages).unwrap();
		let body: Value = serde_json::from_slice(&body).unwrap();
		assert_eq!(
			body["tools"].as_array().unwrap().len(),
			1,
			"the tool is written out once"
		);
		assert_eq!(body["tools"][0]["max_uses"], anthropic::SEARCHES_BOUND);
	}

	#[tokio::test]
	async fn a_stream_cut_halfway_ends_with_an_error_event() {
		let mut s = Setup::new().await;
		let response = s
			.post("/ai/v1/chat/completions", chat("cut", true))
			.send()
			.await
			.unwrap();
		assert_eq!(response.status(), 200, "the first byte had gone out");
		let body = response.text().await.unwrap();
		let last = data_of(&body).pop().unwrap();
		let error: Value = serde_json::from_str(&last).unwrap();
		assert_eq!(error["error"]["code"], "transport", "{body}");
		assert_eq!(s.call().await.outcome, Outcome::Failed(Class::Transport));
	}

	#[tokio::test]
	async fn a_caller_that_goes_away_cancels_the_provider_call() {
		let mut s = Setup::new().await;
		let mut response = s
			.post("/ai/v1/chat/completions", chat("slow", true))
			.send()
			.await
			.unwrap();
		let first = response.chunk().await.unwrap().unwrap();
		assert!(!first.is_empty());
		drop(response);
		let call = s.call().await;
		assert_eq!(call.outcome, Outcome::Cancelled);
		// The fake sees its stream dropped, long before the 40 chunks it meant to send.
		let deadline = Instant::now() + Duration::from_secs(3);
		while s.fake.cancelled() == 0 {
			assert!(
				Instant::now() < deadline,
				"the provider's stream was never dropped"
			);
			tokio::time::sleep(Duration::from_millis(20)).await;
		}
	}

	/// The chunks of a translated stream, parsed.
	fn chunks(body: &str) -> Vec<Value> {
		data_of(body)
			.iter()
			.filter(|d| *d != "[DONE]")
			.map(|d| serde_json::from_str(d).unwrap())
			.collect()
	}

	fn breakpointed(model: &str) -> Value {
		json!({"model": model, "messages": [
			{"role": "system", "content": [{"type": "text", "text": "Schema.", "cache_control": {"type": "ephemeral"}}]},
			{"role": "user", "content": "hello there", "cache_control": {"type": "ephemeral"}}
		], "tools": [{"type": "function", "cache_control": {"type": "ephemeral"}, "function": {"name": "f"}}]})
	}

	#[tokio::test]
	async fn an_openai_model_gets_a_breakpoint_only_if_it_caches() {
		// A model that does not cache can refuse a request that carries a breakpoint.
		let mut s = Setup::new().await;
		let ok = s
			.post("/ai/v1/chat/completions", breakpointed("ok"))
			.send()
			.await
			.unwrap();
		assert_eq!(ok.status(), 200);
		let seen = s.fake.last().body;
		assert!(!seen.to_string().contains("cache_control"), "{seen}");
		assert_eq!(seen["messages"][0]["content"][0]["text"], "Schema.");
		s.call().await;
		s.post("/ai/v1/chat/completions", breakpointed("cached"))
			.send()
			.await
			.unwrap();
		let seen = s.fake.last().body;
		assert_eq!(
			seen,
			{
				let mut sent = breakpointed("cached");
				sent["model"] = json!("fake-ok");
				sent
			},
			"a model that caches gets the request as sent"
		);
	}

	#[tokio::test]
	async fn an_anthropic_model_answers_in_openai_s_shape() {
		let mut s = Setup::new().await;
		let response = s
			.post("/ai/v1/chat/completions", breakpointed("claude"))
			.send()
			.await
			.unwrap();
		assert_eq!(response.status(), 200);
		assert_eq!(response.headers()["x-psyche-provider"], "anthropic");
		let body: Value = response.json().await.unwrap();
		assert_eq!(body["object"], "chat.completion");
		assert_eq!(
			body["choices"][0]["message"]["content"],
			"Hello from the fake."
		);
		assert_eq!(body["choices"][0]["finish_reason"], "stop");
		assert_eq!(body["usage"]["prompt_tokens"], 129);
		assert_eq!(body["usage"]["cache_read_input_tokens"], 100);
		let seen = s.fake.last();
		assert_eq!(seen.path, "/v1/messages");
		assert_eq!(seen.api_key.as_deref(), Some(Fake::KEY));
		assert_eq!(
			seen.authorization, None,
			"Anthropic takes its key in x-api-key only"
		);
		assert_eq!(seen.anthropic_version.as_deref(), Some(anthropic::VERSION));
		assert_eq!(seen.body["system"][0]["cache_control"]["type"], "ephemeral");
		assert_eq!(seen.body["max_tokens"], anthropic::DEFAULT_MAX_TOKENS);
		let call = s.call().await;
		assert_eq!(call.outcome, Outcome::Completed);
		assert_eq!(call.usage.unwrap().cache_write_tokens, Some(20));

		// A model the catalogue does not mark as caching gets no breakpoint on Anthropic either.
		s.post("/ai/v1/chat/completions", breakpointed("claude-tool"))
			.send()
			.await
			.unwrap();
		assert!(!s.fake.last().body.to_string().contains("cache_control"));
	}

	#[tokio::test]
	async fn an_anthropic_stream_is_translated_as_it_arrives() {
		let mut s = Setup::new().await;
		let mut request = chat("claude-tool", true);
		request["stream_options"] = json!({"include_usage": true});
		let response = s
			.post("/ai/v1/chat/completions", request)
			.send()
			.await
			.unwrap();
		assert_eq!(response.status(), 200);
		let body = response.text().await.unwrap();
		assert!(body.ends_with("data: [DONE]\n\n"), "{body}");
		assert!(body.contains(": ping\n\n"), "keep-alives pass through");
		let chunks = chunks(&body);
		let text: String = chunks
			.iter()
			.filter_map(|c| {
				c.pointer("/choices/0/delta/content")
					.and_then(Value::as_str)
			})
			.collect();
		assert_eq!(text, "Let me look.");
		let arguments: String = chunks
			.iter()
			.filter_map(|c| {
				c.pointer("/choices/0/delta/tool_calls/0/function/arguments")
					.and_then(Value::as_str)
			})
			.collect();
		assert_eq!(
			serde_json::from_str::<Value>(&arguments).unwrap()["sql"],
			"select 1"
		);
		assert!(
			chunks
				.iter()
				.any(|c| c["choices"][0]["finish_reason"] == "tool_calls")
		);
		let usage = chunks.last().unwrap();
		assert_eq!(usage["usage"]["completion_tokens"], 4);
		assert_eq!(s.fake.last().body["stream"], true);
		let call = s.call().await;
		assert_eq!(call.outcome, Outcome::Completed);
		assert_eq!(call.usage.unwrap().prompt_tokens, 129);

		// Without asking, no usage chunk, and the call is still counted.
		let body = s
			.post("/ai/v1/chat/completions", chat("claude", true))
			.send()
			.await
			.unwrap()
			.text()
			.await
			.unwrap();
		assert!(!body.contains("\"usage\""), "{body}");
		assert_eq!(s.call().await.usage.unwrap().completion_tokens, 4);
	}

	#[tokio::test]
	async fn a_target_that_keeps_failing_cools_down_and_a_route_asks_the_healthy_one_first() {
		let mut s = Setup::new().await;
		// Two calls: `fails` fails four times (each 5xx is asked once more), and `ok` answers both.
		for _ in 0..2 {
			let reply = s
				.post("/ai/v1/chat/completions", chat("falls-over", false))
				.send()
				.await
				.unwrap();
			assert_eq!(reply.status(), 200);
			for _ in 0..3 {
				s.call().await;
			}
		}
		// Now cooling: the route asks `ok` first, and `fails` is not asked at all.
		let sent = s.fake.count();
		let reply = s
			.post("/ai/v1/chat/completions", chat("falls-over", false))
			.send()
			.await
			.unwrap();
		assert_eq!(reply.status(), 200);
		assert_eq!(s.fake.count(), sent + 1);
		let call = s.call().await;
		assert_eq!((call.model.as_str(), call.attempt), ("ok", 1));

		// Named alone, a cooling target is still asked: a cooldown never fails a call by itself.
		let alone = s
			.post("/ai/v1/chat/completions", chat("fails", false))
			.send()
			.await
			.unwrap();
		assert_eq!(alone.status(), 503);
		s.call().await;
		s.call().await;

		// The service key reads it; nobody else does.
		let health: Value = s
			.client
			.get(format!("{}/ai/v1/admin/health", s.base))
			.bearer_auth(jwt::sign(&json!({"role": "service_role"}), SECRET))
			.send()
			.await
			.unwrap()
			.json()
			.await
			.unwrap();
		let target = |name: &str| {
			health["data"]
				.as_array()
				.unwrap()
				.iter()
				.find(|t| t["model"] == name)
				.cloned()
				.unwrap()
		};
		let fails = target("fails");
		assert_eq!(fails["state"], "cooling", "{health}");
		assert_eq!(fails["provider"], "fake");
		assert_eq!(fails["failed"], 6);
		assert_eq!(fails["last_failure"], "upstream_error");
		assert!(fails["cooling_until"].as_u64().is_some());
		let ok = target("ok");
		assert_eq!(
			(ok["state"].as_str(), ok["requests"].as_u64()),
			(Some("ok"), Some(3))
		);
		assert!(ok["first_byte_ms"].as_f64().is_some());
		let user = s
			.client
			.get(format!("{}/ai/v1/admin/health", s.base))
			.bearer_auth(jwt::sign(
				&json!({"role": "authenticated", "sub": "u"}),
				SECRET,
			))
			.send()
			.await
			.unwrap();
		assert!(user.status().is_client_error(), "{}", user.status());
	}

	#[tokio::test]
	async fn auto_chooses_the_cheapest_model_whose_facts_fit_and_says_why_in_the_log() {
		let mut s = Setup::new().await;
		let label = |r: &reqwest::Response, h: &str| {
			r.headers().get(h).map(|v| v.to_str().unwrap().to_string())
		};
		// The calls borrow these rather than `s`, which the call log needs mutably.
		let (client, base) = (s.client.clone(), s.base.clone());
		let ask = async |body: Value| {
			let reply = client
				.post(format!("{base}/ai/v1/chat/completions"))
				.bearer_auth(jwt::sign(&json!({"role": "service_role"}), SECRET))
				.json(&body)
				.send()
				.await
				.unwrap();
			let status = reply.status();
			let model = label(&reply, "x-psyche-model");
			let route = label(&reply, "x-psyche-route");
			(status, model, route, reply.text().await.unwrap())
		};

		// A plain question: the cheapest of the three that wrote their facts. Models without facts
		// (`ok`, `ok-cheap`) are never chosen, however cheap.
		let (status, model, route, _) = ask(chat("auto", false)).await;
		assert_eq!(status, 200);
		assert_eq!(model.as_deref(), Some("small-text"));
		assert_eq!(route.as_deref(), Some("auto"));
		let call = s.call().await;
		assert_eq!(call.route.as_deref(), Some("auto"));
		assert_eq!(
			call.routing.as_deref(),
			Some("cheapest first: small-text, mid-images, big-tools")
		);

		// Tools: only the model that takes them, and the log names the two left out and why.
		let mut tools = chat("auto", false);
		tools["tools"] =
			json!([{"type": "function", "function": {"name": "run_sql", "parameters": {}}}]);
		let (status, model, _, _) = ask(tools).await;
		assert_eq!(
			(status.as_u16(), model.as_deref()),
			(200, Some("big-tools"))
		);
		assert_eq!(
			s.call().await.routing.as_deref(),
			Some(
				"cheapest first: big-tools; left out small-text (takes no tools), mid-images (takes no tools)"
			)
		);

		// An image: the cheapest that takes images.
		let image = json!({"model": "auto", "messages": [{"role": "user", "content": [
			{"type": "text", "text": "what is this"},
			{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
		]}]});
		let (status, model, _, _) = ask(image).await;
		assert_eq!(
			(status.as_u16(), model.as_deref()),
			(200, Some("mid-images"))
		);
		s.call().await;

		// A reply longer than the small window leaves the small model out.
		let mut long = chat("auto", false);
		long["max_tokens"] = json!(5000);
		let (status, model, _, _) = ask(long).await;
		assert_eq!(
			(status.as_u16(), model.as_deref()),
			(200, Some("mid-images"))
		);
		let routing = s.call().await.routing.unwrap();
		assert!(routing.contains("small-text (needs about"), "{routing}");

		// The two variants: the cheapest, and the fastest (none measured yet, so the order given).
		let (status, model, route, _) = ask(chat("auto:floor", false)).await;
		assert_eq!(
			(status.as_u16(), model.as_deref(), route.as_deref()),
			(200, Some("small-text"), Some("auto:floor"))
		);
		s.call().await;
		let (status, _, route, _) = ask(chat("auto:nitro", true)).await;
		assert_eq!(
			(status.as_u16(), route.as_deref()),
			(200, Some("auto:nitro"))
		);
		s.call().await;

		// Nothing can take it: refused as the caller's request, with every model's reason, and
		// nothing is sent.
		let sent = s.fake.count();
		let mut impossible = chat("auto", false);
		impossible["max_tokens"] = json!(300_000);
		let (status, _, _, text) = ask(impossible).await;
		assert_eq!(status, 400, "{text}");
		assert!(
			text.contains("None of the models auto stands for can take this request")
				&& text.contains("big-tools needs about"),
			"{text}"
		);
		assert_eq!(s.fake.count(), sent);

		// A plain route checks the same facts: the small model takes no tools, so it is skipped
		// rather than tried and failed.
		let mut routed = chat("small-then-big", false);
		routed["tools"] =
			json!([{"type": "function", "function": {"name": "run_sql", "parameters": {}}}]);
		let (status, model, _, _) = ask(routed).await;
		assert_eq!(
			(status.as_u16(), model.as_deref()),
			(200, Some("big-tools"))
		);
		let call = s.call().await;
		assert_eq!(call.attempt, 1);
		assert_eq!(
			call.routing.as_deref(),
			Some("in the order written: big-tools; left out small-text (takes no tools)")
		);

		// Named alone, a model goes as the caller asked, whatever its facts say.
		let mut alone = chat("small-text", false);
		alone["tools"] =
			json!([{"type": "function", "function": {"name": "run_sql", "parameters": {}}}]);
		let (status, model, route, _) = ask(alone).await;
		assert_eq!(
			(status.as_u16(), model.as_deref(), route),
			(200, Some("small-text"), None)
		);
		assert_eq!(s.call().await.routing, None);
	}

	/// A status, the `x-psyche-model` header and the whole body, without holding `s`.
	async fn asked(s: &Setup, body: Value) -> (u16, Option<String>, Value) {
		let reply = s
			.post("/ai/v1/chat/completions", body)
			.send()
			.await
			.unwrap();
		let status = reply.status().as_u16();
		let model = reply
			.headers()
			.get("x-psyche-model")
			.map(|v| v.to_str().unwrap().to_string());
		let text = reply.text().await.unwrap();
		(
			status,
			model,
			serde_json::from_str(&text).unwrap_or(json!(text)),
		)
	}

	fn with(mut body: Value, field: &str, value: Value) -> Value {
		body[field] = value;
		body
	}

	fn web_tool() -> Value {
		json!([{"type": "web_search_20250305", "name": "web_search"}])
	}

	#[tokio::test]
	async fn a_web_tool_reaches_only_a_model_at_the_web_level() {
		let mut s = Setup::new().await;
		// Named alone at the default level: a 400 that says which and why, and nothing sent.
		let sent = s.fake.count();
		let (status, _, body) = asked(&s, with(chat("ok", false), "tools", web_tool())).await;
		assert_eq!(status, 400, "{body}");
		assert_eq!(body["error"]["code"], "bad_request");
		assert_eq!(
			body["error"]["message"],
			"ok cannot take this request: its internet level is provider, which allows no web tools, and this asks for web_search_20250305."
		);
		assert_eq!(s.fake.count(), sent);

		// A function that is merely called web_search is the caller's own.
		let own =
			json!([{"type": "function", "function": {"name": "web_search", "parameters": {}}}]);
		assert_eq!(
			asked(&s, with(chat("ok", false), "tools", own)).await.0,
			200
		);
		s.call().await;

		// A route leaves it out, with the reason in the log.
		let (status, model, _) = asked(
			&s,
			with(chat("web-first", false), "web_search_options", json!({})),
		)
		.await;
		assert_eq!((status, model.as_deref()), (200, Some("web-ok")));
		let routing = s.call().await.routing.unwrap();
		assert!(
			routing.contains("left out ok (its internet level is provider"),
			"{routing}"
		);
		assert!(s.fake.last().body.get("web_search_options").is_some());

		// `auto` likewise: none of its models has the level, so nothing is sent.
		let sent = s.fake.count();
		let (status, _, body) = asked(&s, with(chat("auto", false), "tools", web_tool())).await;
		assert_eq!(status, 400, "{body}");
		assert_eq!(s.fake.count(), sent);

		// A model at the level gets Anthropic's server tool, translated through as written.
		let (status, _, _) = asked(&s, with(chat("web-claude", false), "tools", web_tool())).await;
		assert_eq!(status, 200);
		s.call().await;
		assert_eq!(s.fake.last().body["tools"], web_tool());

		// `:online` is that provider's own web search, and the suffix never leaves.
		let (status, model, _) = asked(&s, chat("web-claude:online", false)).await;
		assert_eq!((status, model.as_deref()), (200, Some("web-claude")));
		s.call().await;
		let seen = s.fake.last().body;
		assert_eq!(seen["model"], "fake-ok");
		assert_eq!(seen["tools"][0]["name"], "web_search");
		// And on an OpenAI-style server with nothing else to go on, OpenAI's field.
		assert_eq!(asked(&s, chat("web-ok:online", false)).await.0, 200);
		s.call().await;
		assert!(s.fake.last().body.get("web_search_options").is_some());
		// Below the level, `:online` is refused just the same.
		assert_eq!(asked(&s, chat("ok:online", false)).await.0, 400);

		// Bedrock runs no web tool: left out with a reason, never sent without it.
		let (status, model, _) = asked(
			&s,
			with(chat("web-bedrock-first", false), "tools", web_tool()),
		)
		.await;
		assert_eq!((status, model.as_deref()), (200, Some("web-claude")));
		let routing = s.call().await.routing.unwrap();
		assert!(routing.contains("web-bedrock (its provider"), "{routing}");
		let (status, _, body) =
			asked(&s, with(chat("web-bedrock", false), "tools", web_tool())).await;
		assert_eq!(status, 400, "{body}");
		assert!(
			body["error"]["message"]
				.as_str()
				.unwrap()
				.contains("cannot take")
		);
	}

	#[tokio::test]
	async fn a_model_at_level_none_reaches_only_a_private_address() {
		let mut s = Setup::new().await;
		// On a private address under private egress it answers, through the sandboxed client.
		let (status, model, _) = asked(&s, chat("local", false)).await;
		assert_eq!((status, model.as_deref()), (200, Some("local")));
		s.call().await;

		// A public address or a hosted provider: a 400 naming it, and nothing sent.
		let sent = s.fake.count();
		let (status, _, body) = asked(&s, chat("none-public", false)).await;
		assert_eq!(status, 400, "{body}");
		assert_eq!(
			body["error"]["message"],
			"none-public cannot take this request: its internet level is none, and 8.8.8.8 is a public address."
		);
		let (status, _, body) = asked(&s, chat("none-claude", false)).await;
		assert_eq!(status, 400, "{body}");
		assert!(
			body["error"]["message"]
				.as_str()
				.unwrap()
				.contains("its internet level is none")
		);

		// A route leaves it out with the reason in the log; all left out is a 400.
		let (status, model, _) = asked(&s, chat("sandbox-first", false)).await;
		assert_eq!((status, model.as_deref()), (200, Some("ok")));
		let routing = s.call().await.routing.unwrap();
		assert!(
			routing.contains("none-public (its internet level is none"),
			"{routing}"
		);
		let sent_now = s.fake.count();
		assert_eq!(sent_now, sent + 1);
		assert_eq!(asked(&s, chat("sandbox-only", false)).await.0, 400);
		assert_eq!(s.fake.count(), sent_now);
	}

	#[tokio::test]
	async fn openrouter_s_controls_narrow_a_route_and_never_reach_the_provider() {
		let mut s = Setup::new().await;
		let controls = |value: Value| with(chat("cheapest", false), "provider", value);

		// `ignore` leaves one out, and `order` puts the named provider first; neither is sent on.
		let (status, model, _) = asked(&s, controls(json!({"sort": "price"}))).await;
		assert_eq!((status, model.as_deref()), (200, Some("ok-cheap")));
		s.call().await;
		assert!(s.fake.last().body.get("provider").is_none());

		// Nothing the controls leave standing is a 404 with a sentence.
		let (status, _, body) = asked(&s, controls(json!({"only": ["nobody"]}))).await;
		assert_eq!(status, 404, "{body}");
		assert!(
			body["error"]["message"]
				.as_str()
				.unwrap()
				.contains("provider.only")
		);

		// A ceiling below every price, by name.
		let (status, _, body) = asked(&s, controls(json!({"max_price": {"prompt": 0.01}}))).await;
		assert_eq!(status, 404, "{body}");

		// They cannot name anything into the route: only what is already there is tried.
		let (status, model, _) =
			asked(&s, controls(json!({"only": ["fake"], "order": ["fake"]}))).await;
		assert_eq!(status, 200);
		assert!(matches!(model.as_deref(), Some("ok-priced" | "ok-cheap")));
		assert!(
			s.call()
				.await
				.routing
				.unwrap()
				.starts_with("in the caller's provider.order")
		);

		// A promise that cannot be kept is refused by name.
		assert_eq!(asked(&s, controls(json!({"zdr": true}))).await.0, 400);

		// The suffixes: `:floor` on a route is its price order, and an unknown suffix is no model.
		let (status, model, _) = asked(&s, chat("falls-over:floor", false)).await;
		assert_eq!(status, 200, "{model:?}");
		for _ in 0..3 {
			if s.call().await.outcome == Outcome::Completed {
				break;
			}
		}
		assert_eq!(asked(&s, chat("ok:sideways", false)).await.0, 404);

		// `require_parameters`: a target that says nothing about tools cannot be shown to take them.
		let body = with(
			with(
				chat("cheapest", false),
				"provider",
				json!({"require_parameters": true}),
			),
			"tools",
			json!([{"type": "function", "function": {"name": "f", "parameters": {}}}]),
		);
		let (status, _, body) = asked(&s, body).await;
		assert_eq!(status, 404, "{body}");
		assert!(
			body["error"]["message"]
				.as_str()
				.unwrap()
				.contains("require_parameters")
		);
	}

	#[tokio::test]
	async fn a_route_s_constraints_and_weights_are_held() {
		let mut s = Setup::new().await;
		// `ok` says nothing about tools, so the route's requirement leaves it out.
		let (status, model, _) = asked(&s, chat("must-have-tools", false)).await;
		assert_eq!((status, model.as_deref()), (200, Some("big-tools")));
		let routing = s.call().await.routing.unwrap();
		assert!(
			routing.contains("ok (the route requires tools)"),
			"{routing}"
		);

		// Three to one, drawn.
		let mut first = 0;
		for _ in 0..200 {
			let (status, model, _) = asked(&s, chat("only-three-to-one", false)).await;
			assert_eq!(status, 200);
			first += u32::from(model.as_deref() == Some("weighted-a"));
			s.call().await;
		}
		assert!((120..=180).contains(&first), "{first} of 200 at 3 to 1");
	}

	#[tokio::test]
	async fn a_route_goes_round_again_while_time_remains() {
		let mut s = Setup::new().await;
		// Each target fails its first two calls and answers its third: the first pass fails them
		// all, and the second finds one answering.
		let (status, model, _) = asked(&s, chat("flaky", false)).await;
		assert_eq!((status, model.as_deref()), (200, Some("flaky-a")));
		let mut outcomes = Vec::new();
		for _ in 0..5 {
			let call = s.call().await;
			outcomes.push((call.model, call.attempt, call.outcome));
		}
		assert_eq!(outcomes[4], ("flaky-a".to_string(), 5, Outcome::Completed));
		assert_eq!(outcomes[3].2, Outcome::Failed(Class::Upstream));
	}

	#[tokio::test]
	async fn traffic_moves_to_the_other_credential_before_the_first_one_answers_429() {
		let mut s = Setup::new().await;
		// `near` is told it may be sent 8 a minute, and the fake starts refusing at 20: the point is
		// that nothing ever gets that far, on two Bedrock accounts behind one route.
		let mut near = 0;
		for _ in 0..60 {
			let (status, model, _) = asked(&s, chat("spread", false)).await;
			assert_eq!(status, 200);
			near += u32::from(model.as_deref() == Some("near"));
			let call = s.call().await;
			assert_eq!(call.outcome, Outcome::Completed, "no call met a 429");
			assert_eq!(call.attempt, 1, "and none had to fall over");
		}
		assert!(near <= 12, "{near} calls went to the full account");
		assert!(near >= 1);
	}

	/// The cache-read tokens of a whole conversation over `two-caches`, each turn longer.
	async fn cache_reads(s: &mut Setup, forget: bool, opening: &str) -> u64 {
		let mut messages = vec![
			json!({"role": "system", "content": "be brief"}),
			json!({"role": "user", "content": opening}),
		];
		let mut reads = 0;
		for turn in 0..24 {
			if forget {
				s.health.forget_conversations();
			}
			let body = json!({"model": "two-caches", "messages": messages});
			let (status, _, reply) = asked(s, body).await;
			assert_eq!(status, 200, "{reply}");
			reads += reply["usage"]["cache_read_input_tokens"]
				.as_u64()
				.unwrap_or(0);
			s.call().await;
			messages.push(json!({"role": "assistant", "content": format!("answer {turn}")}));
			messages.push(json!({"role": "user", "content": format!("question {turn}")}));
		}
		reads
	}

	#[tokio::test]
	async fn a_conversation_that_stays_on_its_cache_reads_more_of_it() {
		let mut s = Setup::new().await;
		let with_sticking = cache_reads(&mut s, false, "first conversation").await;
		let without = cache_reads(&mut s, true, "second conversation").await;
		assert!(
			with_sticking > without,
			"{with_sticking} tokens read staying, {without} moving"
		);
		// Once a hit is seen it stays: all but the first few turns read their whole prefix.
		let whole: u64 = (0..24).map(|turn: u64| turn * 2 * 100).sum();
		assert!(
			with_sticking * 10 >= whole * 8,
			"{with_sticking} of {whole}"
		);

		// A conversation the caller names is stuck from its first answer.
		let named = json!({"model": "two-caches", "session_id": "s-1", "provider": {},
			"messages": [{"role": "user", "content": "named"}]});
		let (_, first, _) = asked(&s, named.clone()).await;
		s.call().await;
		for _ in 0..20 {
			let (_, again, _) = asked(&s, named.clone()).await;
			assert_eq!(again, first);
			s.call().await;
		}
		assert!(s.fake.last().body.get("session_id").is_none());

		// `provider.order` overrides it.
		let moved = with(named, "provider", json!({"order": ["cache-b"]}));
		let (_, model, _) = asked(&s, moved).await;
		assert_eq!(model.as_deref(), Some("cache-on-b"));
	}

	#[tokio::test]
	async fn a_models_internet_level_is_kept_and_an_unknown_one_refused() {
		let Some((s, _db, _key, _id)) = with_budgets("internet").await else {
			return;
		};
		let service = jwt::sign(&json!({"role": "service_role"}), SECRET);
		let put = async |body: Value| {
			let reply = s
				.client
				.put(format!("{}/ai/v1/admin/models/api-net", s.base))
				.bearer_auth(&service)
				.json(&body)
				.send()
				.await
				.unwrap();
			let status = reply.status().as_u16();
			(status, reply.json::<Value>().await.unwrap())
		};
		let listed = async || -> Value {
			let listed: Value = s
				.client
				.get(format!("{}/ai/v1/admin/models", s.base))
				.bearer_auth(&service)
				.send()
				.await
				.unwrap()
				.json()
				.await
				.unwrap();
			listed["data"]
				.as_array()
				.unwrap()
				.iter()
				.find(|m| m["name"] == "api-net")
				.unwrap()
				.clone()
		};
		let base = json!({"provider": "fake", "upstream": "fake-ok"});
		// A new model is at `provider`.
		let (status, made) = put(base.clone()).await;
		assert_eq!(
			(status, made["internet"].as_str()),
			(200, Some("provider")),
			"{made}"
		);
		assert_eq!(listed().await["internet"], "provider");
		// Written, read back; left out of a later PUT, kept.
		let (status, made) = put(with(base.clone(), "internet", json!("web"))).await;
		assert_eq!(
			(status, made["internet"].as_str()),
			(200, Some("web")),
			"{made}"
		);
		let (status, made) = put(base.clone()).await;
		assert_eq!(
			(status, made["internet"].as_str()),
			(200, Some("web")),
			"{made}"
		);
		assert_eq!(listed().await["internet"], "web");
		assert_eq!(
			put(with(base.clone(), "internet", json!("none"))).await.0,
			200
		);
		assert_eq!(listed().await["internet"], "none");
		// An unknown level, or the old boolean form, is refused.
		assert_eq!(
			put(with(base.clone(), "internet", json!("lots"))).await.0,
			400
		);
		assert_eq!(
			put(with(base.clone(), "internet", json!(true))).await.0,
			400
		);
		assert_eq!(listed().await["internet"], "none");
		// A quota is kept the same way, and `{}` clears it.
		assert_eq!(
			put(with(base.clone(), "quota", json!({"requests": 60})))
				.await
				.0,
			200
		);
		assert_eq!(put(base.clone()).await.0, 200);
		assert_eq!(listed().await["quota"], json!({"requests": 60}));
		assert_eq!(put(with(base, "quota", json!({}))).await.0, 200);
		assert_eq!(listed().await["quota"], Value::Null);
	}

	#[tokio::test]
	async fn a_models_fits_are_kept_and_auto_is_never_a_models_name() {
		let Some((s, _db, _key, _id)) = with_budgets("fits").await else {
			return;
		};
		let service = jwt::sign(&json!({"role": "service_role"}), SECRET);
		let put = async |path: &str, body: Value| {
			let reply = s
				.client
				.put(format!("{}/ai/v1/admin/{path}", s.base))
				.bearer_auth(&service)
				.json(&body)
				.send()
				.await
				.unwrap();
			(reply.status().as_u16(), reply.text().await.unwrap())
		};
		let fits_of = async |name: &str| -> Value {
			let listed: Value = s
				.client
				.get(format!("{}/ai/v1/admin/models", s.base))
				.bearer_auth(&service)
				.send()
				.await
				.unwrap()
				.json()
				.await
				.unwrap();
			listed["data"]
				.as_array()
				.unwrap()
				.iter()
				.find(|m| m["name"] == name)
				.unwrap()["fits"]
				.clone()
		};

		// Written, read back as written; the file's own are in the table too.
		let (status, text) = put(
			"models/api-fit",
			json!({"provider": "fake", "upstream": "fake-ok", "fits": {"context": 8000, "tools": true}}),
		)
		.await;
		assert_eq!(status, 200, "{text}");
		assert_eq!(
			fits_of("api-fit").await,
			json!({"context": 8000, "tools": true, "images": false, "json_schema": false})
		);
		assert_eq!(fits_of("small-text").await["context"], 1000);
		// Left out of a later write, kept; `{}` clears it.
		assert_eq!(
			put(
				"models/api-fit",
				json!({"provider": "fake", "upstream": "fake-ok", "enabled": true})
			)
			.await
			.0,
			200
		);
		assert_eq!(fits_of("api-fit").await["tools"], true);
		assert_eq!(
			put(
				"models/api-fit",
				json!({"provider": "fake", "upstream": "fake-ok", "fits": {}})
			)
			.await
			.0,
			200
		);
		assert_eq!(fits_of("api-fit").await, Value::Null);
		// A fact the API does not know is refused, not ignored.
		assert_eq!(
			put(
				"models/api-fit",
				json!({"provider": "fake", "fits": {"vision": true}})
			)
			.await
			.0,
			400
		);

		// `auto` and its variants are never a model's name; `auto` may be a route, its variants not.
		let (status, text) = put(
			"models/auto",
			json!({"provider": "fake", "upstream": "fake-ok"}),
		)
		.await;
		assert_eq!(status, 400, "{text}");
		assert_eq!(
			put("routes/auto:floor", json!({"targets": ["big-tools"]}))
				.await
				.0,
			400
		);
		assert_eq!(
			put(
				"routes/auto",
				json!({"targets": ["big-tools", "small-text"], "sort": "order"})
			)
			.await
			.0,
			200
		);
	}

	#[tokio::test]
	async fn a_route_falls_over_past_what_failed_and_never_past_the_callers_own_mistake() {
		let mut s = Setup::new().await;
		let label = |r: &reqwest::Response, h: &str| {
			r.headers().get(h).map(|v| v.to_str().unwrap().to_string())
		};

		// The first fails twice (a 5xx is asked once more), and the second answers.
		let reply = s
			.post("/ai/v1/chat/completions", chat("falls-over", false))
			.send()
			.await
			.unwrap();
		assert_eq!(reply.status(), 200);
		assert_eq!(label(&reply, "x-psyche-model").as_deref(), Some("ok"));
		assert_eq!(
			label(&reply, "x-psyche-route").as_deref(),
			Some("falls-over")
		);
		let tried: Vec<(String, u16, Outcome, Option<String>)> =
			[s.call().await, s.call().await, s.call().await]
				.into_iter()
				.map(|c| (c.model, c.attempt, c.outcome, c.route))
				.collect();
		let route = Some("falls-over".to_string());
		assert_eq!(
			tried,
			vec![
				(
					"fails".into(),
					1,
					Outcome::Failed(Class::Upstream),
					route.clone()
				),
				(
					"fails".into(),
					2,
					Outcome::Failed(Class::Upstream),
					route.clone()
				),
				("ok".into(), 3, Outcome::Completed, route),
			],
			"the log rows are the routing's explanation"
		);

		// Sorted by price, the cheaper model answers though it is written second.
		let reply = s
			.post("/ai/v1/chat/completions", chat("cheapest", true))
			.send()
			.await
			.unwrap();
		assert_eq!(label(&reply, "x-psyche-model").as_deref(), Some("ok-cheap"));
		reply.text().await.unwrap();
		assert_eq!(s.call().await.model, "ok-cheap");

		// The caller's own bad request is not another model's to answer.
		let sent = s.fake.count();
		let reply = s
			.post("/ai/v1/chat/completions", chat("picky-first", false))
			.send()
			.await
			.unwrap();
		assert_eq!(reply.status(), 400);
		assert_eq!(s.fake.count(), sent + 1, "the next target was not asked");
		assert_eq!(s.call().await.outcome, Outcome::Failed(Class::Rejected));

		// OpenRouter's `models`: a long retry-after is not waited out here, the next is asked, and
		// the list never reaches the provider.
		let mut body = chat("limited", false);
		body["models"] = json!(["ok"]);
		body["route"] = json!("fallback");
		let reply = s
			.post("/ai/v1/chat/completions", body)
			.send()
			.await
			.unwrap();
		assert_eq!(reply.status(), 200);
		assert_eq!(label(&reply, "x-psyche-route").as_deref(), Some("models"));
		let (first, second) = (s.call().await, s.call().await);
		assert_eq!(
			(
				first.model.as_str(),
				first.outcome,
				second.model.as_str(),
				second.attempt
			),
			("limited", Outcome::Failed(Class::RateLimited), "ok", 2)
		);
		let sent = &s.fake.last().body;
		assert!(
			sent.get("models").is_none() && sent.get("route").is_none(),
			"{sent}"
		);

		// `models` alone names the first to try.
		let reply = s
			.post(
				"/ai/v1/chat/completions",
				json!({"models": ["ok-cheap", "ok"], "messages": [{"role": "user", "content": "hi"}]}),
			)
			.send()
			.await
			.unwrap();
		assert_eq!(label(&reply, "x-psyche-model").as_deref(), Some("ok-cheap"));
		s.call().await;
		let mut body = chat("ok", false);
		body["models"] = json!(["no-such-model"]);
		let reply = s
			.post("/ai/v1/chat/completions", body)
			.send()
			.await
			.unwrap();
		assert_eq!(
			reply.status(),
			404,
			"every name is checked before anything is sent"
		);

		// A route is listed beside the models, with what it tries.
		let listed: Value = s
			.client
			.get(format!("{}/ai/v1/models", s.base))
			.bearer_auth(jwt::sign(&json!({"role": "service_role"}), SECRET))
			.send()
			.await
			.unwrap()
			.json()
			.await
			.unwrap();
		let cheapest = listed["data"]
			.as_array()
			.unwrap()
			.iter()
			.find(|m| m["id"] == "cheapest")
			.cloned()
			.unwrap();
		assert_eq!(cheapest["owned_by"], "route");
		assert_eq!(cheapest["targets"], json!(["ok-cheap", "ok-priced"]));
	}

	#[tokio::test]
	async fn routes_are_rows_the_admin_api_edits_with_no_restart() {
		let Some((s, _db, _key, _id)) = with_budgets("routes").await else {
			return;
		};
		let service = || jwt::sign(&json!({"role": "service_role"}), SECRET);
		let admin = |method: reqwest::Method, path: &str, body: Option<Value>| {
			let request = s
				.client
				.request(method, format!("{}/ai/v1/admin/{path}", s.base))
				.bearer_auth(service());
			match body {
				Some(body) => request.json(&body).send(),
				None => request.send(),
			}
		};
		let put = |name: &str, body: Value| {
			admin(reqwest::Method::PUT, &format!("routes/{name}"), Some(body))
		};
		let call = |model: &str| s.post("/ai/v1/chat/completions", chat(model, false)).send();
		let routes = || async {
			let listed: Value = admin(reqwest::Method::GET, "routes", None)
				.await
				.unwrap()
				.json()
				.await
				.unwrap();
			listed["data"].as_array().unwrap().clone()
		};

		// The file's routes are seeded, and are the file's.
		let seeded = routes().await;
		let cheapest = seeded.iter().find(|r| r["name"] == "cheapest").unwrap();
		assert_eq!(
			(
				cheapest["origin"].clone(),
				cheapest["tried"].clone(),
				cheapest["served"].clone()
			),
			(json!("file"), json!(["ok-cheap", "ok-priced"]), json!(true))
		);
		let (status, _, _) =
			error_of(put("cheapest", json!({"targets": ["ok"]})).await.unwrap()).await;
		assert_eq!(status, 409);

		// A name is a model's or a route's, never both; every target is a model.
		let (status, _, message) =
			error_of(put("ok", json!({"targets": ["ok-cheap"]})).await.unwrap()).await;
		assert_eq!(status, 409, "{message}");
		let (status, _, message) = error_of(
			put("mine", json!({"targets": ["no-such-model"]}))
				.await
				.unwrap(),
		)
		.await;
		assert!(
			status == 400 && message.contains("no-such-model"),
			"{message}"
		);
		let (status, _, _) = error_of(put("mine", json!({"targets": []})).await.unwrap()).await;
		assert_eq!(status, 400);

		// Made, and served at once.
		let made = put("mine", json!({"targets": ["fails", "ok-cheap"]}))
			.await
			.unwrap();
		assert_eq!(made.status(), 200);
		let reply = call("mine").await.unwrap();
		assert_eq!(reply.status(), 200, "no restart");
		assert_eq!(reply.headers()["x-psyche-model"], "ok-cheap");

		// A key may be limited to a route, as to a model.
		let scoped: Value = admin(
			reqwest::Method::POST,
			"keys",
			Some(json!({"name": "route only", "models": ["mine"]})),
		)
		.await
		.unwrap()
		.json()
		.await
		.unwrap();
		let scoped = scoped["key"].as_str().unwrap().to_string();
		let as_key = |model: &str| {
			s.client
				.post(format!("{}/ai/v1/chat/completions", s.base))
				.bearer_auth(&scoped)
				.json(&chat(model, false))
				.send()
		};
		assert_eq!(as_key("mine").await.unwrap().status(), 200);
		assert_eq!(
			as_key("ok-cheap").await.unwrap().status(),
			403,
			"only the route was named"
		);
		let (status, _, _) = error_of(
			admin(
				reqwest::Method::PUT,
				"models/mine",
				Some(json!({"provider": "fake", "upstream": "fake-ok"})),
			)
			.await
			.unwrap(),
		)
		.await;
		assert_eq!(status, 409, "a model may not take a route's name");

		// A target that is not served is skipped and named; the route keeps the rest.
		let off = admin(reqwest::Method::DELETE, "models/ok-cheap", None)
			.await
			.unwrap();
		assert_eq!(off.status(), 200);
		let mine = routes()
			.await
			.into_iter()
			.find(|r| r["name"] == "mine")
			.unwrap();
		assert_eq!(
			(mine["skipped"].clone(), mine["tried"].clone()),
			(json!(["ok-cheap"]), json!(["fails"]))
		);
		assert_eq!(call("mine").await.unwrap().status(), 503);

		let gone = admin(reqwest::Method::DELETE, "routes/mine", None)
			.await
			.unwrap();
		assert_eq!(gone.status(), 204);
		assert_eq!(call("mine").await.unwrap().status(), 404);
	}

	#[tokio::test]
	async fn anthropic_failures() {
		let mut s = Setup::new().await;
		let overloaded = s
			.post("/ai/v1/chat/completions", chat("claude-529", true))
			.send()
			.await
			.unwrap();
		assert_eq!(
			overloaded.status(),
			503,
			"Anthropic's 529 is the provider failing"
		);
		let body: Value = overloaded.json().await.unwrap();
		assert_eq!(body["error"]["code"], "upstream_error");
		assert!(
			body["error"]["message"]
				.as_str()
				.unwrap()
				.contains("Overloaded")
		);
		// Asked once more first.
		assert_eq!(s.call().await.outcome, Outcome::Failed(Class::Upstream));
		assert_eq!(s.call().await.attempt, 2);

		for model in ["claude-overloaded", "claude-truncated"] {
			let body = s
				.post("/ai/v1/chat/completions", chat(model, true))
				.send()
				.await
				.unwrap()
				.text()
				.await
				.unwrap();
			let last: Value = serde_json::from_str(&data_of(&body).pop().unwrap()).unwrap();
			assert_eq!(last["error"]["code"], "upstream_error", "{model}: {body}");
			assert_eq!(
				s.call().await.outcome,
				Outcome::Failed(Class::Upstream),
				"{model}"
			);
		}

		let fim = s
			.post(
				"/ai/v1/completions",
				json!({"model": "claude", "prompt": "SELECT "}),
			)
			.send()
			.await
			.unwrap();
		assert_eq!(fim.status(), 400);
		let refused = s
			.post(
				"/ai/v1/chat/completions",
				json!({"model": "claude", "n": 3, "messages": [{"role": "user", "content": "hi"}]}),
			)
			.send()
			.await
			.unwrap();
		assert_eq!(refused.status(), 400);
		assert_eq!(
			s.fake.count(),
			4,
			"nothing untranslatable was sent (the 529 was asked twice)"
		);
	}

	#[tokio::test]
	async fn a_bedrock_model_signed_with_sigv4() {
		// The fake checks the signature against the request as it arrived, so this passes only if
		// what Psyche signs (host and port, the path with `:` encoded, the body) is what it sends.
		let mut s = Setup::new().await;
		let response = s
			.post("/ai/v1/chat/completions", chat("qwen", false))
			.send()
			.await
			.unwrap();
		assert_eq!(response.status(), 200);
		assert_eq!(response.headers()["x-psyche-provider"], "bedrock");
		let body: Value = response.json().await.unwrap();
		assert_eq!(body["model"], "qwen");
		assert_eq!(
			body["choices"][0]["message"]["content"],
			"Hello from the fake."
		);
		assert_eq!(body["choices"][0]["finish_reason"], "stop");
		assert_eq!(body["usage"]["prompt_tokens"], 129);
		assert!(body["id"].as_str().unwrap().starts_with("chatcmpl-"));
		let seen = s.fake.last();
		assert_eq!(seen.path, "/model/fake.qwen%3A0/converse");
		assert!(
			seen.authorization
				.unwrap()
				.starts_with("AWS4-HMAC-SHA256 Credential=AKIDFAKEPROVIDER/")
		);
		assert_eq!(
			seen.body["messages"][0]["content"][0]["text"],
			"hello there"
		);
		assert!(seen.body.get("model").is_none());
		let call = s.call().await;
		assert_eq!(call.outcome, Outcome::Completed);
		assert_eq!(call.usage.unwrap().cached_tokens, Some(100));

		// The same signature made with a secret the provider does not hold is refused, and that is
		// Psyche's credential failing, not the caller's.
		let wrong = s
			.post("/ai/v1/chat/completions", chat("bedrock-wrong", false))
			.send()
			.await
			.unwrap();
		assert_eq!(wrong.status(), 503);
		let body: Value = wrong.json().await.unwrap();
		assert_eq!(body["error"]["code"], "provider_auth");
		assert!(!body.to_string().contains("a-secret-the-fake"));
	}

	#[tokio::test]
	async fn a_bedrock_stream_with_a_tool_call() {
		let mut s = Setup::new().await;
		let mut request = chat("bedrock-tool", true);
		request["stream_options"] = json!({"include_usage": true});
		let response = s
			.post("/ai/v1/chat/completions", request)
			.send()
			.await
			.unwrap();
		assert_eq!(response.status(), 200);
		assert_eq!(response.headers()["content-type"], "text/event-stream");
		let body = response.text().await.unwrap();
		assert!(body.ends_with("data: [DONE]\n\n"), "{body}");
		let chunks = chunks(&body);
		let text: String = chunks
			.iter()
			.filter_map(|c| {
				c.pointer("/choices/0/delta/content")
					.and_then(Value::as_str)
			})
			.collect();
		assert_eq!(text, "Let me look.");
		let arguments: String = chunks
			.iter()
			.filter_map(|c| {
				c.pointer("/choices/0/delta/tool_calls/0/function/arguments")
					.and_then(Value::as_str)
			})
			.collect();
		assert_eq!(
			serde_json::from_str::<Value>(&arguments).unwrap()["sql"],
			"select 1"
		);
		assert!(
			chunks
				.iter()
				.any(|c| c["choices"][0]["finish_reason"] == "tool_calls")
		);
		assert_eq!(chunks.last().unwrap()["usage"]["completion_tokens"], 4);
		let ids: std::collections::HashSet<&str> =
			chunks.iter().filter_map(|c| c["id"].as_str()).collect();
		assert_eq!(ids.len(), 1, "one id for the whole stream");
		let seen = s.fake.last();
		assert_eq!(seen.path, "/model/fake-tool/converse-stream");
		assert_eq!(
			seen.authorization.as_deref(),
			Some(format!("Bearer {}", Fake::KEY).as_str()),
			"a Bedrock API key"
		);
		let call = s.call().await;
		assert_eq!(call.outcome, Outcome::Completed);
		assert_eq!(call.usage.unwrap().prompt_tokens, 129);
	}

	#[tokio::test]
	async fn a_bedrock_model_that_caches_by_itself_never_sees_a_cache_point() {
		// gpt-5.6-luna on Bedrock refused every request carrying a cache point.
		let mut s = Setup::new().await;
		let ok = s
			.post("/ai/v1/chat/completions", breakpointed("luna"))
			.send()
			.await
			.unwrap();
		assert_eq!(ok.status(), 200, "{}", ok.text().await.unwrap());
		assert!(!s.fake.last().body.to_string().contains("cachePoint"));
		s.call().await;
		// The fake refuses one as the real model does: a catalogue that says otherwise is wrong.
		let refused = s
			.post(
				"/ai/v1/chat/completions",
				breakpointed("luna-marked-caching"),
			)
			.send()
			.await
			.unwrap();
		assert_eq!(refused.status(), 400);
		let body: Value = refused.json().await.unwrap();
		assert!(
			body["error"]["message"]
				.as_str()
				.unwrap()
				.contains("prompt caching"),
			"{body}"
		);
		assert_eq!(s.call().await.outcome, Outcome::Failed(Class::Rejected));
	}

	#[tokio::test]
	async fn bedrock_failures() {
		let mut s = Setup::new().await;
		for model in ["bedrock-throttle", "bedrock-unavailable"] {
			let response = s
				.post("/ai/v1/chat/completions", chat(model, false))
				.send()
				.await
				.unwrap();
			assert_eq!(response.status(), 429, "{model}: a limit to route around");
			let body: Value = response.json().await.unwrap();
			assert_eq!(body["error"]["code"], "rate_limited", "{model}");
			// With no retry-after it is asked once more, a moment later.
			let (first, second) = (s.call().await, s.call().await);
			assert_eq!((first.attempt, second.attempt), (1, 2), "{model}");
			assert_eq!(second.outcome, Outcome::Failed(Class::RateLimited));
		}
		for (model, code, class) in [
			(
				"bedrock-stream-throttle",
				"rate_limited",
				Class::RateLimited,
			),
			("bedrock-truncated", "upstream_error", Class::Upstream),
		] {
			let body = s
				.post("/ai/v1/chat/completions", chat(model, true))
				.send()
				.await
				.unwrap()
				.text()
				.await
				.unwrap();
			assert!(body.contains("Hello "), "the reply started: {body}");
			let last: Value = serde_json::from_str(&data_of(&body).pop().unwrap()).unwrap();
			assert_eq!(last["error"]["code"], code, "{model}: {body}");
			assert_eq!(s.call().await.outcome, Outcome::Failed(class), "{model}");
		}
		let fim = s
			.post(
				"/ai/v1/completions",
				json!({"model": "qwen", "prompt": "SELECT "}),
			)
			.send()
			.await
			.unwrap();
		assert_eq!(fim.status(), 400);
	}

	/// The first number of each vector, which the fake makes its text's length.
	fn firsts(body: &Value) -> Vec<f64> {
		body["data"]
			.as_array()
			.unwrap()
			.iter()
			.enumerate()
			.map(|(i, d)| {
				assert_eq!(d["index"], i);
				d["embedding"][0].as_f64().unwrap()
			})
			.collect()
	}

	#[tokio::test]
	async fn embeddings_pass_through_to_an_openai_style_provider() {
		let mut s = Setup::new().await;
		let request = json!({"model": "embed", "input": ["one", "two words"], "encoding_format": "float", "user": "u1"});
		let response = s.post("/ai/v1/embeddings", request).send().await.unwrap();
		assert_eq!(response.status(), 200);
		let body: Value = response.json().await.unwrap();
		assert_eq!(body["model"], "embed");
		assert_eq!(firsts(&body), vec![3.0, 9.0]);
		let seen = s.fake.last();
		assert_eq!(seen.path, "/v1/embeddings");
		assert_eq!(
			seen.body,
			json!({"model": "fake-embed", "input": ["one", "two words"], "encoding_format": "float", "user": "u1"})
		);
		let call = s.call().await;
		assert_eq!(call.operation, Operation::Embeddings);
		assert_eq!(call.usage.unwrap().prompt_tokens, 3);
	}

	#[tokio::test]
	async fn titan_embeds_each_text_in_its_own_signed_call_and_keeps_the_order() {
		let mut s = Setup::new().await;
		let texts: Vec<String> = (1..=20)
			.map(|n| "w ".repeat(n).trim().to_string())
			.collect();
		let response = s
			.post(
				"/ai/v1/embeddings",
				json!({"model": "titan", "input": texts, "dimensions": 3}),
			)
			.send()
			.await
			.unwrap();
		assert_eq!(response.status(), 200);
		assert_eq!(response.headers()["x-psyche-provider"], "bedrock");
		let body: Value = response.json().await.unwrap();
		let expected: Vec<f64> = texts.iter().map(|t| t.len() as f64).collect();
		assert_eq!(firsts(&body), expected);
		assert_eq!(body["data"][0]["embedding"].as_array().unwrap().len(), 3);
		assert_eq!(body["usage"]["prompt_tokens"], 210);
		let seen = s.fake.since(0);
		assert_eq!(seen.len(), 20);
		assert!(seen.iter().all(|r| {
			r.path == "/model/amazon.titan-embed-text-v2%3A0/invoke"
				&& r.authorization
					.as_deref()
					.unwrap_or("")
					.starts_with("AWS4-HMAC-SHA256")
		}));
		assert_eq!(seen[0].body["dimensions"], 3);
		let call = s.call().await;
		assert_eq!(call.outcome, Outcome::Completed);
		assert_eq!(call.usage.unwrap().prompt_tokens, 210);
		// Priced from the shipped catalogue: Titan v2 in us-west-2, $0.02 per million tokens.
		assert_eq!(call.cost().unwrap().to_string(), "0.000004200000");

		// base64 is the vector's numbers as little-endian f32s, OpenAI's form.
		let body: Value = s
			.post(
				"/ai/v1/embeddings",
				json!({"model": "titan", "input": "four", "encoding_format": "base64", "dimensions": 2}),
			)
			.send()
			.await
			.unwrap()
			.json()
			.await
			.unwrap();
		use base64::Engine;
		let bytes = base64::engine::general_purpose::STANDARD
			.decode(body["data"][0]["embedding"].as_str().unwrap())
			.unwrap();
		assert_eq!(
			bytes,
			[4.0f32.to_le_bytes(), 0.25f32.to_le_bytes()].concat()
		);
	}

	#[tokio::test]
	async fn cohere_takes_texts_in_batches_and_its_own_refusals_reach_the_caller() {
		let mut s = Setup::new().await;
		let texts: Vec<String> = (0..100).map(|i| format!("text number {i}")).collect();
		let response = s
			.post(
				"/ai/v1/embeddings",
				json!({"model": "cohere", "input": texts}),
			)
			.send()
			.await
			.unwrap();
		assert_eq!(response.status(), 200);
		let body: Value = response.json().await.unwrap();
		let expected: Vec<f64> = texts.iter().map(|t| t.len() as f64).collect();
		assert_eq!(firsts(&body), expected);
		assert_eq!(body["usage"]["prompt_tokens"], 300);
		let seen = s.fake.since(0);
		assert_eq!(seen.len(), 2);
		assert_eq!(seen[0].path, "/model/cohere.embed-english-v3/invoke");
		assert_eq!(seen[0].body["texts"].as_array().unwrap().len(), 96);
		assert_eq!(seen[0].body["input_type"], "search_document");
		assert_eq!(s.call().await.outcome, Outcome::Completed);

		// v3 has no output_dimension, and says so itself; the caller gets Bedrock's 400.
		let response = s
			.post(
				"/ai/v1/embeddings",
				json!({"model": "cohere", "input": "q", "input_type": "search_query", "dimensions": 256}),
			)
			.send()
			.await
			.unwrap();
		assert_eq!(response.status(), 400);
		let body: Value = response.json().await.unwrap();
		assert!(
			body["error"]["message"]
				.as_str()
				.unwrap()
				.contains("output_dimension")
		);
		assert_eq!(s.fake.last().body["input_type"], "search_query");
		assert_eq!(s.call().await.outcome, Outcome::Failed(Class::Rejected));

		let response = s
			.post(
				"/ai/v1/embeddings",
				json!({"model": "cohere-throttle", "input": "q"}),
			)
			.send()
			.await
			.unwrap();
		assert_eq!(response.status(), 429);
		assert_eq!(s.call().await.outcome, Outcome::Failed(Class::RateLimited));
	}

	#[tokio::test]
	async fn embeddings_that_cannot_be_sent_are_refused_before_anything_is() {
		let s = Setup::new().await;
		for (request, says) in [
			(
				json!({"model": "titan", "input": "a", "stream": true}),
				"not streamed",
			),
			(
				json!({"model": "claude", "input": "a"}),
				"no embeddings API",
			),
			(
				json!({"model": "qwen", "input": "a"}),
				"not an embedding model",
			),
			(json!({"model": "titan", "input": [[1, 2]]}), "Token ids"),
			(
				json!({"model": "titan", "input": "a", "input_type": "search_query"}),
				"Cohere's",
			),
		] {
			let response = s.post("/ai/v1/embeddings", request).send().await.unwrap();
			assert_eq!(response.status(), 400, "{says}");
			let body: Value = response.json().await.unwrap();
			assert!(
				body["error"]["message"].as_str().unwrap().contains(says),
				"{body}"
			);
		}
		assert_eq!(s.fake.count(), 0);
	}
}
