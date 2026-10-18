//! The OpenAI-compatible API: `/ai/v1/chat/completions`, `/ai/v1/completions`, `/ai/v1/embeddings`
//! and `/ai/v1/models` (each also at `/v1/…`, for clients whose base URL ends in `/v1`).
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
//!    answered 502, never passed on as if the caller's own key were wrong.

use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::rejection::BytesRejection;
use axum::extract::{DefaultBodyLimit, Path, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post, put};
use futures_util::{Stream, StreamExt, TryStreamExt};
use serde_json::{Map, Value, json};

use crate::budgets::{self, Budgets, Reservation};
use crate::config::Timeouts;
use crate::error::{ApiError, Class, scrub};
use crate::jwt::{self, Role};
use crate::keys::{self, Key, KeyId, Refusal};
use crate::limits::{self, Limits, Taken};
use crate::net;
use crate::price::{Dollars, Price};
use crate::providers::{Catalogue, Kind, Model, Provider};
use crate::sse::{Event, Parser};
use crate::translate::{Out, Translator};
use crate::usage::Usage;
use crate::who::{Members, UserId, Who};
use crate::{anthropic, bedrock, embeddings, sigv4};

/// How much of a provider's error body is read. Errors are a few hundred bytes.
const MAX_ERROR_BODY: usize = 64 * 1024;
/// How much of a non-streamed reply is read. A long reply is well under a megabyte.
const MAX_REPLY_BODY: usize = 64 * 1024 * 1024;
/// How many of one request's embedding calls are in flight at once.
const EMBED_CONCURRENCY: usize = 8;

/// The server: the catalogue, the one outbound client, and the rules for callers.
pub struct Proxy {
	pub catalogue: Catalogue,
	pub http: reqwest::Client,
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
}

impl Operation {
	fn path(self) -> &'static str {
		match self {
			Operation::Chat => "chat/completions",
			Operation::Completions => "completions",
			Operation::Embeddings => "embeddings",
		}
	}

	pub fn name(self) -> &'static str {
		match self {
			Operation::Chat => "chat",
			Operation::Completions => "completions",
			Operation::Embeddings => "embeddings",
		}
	}
}

/// What a caller sent, read only as far as Psyche needs.
#[derive(Debug, Clone)]
pub struct Inbound {
	pub body: Map<String, Value>,
	pub model: String,
	pub stream: bool,
	/// Whether the caller asked for the usage chunk itself.
	pub wants_usage: bool,
}

impl Inbound {
	pub fn parse(bytes: &[u8]) -> Result<Self, ApiError> {
		let value: Value = serde_json::from_slice(bytes).map_err(|e| {
			ApiError::new(
				Class::BadRequest,
				format!("The request is not valid JSON: {e}"),
			)
		})?;
		let Value::Object(body) = value else {
			return Err(ApiError::new(
				Class::BadRequest,
				"The request is a JSON object.",
			));
		};
		let model = match body.get("model") {
			Some(Value::String(model)) if !model.is_empty() => model.clone(),
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
			stream,
			wants_usage,
		})
	}

	/// The body the provider gets, for this operation.
	fn upstream(&self, model: &Model, operation: Operation) -> Result<Vec<u8>, ApiError> {
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
			_ => anthropic::request(&self.body, model, self.stream)?,
		};
		Ok(serde_json::to_vec(&body).unwrap_or_default())
	}

	/// To an OpenAI-style provider: the provider's name for the model, usage asked for, and no
	/// breakpoint for a model that does not cache.
	fn openai(&self, model: &Model) -> Vec<u8> {
		let mut body = self.body.clone();
		body.insert("model".into(), Value::String(model.upstream.clone()));
		if !model.prompt_caching {
			strip_cache_control(&mut body);
		}
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
		["max_completion_tokens", "max_tokens"]
			.iter()
			.find_map(|f| inbound.body.get(*f).and_then(Value::as_u64))
	})
	.or(model.max_output_tokens)
	.unwrap_or(budgets::UNCAPPED_OUTPUT)
}

/// A call's worst-case cost: every prompt token at the dearer of the plain and cache-write prices,
/// and `output` reply tokens.
fn worst(price: &Price, prompt: u64, output: u64) -> u64 {
	let at = |write: bool| {
		price.cost(&Usage {
			prompt_tokens: prompt,
			completion_tokens: output,
			cache_write_tokens: write.then_some(prompt),
			..Usage::default()
		})
	};
	at(false).max(at(true))
}

/// Where a reply cap goes in what the caller sent, so each translation carries it on: OpenAI's own
/// API (and Azure's) refuse `max_tokens` for reasoning models and take `max_completion_tokens`;
/// the legacy completions API and the other OpenAI-style servers take `max_tokens`.
fn cap_field(provider: &Provider, operation: Operation) -> &'static str {
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

/// Removes every `cache_control` breakpoint from a chat request: on a message, on a content part,
/// on a tool (a model that does not cache can answer a request that carries one with an error,
/// whatever the gateway's settings say).
fn strip_cache_control(body: &mut Map<String, Value>) {
	for list in ["messages", "tools"] {
		let Some(Value::Array(items)) = body.get_mut(list) else {
			continue;
		};
		for item in items.iter_mut().filter_map(Value::as_object_mut) {
			item.remove("cache_control");
			if let Some(Value::Array(parts)) = item.get_mut("content") {
				for part in parts.iter_mut().filter_map(Value::as_object_mut) {
					part.remove("cache_control");
				}
			}
		}
	}
}

pub fn router(proxy: Arc<Proxy>) -> Router {
	let api = Router::new()
		.route("/chat/completions", post(chat))
		.route("/completions", post(completions))
		.route("/embeddings", post(embed))
		.route("/models", get(models))
		.route("/admin/keys", get(list_keys).post(create_key))
		.route("/admin/keys/{id}", delete(revoke_key))
		.route("/admin/budgets", get(list_budgets).post(create_budget))
		.route("/admin/budgets/{id}", delete(delete_budget))
		.route("/admin/limits", get(list_limits).post(create_limit))
		.route("/admin/limits/{id}", delete(delete_limit))
		.route("/admin/groups", get(list_groups).post(create_group))
		.route("/admin/groups/{id}", delete(delete_group))
		.route(
			"/admin/members/{user_id}",
			put(set_member).delete(remove_member),
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

async fn models(State(proxy): State<Arc<Proxy>>, headers: HeaderMap) -> Response {
	let caller = match proxy.authorize(&headers) {
		Ok(caller) => caller,
		Err(error) => return error.into_response(),
	};
	let data: Vec<Value> = proxy
		.catalogue
		.models()
		.iter()
		.filter(|m| caller.may_use(&m.name))
		.map(
			|m| json!({"id": m.name, "object": "model", "created": 0, "owned_by": m.provider.name}),
		)
		.collect();
	axum::Json(json!({"object": "list", "data": data})).into_response()
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
		if let Some(unknown) = new
			.models
			.iter()
			.flatten()
			.find(|m| proxy.catalogue.model(m).is_none())
		{
			return Err(ApiError::new(
				Class::BadRequest,
				format!("There is no model named {unknown} here. GET /ai/v1/models lists them."),
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
	#[serde(flatten)]
	whom: Whom,
	#[serde(default = "yes")]
	hard: bool,
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
const BUDGET_FIELDS: &[&str] = &["name", "amount", "period", "days", "hard"];
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
				 key_id::text, every_user, user_id::text, group_id::text \
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
		if !new.amount.is_finite() || !(0.0..1_000_000.0).contains(&new.amount) {
			return Err(ApiError::new(
				Class::BadRequest,
				"amount is US dollars, from 0 to under 1,000,000.",
			));
		}
		let (key, every_user, user, group) = new.whom.columns()?;
		// Six decimals is a millionth of a dollar, finer than any budget needs; numeric takes it
		// exactly from text.
		let amount = format!("{:.6}", new.amount);
		let row = client
			.query_one(
				"INSERT INTO psyche.budgets (name, amount, period, days, hard, key_id, every_user, \
				 user_id, group_id) VALUES ($1, $2::text::numeric, $3, $4, $5, $6, $7, $8, $9) RETURNING id::text",
				&[&new.name.trim(), &amount, &new.period, &new.days, &new.hard, &key, &every_user, &user, &group],
			)
			.await
			.map_err(|e| write_error(&e))?;
		reload_budgets(&proxy, &client).await;
		Ok::<_, ApiError>(merge(
			json!({"id": row.get::<_, String>(0), "name": new.name.trim(), "amount": amount,
				"period": new.period, "days": new.days, "hard": new.hard}),
			new.whom.json(),
		))
	}
	.await;
	match result {
		Ok(made) => (StatusCode::CREATED, axum::Json(made)).into_response(),
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
				let mut error = ApiError::new(Class::Limited, refused.message);
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
		let amount = worst(&price, prompt, output);
		budgets
			.reserve(who, amount, SystemTime::now())
			.map(Some)
			.map_err(|over| ApiError::new(Class::Budget, over.message))
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
	) -> Option<u64> {
		let named = ["max_tokens", "max_completion_tokens"]
			.iter()
			.any(|f| inbound.body.get(*f).is_some_and(|v| !v.is_null()));
		if named || model.max_output_tokens.is_some() {
			return None;
		}
		let room = self.budgets.as_ref()?.room(who, SystemTime::now())?;
		let (price, prompt) = (model.price?, prompt_bound?);
		let base = worst(&price, prompt, 0);
		let per_token = worst(&price, prompt, 1).saturating_sub(base);
		let fits = room
			.saturating_sub(base)
			.checked_div(per_token)
			.unwrap_or(budgets::UNCAPPED_OUTPUT);
		Some(fits.clamp(budgets::MIN_SENT_OUTPUT, budgets::UNCAPPED_OUTPUT))
	}

	/// The caller, from `Authorization: Bearer` or the `apikey` header (the front door sends both):
	/// a Psyche key, the project's service key, or a signed-in user's token, which is taken only
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

	async fn forward(
		&self,
		headers: &HeaderMap,
		body: Result<Bytes, BytesRejection>,
		operation: Operation,
	) -> Result<Response, ApiError> {
		let caller = self.authorize(headers)?;
		let who = caller.who(self.keys.as_ref().map(|k| &k.members));
		let body = body.map_err(|rejection| {
			let mut error = ApiError::new(
				Class::BadRequest,
				format!("The request could not be read: {}", rejection.body_text()),
			);
			error.status = Some(rejection.status());
			error
		})?;
		let mut inbound = Inbound::parse(&body)?;
		let model = self
			.catalogue
			.model(&inbound.model)
			.cloned()
			.ok_or_else(|| {
				ApiError::new(
					Class::ModelNotFound,
					format!(
						"There is no model named {} here. GET /ai/v1/models lists them.",
						inbound.model
					),
				)
			})?;
		if !caller.may_use(&model.name) {
			return Err(ApiError::new(
				Class::Forbidden,
				format!("This key may not use {}.", model.name),
			));
		}
		// The prompt's upper bound, from what the caller sent, for a call a budget or a limit applies
		// to.
		let budgeted = self.budgets.as_ref().is_some_and(|b| b.applies(&who));
		let limited = self.limits.as_ref().is_some_and(|l| l.applies(&who));
		let prompt_bound = (budgeted || limited).then(|| budgets::prompt_bound(&body));
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
						self.reserve(&who, &model, prompt_bound.filter(|_| budgeted), 0)?;
					let (taken, rate) = self.take(&who, prompt_bound.filter(|_| limited), 0)?;
					let warned = warned(&reservation);
					let out = self
						.embed(&inbound, &model, who, reservation, taken)
						.await?;
					return Ok(headed(warn(labelled(out, &model), warned), rate));
				}
			}
		}
		// Under a hard budget a reply is never left uncapped: the provider is told the most it may
		// write, which is what is reserved.
		if budgeted && let Some(cap) = self.sent_cap(&who, &inbound, &model, prompt_bound) {
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
		let reservation = self.reserve(&who, &model, prompt_bound.filter(|_| budgeted), output)?;
		let (taken, rate) = self.take(&who, prompt_bound.filter(|_| limited), output)?;
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
			.http
			.post(url.clone())
			.header(header::CONTENT_TYPE, "application/json");
		request = provider.authorize(request, &url, &body, now());
		request = request.body(body);
		if provider.kind == Kind::Anthropic {
			request = request.header("anthropic-version", anthropic::VERSION);
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
			},
			started,
			reservation,
			taken,
			billable: false,
		};
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

		if !response.status().is_success() {
			let error = self.provider_error(response, &model).await;
			return Err(record.fail(error));
		}
		// From here the provider may be producing tokens, so a call that ends with no usage counted
		// is charged its whole reservation.
		record.billable = true;
		let out = if inbound.stream {
			self.stream(response, &model, inbound.wants_usage, record)
		} else {
			self.whole(response, &model, record).await?
		};
		Ok(headed(warn(labelled(out, &model), warned), rate))
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
		taken: Option<Taken>,
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
			},
			started: Instant::now(),
			reservation,
			taken,
			// Each text is its own call, and some may be billed before another fails.
			billable: true,
		};
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
			.http
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
		wants_usage: bool,
		record: Record,
	) -> Response {
		let state = Streaming {
			upstream: Box::pin(response.bytes_stream()),
			parser: Parser::new(),
			translate: match model.provider.kind {
				Kind::Openai => None,
				Kind::Anthropic => Some(Translator::anthropic(wants_usage, &model.name, now())),
				Kind::Bedrock => Some(Translator::bedrock(
					wants_usage,
					&model.name,
					&bedrock::new_id(),
					now(),
				)),
			},
			wants_usage,
			idle: self.timeouts.idle,
			secrets: self
				.catalogue
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
		let text = scrub(&String::from_utf8_lossy(&body), &self.catalogue.secrets());
		let said = serde_json::from_str::<Value>(&text)
			.ok()
			.and_then(|v| {
				v.pointer("/error/message")
					.or_else(|| v.get("message"))
					.or_else(|| v.get("Message"))
					.or_else(|| v.get("error"))
					.and_then(Value::as_str)
					.map(String::from)
			})
			.unwrap_or_else(|| text.chars().take(500).collect());
		let (class, message) = match status.as_u16() {
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
		error.retry_after = retry_after;
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
struct Record {
	sink: Option<tokio::sync::mpsc::UnboundedSender<Call>>,
	call: Call,
	started: Instant,
	/// The room held in the call's budgets, settled when this is dropped.
	reservation: Option<Reservation>,
	/// What the call took from its rate limits; the tokens it did not use go back when it ends.
	taken: Option<Taken>,
	/// Whether the provider may have produced anything it bills for: false until a successful
	/// response starts, so a refused or failed request releases its reservation and is charged
	/// nothing.
	billable: bool,
}

impl Record {
	fn finish(&mut self, outcome: Outcome) {
		self.call.outcome = outcome;
	}

	fn fail(mut self, error: ApiError) -> ApiError {
		self.finish(Outcome::Failed(error.class));
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
		if let Some(taken) = self.taken.take() {
			taken.settle(self.call.usage.map(|u| u.total()));
		}
		if let Some(reservation) = self.reservation.take() {
			let charge = match (self.billable, self.call.cost()) {
				(false, _) => Some(0),
				(true, cost) => cost.map(|c| c.0),
			};
			reservation.settle(charge, SystemTime::now());
		}
		if let Some(sink) = &self.sink {
			let _ = sink.send(self.call.clone());
		}
	}
}

type Upstream = Pin<Box<dyn Stream<Item = reqwest::Result<Bytes>> + Send>>;

/// The state of one streamed reply.
struct Streaming {
	upstream: Upstream,
	parser: Parser,
	/// For a provider whose stream is not OpenAI's: what turns it into OpenAI's chunks.
	translate: Option<Translator>,
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
						let out = self.pass(events);
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

	/// The bytes of a translated stream's pieces. The first error ends the stream.
	fn render(&mut self, pieces: Vec<Out>) -> Bytes {
		let mut out = Vec::new();
		let mut failed = None;
		for piece in pieces {
			match piece {
				Out::Chunk(chunk) => out.extend_from_slice(format!("data: {chunk}\n\n").as_bytes()),
				Out::Done => out.extend_from_slice(b"data: [DONE]\n\n"),
				Out::KeepAlive => out.extend_from_slice(b": ping\n\n"),
				Out::Error(class, message) => {
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

	/// Ends the stream with an error event, OpenAI's way.
	fn fail(&mut self, class: Class, message: String) -> Bytes {
		self.done = true;
		self.record.finish(Outcome::Failed(class));
		let error = ApiError::new(class, message).from_provider(&self.record.call.provider);
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
		calls: tokio::sync::mpsc::UnboundedReceiver<Call>,
		client: reqwest::Client,
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
			let catalogue = Catalogue::from_json(
				&format!(
					r#"{{"providers": [
						{{"name": "fake", "kind": "openai", "base_url": "{0}/v1", "api_key_env": "FAKE_KEY"}},
						{{"name": "anthropic", "kind": "anthropic", "base_url": "{0}/v1", "api_key_env": "FAKE_KEY"}},
						{{"name": "bedrock", "kind": "bedrock", "region": "us-west-2", "base_url": "{0}",
							"aws": {{"access_key_id_env": "AWS_AK", "secret_access_key_env": "AWS_SK"}}}},
						{{"name": "bedrock-key", "kind": "bedrock", "region": "us-west-2", "base_url": "{0}", "api_key_env": "FAKE_KEY"}},
						{{"name": "bedrock-wrong", "kind": "bedrock", "region": "us-west-2", "base_url": "{0}",
							"aws": {{"access_key_id_env": "AWS_AK", "secret_access_key_env": "AWS_SK_WRONG"}}}}
					],
					"models": [
						{{"name": "ok", "provider": "fake", "upstream": "fake-ok"}},
						{{"name": "ok-priced", "provider": "fake", "upstream": "fake-ok", "price": {{"input": 1, "output": 2.5}}}},
						{{"name": "cached", "provider": "fake", "upstream": "fake-ok", "prompt_caching": true}},
						{{"name": "claude", "provider": "anthropic", "upstream": "fake-ok", "prompt_caching": true}},
						{{"name": "claude-tool", "provider": "anthropic", "upstream": "fake-tool"}},
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
						{{"name": "slow", "provider": "fake", "upstream": "fake-slow"}},
						{{"name": "fails", "provider": "fake", "upstream": "fake-500"}},
						{{"name": "limited", "provider": "fake", "upstream": "fake-429"}},
						{{"name": "picky", "provider": "fake", "upstream": "fake-400"}},
						{{"name": "hangs", "provider": "fake", "upstream": "fake-hang"}},
						{{"name": "cut", "provider": "fake", "upstream": "fake-cut"}},
						{{"name": "embed", "provider": "fake", "upstream": "fake-embed"}},
						{{"name": "titan", "provider": "bedrock", "upstream": "amazon.titan-embed-text-v2:0", "embeddings": "titan"}},
						{{"name": "cohere", "provider": "bedrock-key", "upstream": "cohere.embed-english-v3", "embeddings": "cohere"}},
						{{"name": "cohere-throttle", "provider": "bedrock", "upstream": "cohere.embed-fake-throttle", "embeddings": "cohere"}}
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
			let proxy = Arc::new(Proxy {
				catalogue,
				http: net::client(Egress::Private).unwrap(),
				jwt_secret: SECRET.into(),
				timeouts,
				max_request_bytes: 1024 * 1024,
				calls: Some(tx),
				keys,
				budgets,
				limits,
			});
			let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
			let base = format!("http://{}", listener.local_addr().unwrap());
			tokio::spawn(async move { axum::serve(listener, router(proxy)).await.unwrap() });
			Self {
				base,
				fake,
				calls,
				client: reqwest::Client::new(),
			}
		}

		fn post(&self, path: &str, body: Value) -> reqwest::RequestBuilder {
			self.client
				.post(format!("{}{path}", self.base))
				.bearer_auth(jwt::sign(&json!({"role": "service_role"}), SECRET))
				.json(&body)
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
				..Default::default()
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
		let (db, url) = crate::log::tests::fresh_database(name).await?;
		let store = keys::Store::default();
		let members = Members::default();
		let budgets = Budgets::default();
		let limits = Limits::default();
		let (ready, loaded) = tokio::sync::oneshot::channel();
		tokio::spawn(keys::watch(
			url.clone(),
			keys::Held {
				keys: store.clone(),
				budgets: Some(budgets.clone()),
				limits: Some(limits.clone()),
				members: members.clone(),
			},
			Some(ready),
		));
		loaded.await.unwrap();
		let s = Setup::build_with(
			Fake::KEY,
			Timeouts::default(),
			Some(Keys {
				store,
				members,
				database_url: url,
			}),
			Some(budgets),
			Some(limits),
		)
		.await;
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

	#[test]
	fn openai_s_own_api_takes_max_completion_tokens_and_the_rest_max_tokens() {
		let provider = |kind: Kind, base: &str| Provider {
			name: "p".into(),
			kind,
			base_url: url::Url::parse(base).unwrap(),
			api_key: None,
			aws: None,
			region: None,
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
			502,
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
		assert_eq!(fails.status(), 502);
		assert_eq!(s.call().await.outcome, Outcome::Failed(Class::Upstream));
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
		assert_eq!(response.status(), 504);
		assert_eq!(s.call().await.outcome, Outcome::Failed(Class::Timeout));
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
	async fn anthropic_failures() {
		let mut s = Setup::new().await;
		let overloaded = s
			.post("/ai/v1/chat/completions", chat("claude-529", true))
			.send()
			.await
			.unwrap();
		assert_eq!(
			overloaded.status(),
			502,
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
		assert_eq!(s.call().await.outcome, Outcome::Failed(Class::Upstream));

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
		assert_eq!(s.fake.count(), 3, "nothing untranslatable was sent");
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
		assert_eq!(wrong.status(), 502);
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
			assert_eq!(s.call().await.outcome, Outcome::Failed(Class::RateLimited));
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
