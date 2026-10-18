//! The fake provider: a stand-in for an OpenAI-style provider on 127.0.0.1, so the tests and the
//! bench never spend a token.
//!
//! What it does is chosen by the upstream model a request names:
//!
//! | model | answer |
//! | --- | --- |
//! | `fake-ok` | "Hello from the fake.", whole or streamed (usage included when asked for) |
//! | `fake-slow` | 40 streamed words, 50 ms apart |
//! | `fake-500` | 500, an OpenAI-shaped error |
//! | `fake-429` | 429 with `retry-after: 7` and OpenAI's rate-limit headers |
//! | `fake-400` | 400, the caller's request at fault |
//! | `fake-hang` | nothing, ever |
//! | `fake-cut` | two streamed chunks, 20 ms apart, then the connection drops |
//!
//! `/embeddings` answers any other model with one vector per input, its first number the input's
//! length, so a test can tell the vectors apart and see their order.
//!
//! A request to `…/messages` is answered as Anthropic answers, with its own key header
//! (`x-api-key`) and `anthropic-version` required:
//!
//! | model | answer |
//! | --- | --- |
//! | `fake-ok` | "Hello from the fake.", whole or streamed, with cache reads and writes in the usage |
//! | `fake-tool` | "Let me look." and a `run_sql` tool call, its arguments streamed in two pieces |
//! | `fake-529` | 529, `overloaded_error` |
//! | `fake-overloaded` | a stream that starts, then sends Anthropic's `error` event |
//! | `fake-truncated` | a stream that closes cleanly before `message_stop` |
//!
//! A request to `/model/<id>/converse` or `…/converse-stream` is answered as Bedrock answers. It
//! takes a Bedrock API key ([`Fake::KEY`] as a bearer) or a SigV4 signature made with
//! [`Fake::AWS_SECRET`], which it CHECKS against the request as it arrived (host, path, body), so a
//! test fails if what Psyche signs is not what it sends:
//!
//! A Bedrock id is matched on the name inside it, so `anthropic.fake-tool-v1:0` is `fake-tool`.
//!
//! | model | answer |
//! | --- | --- |
//! | anything else (`fake-ok`, `fake.qwen:0`) | "Hello from the fake.", whole or streamed, cache reads and writes in the usage |
//! | `fake-tool` | "Let me look." and a `run_sql` tool call |
//! | `fake-luna` | 400 `ValidationException` for a request holding a cache point, as OpenAI's models on Bedrock answer one |
//! | `fake-throttle` | 429 `ThrottlingException` |
//! | `fake-unavailable` | 503 `ServiceUnavailableException` |
//! | `fake-stream-throttle` | a stream that starts, then a `throttlingException` |
//! | `fake-truncated` | a stream that closes before `messageStop` |
//!
//! `/model/<id>/invoke` is InvokeModel for embeddings, by family as Bedrock's ids name it: an id
//! holding `titan-embed` takes `inputText` (one text), one holding `cohere.embed` takes `texts` (at
//! most 96) and requires `input_type`, and refuses `output_dimension` if the id holds `-v3`, as
//! Bedrock does. Each vector's first number is its text's length, the token count is the words,
//! in `X-Amzn-Bedrock-Input-Token-Count`. `fake-throttle` and `fake-unavailable` fail as above.
//!
//! A request without `Authorization: Bearer` + [`Fake::KEY`] gets OpenAI's 401, which echoes the
//! key it was given, as OpenAI's does: that is what proves Psyche scrubs its own key from errors.
//! Everything received is kept ([`Fake::last`]), and a stream the client dropped before its end is
//! counted ([`Fake::cancelled`]).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

/// What the fake received.
#[derive(Debug, Clone)]
pub struct Seen {
	pub path: String,
	pub authorization: Option<String>,
	/// Anthropic's key header.
	pub api_key: Option<String>,
	pub anthropic_version: Option<String>,
	pub body: Value,
}

#[derive(Clone, Default)]
struct Shared {
	seen: Arc<Mutex<Vec<Seen>>>,
	cancelled: Arc<AtomicUsize>,
	/// Off for the bench, which sends millions of requests and reads none of them back.
	quiet: bool,
}

pub struct Fake {
	/// `http://127.0.0.1:<port>`; the API is under `/v1`.
	pub origin: String,
	shared: Shared,
}

const REPLY: [&str; 4] = ["Hello ", "from ", "the ", "fake."];

impl Fake {
	/// The key the fake accepts.
	pub const KEY: &'static str = "sk-fake-provider-key-0123456789";
	/// The AWS credentials Bedrock's half checks signatures with.
	pub const AWS_ACCESS_KEY_ID: &'static str = "AKIDFAKEPROVIDER";
	pub const AWS_SECRET: &'static str = "fake-secret-access-key-0123456789";

	pub async fn start() -> Self {
		Self::start_at("127.0.0.1:0").await
	}

	/// On a chosen address, for a harness whose other side is configured before it starts.
	pub async fn start_at(addr: &str) -> Self {
		Self::serve(addr, false).await
	}

	/// `quiet`: keep no record of what was received (the bench, `psyche-bench fake`).
	pub async fn serve(addr: &str, quiet: bool) -> Self {
		let shared = Shared {
			quiet,
			..Shared::default()
		};
		let app = Router::new().fallback(answer).with_state(shared.clone());
		let listener = tokio::net::TcpListener::bind(addr)
			.await
			.expect("a port for the fake provider");
		let origin = format!("http://{}", listener.local_addr().expect("its address"));
		// As Psyche's own listener: a stream's small writes go out as they are made.
		let listener = axum::serve::ListenerExt::tap_io(listener, |stream| {
			let _ = stream.set_nodelay(true);
		});
		tokio::spawn(async move {
			let _ = axum::serve(listener, app).await;
		});
		Self { origin, shared }
	}

	/// The last request received.
	pub fn last(&self) -> Seen {
		self.shared
			.seen
			.lock()
			.expect("the fake's record")
			.last()
			.cloned()
			.expect("the fake received a request")
	}

	/// Every request received after the first `n`.
	pub fn since(&self, n: usize) -> Vec<Seen> {
		self.shared.seen.lock().expect("the fake's record")[n..].to_vec()
	}

	pub fn count(&self) -> usize {
		self.shared.seen.lock().expect("the fake's record").len()
	}

	/// Streams the client dropped before they ended.
	pub fn cancelled(&self) -> usize {
		self.shared.cancelled.load(Ordering::SeqCst)
	}
}

fn error(status: StatusCode, message: &str, kind: &str) -> Response {
	(
		status,
		axum::Json(
			json!({"error": {"message": message, "type": kind, "param": null, "code": null}}),
		),
	)
		.into_response()
}

async fn answer(
	State(shared): State<Shared>,
	uri: Uri,
	headers: HeaderMap,
	body: Bytes,
) -> Response {
	let raw = body;
	let body: Value = serde_json::from_slice(&raw).unwrap_or(Value::Null);
	let header = |name: &str| {
		headers
			.get(name)
			.and_then(|v| v.to_str().ok())
			.map(String::from)
	};
	let authorization = header("authorization");
	if !shared.quiet {
		shared.seen.lock().expect("the fake's record").push(Seen {
			path: uri.path().to_string(),
			authorization: authorization.clone(),
			api_key: header("x-api-key"),
			anthropic_version: header("anthropic-version"),
			body: body.clone(),
		});
	}
	if uri.path().starts_with("/model/") {
		return bedrock(shared, &uri, &headers, &raw, &body);
	}
	if uri.path().ends_with("/messages") {
		return anthropic(shared, &headers, &body);
	}

	let presented = authorization
		.as_deref()
		.and_then(|a| a.strip_prefix("Bearer "))
		.unwrap_or("");
	if presented != Fake::KEY {
		return error(
			StatusCode::UNAUTHORIZED,
			&format!("Incorrect API key provided: {presented}."),
			"invalid_request_error",
		);
	}
	let model = body["model"].as_str().unwrap_or("").to_string();
	let stream = body["stream"].as_bool() == Some(true);
	let usage = body
		.pointer("/stream_options/include_usage")
		.and_then(Value::as_bool)
		== Some(true);
	let chat = uri.path().ends_with("/chat/completions");
	match model.as_str() {
		"fake-500" => error(
			StatusCode::INTERNAL_SERVER_ERROR,
			"The fake fell over.",
			"server_error",
		),
		"fake-429" => {
			let mut response = error(
				StatusCode::TOO_MANY_REQUESTS,
				"Rate limit reached for requests.",
				"requests",
			);
			let headers = response.headers_mut();
			headers.insert(header::RETRY_AFTER, "7".parse().expect("a header"));
			headers.insert(
				"x-ratelimit-remaining-requests",
				"0".parse().expect("a header"),
			);
			headers.insert(
				"x-ratelimit-reset-requests",
				"7s".parse().expect("a header"),
			);
			response
		}
		"fake-400" => error(
			StatusCode::BAD_REQUEST,
			"Invalid value for temperature: must be between 0 and 2.",
			"invalid_request_error",
		),
		"fake-hang" => {
			std::future::pending::<()>().await;
			StatusCode::OK.into_response()
		}
		_ if uri.path().ends_with("/embeddings") => embeddings(&model, &body),
		_ if !stream => whole(&model, chat),
		"fake-slow" => {
			let words: Vec<String> = (0..40).map(|i| format!("w{i} ")).collect();
			streamed(
				shared,
				model,
				chat,
				words,
				usage,
				Duration::from_millis(50),
				false,
			)
		}
		"fake-cut" => streamed(
			shared,
			model,
			chat,
			REPLY.iter().map(|w| w.to_string()).collect(),
			usage,
			// Long enough for the headers and the first chunks to leave before the cut, so the
			// client meets it mid-stream, as a real provider's dropped connection is met.
			Duration::from_millis(20),
			true,
		),
		_ => streamed(
			shared,
			model,
			chat,
			REPLY.iter().map(|w| w.to_string()).collect(),
			usage,
			Duration::ZERO,
			false,
		),
	}
}

/// A vector the test can recognise: its text's length first.
fn vector(text: &str, dimensions: usize) -> Value {
	(0..dimensions.max(1))
		.map(|i| {
			if i == 0 {
				text.len() as f64
			} else {
				i as f64 / 4.0
			}
		})
		.collect()
}

fn words(text: &str) -> u64 {
	text.split_whitespace().count() as u64
}

/// OpenAI's `/embeddings`.
fn embeddings(model: &str, body: &Value) -> Response {
	let texts: Vec<&str> = match &body["input"] {
		Value::String(text) => vec![text],
		Value::Array(items) => items.iter().filter_map(Value::as_str).collect(),
		_ => vec![],
	};
	let data: Vec<Value> = texts
		.iter()
		.enumerate()
		.map(|(i, t)| json!({"object": "embedding", "index": i, "embedding": vector(t, 4)}))
		.collect();
	let tokens: u64 = texts.iter().map(|t| words(t)).sum();
	axum::Json(json!({
		"object": "list", "data": data, "model": model,
		"usage": {"prompt_tokens": tokens, "total_tokens": tokens},
	}))
	.into_response()
}

/// A reply id of its own for every reply, as a provider's are: a gateway that keys its log by the
/// reply's id (one does) would otherwise count a thousand calls as one.
fn reply_id(prefix: &str) -> String {
	static NEXT: AtomicUsize = AtomicUsize::new(0);
	format!("{prefix}-fake-{}", NEXT.fetch_add(1, Ordering::Relaxed))
}

fn usage_of(completion: usize) -> Value {
	json!({"prompt_tokens": 9, "completion_tokens": completion, "total_tokens": 9 + completion})
}

fn whole(model: &str, chat: bool) -> Response {
	let text: String = REPLY.concat();
	let body = if chat {
		json!({
			"id": reply_id("chatcmpl"), "object": "chat.completion", "created": 0, "model": model,
			"choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
			"usage": usage_of(REPLY.len()),
		})
	} else {
		json!({
			"id": reply_id("cmpl"), "object": "text_completion", "created": 0, "model": model,
			"choices": [{"index": 0, "text": "name, email", "finish_reason": "stop"}],
			"usage": usage_of(3),
		})
	};
	axum::Json(body).into_response()
}

/// Counts a stream as cancelled if it is dropped before its last event went out.
struct Watch {
	cancelled: Arc<AtomicUsize>,
	finished: bool,
}

impl Drop for Watch {
	fn drop(&mut self) {
		if !self.finished {
			self.cancelled.fetch_add(1, Ordering::SeqCst);
		}
	}
}

fn streamed(
	shared: Shared,
	model: String,
	chat: bool,
	words: Vec<String>,
	usage: bool,
	gap: Duration,
	cut: bool,
) -> Response {
	let id = reply_id(if chat { "chatcmpl" } else { "cmpl" });
	let chunk = |delta: Value, finish: Value| -> String {
		let body = if chat {
			json!({"id": id, "object": "chat.completion.chunk", "created": 0, "model": model,
				"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
		} else {
			json!({"id": id, "object": "text_completion", "created": 0, "model": model,
				"choices": [{"index": 0, "text": delta.get("content").cloned().unwrap_or(json!("")), "finish_reason": finish}]})
		};
		format!("data: {body}\n\n")
	};
	let mut events = vec![chunk(
		json!({"role": "assistant", "content": ""}),
		Value::Null,
	)];
	for word in &words {
		events.push(chunk(json!({"content": word}), Value::Null));
	}
	events.push(chunk(json!({}), json!("stop")));
	if usage {
		events.push(format!(
			"data: {}\n\n",
			json!({"id": id, "object": "chat.completion.chunk", "created": 0, "model": model,
				"choices": [], "usage": usage_of(words.len())})
		));
	}
	events.push("data: [DONE]\n\n".to_string());
	send(shared, events, gap, cut.then_some(2))
}

/// Sends `events` as a stream, `gap` apart, dropping the connection at event `cut`.
fn send(shared: Shared, events: Vec<String>, gap: Duration, cut: Option<usize>) -> Response {
	let events = events.into_iter().map(String::into_bytes).collect();
	send_bytes(shared, events, gap, cut, "text/event-stream")
}

/// Sends `events` as a stream of `content_type`, `gap` apart, dropping the connection at `cut`.
fn send_bytes(
	shared: Shared,
	events: Vec<Vec<u8>>,
	gap: Duration,
	cut: Option<usize>,
	content_type: &'static str,
) -> Response {
	let total = events.len();
	let watch = Watch {
		cancelled: shared.cancelled.clone(),
		finished: false,
	};
	let body = futures_util::stream::unfold(
		(events.into_iter().enumerate(), watch),
		move |(mut events, mut watch)| async move {
			let (i, event) = events.next()?;
			if cut == Some(i) {
				watch.finished = true;
				return Some((
					Err(std::io::Error::other("the fake cut the stream")),
					(events, watch),
				));
			}
			if i > 0 && !gap.is_zero() {
				tokio::time::sleep(gap).await;
			}
			if i + 1 == total {
				watch.finished = true;
			}
			Some((
				Ok::<Bytes, std::io::Error>(Bytes::from(event)),
				(events, watch),
			))
		},
	);
	(
		StatusCode::OK,
		[(header::CONTENT_TYPE, content_type)],
		Body::from_stream(body),
	)
		.into_response()
}

fn anthropic_error(status: StatusCode, kind: &str, message: &str) -> Response {
	(
		status,
		axum::Json(json!({"type": "error", "error": {"type": kind, "message": message}})),
	)
		.into_response()
}

/// Anthropic's half.
fn anthropic(shared: Shared, headers: &HeaderMap, body: &Value) -> Response {
	let key = headers.get("x-api-key").and_then(|v| v.to_str().ok());
	if key != Some(Fake::KEY) {
		return anthropic_error(
			StatusCode::UNAUTHORIZED,
			"authentication_error",
			"invalid x-api-key",
		);
	}
	if headers.get("anthropic-version").is_none() {
		return anthropic_error(
			StatusCode::BAD_REQUEST,
			"invalid_request_error",
			"anthropic-version: header is required",
		);
	}
	let model = body["model"].as_str().unwrap_or("").to_string();
	let stream = body["stream"].as_bool() == Some(true);
	if model == "fake-529" {
		return anthropic_error(
			StatusCode::from_u16(529).expect("a status"),
			"overloaded_error",
			"Overloaded",
		);
	}
	let tool = model == "fake-tool";
	let usage = json!({"input_tokens": 9, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 20, "output_tokens": 4});
	if !stream {
		let mut content = vec![
			json!({"type": "text", "text": if tool { "Let me look." } else { "Hello from the fake." }}),
		];
		if tool {
			content.push(json!({"type": "tool_use", "id": "toolu_fake", "name": "run_sql", "input": {"sql": "select 1"}}));
		}
		return axum::Json(json!({
			"id": "msg_fake", "type": "message", "role": "assistant", "model": model,
			"content": content, "stop_reason": if tool { "tool_use" } else { "end_turn" },
			"stop_sequence": null, "usage": usage,
		}))
		.into_response();
	}
	let event = |data: Value| {
		format!(
			"event: {}\ndata: {data}\n\n",
			data["type"].as_str().unwrap_or("")
		)
	};
	let mut events = vec![
		event(json!({"type": "message_start", "message": {
			"id": "msg_fake", "type": "message", "role": "assistant", "model": model, "content": [],
			"stop_reason": null, "usage": {"input_tokens": 9, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 20, "output_tokens": 1}
		}})),
		event(
			json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
		),
		event(json!({"type": "ping"})),
	];
	let words: Vec<&str> = if tool {
		vec!["Let me ", "look."]
	} else {
		REPLY.to_vec()
	};
	for word in words {
		events.push(event(json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": word}})));
	}
	if model == "fake-overloaded" {
		events.push(event(
			json!({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}),
		));
		return send(shared, events, Duration::ZERO, None);
	}
	events.push(event(json!({"type": "content_block_stop", "index": 0})));
	if tool {
		events.push(event(json!({"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_fake", "name": "run_sql", "input": {}}})));
		for piece in ["{\"sql\": ", "\"select 1\"}"] {
			events.push(event(json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": piece}})));
		}
		events.push(event(json!({"type": "content_block_stop", "index": 1})));
	}
	if model == "fake-truncated" {
		return send(shared, events, Duration::ZERO, None);
	}
	events.push(event(json!({"type": "message_delta", "delta": {"stop_reason": if tool { "tool_use" } else { "end_turn" }}, "usage": {"output_tokens": 4}})));
	events.push(event(json!({"type": "message_stop"})));
	send(shared, events, Duration::ZERO, None)
}

fn bedrock_error(status: u16, kind: &str, message: &str) -> Response {
	let mut response = (
		StatusCode::from_u16(status).expect("a status"),
		axum::Json(json!({"message": message})),
	)
		.into_response();
	response.headers_mut().insert(
		"x-amzn-errortype",
		format!("{kind}:http://internal.amazon.com/coral/com.amazon.bedrock/")
			.parse()
			.expect("a header"),
	);
	response
}

fn percent_decode(text: &str) -> String {
	let bytes = text.as_bytes();
	let mut out = Vec::with_capacity(bytes.len());
	let mut i = 0;
	while i < bytes.len() {
		if bytes[i] == b'%'
			&& i + 2 < bytes.len()
			&& let Ok(byte) = u8::from_str_radix(&text[i + 1..i + 3], 16)
		{
			out.push(byte);
			i += 3;
			continue;
		}
		out.push(bytes[i]);
		i += 1;
	}
	String::from_utf8_lossy(&out).into_owned()
}

/// InvokeModel for Titan and Cohere embeddings.
fn bedrock_embed(id: &str, body: &Value) -> Response {
	let malformed = |what: &str| {
		bedrock_error(
			400,
			"ValidationException",
			&format!("Malformed input request: {what}, please reformat your input and try again."),
		)
	};
	let dimensions = |name: &str| body[name].as_u64().unwrap_or(4) as usize;
	let (reply, tokens) = if id.contains("titan-embed") {
		let Some(text) = body["inputText"].as_str() else {
			return malformed("#: required key [inputText] not found");
		};
		(
			json!({"embedding": vector(text, dimensions("dimensions")), "inputTextTokenCount": words(text)}),
			words(text),
		)
	} else if id.contains("cohere.embed") {
		if body.get("input_type").is_none() {
			return malformed("#: required key [input_type] not found");
		}
		if body.get("output_dimension").is_some() && id.contains("-v3") {
			return malformed("#: extraneous key [output_dimension] is not permitted");
		}
		let texts: Vec<&str> = body["texts"]
			.as_array()
			.map(|t| t.iter().filter_map(Value::as_str).collect())
			.unwrap_or_default();
		if texts.is_empty() || texts.len() > 96 {
			return malformed("#/texts: expected maximum item count: 96");
		}
		let vectors: Vec<Value> = texts
			.iter()
			.map(|t| vector(t, dimensions("output_dimension")))
			.collect();
		let embeddings = if body.get("embedding_types").is_some() {
			json!({"float": vectors})
		} else {
			Value::Array(vectors)
		};
		(
			json!({"id": "fake", "texts": texts, "embeddings": embeddings, "response_type": "embeddings_by_type"}),
			texts.iter().map(|t| words(t)).sum(),
		)
	} else {
		return bedrock_error(
			400,
			"ValidationException",
			"The provided model identifier is invalid.",
		);
	};
	let mut response = axum::Json(reply).into_response();
	response.headers_mut().insert(
		"x-amzn-bedrock-input-token-count",
		tokens.to_string().parse().expect("a header"),
	);
	response
}

/// Whether the request carries Bedrock's API key or a SigV4 signature that checks out.
fn bedrock_authorized(uri: &Uri, headers: &HeaderMap, raw: &[u8]) -> bool {
	let header = |name: &str| headers.get(name).and_then(|v| v.to_str().ok());
	let Some(authorization) = header("authorization") else {
		return false;
	};
	if authorization.strip_prefix("Bearer ") == Some(Fake::KEY) {
		return true;
	}
	let (Some(host), Some(date)) = (header("host"), header("x-amz-date")) else {
		return false;
	};
	let Some(region) = authorization
		.split("Credential=")
		.nth(1)
		.and_then(|c| c.split('/').nth(2))
	else {
		return false;
	};
	let credentials = crate::sigv4::Credentials {
		access_key_id: Fake::AWS_ACCESS_KEY_ID.into(),
		secret_access_key: crate::providers::Secret::new(Fake::AWS_SECRET),
		session_token: header("x-amz-security-token").map(crate::providers::Secret::new),
	};
	let expected = crate::sigv4::sign(
		&credentials,
		region,
		"bedrock",
		&crate::sigv4::Request {
			method: "POST",
			host,
			path: uri.path(),
			content_type: header("content-type").unwrap_or(""),
			body: raw,
		},
		date,
	);
	expected.authorization == authorization
}

/// Bedrock's half: Converse and ConverseStream.
fn bedrock(shared: Shared, uri: &Uri, headers: &HeaderMap, raw: &[u8], body: &Value) -> Response {
	if !bedrock_authorized(uri, headers, raw) {
		return bedrock_error(
			403,
			"InvalidSignatureException",
			"The request signature we calculated does not match the signature you provided.",
		);
	}
	let mut parts = uri.path().trim_start_matches("/model/").splitn(2, '/');
	let id = percent_decode(parts.next().unwrap_or(""));
	let operation = parts.next().unwrap_or("");
	let stream = operation == "converse-stream";
	// What the fake does is named inside the id, so a test may give it a real family prefix
	// (`anthropic.fake-tool-v1:0`), which a client that reads capabilities from the id needs to
	// believe the model takes tools.
	let model = [
		"fake-throttle",
		"fake-unavailable",
		"fake-luna",
		"fake-tool",
		"fake-stream-throttle",
		"fake-truncated",
	]
	.into_iter()
	.find(|kind| id.contains(kind))
	.unwrap_or("fake-ok")
	.to_string();
	match model.as_str() {
		"fake-throttle" => {
			return bedrock_error(
				429,
				"ThrottlingException",
				"Too many requests, please wait before trying again.",
			);
		}
		"fake-unavailable" => {
			return bedrock_error(
				503,
				"ServiceUnavailableException",
				"Bedrock is unable to process your request.",
			);
		}
		"fake-luna" if body.to_string().contains("cachePoint") => {
			return bedrock_error(
				400,
				"ValidationException",
				"You invoked an unsupported model or your request did not allow prompt caching.",
			);
		}
		_ => {}
	}
	if operation == "invoke" {
		return bedrock_embed(&id, body);
	}
	let tool = model == "fake-tool";
	let text = if tool {
		"Let me look."
	} else {
		"Hello from the fake."
	};
	let usage = json!({"inputTokens": 9, "outputTokens": 4, "totalTokens": 133, "cacheReadInputTokens": 100, "cacheWriteInputTokens": 20});
	let stop = if tool { "tool_use" } else { "end_turn" };
	if !stream {
		let mut content = vec![json!({"text": text})];
		if tool {
			content.push(json!({"toolUse": {"toolUseId": "tooluse_fake", "name": "run_sql", "input": {"sql": "select 1"}}}));
		}
		return axum::Json(json!({
			"output": {"message": {"role": "assistant", "content": content}},
			"stopReason": stop, "usage": usage, "metrics": {"latencyMs": 1},
		}))
		.into_response();
	}
	let event = |kind: &str, payload: Value| {
		crate::eventstream::encode(
			&[
				(":event-type", kind),
				(":content-type", "application/json"),
				(":message-type", "event"),
			],
			payload.to_string().as_bytes(),
		)
	};
	let mut events = vec![event("messageStart", json!({"role": "assistant"}))];
	let words: Vec<&str> = if tool {
		vec!["Let me ", "look."]
	} else {
		REPLY.to_vec()
	};
	for word in words {
		events.push(event(
			"contentBlockDelta",
			json!({"contentBlockIndex": 0, "delta": {"text": word}}),
		));
	}
	if model == "fake-stream-throttle" {
		events.push(crate::eventstream::encode(
			&[
				(":message-type", "exception"),
				(":exception-type", "throttlingException"),
				(":content-type", "application/json"),
			],
			br#"{"message":"Too many tokens, please wait before trying again."}"#,
		));
		return send_bytes(
			shared,
			events,
			Duration::ZERO,
			None,
			"application/vnd.amazon.eventstream",
		);
	}
	events.push(event("contentBlockStop", json!({"contentBlockIndex": 0})));
	if tool {
		events.push(event("contentBlockStart", json!({"contentBlockIndex": 1, "start": {"toolUse": {"toolUseId": "tooluse_fake", "name": "run_sql"}}})));
		for piece in ["{\"sql\": ", "\"select 1\"}"] {
			events.push(event(
				"contentBlockDelta",
				json!({"contentBlockIndex": 1, "delta": {"toolUse": {"input": piece}}}),
			));
		}
		events.push(event("contentBlockStop", json!({"contentBlockIndex": 1})));
	}
	if model != "fake-truncated" {
		events.push(event("messageStop", json!({"stopReason": stop})));
		events.push(event(
			"metadata",
			json!({"usage": usage, "metrics": {"latencyMs": 1}}),
		));
	}
	send_bytes(
		shared,
		events,
		Duration::ZERO,
		None,
		"application/vnd.amazon.eventstream",
	)
}
