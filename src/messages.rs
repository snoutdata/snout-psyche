//! Anthropic's Messages API, taken in: `POST /ai/v1/messages`, for a caller written for Anthropic,
//! such as Claude Code with `ANTHROPIC_BASE_URL` pointed at a project.
//!
//! A call goes through the same keys, budgets, limits, routes, `auto`, needs check, log and usage as
//! a chat completion (`proxy.rs`). What differs is the shape, and only at the two ends:
//!
//!  - **To an Anthropic provider it passes through**: `model` becomes the provider's name for it, a
//!    `cache_control` breakpoint is taken out for a model that does not cache, and the reply goes
//!    back as Anthropic wrote it, with the name the caller asked for.
//!  - **To any other provider it is translated.** [`to_chat`] makes a chat completion request, which
//!    the existing translations then carry (as written to an OpenAI-style server, `bedrock.rs` for
//!    Converse). What comes back is a chat completion whichever it was, so one [`reply`] and one
//!    [`Events`] turn it into Anthropic's: Bedrock's stream reaches them as the chunks
//!    `translate.rs` already makes.
//!
//! The rules this file keeps, which are `anthropic.rs`'s in the other direction:
//!  - **What cannot be translated is refused, by name, before anything is sent.** A document, a
//!    server tool's blocks in the history, a field Psyche does not know. A field that only tunes
//!    Anthropic's own service is dropped ([`DROPPED`]).
//!  - **`thinking` is how hard a model tries, not what it answers**, so to a model whose catalogue
//!    entry says nothing about reasoning it is dropped rather than refused, and otherwise goes in
//!    the form that entry names.
//!  - **Only signed thinking goes back to the caller.** Anthropic's API refuses a thinking block
//!    without its signature, so an OpenAI-style server's `reasoning_content` is not presented as one.
//!  - **A tool result's image follows it**, in a user turn of its own, since a chat tool message
//!    holds text.
//!
//! Tool calls are held until the reply ends, then written out in order: OpenAI's protocol may
//! interleave parallel calls and Anthropic's blocks may not, and nothing can use a call before its
//! input is whole. A quiet stream is kept open with `ping`s in the meantime.

use std::collections::BTreeMap;

use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use serde_json::{Map, Value, json};

use crate::error::{ApiError, Class};
use crate::providers::{Kind, Model, Reasoning};
use crate::usage::Usage;

/// Fields that only tune Anthropic's own service, so dropping them changes nothing about the answer
/// a caller is owed. `cache_control` here is the request-wide switch; a breakpoint on a block is
/// carried.
pub const DROPPED: [&str; 6] = [
	"service_tier",
	"speed",
	"inference_geo",
	"context_management",
	"diagnostics",
	"cache_control",
];

/// A `ping` event.
pub const PING: &[u8] = b"event: ping\ndata: {\"type\": \"ping\"}\n\n";

/// A `ping` for every this many tool call pieces held back.
const PING_EVERY: u32 = 32;

fn bad(message: impl Into<String>) -> ApiError {
	ApiError::new(Class::BadRequest, message)
}

/// Anthropic's `error.type` for a failure.
fn error_type(class: Class) -> &'static str {
	match class {
		Class::BadRequest | Class::Rejected => "invalid_request_error",
		Class::Unauthorized => "authentication_error",
		Class::Forbidden => "permission_error",
		Class::ModelNotFound => "not_found_error",
		Class::RateLimited | Class::Limited => "rate_limit_error",
		Class::Budget | Class::ProviderBilling => "billing_error",
		Class::Timeout => "timeout_error",
		_ => "api_error",
	}
}

/// A failure as Anthropic words one: `{"type": "error", "error": {"type": ..., "message": ...}}`,
/// with the `code`, `provider` and `scope` every Psyche error has.
pub fn error_body(error: &ApiError) -> Value {
	let mut body = error.body();
	body["type"] = json!("error");
	body["error"]["type"] = json!(error_type(error.class));
	body
}

/// The status a failure is answered with: the proxy's, except a budget, which is a 402 as Anthropic
/// bills, so that a client does not wait and try again as it would after a 429.
fn error_status(error: &ApiError) -> StatusCode {
	if error.class == Class::Budget {
		StatusCode::PAYMENT_REQUIRED
	} else {
		error.status()
	}
}

/// A failure before anything was sent to the caller.
pub fn error_response(error: ApiError) -> Response {
	let status = error_status(&error);
	let mut body = error_body(&error);
	if status == StatusCode::PAYLOAD_TOO_LARGE {
		body["error"]["type"] = json!("request_too_large");
	}
	let mut response = (status, axum::Json(body)).into_response();
	if let Some(after) = error
		.retry_after
		.as_deref()
		.and_then(|v| HeaderValue::from_str(v).ok())
	{
		response.headers_mut().insert("retry-after", after);
	}
	response
}

fn event(name: &str, data: &Value) -> Vec<u8> {
	format!("event: {name}\ndata: {data}\n\n").into_bytes()
}

/// A failure once the reply has started, Anthropic's way.
pub fn error_event(error: &ApiError) -> Vec<u8> {
	event("error", &error_body(error))
}

/// What the Messages API requires of a request, checked here so every provider is held to it.
pub fn check(body: &Map<String, Value>) -> Result<(), ApiError> {
	if body.get("max_tokens").and_then(Value::as_u64).is_none() {
		return Err(bad("max_tokens: Field required, a whole number."));
	}
	if !body
		.get("messages")
		.and_then(Value::as_array)
		.is_some_and(|m| !m.is_empty())
	{
		return Err(bad("messages: a list of at least one message."));
	}
	Ok(())
}

/// The chat completion request for a Messages request, for a provider that does not speak it.
/// `body` is the caller's, as it came. The reply cap stays `max_tokens`, for the proxy to name as
/// the provider does.
pub fn to_chat(body: &Map<String, Value>, model: &Model) -> Result<Map<String, Value>, ApiError> {
	let mut out = Map::new();
	let mut thinking = None;
	let mut config = None;
	let mut turns = Vec::new();
	for (field, value) in body {
		if value.is_null() || DROPPED.contains(&field.as_str()) {
			continue;
		}
		match field.as_str() {
			"messages" => {}
			"system" => system(value, &mut turns, &|what| bad(format!("system: {what}")))?,
			// `plugins` and `web_search_options` are the proxy's and the needs check's, as on a chat
			// completion.
			"model" | "stream" | "max_tokens" | "temperature" | "top_p" | "top_k" | "plugins"
			| "web_search_options" => {
				out.insert(field.clone(), value.clone());
			}
			"stop_sequences" => {
				out.insert("stop".into(), value.clone());
			}
			"metadata" => {
				if let Some(user) = value.get("user_id").filter(|u| u.is_string()) {
					out.insert("user".into(), user.clone());
				}
			}
			"tools" => {
				out.insert("tools".into(), tools(value, model)?);
			}
			"tool_choice" => tool_choice(value, &mut out)?,
			"thinking" => thinking = Some(value),
			"output_config" => config = Some(value),
			other => {
				return Err(bad(format!(
					"{other} is not translated for {}, so it would be ignored. Leave it out.",
					model.name
				)));
			}
		}
	}
	reasoning(thinking, config, model, &mut out);
	if let Some(format) = config
		.and_then(|c| c.get("format"))
		.filter(|f| !f.is_null())
	{
		out.insert(
			"response_format".into(),
			json!({"type": "json_schema", "json_schema": {
				"name": "output", "schema": format["schema"], "strict": true,
			}}),
		);
	}
	let keeps_thinking = model.provider.kind == Kind::Bedrock && model.reasoning.is_some();
	let messages = body
		.get("messages")
		.and_then(Value::as_array)
		.ok_or_else(|| bad("messages: a list of at least one message."))?;
	for (i, message) in messages.iter().enumerate() {
		let at = |what: &str| bad(format!("messages[{i}]: {what}"));
		match message.get("role").and_then(Value::as_str) {
			Some("user") => user(message, model, &mut turns, &at)?,
			Some("assistant") => turns.push(assistant(message, model, keeps_thinking, &at)?),
			// A system message mid-conversation; one with no content only changes the effort.
			Some("system") => system(&message["content"], &mut turns, &at)?,
			Some(other) => return Err(at(&format!("{other} is not a role Psyche knows."))),
			None => return Err(at("it has no role.")),
		}
	}
	out.insert("messages".into(), Value::Array(turns));
	Ok(out)
}

/// `thinking` and `output_config.effort`, in the form the catalogue names for this model.
fn reasoning(
	thinking: Option<&Value>,
	config: Option<&Value>,
	model: &Model,
	out: &mut Map<String, Value>,
) {
	let Some(form) = model.reasoning else {
		return;
	};
	let kind = thinking.and_then(|t| t.get("type")).and_then(Value::as_str);
	if kind == Some("disabled") {
		return;
	}
	let effort = config
		.and_then(|c| c.get("effort"))
		.and_then(Value::as_str)
		.map(|effort| match effort {
			"xhigh" | "max" => "high",
			other => other,
		})
		.or_else(|| {
			thinking
				.and_then(|t| t.get("budget_tokens"))
				.and_then(Value::as_u64)
				.map(|budget| match budget {
					0..=1024 => "low",
					1025..=2048 => "medium",
					_ => "high",
				})
		});
	let bedrock = model.provider.kind == Kind::Bedrock;
	let as_written = match (kind, form) {
		(Some("enabled"), Reasoning::Budget) => bedrock,
		(Some("adaptive"), Reasoning::Adaptive) => bedrock && effort.is_none(),
		_ => false,
	};
	if as_written && let Some(thinking) = thinking {
		out.insert("thinking".into(), thinking.clone());
	} else if let Some(effort) = effort {
		out.insert("reasoning_effort".into(), json!(effort));
	}
}

/// A text block as a chat content part; `None` for an empty one, which no provider takes.
fn text_part(block: &Value, at: &dyn Fn(&str) -> ApiError) -> Result<Option<Value>, ApiError> {
	let text = block
		.get("text")
		.and_then(Value::as_str)
		.ok_or_else(|| at("a text block has no text."))?;
	if text.is_empty() {
		return Ok(None);
	}
	let mut part = json!({"type": "text", "text": text});
	if let Some(point) = block.get("cache_control").filter(|c| !c.is_null()) {
		part["cache_control"] = point.clone();
	}
	Ok(Some(part))
}

/// An image block as a chat content part.
fn image_part(block: &Value, at: &dyn Fn(&str) -> ApiError) -> Result<Value, ApiError> {
	let source = &block["source"];
	let url = match source["type"].as_str() {
		Some("base64") => format!(
			"data:{};base64,{}",
			source["media_type"].as_str().unwrap_or("image/png"),
			source["data"].as_str().unwrap_or("")
		),
		Some("url") => source["url"].as_str().unwrap_or("").to_string(),
		other => {
			return Err(at(&format!(
				"an image from a {} source is not translated.",
				other.unwrap_or("nameless")
			)));
		}
	};
	let mut part = json!({"type": "image_url", "image_url": {"url": url}});
	if let Some(point) = block.get("cache_control").filter(|c| !c.is_null()) {
		part["cache_control"] = point.clone();
	}
	Ok(part)
}

/// Content parts as a chat message's `content`: a plain string when one text with nothing on it is
/// all there is, which every server takes, and the list otherwise.
fn content_of(mut parts: Vec<Value>) -> Value {
	if parts.len() == 1 && parts[0]["type"] == "text" && parts[0].get("cache_control").is_none() {
		return parts.remove(0)["text"].clone();
	}
	Value::Array(parts)
}

/// A content field as the list of blocks it stands for.
fn blocks_of(content: &Value, at: &dyn Fn(&str) -> ApiError) -> Result<Vec<Value>, ApiError> {
	match content {
		Value::Null => Ok(Vec::new()),
		Value::String(text) => Ok(vec![json!({"type": "text", "text": text})]),
		Value::Array(blocks) => Ok(blocks.clone()),
		_ => Err(at("content is a string or a list of blocks.")),
	}
}

fn system(
	value: &Value,
	turns: &mut Vec<Value>,
	at: &dyn Fn(&str) -> ApiError,
) -> Result<(), ApiError> {
	let mut parts = Vec::new();
	for block in blocks_of(value, at)? {
		if block.get("type").and_then(Value::as_str) != Some("text") {
			return Err(at("a system prompt holds text only."));
		}
		parts.extend(text_part(&block, at)?);
	}
	if !parts.is_empty() {
		turns.push(json!({"role": "system", "content": content_of(parts)}));
	}
	Ok(())
}

fn user(
	message: &Value,
	model: &Model,
	turns: &mut Vec<Value>,
	at: &dyn Fn(&str) -> ApiError,
) -> Result<(), ApiError> {
	let mut parts = Vec::new();
	let mut from_results = Vec::new();
	for block in blocks_of(&message["content"], at)? {
		match block.get("type").and_then(Value::as_str) {
			Some("text") => parts.extend(text_part(&block, at)?),
			Some("image") => parts.push(image_part(&block, at)?),
			// A tool's answer goes directly after the call it answers, ahead of the turn's own words.
			Some("tool_result") => {
				let (result, images) = tool_message(&block, at)?;
				turns.push(result);
				from_results.extend(images);
			}
			other => return Err(untranslated(other, model, at)),
		}
	}
	from_results.extend(parts);
	if !from_results.is_empty() {
		turns.push(json!({"role": "user", "content": content_of(from_results)}));
	}
	Ok(())
}

fn untranslated(kind: Option<&str>, model: &Model, at: &dyn Fn(&str) -> ApiError) -> ApiError {
	at(&format!(
		"a {} block is not translated for {}.",
		kind.unwrap_or("nameless"),
		model.name
	))
}

/// A `tool_result` as a chat tool message, and the images it held.
fn tool_message(
	block: &Value,
	at: &dyn Fn(&str) -> ApiError,
) -> Result<(Value, Vec<Value>), ApiError> {
	let id = block
		.get("tool_use_id")
		.and_then(Value::as_str)
		.ok_or_else(|| at("a tool_result has no tool_use_id."))?;
	let mut parts = Vec::new();
	let mut images = Vec::new();
	for inner in blocks_of(&block["content"], at)? {
		match inner.get("type").and_then(Value::as_str) {
			Some("text") => parts.extend(text_part(&inner, at)?),
			Some("image") => images.push(image_part(&inner, at)?),
			other => {
				return Err(at(&format!(
					"a {} block in a tool_result is not translated.",
					other.unwrap_or("nameless")
				)));
			}
		}
	}
	if parts.is_empty() && !images.is_empty() {
		parts.push(json!({"type": "text", "text": "The result is the image that follows."}));
	}
	if block["is_error"] == true {
		parts.insert(0, json!({"type": "text", "text": "The tool failed: "}));
	}
	if let (Some(point), Some(last)) = (
		block.get("cache_control").filter(|c| !c.is_null()),
		parts.last_mut(),
	) {
		last["cache_control"] = point.clone();
	}
	let content = if parts.iter().any(|p| p.get("cache_control").is_some()) {
		Value::Array(parts)
	} else {
		let text: String = parts
			.iter()
			.filter_map(|p| p["text"].as_str())
			.collect::<Vec<_>>()
			.join("");
		Value::String(text)
	};
	Ok((
		json!({"role": "tool", "tool_call_id": id, "content": content}),
		images,
	))
}

fn assistant(
	message: &Value,
	model: &Model,
	keeps_thinking: bool,
	at: &dyn Fn(&str) -> ApiError,
) -> Result<Value, ApiError> {
	let mut parts = Vec::new();
	let mut calls = Vec::new();
	let mut thinking = Vec::new();
	for block in blocks_of(&message["content"], at)? {
		match block.get("type").and_then(Value::as_str) {
			Some("text") => parts.extend(text_part(&block, at)?),
			Some("tool_use") => calls.push(json!({
				"id": block["id"], "type": "function",
				"function": {
					"name": block["name"],
					"arguments": block.get("input").unwrap_or(&json!({})).to_string(),
				},
			})),
			Some("thinking" | "redacted_thinking") => {
				if keeps_thinking {
					thinking.push(block);
				}
			}
			other => return Err(untranslated(other, model, at)),
		}
	}
	let mut turn = json!({"role": "assistant", "content": Value::Null});
	if !parts.is_empty() {
		turn["content"] = content_of(parts);
	} else if calls.is_empty() {
		turn["content"] = json!("");
	}
	if !calls.is_empty() {
		turn["tool_calls"] = Value::Array(calls);
	}
	if !thinking.is_empty() {
		turn["thinking_blocks"] = Value::Array(thinking);
	}
	Ok(turn)
}

fn tools(value: &Value, model: &Model) -> Result<Value, ApiError> {
	let list = value.as_array().ok_or_else(|| bad("tools is a list."))?;
	let out = list
		.iter()
		.enumerate()
		.map(|(i, tool)| {
			let kind = tool.get("type").and_then(Value::as_str);
			// Anthropic's web tools go as written: the needs check has said this provider can take them,
			// and `bedrock.rs` refuses them by name.
			if kind.is_some_and(crate::fits::is_anthropic_web_tool) {
				return Ok(tool.clone());
			}
			let (Some(name), Some(schema)) = (
				tool.get("name").filter(|n| n.is_string()),
				tool.get("input_schema"),
			) else {
				return Err(bad(format!(
					"tools[{i}] ({}) is run by Anthropic or written for it, and has no form on {}.",
					kind.unwrap_or("a tool with no name or schema"),
					model.name
				)));
			};
			let mut function = json!({"name": name, "parameters": schema});
			if let Some(description) = tool.get("description").filter(|d| d.is_string()) {
				function["description"] = description.clone();
			}
			if tool.get("strict") == Some(&Value::Bool(true)) {
				function["strict"] = Value::Bool(true);
			}
			let mut out = json!({"type": "function", "function": function});
			if let Some(point) = tool.get("cache_control").filter(|c| !c.is_null()) {
				out["cache_control"] = point.clone();
			}
			Ok(out)
		})
		.collect::<Result<Vec<_>, _>>()?;
	Ok(Value::Array(out))
}

fn tool_choice(value: &Value, out: &mut Map<String, Value>) -> Result<(), ApiError> {
	let choice = match value.get("type").and_then(Value::as_str) {
		Some("auto") => json!("auto"),
		Some("any") => json!("required"),
		Some("none") => json!("none"),
		Some("tool") => json!({"type": "function", "function": {"name": value["name"]}}),
		other => {
			return Err(bad(format!(
				"tool_choice {} is not one Psyche knows.",
				other.unwrap_or("with no type")
			)));
		}
	};
	out.insert("tool_choice".into(), choice);
	if value.get("disable_parallel_tool_use") == Some(&Value::Bool(true)) {
		out.insert("parallel_tool_calls".into(), Value::Bool(false));
	}
	Ok(())
}

/// Anthropic's `stop_reason` for a chat completion's `finish_reason`.
fn stop_reason(finish: Option<&str>, called_a_tool: bool) -> &'static str {
	match finish {
		Some("length") => "max_tokens",
		Some("tool_calls" | "function_call") => "tool_use",
		Some("content_filter") => "refusal",
		// Some servers finish a turn of tool calls with `stop`.
		_ if called_a_tool => "tool_use",
		_ => "end_turn",
	}
}

/// What a call used, as Anthropic counts it: `input_tokens` leaves out the cache.
fn usage_of(usage: &Usage) -> Value {
	let read = usage.cached_tokens.unwrap_or(0);
	let write = usage.cache_write_tokens.unwrap_or(0);
	let mut out = json!({
		"input_tokens": usage.prompt_tokens.saturating_sub(read + write),
		"output_tokens": usage.completion_tokens,
	});
	if let Some(read) = usage.cached_tokens {
		out["cache_read_input_tokens"] = json!(read);
	}
	if let Some(write) = usage.cache_write_tokens {
		out["cache_creation_input_tokens"] = json!(write);
	}
	out
}

/// A tool call's arguments as an `input` object; `{}` for what is not one.
fn input_of(arguments: &Value) -> Value {
	arguments
		.as_str()
		.and_then(|a| serde_json::from_str::<Value>(a).ok())
		.filter(Value::is_object)
		.unwrap_or_else(|| json!({}))
}

/// A Messages reply for a chat completion, or what was wrong with it. `usage` is what the proxy
/// read from the provider, whose own shape says more than the chat completion's does.
pub fn reply(chat: &Value, usage: Option<Usage>, model: &str) -> Result<Value, String> {
	let choice = chat
		.pointer("/choices/0")
		.ok_or("the reply has no choices")?;
	let message = &choice["message"];
	let mut content: Vec<Value> = message["thinking_blocks"]
		.as_array()
		.cloned()
		.unwrap_or_default();
	if let Some(text) = message["content"].as_str().filter(|t| !t.is_empty()) {
		content.push(json!({"type": "text", "text": text}));
	}
	let mut called = false;
	for call in message["tool_calls"].as_array().into_iter().flatten() {
		called = true;
		content.push(json!({
			"type": "tool_use", "id": call["id"], "name": call["function"]["name"],
			"input": input_of(&call["function"]["arguments"]),
		}));
	}
	Ok(json!({
		"id": chat.get("id").cloned().unwrap_or(Value::Null),
		"type": "message",
		"role": "assistant",
		"model": model,
		"content": content,
		"stop_reason": stop_reason(choice["finish_reason"].as_str(), called),
		"stop_sequence": null,
		"usage": usage_of(&usage.unwrap_or_default()),
	}))
}

/// A tool call being received.
#[derive(Debug, Default)]
struct Call {
	id: String,
	name: String,
	arguments: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Open {
	Text,
	Thinking,
}

/// One chat completion stream being written as Anthropic's events.
#[derive(Debug)]
pub struct Events {
	model: String,
	started: bool,
	/// The index the next block takes.
	next: u64,
	open: Option<Open>,
	calls: BTreeMap<u64, Call>,
	/// Tool call pieces held back so far, which the `ping`s are counted from.
	held: u32,
	finish: Option<String>,
	usage: Usage,
	done: bool,
	/// Whether the provider's reasoning ends in signed blocks.
	signs: bool,
	called: bool,
}

impl Events {
	/// `model` is the name the caller asked for, which the reply carries.
	pub fn new(model: &str) -> Self {
		Self {
			model: model.to_string(),
			started: false,
			next: 0,
			open: None,
			calls: BTreeMap::new(),
			held: 0,
			finish: None,
			usage: Usage::default(),
			done: false,
			signs: false,
			called: false,
		}
	}

	/// This provider's reasoning ends in signed `thinking_blocks`, so it is shown as it arrives.
	pub fn signed(&mut self) {
		self.signs = true;
	}

	/// Whether the provider said the reply had finished, which a stream that closes without saying
	/// `[DONE]` still may have.
	pub fn finished(&self) -> bool {
		self.done || self.finish.is_some()
	}

	fn start(&mut self, chunk: Option<&Value>, out: &mut Vec<u8>) {
		if self.started {
			return;
		}
		self.started = true;
		let id = chunk
			.and_then(|c| c.get("id"))
			.filter(|id| id.is_string())
			.cloned()
			.unwrap_or_else(|| json!("msg_psyche"));
		out.extend(event(
			"message_start",
			&json!({"type": "message_start", "message": {
				"id": id, "type": "message", "role": "assistant", "model": self.model,
				"content": [], "stop_reason": null, "stop_sequence": null,
				"usage": {"input_tokens": 0, "output_tokens": 0},
			}}),
		));
	}

	fn close(&mut self, out: &mut Vec<u8>) {
		if self.open.take().is_some() {
			out.extend(event(
				"content_block_stop",
				&json!({"type": "content_block_stop", "index": self.next}),
			));
			self.next += 1;
		}
	}

	/// Opens a block of this kind unless one is open already.
	fn open(&mut self, kind: Open, out: &mut Vec<u8>) {
		if self.open == Some(kind) {
			return;
		}
		self.close(out);
		let block = match kind {
			Open::Text => json!({"type": "text", "text": ""}),
			Open::Thinking => json!({"type": "thinking", "thinking": "", "signature": ""}),
		};
		out.extend(event(
			"content_block_start",
			&json!({"type": "content_block_start", "index": self.next, "content_block": block}),
		));
		self.open = Some(kind);
	}

	fn delta(&self, delta: Value, out: &mut Vec<u8>) {
		out.extend(event(
			"content_block_delta",
			&json!({"type": "content_block_delta", "index": self.next, "delta": delta}),
		));
	}

	/// One whole thinking block, signature included, the one form Anthropic's API takes back.
	fn thinking_block(&mut self, block: &Value, out: &mut Vec<u8>) {
		if block["type"] == "redacted_thinking" {
			self.close(out);
			out.extend(event(
				"content_block_start",
				&json!({"type": "content_block_start", "index": self.next, "content_block": block}),
			));
			out.extend(event(
				"content_block_stop",
				&json!({"type": "content_block_stop", "index": self.next}),
			));
			self.next += 1;
			return;
		}
		if self.open != Some(Open::Thinking) {
			self.open(Open::Thinking, out);
			let text = block["thinking"].as_str().unwrap_or("");
			self.delta(json!({"type": "thinking_delta", "thinking": text}), out);
		}
		self.delta(
			json!({"type": "signature_delta", "signature": block["signature"]}),
			out,
		);
		self.close(out);
	}

	/// The events for one chunk.
	pub fn chunk(&mut self, chunk: &Value) -> Vec<u8> {
		let mut out = Vec::new();
		if self.done {
			return out;
		}
		self.start(Some(chunk), &mut out);
		if let Some(usage) = Usage::from_openai(chunk) {
			self.usage = usage;
			// A Bedrock reply through `translate.rs` names the cache under Anthropic's words too.
			if let Some(write) = chunk
				.pointer("/usage/cache_creation_input_tokens")
				.and_then(Value::as_u64)
			{
				self.usage.cache_write_tokens = Some(write);
			}
		}
		let Some(choice) = chunk.pointer("/choices/0") else {
			return out;
		};
		let delta = &choice["delta"];
		if let Some(blocks) = delta["thinking_blocks"].as_array() {
			for block in blocks {
				self.thinking_block(block, &mut out);
			}
		}
		// Only what will end in a signed block is shown as it arrives: a block's text comes first
		// and its signature with `thinking_blocks`, and only a provider that writes those is one
		// whose reasoning Anthropic's API would take back.
		if self.signs
			&& let Some(piece) = delta["reasoning_content"]
				.as_str()
				.filter(|p| !p.is_empty())
		{
			self.open(Open::Thinking, &mut out);
			self.delta(
				json!({"type": "thinking_delta", "thinking": piece}),
				&mut out,
			);
		}
		if let Some(text) = delta["content"].as_str().filter(|t| !t.is_empty()) {
			self.open(Open::Text, &mut out);
			self.delta(json!({"type": "text_delta", "text": text}), &mut out);
		}
		for piece in delta["tool_calls"].as_array().into_iter().flatten() {
			let call = self
				.calls
				.entry(piece["index"].as_u64().unwrap_or(0))
				.or_default();
			if let Some(id) = piece["id"].as_str() {
				call.id = id.to_string();
			}
			if let Some(name) = piece["function"]["name"].as_str() {
				call.name.push_str(name);
			}
			if let Some(arguments) = piece["function"]["arguments"].as_str() {
				call.arguments.push_str(arguments);
			}
			self.held = self.held.wrapping_add(1);
			if self.held.is_multiple_of(PING_EVERY) {
				out.extend_from_slice(PING);
			}
		}
		if let Some(finish) = choice["finish_reason"].as_str() {
			self.finish = Some(finish.to_string());
		}
		out
	}

	/// The end of the reply: what was held back, the stop reason and the usage, once. Usage comes
	/// after the finish reason in both of the streams this reads, so nothing is written before here.
	pub fn end(&mut self) -> Vec<u8> {
		let mut out = Vec::new();
		if self.done {
			return out;
		}
		self.done = true;
		self.start(None, &mut out);
		self.close(&mut out);
		for call in std::mem::take(&mut self.calls).into_values() {
			out.extend(event(
				"content_block_start",
				&json!({"type": "content_block_start", "index": self.next, "content_block": {
					"type": "tool_use", "id": call.id, "name": call.name, "input": {},
				}}),
			));
			let input = if call.arguments.trim().is_empty() {
				"{}".to_string()
			} else {
				call.arguments
			};
			self.delta(
				json!({"type": "input_json_delta", "partial_json": input}),
				&mut out,
			);
			out.extend(event(
				"content_block_stop",
				&json!({"type": "content_block_stop", "index": self.next}),
			));
			self.next += 1;
			self.called = true;
		}
		out.extend(event(
			"message_delta",
			&json!({"type": "message_delta",
				"delta": {
					"stop_reason": stop_reason(self.finish.as_deref(), self.called),
					"stop_sequence": null,
				},
				"usage": usage_of(&self.usage),
			}),
		));
		out.extend(event("message_stop", &json!({"type": "message_stop"})));
		out
	}
}

#[cfg(test)]
mod tests {
	use std::sync::Arc;

	use super::*;
	use crate::providers::Provider;

	fn model(kind: Kind, reasoning: Option<Reasoning>) -> Model {
		Model {
			name: "m".into(),
			provider: Arc::new(Provider {
				name: "p".into(),
				kind,
				base_url: url::Url::parse("https://api.example.com/v1").unwrap(),
				api_key: None,
				aws: None,
				region: None,
				credential: None,
				auth_header: None,
			}),
			upstream: "up".into(),
			prompt_caching: true,
			max_output_tokens: None,
			reasoning,
			embeddings: None,
			price: None,
			fits: None,
			internet: crate::providers::Internet::Web,
			quota: None,
		}
	}

	fn chat(body: Value) -> Result<Map<String, Value>, ApiError> {
		to_chat(body.as_object().unwrap(), &model(Kind::Openai, None))
	}

	fn claude_code() -> Value {
		json!({
			"model": "m", "max_tokens": 1000, "stream": true, "temperature": 1,
			"metadata": {"user_id": "u-1"},
			"system": [
				{"type": "text", "text": "You are an agent.", "cache_control": {"type": "ephemeral"}},
				{"type": "text", "text": "Be brief."}
			],
			"tools": [{"name": "Read", "description": "Reads.", "input_schema": {"type": "object"}, "cache_control": {"type": "ephemeral"}}],
			"tool_choice": {"type": "auto", "disable_parallel_tool_use": true},
			"messages": [
				{"role": "user", "content": "Read a.txt"},
				{"role": "assistant", "content": [
					{"type": "text", "text": "Reading."},
					{"type": "tool_use", "id": "t1", "name": "Read", "input": {"path": "a.txt"}}
				]},
				{"role": "user", "content": [
					{"type": "tool_result", "tool_use_id": "t1", "cache_control": {"type": "ephemeral"}, "content": [{"type": "text", "text": "hello"}]},
					{"type": "text", "text": "And?"}
				]}
			]
		})
	}

	#[test]
	fn a_claude_code_turn_becomes_a_chat_completion_request() {
		let out = Value::Object(chat(claude_code()).unwrap());
		assert_eq!(out["max_tokens"], 1000);
		assert_eq!(out["stream"], true);
		assert_eq!(out["user"], "u-1");
		assert_eq!(out["tool_choice"], "auto");
		assert_eq!(out["parallel_tool_calls"], false);
		let tool = &out["tools"][0];
		assert_eq!(tool["function"]["name"], "Read");
		assert_eq!(tool["function"]["parameters"]["type"], "object");
		assert_eq!(tool["cache_control"]["type"], "ephemeral");
		let messages = out["messages"].as_array().unwrap();
		assert_eq!(messages.len(), 5, "{messages:#?}");
		assert_eq!(messages[0]["role"], "system");
		assert_eq!(
			messages[0]["content"][0]["cache_control"]["type"],
			"ephemeral"
		);
		assert_eq!(
			messages[1],
			json!({"role": "user", "content": "Read a.txt"})
		);
		assert_eq!(messages[2]["content"], "Reading.");
		let call = &messages[2]["tool_calls"][0];
		assert_eq!(call["id"], "t1");
		let arguments: Value =
			serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
		assert_eq!(arguments["path"], "a.txt");
		assert_eq!(
			messages[3]["role"], "tool",
			"the result, right after its call"
		);
		assert_eq!(
			messages[3]["content"][0]["cache_control"]["type"],
			"ephemeral"
		);
		assert_eq!(messages[4], json!({"role": "user", "content": "And?"}));
	}

	#[test]
	fn the_translation_is_one_bedrock_can_carry() {
		let chat = to_chat(
			claude_code().as_object().unwrap(),
			&model(Kind::Bedrock, Some(Reasoning::Budget)),
		)
		.unwrap();
		let mut bedrock_model = model(Kind::Bedrock, Some(Reasoning::Budget));
		bedrock_model.prompt_caching = true;
		// `disable_parallel_tool_use` has no Converse form, and it says so by name.
		let refused = crate::bedrock::request(&chat, &bedrock_model).unwrap_err();
		assert!(
			refused.message.contains("parallel_tool_calls"),
			"{}",
			refused.message
		);
		let mut allowed = chat;
		allowed.remove("parallel_tool_calls");
		allowed.remove("temperature");
		let converse = crate::bedrock::request(&allowed, &bedrock_model).unwrap();
		assert_eq!(converse["messages"][1]["role"], "assistant");
		assert!(
			converse["messages"][1]["content"][1]
				.get("toolUse")
				.is_some()
		);
	}

	#[test]
	fn images_reach_the_model_also_from_a_tool_result() {
		let out = Value::Object(
			chat(json!({"messages": [{"role": "user", "content": [
				{"type": "tool_result", "tool_use_id": "t1", "is_error": true, "content": "boom"},
				{"type": "tool_result", "tool_use_id": "t2", "content": [
					{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}
				]},
				{"type": "image", "source": {"type": "url", "url": "https://x.example/a.png"}},
				{"type": "text", "text": "Well?"}
			]}]}))
			.unwrap(),
		);
		let messages = out["messages"].as_array().unwrap();
		assert_eq!(messages[0]["content"], "The tool failed: boom");
		assert_eq!(messages[1]["role"], "tool");
		assert_eq!(
			messages[1]["content"],
			"The result is the image that follows."
		);
		let parts = &messages[2]["content"];
		assert_eq!(parts[0]["image_url"]["url"], "data:image/png;base64,AAAA");
		assert_eq!(parts[1]["image_url"]["url"], "https://x.example/a.png");
		assert_eq!(parts[2]["text"], "Well?");
	}

	#[test]
	fn what_cannot_be_translated_is_refused_by_name_and_what_only_tunes_is_dropped() {
		let turn = json!([{"role": "user", "content": "hi"}]);
		let refused = |body: Value| chat(body).unwrap_err().message;
		assert!(refused(json!({"messages": turn, "container": {}})).contains("container"));
		assert!(refused(json!({"messages": turn, "mcp_servers": []})).contains("mcp_servers"));
		assert!(
			refused(json!({"messages": [{"role": "user", "content": [
				{"type": "document", "source": {"type": "text", "data": "x"}}
			]}]}))
			.contains("document")
		);
		assert!(
			refused(json!({"messages": [{"role": "assistant", "content": [
				{"type": "server_tool_use", "id": "s", "name": "web_search", "input": {}}
			]}]}))
			.contains("server_tool_use")
		);
		assert!(
			refused(
				json!({"messages": turn, "tools": [{"type": "bash_20250124", "name": "bash"}]})
			)
			.contains("bash_20250124")
		);
		assert!(refused(json!({"messages": [{"content": "hi"}]})).contains("no role"));
		let out = chat(
			json!({"messages": turn, "service_tier": "auto", "speed": "fast",
			"context_management": {"edits": []}, "cache_control": {"type": "ephemeral"}}),
		)
		.unwrap();
		for field in [
			"service_tier",
			"speed",
			"context_management",
			"cache_control",
		] {
			assert!(!out.contains_key(field), "{field}");
		}
		// A web tool goes as written; the needs check is what decides who may have it.
		let out = Value::Object(
			chat(
				json!({"messages": turn, "tools": [{"type": "web_search_20250305", "name": "web_search"}]}),
			)
			.unwrap(),
		);
		assert_eq!(out["tools"][0]["type"], "web_search_20250305");
	}

	#[test]
	fn thinking_goes_in_the_form_the_catalogue_names_or_not_at_all() {
		let body = |thinking: Value| {
			json!({"thinking": thinking, "output_config": {"effort": "xhigh"},
				"messages": [{"role": "user", "content": "hi"}]})
		};
		let of = |kind, form, body: Value| {
			Value::Object(to_chat(body.as_object().unwrap(), &model(kind, form)).unwrap())
		};
		let enabled = json!({"type": "enabled", "budget_tokens": 2000});
		let out = of(Kind::Openai, None, body(enabled.clone()));
		assert!(out.get("thinking").is_none());
		assert!(
			out.get("reasoning_effort").is_none(),
			"says nothing of reasoning: dropped"
		);
		let out = of(
			Kind::Openai,
			Some(Reasoning::Effort),
			body(json!({"type": "adaptive"})),
		);
		assert_eq!(out["reasoning_effort"], "high");
		let out = of(
			Kind::Bedrock,
			Some(Reasoning::Budget),
			body(enabled.clone()),
		);
		assert_eq!(out["thinking"], enabled, "as written");
		let bare = json!({"thinking": {"type": "enabled", "budget_tokens": 1500},
			"messages": [{"role": "user", "content": "hi"}]});
		let out = of(Kind::Openai, Some(Reasoning::Effort), bare);
		assert_eq!(out["reasoning_effort"], "medium");
		let off = json!({"thinking": {"type": "disabled"}, "output_config": {"effort": "low"},
			"messages": [{"role": "user", "content": "hi"}]});
		let out = of(Kind::Openai, Some(Reasoning::Effort), off);
		assert!(out.get("reasoning_effort").is_none());

		let history = json!({"messages": [{"role": "assistant", "content": [
			{"type": "thinking", "thinking": "x", "signature": "s"},
			{"type": "text", "text": "y"}
		]}]});
		assert!(
			of(Kind::Openai, Some(Reasoning::Effort), history.clone())["messages"][0]
				.get("thinking_blocks")
				.is_none()
		);
		assert_eq!(
			of(Kind::Bedrock, Some(Reasoning::Budget), history)["messages"][0]["thinking_blocks"]
				[0]["signature"],
			"s"
		);
	}

	#[test]
	fn a_chat_completion_becomes_a_message() {
		let out = reply(
			&json!({"id": "c1", "choices": [{"finish_reason": "tool_calls", "message": {
				"content": "Let me look.",
				"thinking_blocks": [{"type": "thinking", "thinking": "x", "signature": "s"}],
				"tool_calls": [{"id": "t1", "function": {"name": "Read", "arguments": "{\"path\":\"a\"}"}}]
			}}]}),
			Some(Usage {
				prompt_tokens: 100,
				completion_tokens: 7,
				cached_tokens: Some(60),
				cache_write_tokens: Some(30),
				..Usage::default()
			}),
			"asked-for",
		)
		.unwrap();
		assert_eq!(out["type"], "message");
		assert_eq!(out["model"], "asked-for");
		assert_eq!(out["stop_reason"], "tool_use");
		assert_eq!(out["content"][0]["type"], "thinking");
		assert_eq!(out["content"][1]["text"], "Let me look.");
		assert_eq!(out["content"][2]["input"]["path"], "a");
		assert_eq!(out["usage"]["input_tokens"], 10, "the cache counted apart");
		assert_eq!(out["usage"]["cache_read_input_tokens"], 60);
		assert_eq!(out["usage"]["cache_creation_input_tokens"], 30);
		assert!(reply(&json!({}), None, "m").is_err());
	}

	/// An SSE body's events as `(name, data)`.
	fn parsed(bytes: &[u8]) -> Vec<(String, Value)> {
		String::from_utf8_lossy(bytes)
			.split("\n\n")
			.filter(|e| !e.is_empty())
			.map(|e| {
				let (name, data) = e.split_once('\n').unwrap();
				(
					name.strip_prefix("event: ").unwrap().to_string(),
					serde_json::from_str(data.strip_prefix("data: ").unwrap()).unwrap(),
				)
			})
			.collect()
	}

	fn chunk(delta: Value, finish: Value) -> Value {
		json!({"id": "c1", "choices": [{"delta": delta, "finish_reason": finish}]})
	}

	#[test]
	fn a_stream_of_chunks_becomes_anthropic_s_events() {
		let mut events = Events::new("asked-for");
		let mut out = Vec::new();
		out.extend(events.chunk(&chunk(
			json!({"role": "assistant", "content": ""}),
			Value::Null,
		)));
		out.extend(events.chunk(&chunk(json!({"content": "Hi"}), Value::Null)));
		out.extend(events.chunk(&chunk(json!({"content": " there"}), Value::Null)));
		// Two parallel calls, interleaved, as OpenAI's protocol allows.
		out.extend(events.chunk(&chunk(
			json!({"tool_calls": [
				{"index": 0, "id": "t0", "function": {"name": "A", "arguments": ""}},
				{"index": 1, "id": "t1", "function": {"name": "B", "arguments": ""}}
			]}),
			Value::Null,
		)));
		out.extend(events.chunk(&chunk(
			json!({"tool_calls": [{"index": 1, "function": {"arguments": "{\"b\":1}"}}]}),
			Value::Null,
		)));
		out.extend(events.chunk(&chunk(
			json!({"tool_calls": [{"index": 0, "function": {"arguments": "{\"a\":1}"}}]}),
			Value::Null,
		)));
		out.extend(events.chunk(&chunk(json!({}), json!("tool_calls"))));
		assert!(events.finished());
		out.extend(events.chunk(&json!({"id": "c1", "choices": [], "usage": {
			"prompt_tokens": 12, "completion_tokens": 5
		}})));
		out.extend(events.end());
		assert!(events.end().is_empty(), "once");
		let out = parsed(&out);
		let names: Vec<&str> = out.iter().map(|(n, _)| n.as_str()).collect();
		assert_eq!(
			names,
			[
				"message_start",
				"content_block_start",
				"content_block_delta",
				"content_block_delta",
				"content_block_stop",
				"content_block_start",
				"content_block_delta",
				"content_block_stop",
				"content_block_start",
				"content_block_delta",
				"content_block_stop",
				"message_delta",
				"message_stop",
			]
		);
		assert_eq!(out[0].1["message"]["model"], "asked-for");
		assert_eq!(out[0].1["message"]["id"], "c1");
		assert_eq!(out[2].1["delta"]["text"], "Hi");
		assert_eq!(out[5].1["index"], 1);
		assert_eq!(out[5].1["content_block"]["name"], "A");
		assert_eq!(out[6].1["delta"]["partial_json"], "{\"a\":1}");
		assert_eq!(out[8].1["content_block"]["name"], "B");
		assert_eq!(out[9].1["delta"]["partial_json"], "{\"b\":1}");
		assert_eq!(out[11].1["delta"]["stop_reason"], "tool_use");
		assert_eq!(out[11].1["usage"]["input_tokens"], 12);
		assert_eq!(out[11].1["usage"]["output_tokens"], 5);
	}

	#[test]
	fn signed_thinking_comes_through_and_unsigned_does_not() {
		let mut events = Events::new("m");
		events.signed();
		let mut out = Vec::new();
		out.extend(events.chunk(&chunk(json!({"reasoning_content": "Count "}), Value::Null)));
		out.extend(events.chunk(&chunk(json!({"reasoning_content": "them."}), Value::Null)));
		out.extend(events.chunk(&chunk(
			json!({"thinking_blocks": [{"type": "thinking", "thinking": "Count them.", "signature": "sig"}]}),
			Value::Null,
		)));
		out.extend(events.chunk(&chunk(
			json!({"thinking_blocks": [{"type": "redacted_thinking", "data": "opaque"}]}),
			Value::Null,
		)));
		out.extend(events.chunk(&chunk(json!({"content": "42"}), json!("stop"))));
		out.extend(events.end());
		let out = parsed(&out);
		assert_eq!(out[1].1["content_block"]["type"], "thinking");
		assert_eq!(out[2].1["delta"]["thinking"], "Count ");
		assert_eq!(out[3].1["delta"]["thinking"], "them.");
		assert_eq!(out[4].1["delta"]["signature"], "sig");
		assert_eq!(out[5].0, "content_block_stop");
		assert_eq!(out[6].1["content_block"]["data"], "opaque");
		assert_eq!(out[6].1["index"], 1);
		assert_eq!(out[8].1["content_block"]["type"], "text");
		assert_eq!(out[8].1["index"], 2);

		let mut plain = Events::new("m");
		let some = plain.chunk(&chunk(json!({"reasoning_content": "hmm"}), Value::Null));
		assert_eq!(parsed(&some).len(), 1, "only the message's start: {some:?}");
	}

	#[test]
	fn a_finish_with_no_tool_calls_named_still_says_tool_use() {
		let mut events = Events::new("m");
		events.chunk(&chunk(
			json!({"tool_calls": [{"index": 0, "id": "t", "function": {"name": "A"}}]}),
			json!("stop"),
		));
		let out = parsed(&events.end());
		assert_eq!(
			out[1].1["delta"]["partial_json"], "{}",
			"no arguments is an empty object"
		);
		assert_eq!(out[3].1["delta"]["stop_reason"], "tool_use");
	}

	#[test]
	fn an_empty_stream_is_still_a_whole_message() {
		let mut events = Events::new("m");
		assert!(!events.finished());
		let out = parsed(&events.end());
		assert_eq!(out[0].0, "message_start");
		assert_eq!(out[1].1["delta"]["stop_reason"], "end_turn");
		assert_eq!(out[2].0, "message_stop");
	}

	#[test]
	fn failures_are_in_anthropic_s_shape() {
		let of = |class, status: Option<StatusCode>| {
			let mut error = ApiError::new(class, "said").from_provider("p");
			error.status = status;
			error
		};
		let body = error_body(&of(Class::ModelNotFound, None));
		assert_eq!(body["type"], "error");
		assert_eq!(body["error"]["type"], "not_found_error");
		assert_eq!(body["error"]["message"], "said");
		assert_eq!(body["error"]["code"], "model_not_found");
		assert_eq!(body["error"]["provider"], "p");
		assert_eq!(
			error_status(&of(Class::Budget, None)),
			StatusCode::PAYMENT_REQUIRED
		);
		assert_eq!(
			error_body(&of(Class::Budget, None))["error"]["type"],
			"billing_error"
		);
		assert_eq!(
			error_body(&of(Class::Limited, None))["error"]["type"],
			"rate_limit_error"
		);
		assert_eq!(
			error_body(&of(Class::Unauthorized, None))["error"]["type"],
			"authentication_error"
		);
		let event = String::from_utf8(error_event(&of(Class::Upstream, None))).unwrap();
		assert!(event.starts_with("event: error\ndata: {"), "{event}");
		assert!(check(json!({"messages": [{"role": "user"}]}).as_object().unwrap()).is_err());
		assert!(
			check(
				json!({"max_tokens": 5, "messages": []})
					.as_object()
					.unwrap()
			)
			.is_err()
		);
		assert!(
			check(
				json!({"max_tokens": 0, "messages": [{}]})
					.as_object()
					.unwrap()
			)
			.is_ok()
		);
	}
}
