//! Anthropic's Messages API, spoken for a caller that speaks OpenAI's Chat Completions.
//!
//! Three translations, one each way plus the stream: [`request`] turns a chat completion request
//! into a Messages request, [`reply`] turns a Messages reply into a chat completion, and [`Stream`]
//! turns Anthropic's stream events into OpenAI's chunks as they arrive, one event at a time, never
//! holding the reply back (PS9).
//!
//! The rules this file keeps:
//!  - **What cannot be translated is refused, by name, before anything is sent.** A field Psyche
//!    does not know, or one Anthropic has no equivalent for that would change the answer (`n`, log
//!    probabilities, a response format), is a 400 that names it. A field that only tunes OpenAI's
//!    own service (`seed`, `store`, the penalties) is dropped, and the list is [`DROPPED`].
//!  - **A `cache_control` breakpoint goes where the caller put it, and only to a model that takes
//!    one** (`Model::prompt_caching`, off unless the catalogue says so). Psyche never adds one. On
//!    a tool message the breakpoint goes on the `tool_result` block itself, never inside its
//!    content, where Anthropic refuses it.
//!  - **Consecutive messages of one role are merged**, since Anthropic wants turns that alternate:
//!    the results of parallel tool calls become one user turn, results first.

use std::collections::HashMap;

use serde_json::{Map, Value, json};

use crate::error::{ApiError, Class};
use crate::providers::{Model, Reasoning};
use crate::sse::Event;
pub use crate::translate::Out;
use crate::usage::Usage;

/// The `anthropic-version` header every request carries.
pub const VERSION: &str = "2023-06-01";

/// The reply length asked for when neither the caller nor the catalogue names one. Anthropic
/// requires one on every request; 4096 is the usual default of OpenAI-compatible gateways, so a
/// caller moving over sees no change.
pub const DEFAULT_MAX_TOKENS: u64 = 4096;

/// OpenAI fields with no Anthropic equivalent that only tune OpenAI's own service, so dropping them
/// changes nothing about the answer a caller is owed.
pub const DROPPED: [&str; 11] = [
	"frequency_penalty",
	"presence_penalty",
	"seed",
	"logit_bias",
	"store",
	"metadata",
	"service_tier",
	"stream_options",
	"prompt_cache_key",
	"safety_identifier",
	"prediction",
];

fn bad(message: impl Into<String>) -> ApiError {
	ApiError::new(Class::BadRequest, message)
}

/// The Messages request for a chat completion request. `body` is the caller's, as it came.
pub fn request(body: &Map<String, Value>, model: &Model, stream: bool) -> Result<Value, ApiError> {
	let caching = model.prompt_caching;
	let mut out = Map::new();
	out.insert("model".into(), Value::String(model.upstream.clone()));
	let mut max_tokens = None;
	let mut tool_choice = None;
	let mut parallel = None;
	let mut thinking = None;
	let mut effort = None;
	for (field, value) in body {
		if value.is_null() || DROPPED.contains(&field.as_str()) {
			continue;
		}
		match field.as_str() {
			"model" | "stream" | "messages" => {}
			"max_tokens" | "max_completion_tokens" => {
				max_tokens = Some(
					value
						.as_u64()
						.filter(|n| *n > 0)
						.ok_or_else(|| bad(format!("{field} is a whole number above zero.")))?,
				);
			}
			"temperature" | "top_p" | "top_k" => {
				if !value.is_number() {
					return Err(bad(format!("{field} is a number.")));
				}
				out.insert(field.clone(), value.clone());
			}
			"stop" => {
				let stops = match value {
					Value::String(stop) => vec![Value::String(stop.clone())],
					Value::Array(stops) if stops.iter().all(Value::is_string) => stops.clone(),
					_ => return Err(bad("stop is a string or a list of strings.")),
				};
				out.insert("stop_sequences".into(), Value::Array(stops));
			}
			"user" => {
				let user = value.as_str().ok_or_else(|| bad("user is a string."))?;
				out.insert("metadata".into(), json!({"user_id": user}));
			}
			"tools" => {
				out.insert("tools".into(), tools(value, caching)?);
			}
			"tool_choice" => tool_choice = Some(value),
			"parallel_tool_calls" => {
				parallel = Some(
					value
						.as_bool()
						.ok_or_else(|| bad("parallel_tool_calls is true or false."))?,
				);
			}
			"reasoning_effort" => {
				effort = Some(
					value
						.as_str()
						.ok_or_else(|| bad("reasoning_effort is a string."))?,
				);
			}
			// Anthropic's own field, for a caller that knows it.
			"thinking" => {
				if !value.is_object() {
					return Err(bad("thinking is an object, as Anthropic's API takes it."));
				}
				thinking = Some(value.clone());
			}
			"n" => {
				if value.as_u64() != Some(1) {
					return Err(bad(format!(
						"{} answers one choice per request, so n must be 1.",
						model.name
					)));
				}
			}
			"response_format" => {
				if value.get("type").and_then(Value::as_str) != Some("text") {
					return Err(bad(format!(
						"Structured output (response_format) is not translated for {} yet.",
						model.name
					)));
				}
			}
			"logprobs" | "top_logprobs" => {
				if value != &Value::Bool(false) {
					return Err(bad(format!(
						"{} does not return log probabilities.",
						model.name
					)));
				}
			}
			other => {
				return Err(bad(format!(
					"{other} is not translated for {}, so it would be ignored. \
					 Leave it out.",
					model.name
				)));
			}
		}
	}
	let mut max_tokens = max_tokens
		.or(model.max_output_tokens)
		.unwrap_or(DEFAULT_MAX_TOKENS);
	// reasoning_effort, in the form this model takes (the catalogue's `reasoning`).
	let thinking = match (thinking, effort, model.reasoning) {
		(Some(thinking), _, _) => Some(thinking),
		(None, None | Some("none"), _) => None,
		(None, Some(effort), Some(Reasoning::Budget)) => {
			Some(json!({"type": "enabled", "budget_tokens": budget(effort)?}))
		}
		(None, Some(effort), Some(Reasoning::Adaptive)) => {
			let effort = if effort == "minimal" { "low" } else { effort };
			out.insert("output_config".into(), json!({"effort": effort}));
			Some(json!({"type": "adaptive"}))
		}
		(None, Some(effort), Some(Reasoning::Effort)) => {
			out.insert("reasoning".into(), json!({"effort": effort}));
			None
		}
		(None, Some(_), None) => {
			return Err(bad(format!(
				"The catalogue does not say how {} takes reasoning_effort, so it is not sent. \
				 Leave it out, or set the model's \"reasoning\" (budget, adaptive or effort).",
				model.name
			)));
		}
	};
	if let Some(thinking) = thinking {
		// Thinking counts against max_tokens, so with a budget the answer keeps the length asked
		// for and the budget is added on top.
		if let Some(budget) = thinking.get("budget_tokens").and_then(Value::as_u64) {
			max_tokens += budget;
		}
		for field in ["temperature", "top_k"] {
			if out.contains_key(field) {
				return Err(bad(format!(
					"{} cannot change {field} while it is thinking. Leave {field} out, or reasoning_effort.",
					model.name
				)));
			}
		}
		out.insert("thinking".into(), thinking);
	}
	out.insert("max_tokens".into(), json!(max_tokens));
	if let Some(choice) = self::tool_choice(tool_choice, parallel)? {
		out.insert("tool_choice".into(), choice);
	}
	let (system, messages) = messages(body.get("messages"), caching)?;
	if !system.is_empty() {
		out.insert("system".into(), Value::Array(system));
	}
	out.insert("messages".into(), Value::Array(messages));
	if stream {
		out.insert("stream".into(), Value::Bool(true));
	}
	Ok(Value::Object(out))
}

/// The thinking budget for an OpenAI `reasoning_effort`, the usual mapping of OpenAI-compatible
/// gateways, so a caller moving over gets the same depth.
fn budget(effort: &str) -> Result<u64, ApiError> {
	match effort {
		"minimal" | "low" => Ok(1024),
		"medium" => Ok(2048),
		"high" => Ok(4096),
		other => Err(bad(format!(
			"reasoning_effort is none, minimal, low, medium or high, not {other}."
		))),
	}
}

/// An assistant turn's earlier thinking, sent back as Anthropic gave it (`thinking_blocks`, the
/// field OpenAI-compatible clients already send), signature and all: a tool loop with thinking on
/// is refused without it, and a block changed by a byte fails its signature.
fn thinking_blocks(blocks: &Value, at: &dyn Fn(&str) -> ApiError) -> Result<Vec<Value>, ApiError> {
	let blocks = blocks
		.as_array()
		.ok_or_else(|| at("thinking_blocks is a list."))?;
	blocks
		.iter()
		.map(|block| match block.get("type").and_then(Value::as_str) {
			Some("thinking" | "redacted_thinking") => Ok(block.clone()),
			_ => Err(at(
				"a thinking block is of type thinking or redacted_thinking.",
			)),
		})
		.collect()
}

/// `system` blocks and the turns, from OpenAI's `messages`.
fn messages(messages: Option<&Value>, caching: bool) -> Result<(Vec<Value>, Vec<Value>), ApiError> {
	let messages = messages
		.and_then(Value::as_array)
		.filter(|m| !m.is_empty())
		.ok_or_else(|| bad("messages is a list of at least one message."))?;
	let mut system = Vec::new();
	let mut turns: Vec<(&'static str, Vec<Value>)> = Vec::new();
	for (i, message) in messages.iter().enumerate() {
		let at = |what: &str| bad(format!("messages[{i}]: {what}"));
		let role = message
			.get("role")
			.and_then(Value::as_str)
			.ok_or_else(|| at("it has no role."))?;
		let breakpoint = message.get("cache_control").filter(|_| caching);
		let (role, mut blocks) = match role {
			// Anthropic has one system prompt, so every system message joins it, in order.
			"system" | "developer" => {
				let mut blocks = content(message.get("content"), caching, &at)?;
				if blocks.iter().any(|b| b["type"] != "text") {
					return Err(at("a system message holds text only."));
				}
				mark(&mut blocks, breakpoint);
				system.extend(blocks);
				continue;
			}
			"user" => ("user", content(message.get("content"), caching, &at)?),
			"assistant" => {
				// Thinking goes first in a turn, as Anthropic wrote it.
				let mut blocks = match message.get("thinking_blocks").filter(|b| !b.is_null()) {
					Some(thinking) => thinking_blocks(thinking, &at)?,
					None => Vec::new(),
				};
				blocks.extend(content(message.get("content"), caching, &at)?);
				if let Some(calls) = message.get("tool_calls").filter(|c| !c.is_null()) {
					blocks.extend(tool_uses(calls, &at)?);
				}
				("assistant", blocks)
			}
			"tool" => {
				let id = message
					.get("tool_call_id")
					.and_then(Value::as_str)
					.ok_or_else(|| at("a tool message has no tool_call_id."))?;
				let mut inner = content(message.get("content"), caching, &at)?;
				if inner.iter().any(|b| b["type"] != "text") {
					return Err(at("a tool message holds text only."));
				}
				// A breakpoint inside a tool result's content is refused by Anthropic: it belongs on
				// the tool_result block.
				let mut point = breakpoint.cloned();
				for block in &mut inner {
					if let Some(found) = block
						.as_object_mut()
						.and_then(|b| b.remove("cache_control"))
					{
						point = Some(found);
					}
				}
				let mut result =
					json!({"type": "tool_result", "tool_use_id": id, "content": inner});
				if let Some(point) = point {
					result["cache_control"] = point;
				}
				("user", vec![result])
			}
			other => return Err(at(&format!("{other} is not a role Psyche knows."))),
		};
		mark(&mut blocks, breakpoint);
		if blocks.is_empty() {
			continue;
		}
		match turns.last_mut() {
			Some((last, previous)) if *last == role => previous.extend(blocks),
			_ => turns.push((role, blocks)),
		}
	}
	let turns = turns
		.into_iter()
		.map(|(role, content)| json!({"role": role, "content": content}))
		.collect();
	Ok((system, turns))
}

/// A message-level breakpoint, put on the message's last block.
fn mark(blocks: &mut [Value], breakpoint: Option<&Value>) {
	if let (Some(point), Some(last)) = (breakpoint, blocks.last_mut())
		&& last.get("cache_control").is_none()
	{
		last["cache_control"] = point.clone();
	}
}

/// A message's content as Anthropic blocks. An empty text is left out, since Anthropic refuses an
/// empty text block.
fn content(
	content: Option<&Value>,
	caching: bool,
	at: &dyn Fn(&str) -> ApiError,
) -> Result<Vec<Value>, ApiError> {
	let parts = match content {
		None | Some(Value::Null) => return Ok(Vec::new()),
		Some(Value::String(text)) if text.is_empty() => return Ok(Vec::new()),
		Some(Value::String(text)) => return Ok(vec![json!({"type": "text", "text": text})]),
		Some(Value::Array(parts)) => parts,
		Some(_) => return Err(at("content is a string or a list of parts.")),
	};
	let mut blocks = Vec::new();
	for part in parts {
		let kind = part.get("type").and_then(Value::as_str).unwrap_or("");
		let mut block = match kind {
			"text" => {
				let text = part
					.get("text")
					.and_then(Value::as_str)
					.ok_or_else(|| at("a text part has no text."))?;
				if text.is_empty() {
					continue;
				}
				json!({"type": "text", "text": text})
			}
			"image_url" => image(part.get("image_url"), at)?,
			other => {
				return Err(at(&format!(
					"a {other} part is not translated for Anthropic models yet."
				)));
			}
		};
		if caching && let Some(point) = part.get("cache_control") {
			block["cache_control"] = point.clone();
		}
		blocks.push(block);
	}
	Ok(blocks)
}

fn image(image: Option<&Value>, at: &dyn Fn(&str) -> ApiError) -> Result<Value, ApiError> {
	let url = image
		.and_then(|i| i.get("url").or(Some(i)))
		.and_then(Value::as_str)
		.ok_or_else(|| at("an image_url part has no url."))?;
	if let Some(inline) = url.strip_prefix("data:") {
		let (media_type, data) = inline
			.split_once(";base64,")
			.ok_or_else(|| at("an inline image is a base64 data URL."))?;
		return Ok(json!({"type": "image", "source": {
			"type": "base64", "media_type": media_type, "data": data
		}}));
	}
	if url.starts_with("https://") || url.starts_with("http://") {
		return Ok(json!({"type": "image", "source": {"type": "url", "url": url}}));
	}
	Err(at("an image is an http(s) URL or a base64 data URL."))
}

/// An assistant's `tool_calls` as `tool_use` blocks.
fn tool_uses(calls: &Value, at: &dyn Fn(&str) -> ApiError) -> Result<Vec<Value>, ApiError> {
	let calls = calls
		.as_array()
		.ok_or_else(|| at("tool_calls is a list."))?;
	calls
		.iter()
		.map(|call| {
			let id = call.get("id").and_then(Value::as_str);
			let name = call.pointer("/function/name").and_then(Value::as_str);
			let (Some(id), Some(name)) = (id, name) else {
				return Err(at("a tool call has no id or no function name."));
			};
			let arguments = call
				.pointer("/function/arguments")
				.and_then(Value::as_str)
				.unwrap_or("");
			let input = if arguments.trim().is_empty() {
				json!({})
			} else {
				serde_json::from_str::<Value>(arguments)
					.ok()
					.filter(Value::is_object)
					.ok_or_else(|| {
						at(&format!(
							"the arguments of tool call {id} are not a JSON object."
						))
					})?
			};
			Ok(json!({"type": "tool_use", "id": id, "name": name, "input": input}))
		})
		.collect()
}

fn tools(tools: &Value, caching: bool) -> Result<Value, ApiError> {
	let tools = tools.as_array().ok_or_else(|| bad("tools is a list."))?;
	let out = tools
		.iter()
		.enumerate()
		.map(|(i, tool)| {
			if tool.get("type").and_then(Value::as_str) != Some("function") {
				return Err(bad(format!("tools[{i}] is not a function tool.")));
			}
			let function = tool
				.get("function")
				.ok_or_else(|| bad(format!("tools[{i}] has no function.")))?;
			let name = function
				.get("name")
				.and_then(Value::as_str)
				.ok_or_else(|| bad(format!("tools[{i}] has no name.")))?;
			let mut out = json!({
				"name": name,
				"input_schema": function
					.get("parameters")
					.filter(|p| p.is_object())
					.cloned()
					.unwrap_or_else(|| json!({"type": "object", "properties": {}})),
			});
			if let Some(description) = function.get("description").and_then(Value::as_str) {
				out["description"] = json!(description);
			}
			if caching && let Some(point) = tool.get("cache_control") {
				out["cache_control"] = point.clone();
			}
			Ok(out)
		})
		.collect::<Result<Vec<_>, _>>()?;
	Ok(Value::Array(out))
}

fn tool_choice(choice: Option<&Value>, parallel: Option<bool>) -> Result<Option<Value>, ApiError> {
	let mut out = match choice {
		None => None,
		Some(Value::String(choice)) => Some(match choice.as_str() {
			"auto" => json!({"type": "auto"}),
			"none" => json!({"type": "none"}),
			"required" => json!({"type": "any"}),
			other => return Err(bad(format!("tool_choice {other} is not one Psyche knows."))),
		}),
		Some(choice) => {
			let name = choice
				.pointer("/function/name")
				.and_then(Value::as_str)
				.ok_or_else(|| bad("tool_choice names no function."))?;
			Some(json!({"type": "tool", "name": name}))
		}
	};
	if parallel == Some(false) {
		let choice = out.get_or_insert_with(|| json!({"type": "auto"}));
		if choice["type"] != "none" {
			choice["disable_parallel_tool_use"] = json!(true);
		}
	}
	Ok(out)
}

/// OpenAI's `finish_reason` for Anthropic's `stop_reason`, or Bedrock's `stopReason`, which uses
/// the same names and two of its own for a guardrail or a content filter.
pub(crate) fn finish_reason(stop: Option<&str>) -> Value {
	json!(match stop {
		Some("max_tokens") => "length",
		Some("tool_use") => "tool_calls",
		Some("refusal" | "guardrail_intervened" | "content_filtered") => "content_filter",
		_ => "stop",
	})
}

/// Puts a reply's thinking on its message, as reasoning models' OpenAI-style APIs do:
/// `reasoning_content`, the text a person reads (DeepSeek's field, which Studio carries through its
/// agent loop), and `thinking_blocks`, the blocks as Anthropic wrote them, signatures included, which
/// a caller sends back on the next turn of a tool loop.
pub(crate) fn with_reasoning(message: &mut Value, blocks: Vec<Value>) {
	if blocks.is_empty() {
		return;
	}
	let text: String = blocks
		.iter()
		.filter_map(|b| b.get("thinking").and_then(Value::as_str))
		.collect();
	if !text.is_empty() {
		message["reasoning_content"] = Value::String(text);
	}
	message["thinking_blocks"] = Value::Array(blocks);
}

/// A chat completion for a Messages reply, or what was wrong with the reply. `model` is the name
/// the caller asked for, which is the name the reply carries.
pub fn reply(value: &Value, model: &str, created: u64) -> Result<Value, String> {
	let blocks = value
		.get("content")
		.and_then(Value::as_array)
		.ok_or("the reply has no content")?;
	let mut text = String::new();
	let mut calls = Vec::new();
	let mut thinking = Vec::new();
	for block in blocks {
		match block.get("type").and_then(Value::as_str) {
			Some("text") => text.push_str(block.get("text").and_then(Value::as_str).unwrap_or("")),
			Some("thinking" | "redacted_thinking") => thinking.push(block.clone()),
			Some("tool_use") => calls.push(json!({
				"id": block.get("id").cloned().unwrap_or(Value::Null),
				"type": "function",
				"function": {
					"name": block.get("name").cloned().unwrap_or(Value::Null),
					"arguments": block.get("input").map(Value::to_string).unwrap_or_else(|| "{}".into()),
				},
			})),
			_ => {}
		}
	}
	let mut message = json!({
		"role": "assistant",
		"content": if text.is_empty() && !calls.is_empty() { Value::Null } else { Value::String(text) },
	});
	if !calls.is_empty() {
		message["tool_calls"] = Value::Array(calls);
	}
	with_reasoning(&mut message, thinking);
	let usage = value
		.get("usage")
		.and_then(Usage::from_anthropic)
		.unwrap_or_default();
	Ok(json!({
		"id": value.get("id").cloned().unwrap_or(Value::Null),
		"object": "chat.completion",
		"created": created,
		"model": model,
		"choices": [{
			"index": 0,
			"message": message,
			"finish_reason": finish_reason(value.get("stop_reason").and_then(Value::as_str)),
		}],
		"usage": usage.to_openai(),
	}))
}

/// One streamed reply being translated.
#[derive(Debug)]
pub struct Stream {
	wants_usage: bool,
	created: u64,
	id: Value,
	model: Value,
	usage: Option<Usage>,
	/// Anthropic's content block index → OpenAI's tool call index, for tool_use blocks.
	tools: HashMap<u64, u64>,
	/// Thinking blocks being received, by content block index, passed on whole when they end.
	thinking: HashMap<u64, Value>,
	ended: bool,
}

impl Stream {
	/// `model` is the name the caller asked for, which every chunk carries.
	pub fn new(wants_usage: bool, model: &str, created: u64) -> Self {
		Self {
			wants_usage,
			created,
			id: Value::Null,
			model: Value::String(model.to_string()),
			usage: None,
			tools: HashMap::new(),
			thinking: HashMap::new(),
			ended: false,
		}
	}

	/// What the provider reported using, so far.
	pub fn usage(&self) -> Option<Usage> {
		self.usage
	}

	/// Whether the provider's last event (`message_stop`) arrived. A stream that closes without it
	/// was cut.
	pub fn ended(&self) -> bool {
		self.ended
	}

	fn chunk(&self, delta: Value, finish: Value) -> Out {
		Out::Chunk(json!({
			"id": self.id, "object": "chat.completion.chunk", "created": self.created, "model": self.model,
			"choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
		}))
	}

	/// Folds a usage report into what is known: `message_delta` reports the output so far and, on
	/// newer versions, the input again.
	fn count(&mut self, usage: Option<&Value>) {
		let Some(report) = usage.and_then(Usage::from_anthropic) else {
			return;
		};
		let has_input = usage
			.and_then(|u| u.get("input_tokens"))
			.is_some_and(|v| !v.is_null());
		let mut known = self.usage.unwrap_or_default();
		if has_input {
			known.prompt_tokens = report.prompt_tokens;
			known.cached_tokens = report.cached_tokens.or(known.cached_tokens);
			known.cache_write_tokens = report.cache_write_tokens.or(known.cache_write_tokens);
		}
		known.completion_tokens = report.completion_tokens.max(known.completion_tokens);
		self.usage = Some(known);
	}

	pub fn event(&mut self, event: &Event) -> Vec<Out> {
		let Some(data) = event.data.as_deref() else {
			return Vec::new();
		};
		let Ok(value) = serde_json::from_str::<Value>(data) else {
			return vec![Out::Error(
				Class::Upstream,
				"Anthropic sent a stream event that is not JSON.".into(),
			)];
		};
		match value.get("type").and_then(Value::as_str).unwrap_or("") {
			"message_start" => {
				let message = &value["message"];
				self.id = message.get("id").cloned().unwrap_or(Value::Null);
				self.count(message.get("usage"));
				vec![self.chunk(json!({"role": "assistant", "content": ""}), Value::Null)]
			}
			"content_block_start" => {
				let block = &value["content_block"];
				let index = value["index"].as_u64().unwrap_or(0);
				if block["type"] == "thinking" || block["type"] == "redacted_thinking" {
					self.thinking.insert(index, block.clone());
					return Vec::new();
				}
				if block["type"] != "tool_use" {
					return Vec::new();
				}
				let tool = self.tools.len() as u64;
				self.tools.insert(index, tool);
				vec![self.chunk(
					json!({"tool_calls": [{
						"index": tool, "id": block["id"], "type": "function",
						"function": {"name": block["name"], "arguments": ""},
					}]}),
					Value::Null,
				)]
			}
			"content_block_delta" => {
				let delta = &value["delta"];
				match delta["type"].as_str() {
					Some("text_delta") => {
						vec![self.chunk(json!({"content": delta["text"]}), Value::Null)]
					}
					Some("thinking_delta") => {
						let index = value["index"].as_u64().unwrap_or(0);
						let piece = delta["thinking"].as_str().unwrap_or("");
						if let Some(block) = self.thinking.get_mut(&index) {
							let joined =
								format!("{}{piece}", block["thinking"].as_str().unwrap_or(""));
							block["thinking"] = Value::String(joined);
						}
						vec![self.chunk(json!({"reasoning_content": piece}), Value::Null)]
					}
					Some("signature_delta") => {
						let index = value["index"].as_u64().unwrap_or(0);
						if let Some(block) = self.thinking.get_mut(&index) {
							let joined = format!(
								"{}{}",
								block["signature"].as_str().unwrap_or(""),
								delta["signature"].as_str().unwrap_or("")
							);
							block["signature"] = Value::String(joined);
						}
						Vec::new()
					}
					Some("input_json_delta") => {
						let index = value["index"].as_u64().unwrap_or(0);
						let Some(tool) = self.tools.get(&index).copied() else {
							return Vec::new();
						};
						vec![self.chunk(
							json!({"tool_calls": [{
								"index": tool, "function": {"arguments": delta["partial_json"]},
							}]}),
							Value::Null,
						)]
					}
					_ => Vec::new(),
				}
			}
			"content_block_stop" => {
				let index = value["index"].as_u64().unwrap_or(0);
				match self.thinking.remove(&index) {
					// The whole block, signature included, once it is complete.
					Some(block) => {
						vec![self.chunk(json!({"thinking_blocks": [block]}), Value::Null)]
					}
					None => Vec::new(),
				}
			}
			"message_delta" => {
				self.count(value.get("usage"));
				vec![self.chunk(
					json!({}),
					finish_reason(value.pointer("/delta/stop_reason").and_then(Value::as_str)),
				)]
			}
			"message_stop" => {
				self.ended = true;
				let mut out = Vec::new();
				if self.wants_usage {
					out.push(Out::Chunk(json!({
						"id": self.id, "object": "chat.completion.chunk", "created": self.created,
						"model": self.model, "choices": [],
						"usage": self.usage.unwrap_or_default().to_openai(),
					})));
				}
				out.push(Out::Done);
				out
			}
			"ping" => vec![Out::KeepAlive],
			"error" => {
				let kind = value.pointer("/error/type").and_then(Value::as_str);
				let message = value
					.pointer("/error/message")
					.and_then(Value::as_str)
					.unwrap_or("an error with no message");
				let class = match kind {
					Some("rate_limit_error") => Class::RateLimited,
					Some("invalid_request_error") => Class::Rejected,
					_ => Class::Upstream,
				};
				vec![Out::Error(class, message.to_string())]
			}
			// content_block_stop, and anything added after this was written.
			_ => Vec::new(),
		}
	}
}

#[cfg(test)]
mod tests {
	use std::sync::Arc;

	use super::*;
	use crate::providers::{Kind, Provider};

	fn model(caching: bool) -> Model {
		Model {
			name: "claude".into(),
			provider: Arc::new(Provider {
				name: "anthropic".into(),
				kind: Kind::Anthropic,
				base_url: url::Url::parse("https://api.anthropic.com/v1").unwrap(),
				api_key: None,
				aws: None,
				region: None,
				credential: None,
				auth_header: None,
			}),
			upstream: "claude-sonnet-4-5".into(),
			prompt_caching: caching,
			max_output_tokens: None,
			reasoning: Some(Reasoning::Budget),
			embeddings: None,
			price: None,
		}
	}

	fn translate(body: Value, caching: bool) -> Result<Value, ApiError> {
		request(body.as_object().unwrap(), &model(caching), false)
	}

	/// Studio's agent loop as it sends it: a system
	/// prompt with a breakpoint, a question, a turn of two parallel tool calls, both results, the
	/// last one carrying the rolling breakpoint.
	fn agent_loop() -> Value {
		json!({
			"model": "claude", "stream": true, "max_tokens": 1000, "user": "u-1",
			"stream_options": {"include_usage": true},
			"messages": [
				{"role": "system", "content": [{"type": "text", "text": "You are an assistant.", "cache_control": {"type": "ephemeral"}}]},
				{"role": "user", "content": "How many users?"},
				{"role": "assistant", "content": null, "tool_calls": [
					{"id": "call_1", "type": "function", "function": {"name": "run_sql", "arguments": "{\"sql\":\"select count(*) from users\"}"}},
					{"id": "call_2", "type": "function", "function": {"name": "describe", "arguments": ""}}
				]},
				{"role": "tool", "tool_call_id": "call_1", "content": "42"},
				{"role": "tool", "tool_call_id": "call_2", "content": [{"type": "text", "text": "users(id, email)", "cache_control": {"type": "ephemeral"}}]}
			],
			"tools": [{"type": "function", "function": {"name": "run_sql", "description": "Runs SQL.", "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}}}}]
		})
	}

	#[test]
	fn an_agent_loop_becomes_alternating_turns() {
		let out = translate(agent_loop(), true).unwrap();
		assert_eq!(out["model"], "claude-sonnet-4-5");
		assert_eq!(out["max_tokens"], 1000);
		assert_eq!(out["metadata"]["user_id"], "u-1");
		assert!(out.get("stream_options").is_none());
		assert_eq!(out["system"][0]["text"], "You are an assistant.");
		assert_eq!(out["system"][0]["cache_control"]["type"], "ephemeral");
		let messages = out["messages"].as_array().unwrap();
		assert_eq!(messages.len(), 3, "{messages:#?}");
		assert_eq!(
			messages[0],
			json!({"role": "user", "content": [{"type": "text", "text": "How many users?"}]})
		);
		let calls = &messages[1]["content"];
		assert_eq!(messages[1]["role"], "assistant");
		assert_eq!(calls[0]["type"], "tool_use");
		assert_eq!(calls[0]["input"]["sql"], "select count(*) from users");
		assert_eq!(
			calls[1]["input"],
			json!({}),
			"empty arguments are no arguments"
		);
		let results = &messages[2]["content"];
		assert_eq!(messages[2]["role"], "user", "both results, one turn");
		assert_eq!(results[0]["tool_use_id"], "call_1");
		assert_eq!(results[1]["tool_use_id"], "call_2");
		// The breakpoint is on the tool_result, never inside its content.
		assert_eq!(results[1]["cache_control"]["type"], "ephemeral");
		assert!(results[1]["content"][0].get("cache_control").is_none());
		assert_eq!(
			out["tools"][0]["input_schema"]["properties"]["sql"]["type"],
			"string"
		);
	}

	#[test]
	fn a_model_without_caching_never_gets_a_breakpoint() {
		// A model that does not cache can answer a request that
		// carries cache_control with an error. Off unless the catalogue says so.
		let mut body = agent_loop();
		body["tools"][0]["cache_control"] = json!({"type": "ephemeral"});
		body["messages"][1]["cache_control"] = json!({"type": "ephemeral"});
		let out = translate(body.clone(), false).unwrap();
		assert!(!out.to_string().contains("cache_control"), "{out}");
		let out = translate(body, true).unwrap();
		assert_eq!(out["tools"][0]["cache_control"]["type"], "ephemeral");
		assert_eq!(
			out["messages"][0]["content"][0]["cache_control"]["type"], "ephemeral",
			"a message-level breakpoint lands on its last block"
		);
	}

	#[test]
	fn the_rest_of_a_request() {
		let out = translate(
			json!({"model": "claude", "seed": 7, "temperature": 0.2, "stop": "END",
			"tool_choice": "required", "parallel_tool_calls": false, "n": 1,
			"messages": [
				{"role": "developer", "content": "Be brief."},
				{"role": "user", "content": [
					{"type": "text", "text": "What is this?"},
					{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
					{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
				]},
				{"role": "assistant", "content": ""},
				{"role": "user", "content": "Well?"}
			]}),
			false,
		)
		.unwrap();
		assert_eq!(out["max_tokens"], DEFAULT_MAX_TOKENS);
		assert!(out.get("seed").is_none());
		assert_eq!(out["stop_sequences"], json!(["END"]));
		assert_eq!(
			out["tool_choice"],
			json!({"type": "any", "disable_parallel_tool_use": true})
		);
		assert_eq!(out["system"][0]["text"], "Be brief.");
		let content = &out["messages"][0]["content"];
		assert_eq!(
			content[1]["source"],
			json!({"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="})
		);
		assert_eq!(content[2]["source"]["type"], "url");
		assert_eq!(
			out["messages"].as_array().unwrap().len(),
			1,
			"an empty assistant turn is dropped, and the two user turns around it merge"
		);
		assert!(out.get("stream").is_none());

		let mut catalogue_default = model(false);
		catalogue_default.max_output_tokens = Some(64_000);
		let out = request(
			json!({"messages": [{"role": "user", "content": "hi"}]})
				.as_object()
				.unwrap(),
			&catalogue_default,
			true,
		)
		.unwrap();
		assert_eq!(out["max_tokens"], 64_000);
		assert_eq!(out["stream"], true);
	}

	#[test]
	fn what_cannot_be_translated_is_refused_by_name() {
		for (field, value, said) in [
			("n", json!(2), "n must be 1"),
			(
				"response_format",
				json!({"type": "json_object"}),
				"response_format",
			),
			("logprobs", json!(true), "log probabilities"),
			("reasoning_effort", json!("deep"), "reasoning_effort"),
			("functions", json!([]), "functions"),
		] {
			let mut body = json!({"messages": [{"role": "user", "content": "hi"}]});
			body[field] = value;
			let error = translate(body, false).unwrap_err();
			assert_eq!(error.class, Class::BadRequest);
			assert!(error.message.contains(said), "{field}: {}", error.message);
		}
		for (messages, said) in [
			(json!([]), "at least one"),
			(json!([{"content": "hi"}]), "no role"),
			(json!([{"role": "tool", "content": "x"}]), "tool_call_id"),
			(
				json!([{"role": "user", "content": [{"type": "input_audio"}]}]),
				"input_audio",
			),
			(
				json!([{"role": "assistant", "tool_calls": [{"id": "c", "function": {"name": "f", "arguments": "[1]"}}]}]),
				"not a JSON object",
			),
			(
				json!([{"role": "system", "content": [{"type": "image_url", "image_url": "https://x/y.png"}]}]),
				"text only",
			),
		] {
			let error = translate(json!({"messages": messages}), false).unwrap_err();
			assert!(error.message.contains(said), "{said}: {}", error.message);
		}
	}

	#[test]
	fn a_reply_becomes_a_chat_completion() {
		let reply = reply(
			&json!({
				"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-4-5",
				"content": [
					{"type": "text", "text": "Let me look."},
					{"type": "tool_use", "id": "toolu_1", "name": "run_sql", "input": {"sql": "select 1"}}
				],
				"stop_reason": "tool_use",
				"usage": {"input_tokens": 10, "cache_read_input_tokens": 90, "output_tokens": 7}
			}),
			"claude",
			1,
		)
		.unwrap();
		assert_eq!(reply["model"], "claude", "the name the caller asked for");
		let choice = &reply["choices"][0];
		assert_eq!(choice["finish_reason"], "tool_calls");
		assert_eq!(choice["message"]["content"], "Let me look.");
		let call = &choice["message"]["tool_calls"][0];
		assert_eq!(call["function"]["name"], "run_sql");
		let arguments: Value =
			serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
		assert_eq!(arguments["sql"], "select 1");
		assert_eq!(reply["usage"]["prompt_tokens"], 100);
		assert_eq!(reply["usage"]["cache_read_input_tokens"], 90);
		assert!(super::reply(&json!({"type": "message"}), "claude", 1).is_err());
	}

	fn event(data: Value) -> Event {
		Event {
			event: data["type"].as_str().map(String::from),
			data: Some(data.to_string()),
			raw: bytes::Bytes::new(),
		}
	}

	#[test]
	fn a_stream_becomes_chunks_as_it_arrives() {
		let mut stream = Stream::new(true, "claude", 1);
		let events = [
			json!({"type": "message_start", "message": {"id": "msg_1", "model": "claude-sonnet-4-5", "usage": {"input_tokens": 10, "cache_creation_input_tokens": 50, "output_tokens": 1}}}),
			json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
			json!({"type": "ping"}),
			json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}}),
			json!({"type": "content_block_stop", "index": 0}),
			json!({"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "run_sql", "input": {}}}),
			json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"sql\":"}}),
			json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "\"select 1\"}"}}),
			json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 12}}),
		];
		let out: Vec<Out> = events
			.iter()
			.flat_map(|e| stream.event(&event(e.clone())))
			.collect();
		assert!(!stream.ended());
		let chunks: Vec<&Value> = out
			.iter()
			.filter_map(|o| match o {
				Out::Chunk(v) => Some(v),
				_ => None,
			})
			.collect();
		assert!(out.contains(&Out::KeepAlive));
		assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
		assert_eq!(chunks[0]["id"], "msg_1");
		assert_eq!(chunks[1]["choices"][0]["delta"]["content"], "Hi");
		let tool = &chunks[2]["choices"][0]["delta"]["tool_calls"][0];
		assert_eq!(
			(tool["index"].clone(), tool["id"].clone()),
			(json!(0), json!("toolu_1"))
		);
		let arguments: String = chunks[3..5]
			.iter()
			.map(|c| {
				c["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
					.as_str()
					.unwrap()
			})
			.collect();
		assert_eq!(arguments, "{\"sql\":\"select 1\"}");
		assert_eq!(chunks[5]["choices"][0]["finish_reason"], "tool_calls");

		let end = stream.event(&event(json!({"type": "message_stop"})));
		assert!(stream.ended());
		let Out::Chunk(usage) = &end[0] else {
			panic!("{end:?}")
		};
		assert_eq!(usage["usage"]["prompt_tokens"], 60);
		assert_eq!(usage["usage"]["completion_tokens"], 12);
		assert_eq!(usage["usage"]["cache_creation_input_tokens"], 50);
		assert_eq!(end[1], Out::Done);

		let mut quiet = Stream::new(false, "claude", 1);
		assert_eq!(
			quiet.event(&event(json!({"type": "message_stop"}))),
			vec![Out::Done],
			"no usage chunk unless asked"
		);
	}

	#[test]
	fn a_stream_error_is_classified() {
		let mut stream = Stream::new(false, "claude", 1);
		let out = stream.event(&event(
			json!({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}),
		));
		assert_eq!(out, vec![Out::Error(Class::Upstream, "Overloaded".into())]);
		let out = stream.event(&Event {
			event: None,
			data: Some("{not json".into()),
			raw: bytes::Bytes::new(),
		});
		assert!(matches!(out[0], Out::Error(Class::Upstream, _)));
	}

	#[test]
	fn reasoning_becomes_thinking_and_comes_back_with_its_signature() {
		let out = translate(
			json!({"reasoning_effort": "medium", "max_tokens": 500, "messages": [
				{"role": "user", "content": "How many users?"},
				{"role": "assistant", "content": null,
					"thinking_blocks": [{"type": "thinking", "thinking": "Count them.", "signature": "sig-1"}],
					"tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "run_sql", "arguments": "{}"}}]},
				{"role": "tool", "tool_call_id": "call_1", "content": "42"}
			]}),
			false,
		)
		.unwrap();
		assert_eq!(
			out["thinking"],
			json!({"type": "enabled", "budget_tokens": 2048})
		);
		assert_eq!(
			out["max_tokens"], 2548,
			"the answer keeps its 500, thinking on top"
		);
		assert_eq!(
			out["messages"][1]["content"][0],
			json!({"type": "thinking", "thinking": "Count them.", "signature": "sig-1"}),
			"sent back byte for byte, first in its turn"
		);
		assert_eq!(out["messages"][1]["content"][1]["type"], "tool_use");
		let none = translate(
			json!({"reasoning_effort": "none", "messages": [{"role": "user", "content": "hi"}]}),
			false,
		)
		.unwrap();
		assert!(none.get("thinking").is_none());
		let error = translate(
			json!({"reasoning_effort": "low", "temperature": 0.2, "messages": [{"role": "user", "content": "hi"}]}),
			false,
		)
		.unwrap_err();
		assert!(error.message.contains("temperature"), "{}", error.message);

		let reply = reply(
			&json!({"id": "msg_1", "content": [
				{"type": "thinking", "thinking": "Count ", "signature": "sig-2"},
				{"type": "redacted_thinking", "data": "opaque"},
				{"type": "text", "text": "42."}
			], "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}}),
			"claude",
			1,
		)
		.unwrap();
		let message = &reply["choices"][0]["message"];
		assert_eq!(message["content"], "42.");
		assert_eq!(message["reasoning_content"], "Count ");
		assert_eq!(
			message["thinking_blocks"][1],
			json!({"type": "redacted_thinking", "data": "opaque"})
		);
	}

	#[test]
	fn streamed_thinking_comes_as_text_then_as_a_whole_block() {
		let mut stream = Stream::new(false, "claude", 1);
		let events = [
			json!({"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 1}}}),
			json!({"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}}),
			json!({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Count "}}),
			json!({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "them."}}),
			json!({"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig-3"}}),
			json!({"type": "content_block_stop", "index": 0}),
		];
		let chunks: Vec<Value> = events
			.iter()
			.flat_map(|e| stream.event(&event(e.clone())))
			.filter_map(|o| match o {
				Out::Chunk(v) => Some(v),
				_ => None,
			})
			.collect();
		let text: String = chunks
			.iter()
			.filter_map(|c| {
				c.pointer("/choices/0/delta/reasoning_content")
					.and_then(Value::as_str)
			})
			.collect();
		assert_eq!(text, "Count them.");
		assert_eq!(
			chunks.last().unwrap()["choices"][0]["delta"]["thinking_blocks"][0],
			json!({"type": "thinking", "thinking": "Count them.", "signature": "sig-3"})
		);
	}
}
