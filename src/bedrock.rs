//! Amazon Bedrock's Converse API, spoken for a caller that speaks OpenAI's Chat Completions.
//!
//! Converse is the one Bedrock API every chat model there answers (Claude, OpenAI's models on
//! Bedrock, Qwen, Llama, …), so it is the only one Psyche speaks. A request is built in two steps:
//! [`anthropic::request`] first, which does the checking, the refusals by name, the turn merging
//! and the `cache_control` rule (a breakpoint only for a model whose catalogue entry says
//! `prompt_caching`), and then the Anthropic shape is rewritten as Converse's, block for block. The
//! two APIs were designed alike, so this keeps one set of rules for both rather than two that drift.
//!
//! What Converse does differently, and this file keeps:
//!  - **A cache point is a block of its own**, placed after the block it caches, in `system`, in a
//!    message's content or in `toolConfig.tools`. A model that does not take one (OpenAI's models
//!    on Bedrock cache automatically, and refuse the whole request when they see one) never gets
//!    one, because its catalogue entry does not
//!    say `prompt_caching`.
//!  - **`maxTokens` is optional**, so it is sent only when the caller or the catalogue names one.
//!  - **Images go inline only**: an image URL is refused, since Converse takes bytes (or S3).
//!  - **`tool_choice: none` and `parallel_tool_calls: false` have no Converse equivalent** and are
//!    refused by name rather than ignored.
//!  - **The stream is AWS's event-stream framing** (`eventstream.rs`), and a failure inside it is an
//!    `exception` message: throttling and service-unavailable become `rate_limited`, so the router
//!    can take another target instead of passing them on as a normal close.

use std::collections::HashMap;

use serde_json::{Map, Value, json};

use crate::anthropic;
use crate::error::{ApiError, Class};
use crate::eventstream::Message;
use crate::providers::Model;
pub use crate::translate::Out;
use crate::usage::Usage;

fn bad(message: impl Into<String>) -> ApiError {
	ApiError::new(Class::BadRequest, message)
}

/// The Converse request for a chat completion request. `body` is the caller's, as it came.
pub fn request(body: &Map<String, Value>, model: &Model) -> Result<Value, ApiError> {
	if body.get("parallel_tool_calls") == Some(&Value::Bool(false)) {
		return Err(bad(format!(
			"{} is on Bedrock, whose Converse API cannot turn off parallel tool calls. Leave parallel_tool_calls out.",
			model.name
		)));
	}
	if body.get("tool_choice").and_then(Value::as_str) == Some("none") {
		return Err(bad(format!(
			"{} is on Bedrock, whose Converse API has no tool_choice none. Send the request without tool_choice instead.",
			model.name
		)));
	}
	let shaped = anthropic::request(body, model, false)?;
	let mut additional = Map::new();
	let mut out = Map::new();

	let mut inference = Map::new();
	// Thinking counts against maxTokens, so with it on the limit is always sent.
	let named = ["max_tokens", "max_completion_tokens"]
		.iter()
		.any(|f| body.get(*f).is_some_and(|v| !v.is_null()))
		|| model.max_output_tokens.is_some()
		|| shaped.get("thinking").is_some();
	if named {
		inference.insert("maxTokens".into(), shaped["max_tokens"].clone());
	}
	for (from, to) in [
		("temperature", "temperature"),
		("top_p", "topP"),
		("stop_sequences", "stopSequences"),
	] {
		if let Some(value) = shaped.get(from) {
			inference.insert(to.into(), value.clone());
		}
	}
	if !inference.is_empty() {
		out.insert("inferenceConfig".into(), Value::Object(inference));
	}
	// Reasoning goes where Converse passes a model's own fields, in the form the catalogue named.
	for field in ["top_k", "thinking", "output_config", "reasoning"] {
		if let Some(value) = shaped.get(field) {
			additional.insert(field.into(), value.clone());
		}
	}
	if !additional.is_empty() {
		out.insert(
			"additionalModelRequestFields".into(),
			Value::Object(additional),
		);
	}

	if let Some(system) = shaped.get("system").and_then(Value::as_array) {
		let mut blocks = Vec::new();
		for block in system {
			blocks.push(json!({"text": block["text"]}));
			cache_point(&mut blocks, block);
		}
		out.insert("system".into(), Value::Array(blocks));
	}

	let mut messages = Vec::new();
	for message in shaped["messages"].as_array().into_iter().flatten() {
		let mut content = Vec::new();
		for block in message["content"].as_array().into_iter().flatten() {
			content.push(converse_block(block, model)?);
			cache_point(&mut content, block);
		}
		messages.push(json!({"role": message["role"], "content": content}));
	}
	out.insert("messages".into(), Value::Array(messages));

	if let Some(tools) = shaped.get("tools").and_then(Value::as_array) {
		let mut specs = Vec::new();
		for tool in tools {
			let mut spec =
				json!({"name": tool["name"], "inputSchema": {"json": tool["input_schema"]}});
			if let Some(description) = tool.get("description") {
				spec["description"] = description.clone();
			}
			specs.push(json!({"toolSpec": spec}));
			cache_point(&mut specs, tool);
		}
		let mut config = json!({"tools": specs});
		if let Some(choice) = shaped.get("tool_choice") {
			config["toolChoice"] = match choice["type"].as_str() {
				Some("any") => json!({"any": {}}),
				Some("tool") => json!({"tool": {"name": choice["name"]}}),
				_ => json!({"auto": {}}),
			};
		}
		out.insert("toolConfig".into(), config);
	}
	Ok(Value::Object(out))
}

/// A cache point after `block`, when the Anthropic shape put a breakpoint on it.
fn cache_point(blocks: &mut Vec<Value>, block: &Value) {
	if block.get("cache_control").is_some() {
		blocks.push(json!({"cachePoint": {"type": "default"}}));
	}
}

/// One Anthropic content block as Converse's.
fn converse_block(block: &Value, model: &Model) -> Result<Value, ApiError> {
	Ok(match block["type"].as_str() {
		Some("text") => json!({"text": block["text"]}),
		Some("image") => {
			let source = &block["source"];
			if source["type"] != "base64" {
				return Err(bad(format!(
					"{} is on Bedrock, which takes images inline only: send a base64 data URL, not a link.",
					model.name
				)));
			}
			let format = source["media_type"]
				.as_str()
				.and_then(|m| m.strip_prefix("image/"))
				.map(|f| if f == "jpg" { "jpeg" } else { f })
				.filter(|f| ["png", "jpeg", "gif", "webp"].contains(f))
				.ok_or_else(|| bad("Bedrock takes png, jpeg, gif or webp images."))?;
			json!({"image": {"format": format, "source": {"bytes": source["data"]}}})
		}
		Some("tool_use") => json!({"toolUse": {
			"toolUseId": block["id"], "name": block["name"], "input": block["input"],
		}}),
		Some("thinking") => json!({"reasoningContent": {"reasoningText": {
			"text": block["thinking"], "signature": block["signature"],
		}}}),
		Some("redacted_thinking") => {
			json!({"reasoningContent": {"redactedContent": block["data"]}})
		}
		Some("tool_result") => {
			let mut content: Vec<Value> = block["content"]
				.as_array()
				.into_iter()
				.flatten()
				.map(|b| json!({"text": b["text"]}))
				.collect();
			if content.is_empty() {
				content.push(json!({"text": ""}));
			}
			json!({"toolResult": {"toolUseId": block["tool_use_id"], "content": content}})
		}
		other => {
			return Err(bad(format!(
				"a {} block is not translated for Bedrock.",
				other.unwrap_or("nameless")
			)));
		}
	})
}

/// Converse's `reasoningContent` as an Anthropic thinking block, the one shape a caller sends back
/// whichever provider wrote it.
fn thinking_block(reasoning: &Value) -> Value {
	match reasoning.get("redactedContent") {
		Some(data) => json!({"type": "redacted_thinking", "data": data}),
		None => json!({
			"type": "thinking",
			"thinking": reasoning.pointer("/reasoningText/text").cloned().unwrap_or(json!("")),
			"signature": reasoning.pointer("/reasoningText/signature").cloned().unwrap_or(Value::Null),
		}),
	}
}

/// A chat completion for a Converse reply, or what was wrong with the reply. `model` is the name
/// the caller asked for; Converse has no id of its own for a reply, so `id` is Psyche's.
pub fn reply(value: &Value, model: &str, id: &str, created: u64) -> Result<Value, String> {
	let blocks = value
		.pointer("/output/message/content")
		.and_then(Value::as_array)
		.ok_or("the reply has no output message")?;
	let mut text = String::new();
	let mut calls = Vec::new();
	let mut thinking = Vec::new();
	for block in blocks {
		if let Some(piece) = block.get("text").and_then(Value::as_str) {
			text.push_str(piece);
		} else if let Some(reasoning) = block.get("reasoningContent") {
			thinking.push(thinking_block(reasoning));
		} else if let Some(call) = block.get("toolUse") {
			calls.push(json!({
				"id": call["toolUseId"], "type": "function",
				"function": {"name": call["name"], "arguments": call.get("input").map(Value::to_string).unwrap_or_else(|| "{}".into())},
			}));
		}
	}
	let mut message = json!({
		"role": "assistant",
		"content": if text.is_empty() && !calls.is_empty() { Value::Null } else { Value::String(text) },
	});
	if !calls.is_empty() {
		message["tool_calls"] = Value::Array(calls);
	}
	anthropic::with_reasoning(&mut message, thinking);
	let usage = value
		.get("usage")
		.and_then(Usage::from_bedrock)
		.unwrap_or_default();
	Ok(json!({
		"id": id, "object": "chat.completion", "created": created, "model": model,
		"choices": [{
			"index": 0, "message": message,
			"finish_reason": anthropic::finish_reason(value.get("stopReason").and_then(Value::as_str)),
		}],
		"usage": usage.to_openai(),
	}))
}

/// The class of a failure Bedrock reports, by its exception's name (in the stream's
/// `:exception-type`, or an HTTP error's `x-amzn-ErrorType`), any case.
pub fn class_of(exception: &str) -> Class {
	let name = exception
		.split(':')
		.next()
		.unwrap_or("")
		.to_ascii_lowercase();
	match name.as_str() {
		"throttlingexception" | "serviceunavailableexception" => Class::RateLimited,
		"validationexception" => Class::Rejected,
		"modeltimeoutexception" => Class::Timeout,
		"accessdeniedexception" | "unrecognizedclientexception" => Class::ProviderAuth,
		_ => Class::Upstream,
	}
}

/// A new id for a reply, since Converse gives none.
pub fn new_id() -> String {
	use ring::rand::SecureRandom;
	let mut bytes = [0u8; 12];
	let _ = ring::rand::SystemRandom::new().fill(&mut bytes);
	format!(
		"chatcmpl-{}",
		bytes.iter().map(|b| format!("{b:02x}")).collect::<String>()
	)
}

/// One streamed reply being translated.
#[derive(Debug)]
pub struct Stream {
	wants_usage: bool,
	created: u64,
	id: String,
	model: String,
	usage: Option<Usage>,
	/// Converse's content block index → OpenAI's tool call index, for toolUse blocks.
	tools: HashMap<u64, u64>,
	/// Reasoning blocks being received, by content block index, passed on whole when they end.
	thinking: HashMap<u64, Value>,
	/// `messageStop` arrived.
	stopped: bool,
	/// `metadata`, the last event, arrived.
	ended: bool,
}

impl Stream {
	pub fn new(wants_usage: bool, model: &str, id: &str, created: u64) -> Self {
		Self {
			wants_usage,
			created,
			id: id.to_string(),
			model: model.to_string(),
			usage: None,
			tools: HashMap::new(),
			thinking: HashMap::new(),
			stopped: false,
			ended: false,
		}
	}

	pub fn usage(&self) -> Option<Usage> {
		self.usage
	}

	fn chunk(&self, delta: Value, finish: Value) -> Out {
		Out::Chunk(json!({
			"id": self.id, "object": "chat.completion.chunk", "created": self.created, "model": self.model,
			"choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
		}))
	}

	/// The provider closed the stream: what is still owed, or `None` if the reply was cut. Usage
	/// comes after `messageStop`, so a stream that stopped but sent no usage is complete, uncounted.
	pub fn end(&mut self) -> Option<Vec<Out>> {
		if self.ended {
			Some(Vec::new())
		} else if self.stopped {
			self.ended = true;
			Some(vec![Out::Done])
		} else {
			None
		}
	}

	pub fn message(&mut self, message: &Message) -> Vec<Out> {
		let payload: Value = serde_json::from_slice(&message.payload).unwrap_or(Value::Null);
		match message.header(":message-type") {
			Some("event") => {}
			Some("exception") => {
				let kind = message.header(":exception-type").unwrap_or("");
				let said = payload
					.get("message")
					.or_else(|| payload.get("Message"))
					.and_then(Value::as_str)
					.unwrap_or("an error with no message");
				return vec![Out::Error(class_of(kind), format!("{kind}: {said}"))];
			}
			_ => {
				let code = message.header(":error-code").unwrap_or("error");
				let said = message.header(":error-message").unwrap_or("");
				return vec![Out::Error(Class::Upstream, format!("{code}: {said}"))];
			}
		}
		if payload.is_null() {
			return vec![Out::Error(
				Class::Upstream,
				"Bedrock sent a stream event that is not JSON.".into(),
			)];
		}
		match message.header(":event-type").unwrap_or("") {
			"messageStart" => {
				vec![self.chunk(json!({"role": "assistant", "content": ""}), Value::Null)]
			}
			"contentBlockStart" => {
				let Some(call) = payload.pointer("/start/toolUse") else {
					return Vec::new();
				};
				let index = payload["contentBlockIndex"].as_u64().unwrap_or(0);
				let tool = self.tools.len() as u64;
				self.tools.insert(index, tool);
				vec![self.chunk(
					json!({"tool_calls": [{
						"index": tool, "id": call["toolUseId"], "type": "function",
						"function": {"name": call["name"], "arguments": ""},
					}]}),
					Value::Null,
				)]
			}
			"contentBlockDelta" => {
				let delta = &payload["delta"];
				if let Some(text) = delta.get("text") {
					return vec![self.chunk(json!({"content": text}), Value::Null)];
				}
				let index = payload["contentBlockIndex"].as_u64().unwrap_or(0);
				if let Some(reasoning) = delta.get("reasoningContent") {
					let block = self.thinking.entry(index).or_insert_with(
						|| json!({"type": "thinking", "thinking": "", "signature": ""}),
					);
					if let Some(data) = reasoning.get("redactedContent") {
						*block = json!({"type": "redacted_thinking", "data": data});
						return Vec::new();
					}
					if let Some(signature) = reasoning.get("signature").and_then(Value::as_str) {
						let joined =
							format!("{}{signature}", block["signature"].as_str().unwrap_or(""));
						block["signature"] = json!(joined);
						return Vec::new();
					}
					let piece = reasoning.get("text").and_then(Value::as_str).unwrap_or("");
					let joined = format!("{}{piece}", block["thinking"].as_str().unwrap_or(""));
					block["thinking"] = json!(joined);
					return vec![self.chunk(json!({"reasoning_content": piece}), Value::Null)];
				}
				if let Some(input) = delta.pointer("/toolUse/input") {
					let Some(tool) = self.tools.get(&index).copied() else {
						return Vec::new();
					};
					return vec![self.chunk(
						json!({"tool_calls": [{"index": tool, "function": {"arguments": input}}]}),
						Value::Null,
					)];
				}
				Vec::new()
			}
			"contentBlockStop" => {
				let index = payload["contentBlockIndex"].as_u64().unwrap_or(0);
				match self.thinking.remove(&index) {
					Some(block) => {
						vec![self.chunk(json!({"thinking_blocks": [block]}), Value::Null)]
					}
					None => Vec::new(),
				}
			}
			"messageStop" => {
				self.stopped = true;
				vec![self.chunk(
					json!({}),
					anthropic::finish_reason(payload.get("stopReason").and_then(Value::as_str)),
				)]
			}
			"metadata" => {
				self.usage = payload.get("usage").and_then(Usage::from_bedrock);
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
			// Anything added after this was written.
			_ => Vec::new(),
		}
	}
}

#[cfg(test)]
mod tests {
	use std::sync::Arc;

	use super::*;
	use crate::eventstream::{Decoder, encode};
	use crate::providers::{Kind, Provider};

	fn model(caching: bool) -> Model {
		Model {
			name: "claude".into(),
			provider: Arc::new(Provider {
				name: "bedrock".into(),
				kind: Kind::Bedrock,
				base_url: url::Url::parse("https://bedrock-runtime.us-west-2.amazonaws.com")
					.unwrap(),
				api_key: None,
				aws: None,
				region: None,
				credential: None,
			}),
			upstream: "us.anthropic.claude-sonnet-5".into(),
			prompt_caching: caching,
			max_output_tokens: None,
			reasoning: Some(crate::providers::Reasoning::Budget),
			embeddings: None,
			price: None,
		}
	}

	fn translate(body: Value, caching: bool) -> Result<Value, ApiError> {
		request(body.as_object().unwrap(), &model(caching))
	}

	/// Studio's agent loop, breakpoints included.
	fn agent_loop() -> Value {
		json!({
			"model": "claude", "stream": true, "temperature": 0.2, "stop": "END", "user": "u-1",
			"messages": [
				{"role": "system", "content": [{"type": "text", "text": "You are an assistant.", "cache_control": {"type": "ephemeral"}}]},
				{"role": "user", "content": "How many users?"},
				{"role": "assistant", "content": null, "tool_calls": [
					{"id": "call_1", "type": "function", "function": {"name": "run_sql", "arguments": "{\"sql\":\"select count(*) from users\"}"}}
				]},
				{"role": "tool", "tool_call_id": "call_1", "content": [{"type": "text", "text": "42", "cache_control": {"type": "ephemeral"}}]}
			],
			"tools": [{"type": "function", "cache_control": {"type": "ephemeral"}, "function": {"name": "run_sql", "description": "Runs SQL.", "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}}}}],
			"tool_choice": "required"
		})
	}

	#[test]
	fn an_agent_loop_becomes_a_converse_request() {
		let out = translate(agent_loop(), true).unwrap();
		assert!(out.get("model").is_none(), "the model is in the path");
		assert_eq!(
			out["inferenceConfig"],
			json!({"temperature": 0.2, "stopSequences": ["END"]})
		);
		assert_eq!(
			out["system"],
			json!([{"text": "You are an assistant."}, {"cachePoint": {"type": "default"}}])
		);
		let messages = out["messages"].as_array().unwrap();
		assert_eq!(
			messages[0],
			json!({"role": "user", "content": [{"text": "How many users?"}]})
		);
		assert_eq!(
			messages[1]["content"][0]["toolUse"],
			json!({"toolUseId": "call_1", "name": "run_sql", "input": {"sql": "select count(*) from users"}})
		);
		assert_eq!(
			messages[2]["content"],
			json!([
				{"toolResult": {"toolUseId": "call_1", "content": [{"text": "42"}]}},
				{"cachePoint": {"type": "default"}}
			]),
			"the cache point follows the tool result, never inside it"
		);
		let tools = &out["toolConfig"]["tools"];
		assert_eq!(
			tools[0]["toolSpec"]["inputSchema"]["json"]["properties"]["sql"]["type"],
			"string"
		);
		assert_eq!(tools[1], json!({"cachePoint": {"type": "default"}}));
		assert_eq!(out["toolConfig"]["toolChoice"], json!({"any": {}}));
		assert!(
			!out.to_string().contains("u-1"),
			"user has no Converse field"
		);
	}

	#[test]
	fn a_model_that_caches_by_itself_never_gets_a_cache_point() {
		// gpt-5.6-luna on Bedrock answered every request carrying a cache point with a 500.
		let out = translate(agent_loop(), false).unwrap();
		assert!(!out.to_string().contains("cachePoint"), "{out}");
		assert!(!out.to_string().contains("cache_control"), "{out}");
	}

	#[test]
	fn the_rest_of_a_request() {
		let out = translate(
			json!({"max_tokens": 50, "top_k": 5, "messages": [{"role": "user", "content": [
				{"type": "text", "text": "What is this?"},
				{"type": "image_url", "image_url": {"url": "data:image/jpg;base64,AAAA"}}
			]}]}),
			false,
		)
		.unwrap();
		assert_eq!(out["inferenceConfig"]["maxTokens"], 50);
		assert_eq!(out["additionalModelRequestFields"]["top_k"], 5);
		assert_eq!(
			out["messages"][0]["content"][1],
			json!({"image": {"format": "jpeg", "source": {"bytes": "AAAA"}}})
		);
		let out = translate(
			json!({"messages": [{"role": "user", "content": "hi"}]}),
			false,
		)
		.unwrap();
		assert!(
			out.get("inferenceConfig").is_none(),
			"maxTokens only when named"
		);

		for (body, said) in [
			(json!({"parallel_tool_calls": false}), "parallel"),
			(json!({"tool_choice": "none"}), "tool_choice none"),
			(
				json!({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}]}]}),
				"inline only",
			),
			(json!({"n": 2}), "n must be 1"),
		] {
			let mut body = body;
			if body.get("messages").is_none() {
				body["messages"] = json!([{"role": "user", "content": "hi"}]);
			}
			let error = translate(body, false).unwrap_err();
			assert!(error.message.contains(said), "{said}: {}", error.message);
		}
	}

	#[test]
	fn a_reply_becomes_a_chat_completion() {
		let reply = reply(
			&json!({
				"output": {"message": {"role": "assistant", "content": [
					{"reasoningContent": {"reasoningText": {"text": "thinking"}}},
					{"text": "Let me look."},
					{"toolUse": {"toolUseId": "tooluse_1", "name": "run_sql", "input": {"sql": "select 1"}}}
				]}},
				"stopReason": "tool_use",
				"usage": {"inputTokens": 10, "outputTokens": 7, "totalTokens": 137, "cacheReadInputTokens": 100, "cacheWriteInputTokens": 20}
			}),
			"claude",
			"chatcmpl-x",
			1,
		)
		.unwrap();
		assert_eq!(reply["model"], "claude");
		assert_eq!(reply["id"], "chatcmpl-x");
		let choice = &reply["choices"][0];
		assert_eq!(choice["finish_reason"], "tool_calls");
		assert_eq!(
			choice["message"]["content"], "Let me look.",
			"reasoning is not passed on"
		);
		assert_eq!(choice["message"]["tool_calls"][0]["id"], "tooluse_1");
		assert_eq!(reply["usage"]["prompt_tokens"], 130);
		assert_eq!(reply["usage"]["cache_creation_input_tokens"], 20);
		assert!(super::reply(&json!({}), "m", "i", 1).is_err());
	}

	fn event(kind: &str, payload: Value) -> Vec<u8> {
		encode(
			&[
				(":event-type", kind),
				(":content-type", "application/json"),
				(":message-type", "event"),
			],
			payload.to_string().as_bytes(),
		)
	}

	fn translate_stream(stream: &mut Stream, bytes: &[u8]) -> Vec<Out> {
		let mut decoder = Decoder::new();
		decoder
			.feed(bytes)
			.unwrap()
			.iter()
			.flat_map(|m| stream.message(m))
			.collect()
	}

	#[test]
	fn a_stream_becomes_chunks() {
		let mut bytes = event("messageStart", json!({"role": "assistant"}));
		bytes.extend(event(
			"contentBlockDelta",
			json!({"contentBlockIndex": 0, "delta": {"text": "Hi"}}),
		));
		bytes.extend(event("contentBlockStop", json!({"contentBlockIndex": 0})));
		bytes.extend(event("contentBlockStart", json!({"contentBlockIndex": 1, "start": {"toolUse": {"toolUseId": "tooluse_1", "name": "run_sql"}}})));
		bytes.extend(event(
			"contentBlockDelta",
			json!({"contentBlockIndex": 1, "delta": {"toolUse": {"input": "{\"sql\":"}}}),
		));
		bytes.extend(event(
			"contentBlockDelta",
			json!({"contentBlockIndex": 1, "delta": {"toolUse": {"input": "\"select 1\"}"}}}),
		));
		bytes.extend(event("messageStop", json!({"stopReason": "tool_use"})));
		bytes.extend(event("metadata", json!({"usage": {"inputTokens": 9, "outputTokens": 4, "cacheReadInputTokens": 100}, "metrics": {"latencyMs": 5}})));
		let mut stream = Stream::new(true, "claude", "chatcmpl-x", 1);
		let out = translate_stream(&mut stream, &bytes);
		let chunks: Vec<&Value> = out
			.iter()
			.filter_map(|o| match o {
				Out::Chunk(v) => Some(v),
				_ => None,
			})
			.collect();
		assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
		assert_eq!(chunks[1]["choices"][0]["delta"]["content"], "Hi");
		assert_eq!(
			chunks[2]["choices"][0]["delta"]["tool_calls"][0]["id"],
			"tooluse_1"
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
		assert_eq!(chunks[6]["usage"]["prompt_tokens"], 109);
		assert_eq!(out.last(), Some(&Out::Done));
		assert!(
			chunks
				.iter()
				.all(|c| c["model"] == "claude" && c["id"] == "chatcmpl-x")
		);
		assert_eq!(stream.end(), Some(Vec::new()));
	}

	#[test]
	fn a_stream_without_usage_or_without_its_end() {
		let mut stopped = Stream::new(false, "m", "i", 1);
		translate_stream(
			&mut stopped,
			&event("messageStop", json!({"stopReason": "end_turn"})),
		);
		assert_eq!(
			stopped.end(),
			Some(vec![Out::Done]),
			"stopped, uncounted, complete"
		);
		let mut cut = Stream::new(false, "m", "i", 1);
		translate_stream(
			&mut cut,
			&event("messageStart", json!({"role": "assistant"})),
		);
		assert_eq!(cut.end(), None);
	}

	#[test]
	fn exceptions_are_classified() {
		let throttled = encode(
			&[
				(":message-type", "exception"),
				(":exception-type", "throttlingException"),
			],
			br#"{"message":"Too many tokens, please wait before trying again."}"#,
		);
		let out = translate_stream(&mut Stream::new(false, "m", "i", 1), &throttled);
		assert!(
			matches!(&out[0], Out::Error(Class::RateLimited, m) if m.contains("Too many tokens")),
			"{out:?}"
		);
		assert_eq!(class_of("ServiceUnavailableException"), Class::RateLimited);
		assert_eq!(
			class_of("ValidationException:http://internal.amazon.com/coral/com.amazon.bedrock/"),
			Class::Rejected
		);
		assert_eq!(class_of("modelStreamErrorException"), Class::Upstream);
		assert_eq!(class_of("AccessDeniedException"), Class::ProviderAuth);
	}

	#[test]
	fn reasoning_for_claude_and_for_the_others() {
		let out = translate(
			json!({"reasoning_effort": "low", "messages": [
				{"role": "user", "content": "hi"},
				{"role": "assistant", "content": "Hello.", "thinking_blocks": [
					{"type": "thinking", "thinking": "Greet.", "signature": "sig-1"},
					{"type": "redacted_thinking", "data": "opaque"}
				]},
				{"role": "user", "content": "again"}
			]}),
			false,
		)
		.unwrap();
		assert_eq!(
			out["additionalModelRequestFields"]["thinking"],
			json!({"type": "enabled", "budget_tokens": 1024})
		);
		assert_eq!(
			out["inferenceConfig"]["maxTokens"],
			1024 + 4096,
			"sent whenever thinking is on"
		);
		assert_eq!(
			out["messages"][1]["content"],
			json!([
				{"reasoningContent": {"reasoningText": {"text": "Greet.", "signature": "sig-1"}}},
				{"reasoningContent": {"redactedContent": "opaque"}},
				{"text": "Hello."}
			])
		);
		let mut luna = model(false);
		luna.upstream = "us.openai.gpt-5.6-luna".into();
		luna.reasoning = Some(crate::providers::Reasoning::Effort);
		let out = request(
			json!({"reasoning_effort": "high", "temperature": 0.3, "messages": [{"role": "user", "content": "hi"}]})
				.as_object()
				.unwrap(),
			&luna,
		)
		.unwrap();
		assert_eq!(
			out["additionalModelRequestFields"],
			json!({"reasoning": {"effort": "high"}})
		);
		assert_eq!(
			out["inferenceConfig"],
			json!({"temperature": 0.3}),
			"no thinking budget, no forced maxTokens"
		);

		let reply = reply(
			&json!({"output": {"message": {"role": "assistant", "content": [
				{"reasoningContent": {"reasoningText": {"text": "Count.", "signature": "sig-2"}}},
				{"text": "42."}
			]}}, "stopReason": "end_turn", "usage": {"inputTokens": 1, "outputTokens": 1}}),
			"claude",
			"i",
			1,
		)
		.unwrap();
		let message = &reply["choices"][0]["message"];
		assert_eq!(message["reasoning_content"], "Count.");
		assert_eq!(
			message["thinking_blocks"][0],
			json!({"type": "thinking", "thinking": "Count.", "signature": "sig-2"})
		);
	}

	#[test]
	fn streamed_reasoning() {
		let mut bytes = event("messageStart", json!({"role": "assistant"}));
		bytes.extend(event(
			"contentBlockDelta",
			json!({"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "Count "}}}),
		));
		bytes.extend(event(
			"contentBlockDelta",
			json!({"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "them."}}}),
		));
		bytes.extend(event(
			"contentBlockDelta",
			json!({"contentBlockIndex": 0, "delta": {"reasoningContent": {"signature": "sig-3"}}}),
		));
		bytes.extend(event("contentBlockStop", json!({"contentBlockIndex": 0})));
		let mut stream = Stream::new(false, "claude", "i", 1);
		let chunks: Vec<Value> = translate_stream(&mut stream, &bytes)
			.into_iter()
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
