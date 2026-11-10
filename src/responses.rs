//! OpenAI's Responses API, taken in: `POST /ai/v1/responses`, for a caller written for it, such as
//! Codex with `wire_api = "responses"` and a Psyche key.
//!
//! A call goes through the same keys, budgets, limits, routes, `auto`, needs check, internet level,
//! log and usage as a chat completion (`proxy.rs`). What differs is the shape, and only at the two
//! ends: [`to_chat`] makes a chat completion request for every provider, which the existing
//! translations then carry, and what comes back is a chat completion whichever it was, so one
//! [`reply`] and one [`Events`] turn it into a `response`.
//!
//! The rules this file keeps, which are `messages.rs`'s:
//!  - **Nothing is stored.** `previous_response_id` is refused by name ([`check`]), so a caller keeps
//!    the conversation and sends it whole; `store` changes nothing.
//!  - **What cannot be translated is refused, by name, before anything is sent.** A hosted tool, a
//!    file, an item or a field Psyche does not know. A field that only tunes the service is dropped
//!    ([`DROPPED`]).
//!  - **`reasoning` is how hard a model tries, not what it answers**, so to a model whose catalogue
//!    entry says nothing about reasoning it is dropped rather than refused, and otherwise goes as
//!    `reasoning_effort`, which the existing translations put in the form that entry names. A
//!    `reasoning` item in the history is dropped: a provider's signature does not come back here.
//!    So a model that needs its signed thinking to go on with a tool loop (Claude's) is not asked to
//!    think on a turn that continues one.
//!  - **A `web_search` tool is an offer, held to the model's internet level.** Codex sends one on every
//!    call whatever the model, so it refuses nothing: a model at the level `web` whose provider takes
//!    OpenAI's form gets it in the form its chat completion takes, and any other is sent the call
//!    without it, and so stays off the web.
//!  - **A tool result's image follows it**, in a user turn of its own, after the whole run of results,
//!    since a chat tool message holds text and every result must directly follow its call.
//!  - **A namespace's tools are functions named `<namespace>__<name>`**, and come back with their
//!    `namespace` and their own `name`, since a chat completion has no namespaces.
//!  - **A custom tool is a function taking one string, `input`**, and comes back as a
//!    `custom_tool_call`; its grammar goes in that parameter's description, which is as much of it as
//!    a function can carry.
//!
//! The stream always ends in exactly one `response.completed`, `response.incomplete` or
//! `response.failed`, which carries the whole response: a caller errors out on a stream that closes
//! without one. Parallel tool calls may interleave, so each is its own output item from the moment
//! its name is known, told apart by `output_index`. A custom tool's input is one JSON string the
//! provider sends in pieces, so it is written whole once the reply has ended.

use std::collections::{BTreeMap, HashMap, HashSet};

use serde_json::{Map, Value, json};

use crate::bedrock;
use crate::error::{ApiError, Class};
use crate::fits::{self, WebWay};
use crate::providers::{Kind, Model, Reasoning};
use crate::usage::Usage;

/// Fields that only tune OpenAI's own service, so dropping them changes nothing about the answer a
/// caller is owed. `prompt_cache_key` has done its work by here: a route's conversations stay on one
/// target by it (`health::conversation`).
pub const DROPPED: [&str; 9] = [
	"client_metadata",
	"store",
	"metadata",
	"service_tier",
	"truncation",
	"prompt_cache_key",
	"prompt_cache_retention",
	"safety_identifier",
	"stream_options",
];

/// A comment, which an SSE reader skips and which keeps a quiet stream open.
pub const KEEP_ALIVE: &[u8] = b": ping\n\n";

/// A keep-alive for every this many pieces of a custom tool's input held back.
const KEEP_ALIVE_EVERY: u32 = 32;

fn bad(message: impl Into<String>) -> ApiError {
	ApiError::new(Class::BadRequest, message)
}

/// What the Responses API requires of a request, checked here so every provider is held to it.
pub fn check(body: &Map<String, Value>) -> Result<(), ApiError> {
	if body
		.get("previous_response_id")
		.is_some_and(|id| !id.is_null())
	{
		return Err(bad(
			"previous_response_id is not taken: nothing is stored here. Send the whole conversation in input.",
		));
	}
	match body.get("input") {
		Some(Value::String(_)) => Ok(()),
		Some(Value::Array(items)) if !items.is_empty() => Ok(()),
		_ => Err(bad(
			"input: Field required, a string or a list of at least one item.",
		)),
	}
}

/// What a chat completion's function is called for a tool in a namespace; it has no namespaces.
fn flat_name(namespace: &str, name: &str) -> String {
	format!("{namespace}__{name}")
}

/// What the reply has to know of the request's tools.
#[derive(Debug, Default, Clone)]
pub struct ToolNames {
	/// The custom tools, whose calls go back as `custom_tool_call`.
	custom: HashSet<String>,
	/// A namespaced tool's [`flat_name`], and the namespace and name it goes back as.
	namespaced: HashMap<String, (String, String)>,
}

impl ToolNames {
	fn is_custom(&self, name: &str) -> bool {
		self.custom.contains(name)
	}

	/// The namespace and name of a function the model called, when it is one in a namespace.
	fn namespaced(&self, name: &str) -> Option<(String, String)> {
		self.namespaced.get(name).cloned()
	}
}

#[cfg(test)]
impl FromIterator<String> for ToolNames {
	fn from_iter<I: IntoIterator<Item = String>>(custom: I) -> Self {
		Self {
			custom: custom.into_iter().collect(),
			namespaced: HashMap::new(),
		}
	}
}

/// The names of the request's custom and namespaced tools.
pub fn custom_tools(body: &Map<String, Value>) -> ToolNames {
	let mut names = ToolNames::default();
	for tool in body
		.get("tools")
		.and_then(Value::as_array)
		.into_iter()
		.flatten()
	{
		match (tool["type"].as_str(), tool["name"].as_str()) {
			(Some("custom"), Some(name)) => {
				names.custom.insert(name.to_string());
			}
			(Some("namespace"), Some(namespace)) => {
				for inner in tool["tools"].as_array().into_iter().flatten() {
					let Some(name) = inner["name"].as_str() else {
						continue;
					};
					let flat = flat_name(namespace, name);
					if inner["type"] == "custom" {
						names.custom.insert(flat.clone());
					}
					names
						.namespaced
						.insert(flat, (namespace.to_string(), name.to_string()));
				}
			}
			_ => {}
		}
	}
	names
}

/// The chat completion request for a Responses request. `body` is the caller's, as it came. The
/// reply cap becomes `max_tokens`, for the proxy to name as the provider does.
pub fn to_chat(body: &Map<String, Value>, model: &Model) -> Result<Map<String, Value>, ApiError> {
	let mut out = Map::new();
	let mut effort = None;
	let mut listed = None;
	for (field, value) in body {
		if value.is_null() || DROPPED.contains(&field.as_str()) {
			continue;
		}
		match field.as_str() {
			"input" | "instructions" => {}
			// `plugins` and `web_search_options` are the proxy's and the needs check's, as on a chat
			// completion.
			"model" | "stream" | "temperature" | "top_p" | "user" | "plugins"
			| "web_search_options" => {
				out.insert(field.clone(), value.clone());
			}
			"parallel_tool_calls" => {
				// Converse cannot say this one, and `bedrock.rs` would refuse the whole call; several
				// calls in a turn are still each answered in order by a caller that asked for none.
				if model.provider.kind != Kind::Bedrock || value != &Value::Bool(false) {
					out.insert(field.clone(), value.clone());
				}
			}
			"max_output_tokens" => {
				out.insert("max_tokens".into(), value.clone());
			}
			"tools" => listed = Some(value),
			"tool_choice" => tool_choice(value, &mut out)?,
			"reasoning" => effort = value.get("effort").and_then(Value::as_str),
			"text" => text(value, &mut out)?,
			"include" => include(value)?,
			"background" if value == &Value::Bool(false) => {}
			other => {
				return Err(bad(format!(
					"{other} is not translated for {}, so it would be ignored. Leave it out.",
					model.name
				)));
			}
		}
	}
	if let Some(value) = listed {
		tools(value, model, &mut out)?;
	}
	// A chat completion takes neither of these without a function to apply them to.
	if !out.contains_key("tools") {
		out.remove("tool_choice");
		out.remove("parallel_tool_calls");
	}
	let mut turns = Vec::new();
	if let Some(value) = body.get("instructions").filter(|i| !i.is_null()) {
		let text = value
			.as_str()
			.ok_or_else(|| bad("instructions is a string."))?;
		system(text, &mut turns);
	}
	match body.get("input") {
		Some(Value::String(text)) => turns.push(json!({"role": "user", "content": text})),
		Some(Value::Array(items)) => input(items, model, &mut turns)?,
		_ => {
			return Err(bad(
				"input: Field required, a string or a list of at least one item.",
			));
		}
	}
	if turns.iter().all(|turn| turn["role"] == "system") {
		return Err(bad("input holds nothing for the model to answer."));
	}
	reasoning(effort, model, &turns, &mut out);
	out.insert("messages".into(), Value::Array(turns));
	Ok(out)
}

/// `reasoning.effort`, for a model whose catalogue entry names a form for it. A signed thinking block
/// cannot come back through this API, and Claude refuses a tool loop's next turn with thinking on and
/// none on the turn it continues.
fn reasoning(effort: Option<&str>, model: &Model, turns: &[Value], out: &mut Map<String, Value>) {
	let (Some(effort), Some(form)) = (effort, model.reasoning) else {
		return;
	};
	let signs = model.provider.kind != Kind::Openai
		&& matches!(form, Reasoning::Budget | Reasoning::Adaptive);
	let continues_tools = turns
		.iter()
		.rev()
		.find(|turn| turn["role"] == "assistant")
		.is_some_and(|turn| turn.get("tool_calls").is_some());
	if signs && continues_tools {
		return;
	}
	let effort = match effort {
		"xhigh" | "max" => "high",
		other => other,
	};
	out.insert("reasoning_effort".into(), json!(effort));
}

fn include(value: &Value) -> Result<(), ApiError> {
	let list = value
		.as_array()
		.ok_or_else(|| bad("include is a list of names."))?;
	match list
		.iter()
		.find(|name| name.as_str() != Some("reasoning.encrypted_content"))
	{
		// There is no encrypted reasoning to give back, which a caller reads as there being none.
		None => Ok(()),
		Some(name) => Err(bad(format!(
			"include {} is not translated, so it would be ignored. Leave it out.",
			name.as_str().unwrap_or("that is not a name")
		))),
	}
}

fn text(value: &Value, out: &mut Map<String, Value>) -> Result<(), ApiError> {
	let settings = value.as_object().ok_or_else(|| bad("text is an object."))?;
	for (field, setting) in settings {
		if setting.is_null() || field == "verbosity" {
			continue;
		}
		if field != "format" {
			return Err(bad(format!(
				"text.{field} is not translated, so it would be ignored. Leave it out."
			)));
		}
		match setting.get("type").and_then(Value::as_str) {
			Some("text") => {}
			Some("json_object") => {
				out.insert("response_format".into(), json!({"type": "json_object"}));
			}
			Some("json_schema") => {
				let mut schema = json!({
					"name": setting.get("name").filter(|n| n.is_string()).cloned().unwrap_or(json!("output")),
					"schema": setting["schema"],
				});
				for carried in ["strict", "description"] {
					if let Some(given) = setting.get(carried).filter(|v| !v.is_null()) {
						schema[carried] = given.clone();
					}
				}
				out.insert(
					"response_format".into(),
					json!({"type": "json_schema", "json_schema": schema}),
				);
			}
			other => {
				return Err(bad(format!(
					"text.format {} is not one Psyche knows.",
					other.unwrap_or("with no type")
				)));
			}
		}
	}
	Ok(())
}

fn tools(value: &Value, model: &Model, out: &mut Map<String, Value>) -> Result<(), ApiError> {
	let list = value.as_array().ok_or_else(|| bad("tools is a list."))?;
	let mut functions = Vec::with_capacity(list.len());
	for (i, tool) in list.iter().enumerate() {
		match tool.get("type").and_then(Value::as_str) {
			Some("function") => functions.push(function_tool(tool, i)?),
			Some("custom") => functions.push(custom_tool(tool, i)?),
			Some("namespace") => namespace_tools(tool, i, &mut functions)?,
			// What `:online` wrote for Anthropic, with its bound.
			Some(kind) if fits::is_anthropic_web_tool(kind) => functions.push(tool.clone()),
			Some(kind) if kind.starts_with("web_search") => {
				if fits::takes_offered_search(model) {
					web_tool(tool, i, model, out, &mut functions)?;
				}
			}
			other => {
				return Err(bad(format!(
					"tools[{i}] ({}) is run by OpenAI, and has no form on {}.",
					other.unwrap_or("a tool with no type"),
					model.name
				)));
			}
		}
	}
	if !functions.is_empty() {
		out.insert("tools".into(), Value::Array(functions));
	}
	Ok(())
}

/// A tool's `name`, which every kind but the web's must have.
fn tool_name(tool: &Value, i: usize) -> Result<&Value, ApiError> {
	tool.get("name")
		.filter(|n| n.is_string())
		.ok_or_else(|| bad(format!("tools[{i}] has no name.")))
}

fn function_tool(tool: &Value, i: usize) -> Result<Value, ApiError> {
	let mut function = json!({
		"name": tool_name(tool, i)?,
		"parameters": tool
			.get("parameters")
			.filter(|p| p.is_object())
			.cloned()
			.unwrap_or_else(|| json!({"type": "object", "properties": {}})),
	});
	if let Some(description) = tool.get("description").filter(|d| d.is_string()) {
		function["description"] = description.clone();
	}
	if tool.get("strict") == Some(&Value::Bool(true)) {
		function["strict"] = Value::Bool(true);
	}
	Ok(json!({"type": "function", "function": function}))
}

/// Each tool of a namespace, as a function of its own under its [`flat_name`].
fn namespace_tools(tool: &Value, i: usize, functions: &mut Vec<Value>) -> Result<(), ApiError> {
	let namespace = tool_name(tool, i)?.as_str().unwrap_or_default();
	let inner = tool["tools"]
		.as_array()
		.ok_or_else(|| bad(format!("tools[{i}] is a namespace with no tools.")))?;
	for member in inner {
		let mut function = match member["type"].as_str() {
			Some("function") => function_tool(member, i)?,
			Some("custom") => custom_tool(member, i)?,
			other => {
				return Err(bad(format!(
					"tools[{i}] holds a tool of type {}, which has no translation.",
					other.unwrap_or("none")
				)));
			}
		};
		let name = tool_name(member, i)?.as_str().unwrap_or_default();
		function["function"]["name"] = json!(flat_name(namespace, name));
		functions.push(function);
	}
	Ok(())
}

fn custom_tool(tool: &Value, i: usize) -> Result<Value, ApiError> {
	let mut input = json!({"type": "string"});
	if let Some(definition) = tool.pointer("/format/definition").and_then(Value::as_str) {
		let syntax = tool
			.pointer("/format/syntax")
			.and_then(Value::as_str)
			.unwrap_or("a");
		input["description"] = json!(format!("Written in this {syntax} grammar:\n{definition}"));
	}
	let mut function = json!({
		"name": tool_name(tool, i)?,
		"parameters": {
			"type": "object",
			"properties": {"input": input},
			"required": ["input"],
			"additionalProperties": false,
		},
	});
	if let Some(description) = tool.get("description").filter(|d| d.is_string()) {
		function["description"] = description.clone();
	}
	Ok(json!({"type": "function", "function": function}))
}

/// A web tool, in the form this provider's chat completion takes. The needs check has already said
/// the model may have it; one whose provider has no such form was left out there, and any other
/// server takes it as written, which is its owner's to say.
fn web_tool(
	tool: &Value,
	i: usize,
	model: &Model,
	out: &mut Map<String, Value>,
	functions: &mut Vec<Value>,
) -> Result<(), ApiError> {
	if let Some((field, _)) = tool
		.as_object()
		.into_iter()
		.flatten()
		.find(|(field, value)| {
			!value.is_null()
				&& !matches!(
					field.as_str(),
					"type" | "description" | "search_context_size" | "external_web_access"
				)
		}) {
		return Err(bad(format!(
			"tools[{i}].{field} is not translated, so it would be ignored. Leave it out."
		)));
	}
	match WebWay::of(&model.provider) {
		WebWay::OpenAi => {
			let options = out.entry("web_search_options").or_insert_with(|| json!({}));
			if let Some(size) = tool.get("search_context_size").filter(|s| s.is_string()) {
				options["search_context_size"] = size.clone();
			}
		}
		WebWay::OpenRouter => {
			let plugins = out.entry("plugins").or_insert_with(|| json!([]));
			if let Value::Array(plugins) = plugins
				&& !plugins.iter().any(|p| p["id"] == "web")
			{
				plugins.push(json!({"id": "web"}));
			}
		}
		_ => functions.push(tool.clone()),
	}
	Ok(())
}

fn tool_choice(value: &Value, out: &mut Map<String, Value>) -> Result<(), ApiError> {
	let choice = match value {
		Value::String(mode) if matches!(mode.as_str(), "auto" | "none" | "required") => {
			value.clone()
		}
		_ => match value.get("type").and_then(Value::as_str) {
			Some("function" | "custom") => {
				json!({"type": "function", "function": {"name": value["name"]}})
			}
			other => {
				return Err(bad(format!(
					"tool_choice {} is not translated.",
					other.or(value.as_str()).unwrap_or("with no type")
				)));
			}
		},
	};
	out.insert("tool_choice".into(), choice);
	Ok(())
}

/// A system prompt, joined to the one before it: many servers take one, and only first.
fn system(text: &str, turns: &mut Vec<Value>) {
	if text.is_empty() {
		return;
	}
	if let Some(last) = turns.last_mut().filter(|t| t["role"] == "system")
		&& let Some(before) = last["content"].as_str()
	{
		last["content"] = json!(format!("{before}\n\n{text}"));
		return;
	}
	turns.push(json!({"role": "system", "content": text}));
}

fn input(items: &[Value], model: &Model, turns: &mut Vec<Value>) -> Result<(), ApiError> {
	// Images from tool results, held until the run of results ends.
	let mut images = Vec::new();
	for (i, item) in items.iter().enumerate() {
		let at = |what: &str| bad(format!("input[{i}]: {what}"));
		let kind = item
			.get("type")
			.and_then(Value::as_str)
			.or(item.get("role").map(|_| "message"));
		match kind {
			Some("reasoning") => continue,
			Some("function_call_output" | "custom_tool_call_output") => {
				tool_output(item, turns, &mut images, &at)?;
				continue;
			}
			_ => follow_with_images(&mut images, turns),
		}
		match kind {
			Some("message") => message(item, turns, &at)?,
			Some("function_call") => {
				let arguments = item
					.get("arguments")
					.and_then(Value::as_str)
					.ok_or_else(|| at("a function_call has no arguments."))?;
				match (item["namespace"].as_str(), item["name"].as_str()) {
					(Some(namespace), Some(name)) => {
						let mut flat = item.clone();
						flat["name"] = json!(flat_name(namespace, name));
						tool_call(&flat, arguments, turns, &at)?;
					}
					_ => tool_call(item, arguments, turns, &at)?,
				}
			}
			Some("custom_tool_call") => {
				let input = item
					.get("input")
					.and_then(Value::as_str)
					.ok_or_else(|| at("a custom_tool_call has no input."))?;
				tool_call(item, &json!({"input": input}).to_string(), turns, &at)?;
			}
			other => {
				return Err(at(&format!(
					"a {} item is not translated for {}.",
					other.unwrap_or("nameless"),
					model.name
				)));
			}
		}
	}
	follow_with_images(&mut images, turns);
	Ok(())
}

fn follow_with_images(images: &mut Vec<Value>, turns: &mut Vec<Value>) {
	if !images.is_empty() {
		turns.push(json!({"role": "user", "content": std::mem::take(images)}));
	}
}

/// An image part as a chat content part. `detail` only tunes the service unless it is one of the two
/// a chat completion names.
fn image_part(part: &Value, at: &dyn Fn(&str) -> ApiError) -> Result<Value, ApiError> {
	let url = part
		.get("image_url")
		.and_then(Value::as_str)
		.filter(|url| !url.is_empty())
		.ok_or_else(|| at("an input_image with no image_url is not translated."))?;
	let mut image = json!({"url": url});
	if let Some(detail @ ("low" | "high")) = part.get("detail").and_then(Value::as_str) {
		image["detail"] = json!(detail);
	}
	Ok(json!({"type": "image_url", "image_url": image}))
}

/// Content as chat parts; an empty text is left out, which no provider takes.
fn parts_of(content: &Value, at: &dyn Fn(&str) -> ApiError) -> Result<Vec<Value>, ApiError> {
	let list = match content {
		Value::Null => return Ok(Vec::new()),
		Value::String(text) if text.is_empty() => return Ok(Vec::new()),
		Value::String(text) => return Ok(vec![json!({"type": "text", "text": text})]),
		Value::Array(list) => list,
		_ => return Err(at("content is a string or a list of parts.")),
	};
	let mut parts = Vec::with_capacity(list.len());
	for part in list {
		match part.get("type").and_then(Value::as_str) {
			Some("input_text" | "output_text") => {
				let text = part
					.get("text")
					.and_then(Value::as_str)
					.ok_or_else(|| at("a text part has no text."))?;
				if !text.is_empty() {
					parts.push(json!({"type": "text", "text": text}));
				}
			}
			Some("input_image") => parts.push(image_part(part, at)?),
			other => {
				return Err(at(&format!(
					"a {} part is not translated.",
					other.unwrap_or("nameless")
				)));
			}
		}
	}
	Ok(parts)
}

/// The text of chat parts, joined.
fn text_of(parts: &[Value]) -> String {
	parts
		.iter()
		.filter_map(|part| part["text"].as_str())
		.collect::<Vec<_>>()
		.join("")
}

fn message(
	item: &Value,
	turns: &mut Vec<Value>,
	at: &dyn Fn(&str) -> ApiError,
) -> Result<(), ApiError> {
	let parts = parts_of(&item["content"], at)?;
	let role = match item.get("role").and_then(Value::as_str) {
		Some(role @ ("user" | "assistant")) => role,
		Some("system" | "developer") => {
			if parts.iter().any(|part| part["type"] != "text") {
				return Err(at("a system message holds text only."));
			}
			system(&text_of(&parts), turns);
			return Ok(());
		}
		Some(other) => return Err(at(&format!("{other} is not a role Psyche knows."))),
		None => return Err(at("it has no role.")),
	};
	if parts.is_empty() {
		return Ok(());
	}
	let content = if parts.iter().all(|part| part["type"] == "text") {
		json!(text_of(&parts))
	} else {
		Value::Array(parts)
	};
	turns.push(json!({"role": role, "content": content}));
	Ok(())
}

/// A call, on the assistant turn it belongs to: parallel calls are one turn.
fn tool_call(
	item: &Value,
	arguments: &str,
	turns: &mut Vec<Value>,
	at: &dyn Fn(&str) -> ApiError,
) -> Result<(), ApiError> {
	let (Some(id), Some(name)) = (
		item.get("call_id").filter(|id| id.is_string()),
		item.get("name").filter(|name| name.is_string()),
	) else {
		return Err(at("a tool call has no call_id or no name."));
	};
	let call = json!({
		"id": id, "type": "function",
		"function": {"name": name, "arguments": arguments},
	});
	if let Some(last) = turns.last_mut().filter(|t| t["role"] == "assistant") {
		match last.get_mut("tool_calls").and_then(Value::as_array_mut) {
			Some(calls) => calls.push(call),
			None => last["tool_calls"] = json!([call]),
		}
		return Ok(());
	}
	turns.push(json!({"role": "assistant", "content": null, "tool_calls": [call]}));
	Ok(())
}

/// A call's output as a chat tool message; any image it held waits for the run of results to end.
fn tool_output(
	item: &Value,
	turns: &mut Vec<Value>,
	images: &mut Vec<Value>,
	at: &dyn Fn(&str) -> ApiError,
) -> Result<(), ApiError> {
	let id = item
		.get("call_id")
		.filter(|id| id.is_string())
		.ok_or_else(|| at("a tool output has no call_id."))?;
	let parts = parts_of(&item["output"], at)?;
	let mut text = text_of(&parts);
	let held = parts.into_iter().filter(|part| part["type"] != "text");
	let before = images.len();
	images.extend(held);
	if text.is_empty() && images.len() > before {
		text = "The result is the image that follows.".to_string();
	}
	turns.push(json!({"role": "tool", "tool_call_id": id, "content": text}));
	Ok(())
}

/// What a call used, as the Responses API counts it: the cache is inside `input_tokens`.
fn usage_of(usage: &Usage) -> Value {
	json!({
		"input_tokens": usage.prompt_tokens,
		"input_tokens_details": {"cached_tokens": usage.cached_tokens.unwrap_or(0)},
		"output_tokens": usage.completion_tokens,
		"output_tokens_details": {"reasoning_tokens": usage.reasoning_tokens.unwrap_or(0)},
		"total_tokens": usage.total(),
	})
}

/// A `response` with nothing in it yet.
fn response_shell(id: &str, created_at: u64, model: &str, status: &str) -> Value {
	json!({
		"id": id,
		"object": "response",
		"created_at": created_at,
		"status": status,
		"error": null,
		"incomplete_details": null,
		"model": model,
		"output": [],
		"usage": null,
	})
}

/// The status a chat completion's `finish_reason` ends a response in, and why if it stopped short.
fn status_of(finish: Option<&str>) -> (&'static str, Value) {
	match finish {
		Some("length") => ("incomplete", json!({"reason": "max_output_tokens"})),
		Some("content_filter") => ("incomplete", json!({"reason": "content_filter"})),
		_ => ("completed", Value::Null),
	}
}

fn new_id() -> String {
	bedrock::new_id().replacen("chatcmpl-", "", 1)
}

fn message_item(id: &str, text: &str) -> Value {
	json!({
		"id": id, "type": "message", "status": "completed", "role": "assistant",
		"content": [{"type": "output_text", "text": text, "annotations": []}],
	})
}

/// The reasoning a person can read. It has no `encrypted_content`, so it cannot be sent back to be
/// continued from, and there is nothing to send back: a `reasoning` item in the history is dropped.
fn reasoning_item(id: &str, text: &str) -> Value {
	json!({
		"id": id, "type": "reasoning", "summary": [],
		"content": [{"type": "reasoning_text", "text": text}],
	})
}

/// A tool call being received, or whole.
#[derive(Debug, Default)]
struct Call {
	/// The output item it is, once it has been announced.
	index: Option<u64>,
	item: String,
	id: String,
	name: String,
	arguments: String,
	custom: bool,
	/// Set when the model called a tool in a namespace; `name` is then the tool's own.
	namespace: Option<String>,
}

impl Call {
	/// No arguments is an empty object.
	fn arguments(&self) -> &str {
		if self.arguments.trim().is_empty() {
			"{}"
		} else {
			&self.arguments
		}
	}

	/// What a custom tool was given, from the one string parameter it is called with here.
	fn input(&self) -> String {
		serde_json::from_str::<Value>(&self.arguments)
			.ok()
			.and_then(|arguments| arguments["input"].as_str().map(String::from))
			.unwrap_or_else(|| self.arguments.clone())
	}

	fn item_json(&self, status: &str) -> Value {
		let done = status == "completed";
		if self.custom {
			json!({
				"id": self.item, "type": "custom_tool_call", "status": status,
				"call_id": self.id, "name": self.name,
				"input": if done { self.input() } else { String::new() },
			})
		} else {
			let mut item = json!({
				"id": self.item, "type": "function_call", "status": status,
				"call_id": self.id, "name": self.name,
				"arguments": if done { self.arguments() } else { "" },
			});
			if let Some(namespace) = &self.namespace {
				item["namespace"] = json!(namespace);
			}
			item
		}
	}
}

/// A `response` for a chat completion, or what was wrong with it. `usage` is what the proxy read from
/// the provider, whose own shape says more than the chat completion's does.
pub fn reply(
	chat: &Value,
	usage: Option<Usage>,
	model: &str,
	custom: &ToolNames,
	created_at: u64,
) -> Result<Value, String> {
	let choice = chat
		.pointer("/choices/0")
		.ok_or("the reply has no choices")?;
	let message = &choice["message"];
	let suffix = new_id();
	let mut output = Vec::new();
	if let Some(text) = message["reasoning_content"]
		.as_str()
		.filter(|t| !t.is_empty())
	{
		output.push(reasoning_item(&format!("rs_{suffix}"), text));
	}
	if let Some(text) = message["content"].as_str().filter(|t| !t.is_empty()) {
		output.push(message_item(&format!("msg_{suffix}"), text));
	}
	for (n, call) in message["tool_calls"]
		.as_array()
		.into_iter()
		.flatten()
		.enumerate()
	{
		let name = call["function"]["name"].as_str().unwrap_or("").to_string();
		let id = call["id"]
			.as_str()
			.filter(|id| !id.is_empty())
			.map_or_else(|| format!("call_{suffix}_{n}"), String::from);
		let (namespace, name) = match custom.namespaced(&name) {
			Some((namespace, own)) => (Some(namespace), own),
			None => (None, name),
		};
		let flat = namespace
			.as_deref()
			.map_or_else(|| name.clone(), |n| flat_name(n, &name));
		let call = Call {
			index: None,
			item: format!("fc_{suffix}_{n}"),
			id,
			custom: custom.is_custom(&flat),
			namespace,
			name,
			arguments: call["function"]["arguments"]
				.as_str()
				.unwrap_or("")
				.to_string(),
		};
		output.push(call.item_json("completed"));
	}
	let (status, incomplete) = status_of(choice["finish_reason"].as_str());
	let mut response = response_shell(&format!("resp_{suffix}"), created_at, model, status);
	response["incomplete_details"] = incomplete;
	response["output"] = Value::Array(output);
	response["usage"] = usage_of(&usage.unwrap_or_default());
	Ok(response)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kept {
	Text,
	Reasoning,
}

impl Kept {
	/// The content part an item of this kind holds.
	fn part(self, text: &str) -> Value {
		match self {
			Kept::Text => json!({"type": "output_text", "text": text, "annotations": []}),
			Kept::Reasoning => json!({"type": "reasoning_text", "text": text}),
		}
	}

	fn event(self, what: &str) -> String {
		match self {
			Kept::Text => format!("response.output_text.{what}"),
			Kept::Reasoning => format!("response.reasoning_text.{what}"),
		}
	}
}

/// The text or reasoning item being written.
#[derive(Debug)]
struct Open {
	kind: Kept,
	id: String,
	index: u64,
	text: String,
}

/// One chat completion stream being written as the Responses API's events.
#[derive(Debug)]
pub struct Events {
	suffix: String,
	created_at: u64,
	model: String,
	custom: ToolNames,
	sequence: u64,
	started: bool,
	/// The `output_index` the next item takes.
	next: u64,
	open: Option<Open>,
	calls: BTreeMap<u64, Call>,
	/// Finished items, by `output_index`.
	items: BTreeMap<u64, Value>,
	/// Pieces of custom tool input held back so far, which the keep-alives are counted from.
	held: u32,
	finish: Option<String>,
	usage: Usage,
	done: bool,
}

impl Events {
	/// `model` is the name the caller asked for, which the reply carries. `custom` is
	/// [`custom_tools`] of the request.
	pub fn new(model: &str, custom: ToolNames, created_at: u64) -> Self {
		Self {
			suffix: new_id(),
			created_at,
			model: model.to_string(),
			custom,
			sequence: 0,
			started: false,
			next: 0,
			open: None,
			calls: BTreeMap::new(),
			items: BTreeMap::new(),
			held: 0,
			finish: None,
			usage: Usage::default(),
			done: false,
		}
	}

	/// Whether the provider said the reply had finished, which a stream that closes without saying
	/// `[DONE]` still may have.
	pub fn finished(&self) -> bool {
		self.done || self.finish.is_some()
	}

	fn emit(&mut self, kind: &str, mut data: Value, out: &mut Vec<u8>) {
		data["type"] = json!(kind);
		data["sequence_number"] = json!(self.sequence);
		self.sequence += 1;
		out.extend(format!("event: {kind}\ndata: {data}\n\n").into_bytes());
	}

	fn shell(&self, status: &str) -> Value {
		response_shell(
			&format!("resp_{}", self.suffix),
			self.created_at,
			&self.model,
			status,
		)
	}

	fn start(&mut self, out: &mut Vec<u8>) {
		if self.started {
			return;
		}
		self.started = true;
		let response = self.shell("in_progress");
		self.emit(
			"response.created",
			json!({"response": response.clone()}),
			out,
		);
		self.emit("response.in_progress", json!({"response": response}), out);
	}

	fn take_index(&mut self) -> u64 {
		let index = self.next;
		self.next += 1;
		index
	}

	/// Ends the text or reasoning item being written.
	fn close(&mut self, out: &mut Vec<u8>) {
		let Some(open) = self.open.take() else {
			return;
		};
		let place = json!({
			"item_id": open.id, "output_index": open.index, "content_index": 0,
		});
		let with = |field: &str, value: Value| {
			let mut data = place.clone();
			data[field] = value;
			data
		};
		self.emit(
			&open.kind.event("done"),
			with("text", json!(open.text)),
			out,
		);
		self.emit(
			"response.content_part.done",
			with("part", open.kind.part(&open.text)),
			out,
		);
		let item = match open.kind {
			Kept::Text => message_item(&open.id, &open.text),
			Kept::Reasoning => reasoning_item(&open.id, &open.text),
		};
		self.emit(
			"response.output_item.done",
			json!({"output_index": open.index, "item": item}),
			out,
		);
		self.items.insert(open.index, item);
	}

	/// Writes one piece of text or reasoning, opening its item unless it is the one being written.
	fn write(&mut self, kind: Kept, piece: &str, out: &mut Vec<u8>) {
		if self.open.as_ref().map(|open| open.kind) != Some(kind) {
			self.close(out);
			let index = self.take_index();
			let (id, item) = match kind {
				Kept::Text => {
					let id = format!("msg_{}_{index}", self.suffix);
					let item = json!({
						"id": id, "type": "message", "status": "in_progress",
						"role": "assistant", "content": [],
					});
					(id, item)
				}
				Kept::Reasoning => {
					let id = format!("rs_{}_{index}", self.suffix);
					let item = json!({
						"id": id, "type": "reasoning", "summary": [], "content": [],
					});
					(id, item)
				}
			};
			self.emit(
				"response.output_item.added",
				json!({"output_index": index, "item": item}),
				out,
			);
			self.emit(
				"response.content_part.added",
				json!({
					"item_id": id, "output_index": index, "content_index": 0,
					"part": kind.part(""),
				}),
				out,
			);
			self.open = Some(Open {
				kind,
				id,
				index,
				text: String::new(),
			});
		}
		let Some(open) = self.open.as_mut() else {
			return;
		};
		open.text.push_str(piece);
		let (id, index) = (open.id.clone(), open.index);
		self.emit(
			&kind.event("delta"),
			json!({
				"item_id": id, "output_index": index, "content_index": 0, "delta": piece,
			}),
			out,
		);
	}

	/// Makes a call an output item, with what has come of its arguments so far.
	fn announce(&mut self, call: &mut Call, out: &mut Vec<u8>) {
		self.close(out);
		let index = self.take_index();
		call.index = Some(index);
		call.custom = self.custom.is_custom(&call.name);
		if let Some((namespace, name)) = self.custom.namespaced(&call.name) {
			call.namespace = Some(namespace);
			call.name = name;
		}
		call.item = format!("fc_{}_{index}", self.suffix);
		if call.id.is_empty() {
			call.id = format!("call_{}_{index}", self.suffix);
		}
		self.emit(
			"response.output_item.added",
			json!({"output_index": index, "item": call.item_json("in_progress")}),
			out,
		);
		if !call.custom && !call.arguments.is_empty() {
			let delta = call.arguments.clone();
			self.arguments_delta(call, &delta, out);
		}
	}

	fn arguments_delta(&mut self, call: &Call, delta: &str, out: &mut Vec<u8>) {
		self.emit(
			"response.function_call_arguments.delta",
			json!({"item_id": call.item, "output_index": call.index, "delta": delta}),
			out,
		);
	}

	fn tool_piece(&mut self, piece: &Value, out: &mut Vec<u8>) {
		let key = piece["index"].as_u64().unwrap_or(0);
		let mut call = self.calls.remove(&key).unwrap_or_default();
		if let Some(id) = piece["id"].as_str() {
			call.id = id.to_string();
		}
		if let Some(name) = piece["function"]["name"].as_str() {
			call.name.push_str(name);
		}
		let arguments = piece["function"]["arguments"].as_str().unwrap_or("");
		call.arguments.push_str(arguments);
		if call.index.is_some() {
			if call.custom {
				self.held = self.held.wrapping_add(1);
				if self.held.is_multiple_of(KEEP_ALIVE_EVERY) {
					out.extend_from_slice(KEEP_ALIVE);
				}
			} else if !arguments.is_empty() {
				self.arguments_delta(&call, arguments, out);
			}
		} else if !call.name.is_empty() {
			self.announce(&mut call, out);
		}
		self.calls.insert(key, call);
	}

	/// The events for one chunk.
	pub fn chunk(&mut self, chunk: &Value) -> Vec<u8> {
		let mut out = Vec::new();
		if self.done {
			return out;
		}
		self.start(&mut out);
		if let Some(usage) = Usage::from_openai(chunk) {
			self.usage = usage;
		}
		let Some(choice) = chunk.pointer("/choices/0") else {
			return out;
		};
		let delta = &choice["delta"];
		if let Some(piece) = delta["reasoning_content"]
			.as_str()
			.filter(|p| !p.is_empty())
		{
			self.write(Kept::Reasoning, piece, &mut out);
		}
		if let Some(text) = delta["content"].as_str().filter(|t| !t.is_empty()) {
			self.write(Kept::Text, text, &mut out);
		}
		for piece in delta["tool_calls"].as_array().into_iter().flatten() {
			self.tool_piece(piece, &mut out);
		}
		if let Some(finish) = choice["finish_reason"].as_str() {
			self.finish = Some(finish.to_string());
		}
		out
	}

	fn finish_call(&mut self, mut call: Call, out: &mut Vec<u8>) {
		if call.index.is_none() {
			self.announce(&mut call, out);
		}
		let place = json!({"item_id": call.item, "output_index": call.index});
		let mut data = place.clone();
		if call.custom {
			let input = call.input();
			if !input.is_empty() {
				data["delta"] = json!(input);
				self.emit("response.custom_tool_call_input.delta", data, out);
				data = place.clone();
			}
			data["input"] = json!(input);
			self.emit("response.custom_tool_call_input.done", data, out);
		} else {
			data["name"] = json!(call.name);
			data["arguments"] = json!(call.arguments());
			self.emit("response.function_call_arguments.done", data, out);
		}
		let item = call.item_json("completed");
		self.emit(
			"response.output_item.done",
			json!({"output_index": call.index, "item": item}),
			out,
		);
		if let Some(index) = call.index {
			self.items.insert(index, item);
		}
	}

	/// The items written so far, in `output_index` order.
	fn take_output(&mut self) -> Value {
		Value::Array(std::mem::take(&mut self.items).into_values().collect())
	}

	/// The end of the reply: the open items, then the one event that says how it ended, once.
	pub fn end(&mut self) -> Vec<u8> {
		let mut out = Vec::new();
		if self.done {
			return out;
		}
		self.done = true;
		self.start(&mut out);
		self.close(&mut out);
		for call in std::mem::take(&mut self.calls).into_values() {
			self.finish_call(call, &mut out);
		}
		let (status, incomplete) = status_of(self.finish.as_deref());
		let mut response = self.shell(status);
		response["incomplete_details"] = incomplete;
		response["output"] = self.take_output();
		response["usage"] = usage_of(&self.usage);
		self.emit(
			&format!("response.{status}"),
			json!({"response": response}),
			&mut out,
		);
		out
	}

	/// A failure once the stream has started: `response.failed`, the one thing a caller reads as the
	/// stream's end when the reply did not finish.
	pub fn failed(&mut self, error: &ApiError) -> Vec<u8> {
		let mut out = Vec::new();
		if self.done {
			return out;
		}
		self.done = true;
		self.start(&mut out);
		let code = match error.class {
			Class::RateLimited => "rate_limit_exceeded",
			class => class.code(),
		};
		let mut response = self.shell("failed");
		response["error"] = json!({"code": code, "message": error.message});
		response["output"] = self.take_output();
		response["usage"] = usage_of(&self.usage);
		self.emit("response.failed", json!({"response": response}), &mut out);
		out
	}
}

#[cfg(test)]
mod tests {
	use std::sync::Arc;

	use super::*;
	use crate::providers::Provider;

	fn model_at(kind: Kind, host: &str, reasoning: Option<Reasoning>) -> Model {
		Model {
			name: "m".into(),
			provider: Arc::new(Provider {
				name: "p".into(),
				kind,
				base_url: url::Url::parse(&format!("https://{host}/v1")).unwrap(),
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

	fn model(kind: Kind, reasoning: Option<Reasoning>) -> Model {
		model_at(kind, "api.example.com", reasoning)
	}

	fn chat(body: Value) -> Result<Value, ApiError> {
		to_chat(body.as_object().unwrap(), &model(Kind::Openai, None)).map(Value::Object)
	}

	fn codex_turn() -> Value {
		json!({
			"model": "m", "stream": true, "store": false, "instructions": "You are an agent.",
			"prompt_cache_key": "conversation-1", "service_tier": "auto", "truncation": "auto",
			"include": ["reasoning.encrypted_content"], "metadata": {"a": "b"},
			"parallel_tool_calls": true, "tool_choice": "auto", "max_output_tokens": 500,
			"text": {"verbosity": "low"}, "reasoning": {"effort": "high", "summary": "auto"},
			"tools": [
				{"type": "function", "name": "shell", "description": "Runs.", "strict": false,
					"parameters": {"type": "object"}},
				{"type": "custom", "name": "apply_patch", "description": "Patches.",
					"format": {"type": "grammar", "syntax": "lark", "definition": "start: /x/"}}
			],
			"input": [
				{"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "Be brief."}]},
				{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "List it."}]},
				{"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "x"},
				{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Looking."}]},
				{"type": "function_call", "name": "shell", "arguments": "{\"c\":1}", "call_id": "c1"},
				{"type": "custom_tool_call", "name": "apply_patch", "input": "*** Begin", "call_id": "c2"},
				{"type": "function_call_output", "call_id": "c1", "output": "a.txt"},
				{"type": "custom_tool_call_output", "call_id": "c2", "output": [
					{"type": "input_text", "text": "done "},
					{"type": "input_image", "image_url": "data:image/png;base64,AA", "detail": "low"}
				]},
				{"type": "message", "role": "user", "content": "And?"}
			]
		})
	}

	#[test]
	fn a_codex_turn_becomes_a_chat_completion_request() {
		let out = chat(codex_turn()).unwrap();
		assert_eq!(out["max_tokens"], 500);
		assert_eq!(out["stream"], true);
		assert_eq!(out["tool_choice"], "auto");
		assert_eq!(out["parallel_tool_calls"], true);
		for gone in [
			"store",
			"prompt_cache_key",
			"service_tier",
			"truncation",
			"include",
			"metadata",
			"text",
			"reasoning",
			"reasoning_effort",
			"instructions",
			"input",
			"max_output_tokens",
		] {
			assert!(out.get(gone).is_none(), "{gone}");
		}
		let shell = &out["tools"][0]["function"];
		assert_eq!(shell["name"], "shell");
		assert!(shell.get("strict").is_none(), "only strict: true is sent");
		let patch = &out["tools"][1]["function"];
		assert_eq!(patch["parameters"]["required"][0], "input");
		assert!(
			patch["parameters"]["properties"]["input"]["description"]
				.as_str()
				.unwrap()
				.contains("start: /x/")
		);
		let messages = out["messages"].as_array().unwrap();
		assert_eq!(messages.len(), 7, "{messages:#?}");
		assert_eq!(messages[0]["role"], "system");
		assert_eq!(messages[0]["content"], "You are an agent.\n\nBe brief.");
		assert_eq!(messages[1]["content"], "List it.");
		assert_eq!(messages[2]["content"], "Looking.");
		let calls = messages[2]["tool_calls"].as_array().unwrap();
		assert_eq!(calls.len(), 2, "parallel calls are one turn");
		assert_eq!(calls[0]["id"], "c1");
		assert_eq!(calls[0]["function"]["arguments"], "{\"c\":1}");
		let arguments: Value =
			serde_json::from_str(calls[1]["function"]["arguments"].as_str().unwrap()).unwrap();
		assert_eq!(arguments["input"], "*** Begin");
		assert_eq!(messages[3]["role"], "tool");
		assert_eq!(messages[3]["content"], "a.txt");
		assert_eq!(messages[4]["tool_call_id"], "c2");
		assert_eq!(messages[4]["content"], "done ");
		assert_eq!(
			messages[5]["content"][0]["image_url"],
			json!({"url": "data:image/png;base64,AA", "detail": "low"}),
			"the image, after the whole run of results"
		);
		assert_eq!(messages[6], json!({"role": "user", "content": "And?"}));
	}

	#[test]
	fn the_translation_is_one_every_provider_can_carry() {
		let mut body = codex_turn();
		body["parallel_tool_calls"] = json!(false);
		let body = body.as_object().unwrap();
		let translated = to_chat(body, &model(Kind::Bedrock, None)).unwrap();
		assert!(
			translated.get("parallel_tool_calls").is_none(),
			"Converse cannot say it"
		);
		let converse = crate::bedrock::request(&translated, &model(Kind::Bedrock, None)).unwrap();
		assert!(
			converse["messages"][1]["content"][1]
				.get("toolUse")
				.is_some()
		);
		let translated = to_chat(body, &model(Kind::Anthropic, None)).unwrap();
		assert_eq!(translated["parallel_tool_calls"], false);
		let messages =
			crate::anthropic::request(&translated, &model(Kind::Anthropic, None), true).unwrap();
		assert_eq!(messages["tool_choice"]["disable_parallel_tool_use"], true);
	}

	#[test]
	fn a_plain_string_is_the_user_s_turn() {
		let out = chat(json!({"input": "hi"})).unwrap();
		assert_eq!(out["messages"], json!([{"role": "user", "content": "hi"}]));
		let bare = chat(json!({"input": [{"role": "user", "content": "hi"}]})).unwrap();
		assert_eq!(
			bare["messages"][0]["content"], "hi",
			"a message needs no type"
		);
		let image = chat(json!({"input": [{"role": "user", "content": [
			{"type": "input_text", "text": "see"},
			{"type": "input_image", "image_url": "https://x.example/a.png", "detail": "auto"}
		]}]}))
		.unwrap();
		assert_eq!(image["messages"][0]["content"][0]["text"], "see");
		assert_eq!(
			image["messages"][0]["content"][1]["image_url"],
			json!({"url": "https://x.example/a.png"}),
		);
	}

	#[test]
	fn a_tool_s_image_alone_says_so() {
		let out = chat(json!({"input": [
			{"type": "function_call", "name": "see", "arguments": "{}", "call_id": "c"},
			{"type": "function_call_output", "call_id": "c", "output": [
				{"type": "input_image", "image_url": "data:image/png;base64,AA"}
			]}
		]}))
		.unwrap();
		let messages = out["messages"].as_array().unwrap();
		assert_eq!(
			messages[1]["content"],
			"The result is the image that follows."
		);
		assert_eq!(messages[2]["role"], "user");
	}

	#[test]
	fn what_cannot_be_translated_is_refused_by_name_and_what_only_tunes_is_dropped() {
		let refused = |body: Value| chat(body).unwrap_err().message;
		let input = json!("hi");
		assert!(refused(json!({"input": input, "conversation": "c"})).contains("conversation"));
		assert!(refused(json!({"input": input, "max_tool_calls": 1})).contains("max_tool_calls"));
		assert!(refused(json!({"input": input, "background": true})).contains("background"));
		assert!(
			refused(json!({"input": input, "include": ["file_search_call.results"]}))
				.contains("file_search_call.results")
		);
		assert!(
			refused(json!({"input": input, "tools": [{"type": "file_search"}]}))
				.contains("file_search")
		);
		assert!(
			refused(json!({"input": input, "tools": [{"type": "web_search", "filters": {}}]}))
				.contains("filters")
		);
		assert!(
			refused(json!({"input": input, "tool_choice": {"type": "web_search"}}))
				.contains("web_search")
		);
		assert!(
			refused(json!({"input": input, "text": {"format": {"type": "grammar"}}}))
				.contains("grammar")
		);
		assert!(
			refused(json!({"input": [{"type": "item_reference", "id": "x"}]}))
				.contains("item_reference")
		);
		assert!(
			refused(json!({"input": [{"type": "web_search_call"}]})).contains("web_search_call")
		);
		assert!(
			refused(json!({"input": [{"role": "user", "content": [{"type": "input_file"}]}]}))
				.contains("input_file")
		);
		assert!(
			refused(json!({"input": [{"role": "user", "content": [
				{"type": "input_image", "file_id": "f"}
			]}]}))
			.contains("input_image")
		);
		assert!(refused(json!({"input": [{"role": "tool", "content": "x"}]})).contains("tool"));
		assert!(refused(json!({"input": [{"type": "reasoning"}]})).contains("nothing"));
		let out = chat(
			json!({"input": input, "background": false, "service_tier": "flex",
			"text": {"format": {"type": "text"}, "verbosity": "high"}}),
		)
		.unwrap();
		for field in ["background", "service_tier", "response_format"] {
			assert!(out.get(field).is_none(), "{field}");
		}
	}

	#[test]
	fn previous_response_id_and_a_missing_input_are_refused_first() {
		let of = |body: Value| check(body.as_object().unwrap());
		let refused = of(json!({"input": "hi", "previous_response_id": "resp_1"})).unwrap_err();
		assert_eq!(refused.class, Class::BadRequest);
		assert!(refused.message.contains("previous_response_id"));
		assert!(of(json!({"input": "hi", "previous_response_id": null})).is_ok());
		assert!(of(json!({"model": "m"})).is_err());
		assert!(of(json!({"input": []})).is_err());
		assert!(of(json!({"input": [{"role": "user", "content": "hi"}]})).is_ok());
	}

	#[test]
	fn effort_goes_where_the_catalogue_names_a_form_and_nowhere_else() {
		let of = |kind, form, body: &Value| {
			Value::Object(to_chat(body.as_object().unwrap(), &model(kind, form)).unwrap())
		};
		let body = json!({"input": "hi", "reasoning": {"effort": "xhigh", "summary": "auto"}});
		assert!(
			of(Kind::Openai, None, &body)
				.get("reasoning_effort")
				.is_none()
		);
		assert_eq!(
			of(Kind::Openai, Some(Reasoning::Effort), &body)["reasoning_effort"],
			"high"
		);
		assert_eq!(
			of(Kind::Anthropic, Some(Reasoning::Budget), &body)["reasoning_effort"],
			"high"
		);
		let looping = json!({"reasoning": {"effort": "low"}, "input": [
			{"type": "function_call", "name": "a", "arguments": "{}", "call_id": "c"},
			{"type": "function_call_output", "call_id": "c", "output": "x"}
		]});
		for (kind, form) in [
			(Kind::Anthropic, Reasoning::Budget),
			(Kind::Bedrock, Reasoning::Adaptive),
		] {
			assert!(
				of(kind, Some(form), &looping)
					.get("reasoning_effort")
					.is_none(),
				"no signed thinking to go on from"
			);
		}
		assert_eq!(
			of(Kind::Bedrock, Some(Reasoning::Effort), &looping)["reasoning_effort"],
			"low",
			"a form that signs nothing"
		);
	}

	#[test]
	fn a_json_schema_format_becomes_response_format() {
		let out = chat(json!({"input": "hi", "text": {"format": {
			"type": "json_schema", "name": "n", "strict": true, "schema": {"type": "object"}
		}}}))
		.unwrap();
		assert_eq!(
			out["response_format"],
			json!({"type": "json_schema", "json_schema": {
				"name": "n", "schema": {"type": "object"}, "strict": true,
			}})
		);
	}

	#[test]
	fn a_web_tool_goes_in_the_form_the_provider_takes() {
		let body = json!({"input": "hi", "tools": [
			{"type": "web_search", "search_context_size": "low", "external_web_access": false}
		], "tool_choice": "auto"});
		let of = |host: &str| {
			Value::Object(
				to_chat(
					body.as_object().unwrap(),
					&model_at(Kind::Openai, host, None),
				)
				.unwrap(),
			)
		};
		let openai = of("api.openai.com");
		assert_eq!(openai["web_search_options"]["search_context_size"], "low");
		assert!(openai.get("tools").is_none());
		assert!(openai.get("tool_choice").is_none(), "no function to choose");
		let router = of("openrouter.ai");
		assert_eq!(router["plugins"], json!([{"id": "web"}]));
		let other = of("api.example.com");
		assert_eq!(other["tools"][0]["type"], "web_search");
	}

	#[test]
	fn the_web_capability_is_seen_where_the_needs_check_reads_it() {
		let with = |tools: Value| json!({"input": "hi", "tools": tools});
		let web = |body: &Value| fits::Web::of(body.as_object().unwrap(), false);
		assert_ne!(
			web(&with(json!([{"type": "web_search"}]))),
			fits::Web::default()
		);
		assert_eq!(
			web(&with(json!([
				{"type": "function", "name": "web_search", "parameters": {}},
				{"type": "custom", "name": "google_search"}
			]))),
			fits::Web::default()
		);
	}

	#[test]
	fn a_chat_completion_becomes_a_response() {
		let custom = ToolNames::from_iter(["apply_patch".to_string()]);
		let out = reply(
			&json!({"id": "c1", "choices": [{"finish_reason": "tool_calls", "message": {
				"content": "Let me look.", "reasoning_content": "Hm.",
				"tool_calls": [
					{"id": "t1", "function": {"name": "shell", "arguments": "{\"c\":1}"}},
					{"id": "t2", "function": {"name": "apply_patch", "arguments": "{\"input\":\"*** x\"}"}},
					{"id": "t3", "function": {"name": "noop"}}
				]
			}}]}),
			Some(Usage {
				prompt_tokens: 100,
				completion_tokens: 7,
				cached_tokens: Some(60),
				reasoning_tokens: Some(3),
				..Usage::default()
			}),
			"asked-for",
			&custom,
			1234,
		)
		.unwrap();
		assert!(out["id"].as_str().unwrap().starts_with("resp_"));
		assert_eq!(out["object"], "response");
		assert_eq!(out["created_at"], 1234);
		assert_eq!(out["status"], "completed");
		assert_eq!(out["model"], "asked-for");
		let output = out["output"].as_array().unwrap();
		assert_eq!(output[0]["type"], "reasoning");
		assert!(output[0].get("encrypted_content").is_none());
		assert_eq!(output[1]["content"][0]["text"], "Let me look.");
		assert_eq!(output[2]["type"], "function_call");
		assert_eq!(output[2]["call_id"], "t1");
		assert_eq!(output[2]["status"], "completed");
		assert_eq!(output[2]["arguments"], "{\"c\":1}");
		assert_eq!(output[3]["type"], "custom_tool_call");
		assert_eq!(output[3]["input"], "*** x");
		assert_eq!(
			output[4]["arguments"], "{}",
			"no arguments is an empty object"
		);
		assert_eq!(out["usage"]["input_tokens"], 100, "the cache is inside it");
		assert_eq!(out["usage"]["input_tokens_details"]["cached_tokens"], 60);
		assert_eq!(out["usage"]["output_tokens_details"]["reasoning_tokens"], 3);
		assert_eq!(out["usage"]["total_tokens"], 107);
		assert!(reply(&json!({}), None, "m", &custom, 0).is_err());
	}

	#[test]
	fn a_reply_cut_short_is_incomplete_and_says_why() {
		for (finish, why) in [
			("length", "max_output_tokens"),
			("content_filter", "content_filter"),
		] {
			let out = reply(
				&json!({"choices": [{"finish_reason": finish, "message": {"content": "x"}}]}),
				None,
				"m",
				&ToolNames::default(),
				0,
			)
			.unwrap();
			assert_eq!(out["status"], "incomplete");
			assert_eq!(out["incomplete_details"]["reason"], why);
		}
	}

	/// An SSE body's events as `(name, data)`.
	fn parsed(bytes: &[u8]) -> Vec<(String, Value)> {
		String::from_utf8_lossy(bytes)
			.split("\n\n")
			.filter(|e| !e.is_empty() && !e.starts_with(':'))
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

	fn names(events: &[(String, Value)]) -> Vec<&str> {
		events.iter().map(|(n, _)| n.as_str()).collect()
	}

	fn in_order(events: &[(String, Value)]) {
		for (n, (name, data)) in events.iter().enumerate() {
			assert_eq!(data["type"], name.as_str());
			assert_eq!(data["sequence_number"], n, "{name}");
		}
	}

	fn events(custom: &[&str]) -> Events {
		Events::new(
			"asked-for",
			custom.iter().map(|c| c.to_string()).collect(),
			1234,
		)
	}

	#[test]
	fn a_stream_of_text_ends_in_response_completed() {
		let mut writer = events(&[]);
		let mut out = Vec::new();
		out.extend(writer.chunk(&chunk(
			json!({"role": "assistant", "content": ""}),
			Value::Null,
		)));
		out.extend(writer.chunk(&chunk(json!({"content": "Hi"}), Value::Null)));
		out.extend(writer.chunk(&chunk(json!({"content": " there"}), Value::Null)));
		out.extend(writer.chunk(&chunk(json!({}), json!("stop"))));
		assert!(writer.finished());
		out.extend(writer.chunk(&json!({"choices": [], "usage": {
			"prompt_tokens": 12, "completion_tokens": 5,
			"prompt_tokens_details": {"cached_tokens": 2}
		}})));
		out.extend(writer.end());
		assert!(writer.end().is_empty(), "once");
		let out = parsed(&out);
		in_order(&out);
		assert_eq!(
			names(&out),
			[
				"response.created",
				"response.in_progress",
				"response.output_item.added",
				"response.content_part.added",
				"response.output_text.delta",
				"response.output_text.delta",
				"response.output_text.done",
				"response.content_part.done",
				"response.output_item.done",
				"response.completed",
			]
		);
		assert_eq!(out[0].1["response"]["status"], "in_progress");
		assert_eq!(out[4].1["delta"], "Hi");
		assert_eq!(out[6].1["text"], "Hi there");
		assert_eq!(out[7].1["part"]["text"], "Hi there");
		let response = &out[9].1["response"];
		assert_eq!(response["status"], "completed");
		assert_eq!(response["model"], "asked-for");
		assert_eq!(response["id"], out[0].1["response"]["id"]);
		assert_eq!(response["output"][0]["content"][0]["text"], "Hi there");
		assert_eq!(response["usage"]["input_tokens"], 12);
		assert_eq!(response["usage"]["output_tokens"], 5);
		assert_eq!(
			response["usage"]["input_tokens_details"]["cached_tokens"],
			2
		);
	}

	#[test]
	fn parallel_calls_stay_apart_by_output_index() {
		let mut writer = events(&[]);
		let mut out = Vec::new();
		out.extend(writer.chunk(&chunk(json!({"content": "Reading."}), Value::Null)));
		out.extend(writer.chunk(&chunk(
			json!({"tool_calls": [
				{"index": 0, "id": "t0", "function": {"name": "A", "arguments": ""}},
				{"index": 1, "id": "t1", "function": {"name": "B", "arguments": ""}}
			]}),
			Value::Null,
		)));
		out.extend(writer.chunk(&chunk(
			json!({"tool_calls": [{"index": 1, "function": {"arguments": "{\"b\":1}"}}]}),
			Value::Null,
		)));
		out.extend(writer.chunk(&chunk(
			json!({"tool_calls": [{"index": 0, "function": {"arguments": "{\"a\":1}"}}]}),
			Value::Null,
		)));
		out.extend(writer.chunk(&chunk(json!({}), json!("tool_calls"))));
		out.extend(writer.end());
		let out = parsed(&out);
		in_order(&out);
		let added: Vec<&Value> = out
			.iter()
			.filter(|(n, d)| {
				n == "response.output_item.added" && d["item"]["type"] == "function_call"
			})
			.map(|(_, d)| d)
			.collect();
		assert_eq!(added[0]["output_index"], 1);
		assert_eq!(added[0]["item"]["name"], "A");
		assert_eq!(added[0]["item"]["status"], "in_progress");
		assert_eq!(added[1]["output_index"], 2);
		assert_eq!(added[1]["item"]["call_id"], "t1");
		let deltas: Vec<(&Value, &Value)> = out
			.iter()
			.filter(|(n, _)| n == "response.function_call_arguments.delta")
			.map(|(_, d)| (&d["output_index"], &d["delta"]))
			.collect();
		assert_eq!(deltas[0], (&json!(2), &json!("{\"b\":1}")));
		assert_eq!(deltas[1], (&json!(1), &json!("{\"a\":1}")));
		let response = &out.last().unwrap().1["response"];
		assert_eq!(out.last().unwrap().0, "response.completed");
		let output = response["output"].as_array().unwrap();
		assert_eq!(output.len(), 3);
		assert_eq!(output[0]["type"], "message");
		assert_eq!(output[1]["name"], "A");
		assert_eq!(output[1]["arguments"], "{\"a\":1}");
		assert_eq!(output[2]["name"], "B");
		assert_eq!(output[2]["status"], "completed");
	}

	#[test]
	fn a_namespace_s_tools_are_flat_on_the_way_out_and_namespaced_on_the_way_back() {
		let body = json!({"input": [
			{"type": "function_call", "namespace": "multi", "name": "spawn", "arguments": "{}", "call_id": "c"},
			{"type": "function_call_output", "call_id": "c", "output": "ok"}
		], "tools": [{"type": "namespace", "name": "multi", "description": "d", "tools": [
			{"type": "function", "name": "spawn", "parameters": {"type": "object"}}
		]}]});
		let body = body.as_object().unwrap();
		let chat =
			Value::Object(to_chat(body, &model_at(Kind::Openai, "api.example.com", None)).unwrap());
		assert_eq!(chat["tools"][0]["function"]["name"], "multi__spawn");
		assert!(chat.to_string().contains("multi__spawn"));
		let out = reply(
			&json!({"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [
				{"id": "c2", "function": {"name": "multi__spawn", "arguments": "{}"}}
			]}}]}),
			None,
			"m",
			&custom_tools(body),
			0,
		)
		.unwrap();
		assert_eq!(out["output"][0]["name"], "spawn");
		assert_eq!(out["output"][0]["namespace"], "multi");
	}

	#[test]
	fn a_custom_call_is_written_whole_with_its_string_input() {
		let mut writer = events(&["apply_patch"]);
		let mut out = Vec::new();
		out.extend(writer.chunk(&chunk(
			json!({"tool_calls": [{"index": 0, "id": "t", "function": {"name": "apply_patch", "arguments": "{\"inp"}}]}),
			Value::Null,
		)));
		out.extend(writer.chunk(&chunk(
			json!({"tool_calls": [{"index": 0, "function": {"arguments": "ut\":\"*** x\"}"}}]}),
			json!("stop"),
		)));
		out.extend(writer.end());
		let out = parsed(&out);
		in_order(&out);
		assert!(
			!names(&out).contains(&"response.function_call_arguments.delta"),
			"{out:#?}"
		);
		let delta = out
			.iter()
			.find(|(n, _)| n == "response.custom_tool_call_input.delta")
			.unwrap();
		assert_eq!(delta.1["delta"], "*** x");
		let item = &out.last().unwrap().1["response"]["output"][0];
		assert_eq!(item["type"], "custom_tool_call");
		assert_eq!(item["call_id"], "t");
		assert_eq!(item["input"], "*** x");
	}

	#[test]
	fn reasoning_is_its_own_item_with_nothing_encrypted() {
		let mut writer = events(&[]);
		let mut out = Vec::new();
		out.extend(writer.chunk(&chunk(json!({"reasoning_content": "Count "}), Value::Null)));
		out.extend(writer.chunk(&chunk(json!({"reasoning_content": "them."}), Value::Null)));
		out.extend(writer.chunk(&chunk(json!({"content": "42"}), json!("stop"))));
		out.extend(writer.end());
		let out = parsed(&out);
		in_order(&out);
		assert_eq!(out[2].1["item"]["type"], "reasoning");
		assert_eq!(out[4].0, "response.reasoning_text.delta");
		assert_eq!(out[4].1["delta"], "Count ");
		assert_eq!(out[6].0, "response.reasoning_text.done");
		assert_eq!(out[6].1["text"], "Count them.");
		let response = &out.last().unwrap().1["response"];
		assert_eq!(response["output"][0]["content"][0]["text"], "Count them.");
		assert_eq!(response["output"][0]["summary"], json!([]));
		assert!(response["output"][0].get("encrypted_content").is_none());
		assert_eq!(response["output"][1]["type"], "message");
	}

	#[test]
	fn a_call_with_no_arguments_is_an_empty_object() {
		let mut writer = events(&[]);
		writer.chunk(&chunk(
			json!({"tool_calls": [{"index": 0, "function": {"name": "A"}}]}),
			json!("stop"),
		));
		let out = parsed(&writer.end());
		let done = out
			.iter()
			.find(|(n, _)| n == "response.function_call_arguments.done")
			.unwrap();
		assert_eq!(done.1["arguments"], "{}");
		let item = &out.last().unwrap().1["response"]["output"][0];
		assert!(
			item["call_id"].as_str().unwrap().starts_with("call_"),
			"a server that names no id still gets one: {item}"
		);
	}

	#[test]
	fn a_stream_cut_short_is_incomplete_and_a_failure_is_failed() {
		let mut writer = events(&[]);
		writer.chunk(&chunk(json!({"content": "x"}), json!("length")));
		let out = parsed(&writer.end());
		assert_eq!(out.last().unwrap().0, "response.incomplete");
		assert_eq!(
			out.last().unwrap().1["response"]["incomplete_details"]["reason"],
			"max_output_tokens"
		);

		let mut writer = events(&[]);
		let mut out = writer.chunk(&chunk(json!({"content": "x"}), Value::Null));
		assert!(!writer.finished());
		let error = ApiError::new(Class::RateLimited, "slow down");
		out.extend(writer.failed(&error));
		assert!(writer.failed(&error).is_empty(), "once");
		assert!(writer.end().is_empty(), "and nothing after it");
		let out = parsed(&out);
		in_order(&out);
		let (name, data) = out.last().unwrap();
		assert_eq!(name, "response.failed");
		assert_eq!(data["response"]["status"], "failed");
		assert_eq!(data["response"]["error"]["code"], "rate_limit_exceeded");
		assert_eq!(data["response"]["error"]["message"], "slow down");
	}

	#[test]
	fn an_empty_stream_is_still_a_whole_response() {
		let mut writer = events(&[]);
		let out = parsed(&writer.end());
		assert_eq!(
			names(&out),
			[
				"response.created",
				"response.in_progress",
				"response.completed"
			]
		);
		let mut writer = events(&[]);
		let error = ApiError::new(Class::Upstream, "gone");
		let out = parsed(&writer.failed(&error));
		assert_eq!(out[0].0, "response.created");
		assert_eq!(out[2].1["response"]["error"]["code"], "upstream_error");
	}
}
