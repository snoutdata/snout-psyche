//! What a model can take, and what a request needs: the facts `auto` chooses by (PS3, PS15) and the
//! check every route and fallback list makes before it tries a model.
//!
//! A model says what it can take in `fits`, as its owner writes it:
//!
//! ```json
//! "fits": {"context": 200000, "tools": true, "images": true, "json_schema": true}
//! ```
//!
//! `context` is the model's whole window in tokens (prompt and reply together); the three switches
//! are off unless written. A model that writes no `fits` says nothing: a route never skips it for a
//! need, and `auto` never chooses it, since `auto` may only pick a model whose facts it can read.
//!
//! A request's needs are read from what it sends: tools (`tools` or `functions`, not empty), an
//! image in any message or Responses API item, a reply held to a JSON schema
//! (`response_format.type: "json_schema"`, `text.format` on the Responses API), and its length: a
//! third of the prompt's bytes (images counted as the budget gate counts them,
//! `budgets::prompt_bound`) plus the reply it asks for (`max_completion_tokens`, `max_tokens`,
//! `max_output_tokens`, or the model's own `max_output_tokens`). A third is an estimate, not a tokenizer: English runs nearer a
//! quarter, code and other scripts nearer a third, so it rules a model out a little early rather
//! than sending it a prompt it cannot hold.
//!
//! Everything here is a pure function of the request and the catalogue, so the same request routes
//! the same way, and every model left out is named with its reason in the log row (`routing`).
//!
//! `fits` may also say `no_data_collection`, that the account behind the model keeps nothing from
//! its calls; only its owner can know that, so nothing guesses it, and a request or a route that
//! requires it leaves out every model that does not say so.
//!
//! **A model's internet level** is its own (`Model::internet`) and is not a fact in `fits`. At
//! `provider`, the default, a call reaches the model's provider and nothing more: a request that
//! carries a capability a provider runs on the open web ([`Web`]) is not sent to it. At `web` such a
//! request goes through. A route or `auto` leaves out a model that may not take the request, and a
//! call naming it alone is refused. A capability the model's provider cannot run is treated the
//! same way rather than dropped, since an answer written without the web it asked for looks like
//! one that had it. At `none` its calls may not reach any public address at all, which `proxy.rs`
//! and `net.rs` hold it to.

use serde::Deserialize;
use serde_json::{Map, Value};

use crate::price::{PICO, Price, Rates};
use crate::providers::{Internet, Kind, Model, Provider};

/// What a model can take, as written.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct Fits {
	/// The whole window, prompt and reply, in tokens; `None` is not checked.
	#[serde(default)]
	pub context: Option<u64>,
	#[serde(default)]
	pub tools: bool,
	#[serde(default)]
	pub images: bool,
	#[serde(default)]
	pub json_schema: bool,
	/// The account behind this model keeps nothing from its calls, as its owner says.
	#[serde(default, skip_serializing_if = "is_false")]
	pub no_data_collection: bool,
}

fn is_false(value: &bool) -> bool {
	!*value
}

impl Fits {
	/// The fields as a JSON object, as the admin API shows them.
	pub fn to_json(self) -> Value {
		serde_json::to_value(self).unwrap_or(Value::Null)
	}
}

/// The most a model may be sent in a minute, where the owner knows it (PS17): Bedrock sends no
/// rate-limit headers, so its quota can only be written down. Limits belong to a model on an
/// account, which is what a target is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct Quota {
	/// Requests a minute.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub requests: Option<u64>,
	/// Tokens a minute, prompt and reply together.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub tokens: Option<u64>,
}

impl Quota {
	pub fn to_json(self) -> Value {
		serde_json::to_value(self).unwrap_or(Value::Null)
	}

	/// What is wrong with it, for the admin API and the providers file.
	pub fn problem(self) -> Option<&'static str> {
		(self.requests == Some(0) || self.tokens == Some(0))
			.then_some("quota is requests and tokens a minute, each above 0.")
	}
}

/// A price ceiling in OpenRouter's terms: `prompt` and `completion` in dollars per million tokens,
/// `request` in dollars per call. `image` is accepted and holds nothing back, since Psyche counts
/// an image as tokens and no model here has a price per image.
#[derive(Debug, Clone, Copy, PartialEq, Default, Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct MaxPrice {
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub prompt: Option<f64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub completion: Option<f64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub request: Option<f64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub image: Option<f64>,
}

impl MaxPrice {
	/// What is wrong with it, or `None`.
	pub fn problem(&self) -> Option<&'static str> {
		[self.prompt, self.completion, self.request, self.image]
			.iter()
			.flatten()
			.any(|p| !p.is_finite() || *p < 0.0)
			.then_some("max_price holds prices of 0 or more.")
	}

	/// Why `model` is over this ceiling for a prompt of about `prompt` tokens. A model with no price
	/// is over it: a ceiling that cannot be checked is not kept.
	pub fn over(&self, model: &Model, prompt: u64) -> Option<String> {
		let Some(price) = model.price else {
			return Some("has no price, so a price ceiling cannot be kept for it".to_string());
		};
		let rates = charged(&price, prompt);
		let per_million = |picodollars: u64| picodollars as f64 / 1_000_000.0;
		for (name, ceiling, its) in [
			("prompt", self.prompt, per_million(rates.input)),
			("completion", self.completion, per_million(rates.output)),
			("request", self.request, price.request as f64 / PICO as f64),
		] {
			if let Some(ceiling) = ceiling
				&& its > ceiling
			{
				return Some(format!(
					"costs {its} per {name}, over the ceiling of {ceiling}"
				));
			}
		}
		None
	}
}

/// The tier a prompt of about `prompt` tokens is charged at.
fn charged(price: &Price, prompt: u64) -> Rates {
	match price.long_context {
		Some((above, long)) if prompt > above => long,
		_ => price.rates,
	}
}

/// What a route requires of every target, whatever the request is (PS15). Unlike a request's needs,
/// where a model that says nothing is not ruled out, a requirement is a promise the route makes:
/// a target that does not say it meets one is left out.
#[derive(Debug, Clone, PartialEq, Default, Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct Constraints {
	/// The smallest window, in tokens.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub context: Option<u64>,
	#[serde(default, skip_serializing_if = "is_false")]
	pub tools: bool,
	#[serde(default, skip_serializing_if = "is_false")]
	pub images: bool,
	#[serde(default, skip_serializing_if = "is_false")]
	pub json_schema: bool,
	/// The regions a target may be in; a provider that names no region is in none of them.
	#[serde(default, skip_serializing_if = "Vec::is_empty")]
	pub regions: Vec<String>,
	#[serde(default, skip_serializing_if = "is_false")]
	pub no_data_collection: bool,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub max_price: Option<MaxPrice>,
}

impl Constraints {
	/// What is wrong with it, or `None`.
	pub fn problem(&self) -> Option<&'static str> {
		if self.context == Some(0) {
			return Some("constraints.context is a number of tokens above 0.");
		}
		self.max_price.as_ref().and_then(MaxPrice::problem)
	}

	/// Why `model` does not meet the route's requirements, or `None` when it does.
	pub fn rules_out(&self, model: &Model, prompt: u64) -> Option<String> {
		let fits = model.fits.unwrap_or_default();
		let mut why = Vec::new();
		if let Some(context) = self.context
			&& fits.context.is_none_or(|own| own < context)
		{
			why.push(format!("the route requires a window of {context} tokens"));
		}
		for (required, has, what) in [
			(self.tools, fits.tools, "tools"),
			(self.images, fits.images, "images"),
			(self.json_schema, fits.json_schema, "a JSON schema"),
			(
				self.no_data_collection,
				fits.no_data_collection,
				"no data collection",
			),
		] {
			if required && !has {
				why.push(format!("the route requires {what}"));
			}
		}
		if !self.regions.is_empty()
			&& !model
				.provider
				.region
				.as_ref()
				.is_some_and(|r| self.regions.contains(r))
		{
			why.push(format!(
				"the route requires a region of {}",
				self.regions.join(", ")
			));
		}
		if let Some(over) = self
			.max_price
			.as_ref()
			.and_then(|ceiling| ceiling.over(model, prompt))
		{
			why.push(over);
		}
		(!why.is_empty()).then(|| why.join(", "))
	}
}

/// Whose form a web capability is written in, which decides who can run it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Family {
	/// Anthropic's dated server tools (`web_search_20250305`, `web_fetch_20250910` and the others
	/// [`is_anthropic_server_tool`] names) and its MCP connector.
	Anthropic,
	/// OpenAI's `web_search*` tools and `web_search_options`.
	OpenAi,
	/// Gemini's `google_search` and `url_context`.
	Gemini,
	/// OpenRouter's server tool, or its `web` plugin with options of its own.
	OpenRouter,
	/// Web search with nothing said about how: the `:online` suffix, or the plain `web` plugin,
	/// which OpenRouter documents as the same thing. Each provider gets its own form of it.
	Neutral,
}

/// What a provider can do with a web capability, by who it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebWay {
	Anthropic,
	/// Has no hosted web tool in Converse.
	Bedrock,
	/// Takes every form, and `:online` as its `web` plugin.
	OpenRouter,
	/// OpenAI's own API or Azure's.
	OpenAi,
	/// Gemini's OpenAI-compatible endpoint.
	Google,
	/// Any other server that speaks OpenAI's API: what it can do is its owner's to say, by turning
	/// the model's internet level to `web`, so the request goes as written.
	Other,
}

impl WebWay {
	pub fn of(provider: &Provider) -> Self {
		let host = provider.base_url.host_str().unwrap_or("");
		match provider.kind {
			Kind::Anthropic => Self::Anthropic,
			Kind::Bedrock => Self::Bedrock,
			Kind::Openai if host == "openrouter.ai" || host.ends_with(".openrouter.ai") => {
				Self::OpenRouter
			}
			Kind::Openai if host == "api.openai.com" || host.ends_with(".openai.azure.com") => {
				Self::OpenAi
			}
			Kind::Openai if host == "generativelanguage.googleapis.com" => Self::Google,
			Kind::Openai => Self::Other,
		}
	}

	fn takes(self, family: Family) -> bool {
		match self {
			Self::Bedrock => false,
			Self::Anthropic => matches!(family, Family::Anthropic | Family::Neutral),
			Self::OpenAi => matches!(family, Family::OpenAi | Family::Neutral),
			Self::Google => family == Family::Gemini,
			Self::OpenRouter | Self::Other => true,
		}
	}
}

/// Whether `kind` is one of Anthropic's dated web tools. OpenAI's dated names have underscores in
/// their date (`web_search_2025_08_26`), so the two never meet.
pub fn is_anthropic_web_tool(kind: &str) -> bool {
	kind.strip_prefix("web_search_")
		.or_else(|| kind.strip_prefix("web_fetch_"))
		.is_some_and(|date| date.len() == 8 && date.bytes().all(|b| b.is_ascii_digit()))
}

/// The tools Anthropic names by date that the caller's own program runs, and so are not server
/// tools.
const ANTHROPIC_CLIENT_TOOLS: [&str; 5] = [
	"bash",
	"text_editor",
	"computer",
	"computer_toolset",
	"memory",
];

/// Whether `kind` is a tool Anthropic runs itself: a dated name that is not one of the client's. It
/// reaches further than the call's provider can be held to, so every one counts as a web capability.
pub fn is_anthropic_server_tool(kind: &str) -> bool {
	kind == "mcp_toolset"
		|| kind.rsplit_once('_').is_some_and(|(name, date)| {
			date.len() == 8
				&& date.bytes().all(|b| b.is_ascii_digit())
				&& !ANTHROPIC_CLIENT_TOOLS.contains(&name)
		})
}

/// Gemini's tools, in both spellings: `{"google_search": {}}` or `{"type": "google_search"}`.
const GEMINI_TOOLS: [&str; 6] = [
	"google_search",
	"googleSearch",
	"google_search_retrieval",
	"googleSearchRetrieval",
	"url_context",
	"urlContext",
];

/// A provider-hosted web capability one entry of `tools` asks for. A function tool is the caller's
/// own, whatever its name, and is never one.
fn hosted_tool(tool: &Value) -> Option<(String, Family)> {
	let tool = tool.as_object()?;
	if tool.contains_key("function") {
		return None;
	}
	match tool.get("type").and_then(Value::as_str) {
		Some(kind) if is_anthropic_server_tool(kind) || kind.starts_with("web_fetch") => {
			Some((kind.to_string(), Family::Anthropic))
		}
		Some(kind) if kind.starts_with("web_search") => Some((kind.to_string(), Family::OpenAi)),
		Some(kind) if GEMINI_TOOLS.contains(&kind) => Some((kind.to_string(), Family::Gemini)),
		Some(kind) if kind.starts_with("openrouter:web") => {
			Some((kind.to_string(), Family::OpenRouter))
		}
		_ => GEMINI_TOOLS
			.iter()
			.find(|name| tool.contains_key(**name))
			.map(|name| ((*name).to_string(), Family::Gemini)),
	}
}

/// Whether a `plugins` entry is OpenRouter's `web` plugin.
fn is_web_plugin(plugin: &Value) -> bool {
	plugin.get("id").and_then(Value::as_str) == Some("web")
}

/// Whether a `web` plugin says nothing but its id.
fn is_plain_web_plugin(plugin: &Value) -> bool {
	is_web_plugin(plugin) && plugin.as_object().is_some_and(|p| p.len() == 1)
}

/// Takes the plain `web` plugin out of a body, for a provider that gets web search another way;
/// whether there was one. Any other plugin stays where it is.
pub fn take_plain_web_plugin(body: &mut Map<String, Value>) -> bool {
	let Some(Value::Array(plugins)) = body.get_mut("plugins") else {
		return false;
	};
	let before = plugins.len();
	plugins.retain(|p| !is_plain_web_plugin(p));
	let found = plugins.len() != before;
	if found && plugins.is_empty() {
		body.remove("plugins");
	}
	found
}

/// The provider-hosted web capabilities a request carries, read from what it sends. A request with
/// none costs a look at four fields.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Web {
	asked: Vec<(String, Family)>,
}

impl Web {
	/// `online`: the request named a model with the `:online` suffix.
	pub fn of(body: &Map<String, Value>, online: bool) -> Self {
		let mut asked = Vec::new();
		if online {
			asked.push((":online".to_string(), Family::Neutral));
		}
		// `extra_body` is how an OpenAI SDK writes what it has no field for, and Gemini's endpoint
		// reads tools there.
		let lists = [
			body.get("tools"),
			body.get("extra_body").and_then(|e| e.get("tools")),
			body.get("extra_body")
				.and_then(|e| e.get("google"))
				.and_then(|g| g.get("tools")),
		];
		for tools in lists.into_iter().flatten().filter_map(Value::as_array) {
			asked.extend(tools.iter().filter_map(hosted_tool));
		}
		// Anthropic's MCP connector: its servers are reached from Anthropic, at the addresses given.
		if body
			.get("mcp_servers")
			.and_then(Value::as_array)
			.is_some_and(|servers| !servers.is_empty())
		{
			asked.push(("mcp_servers".to_string(), Family::Anthropic));
		}
		if body.get("web_search_options").is_some_and(|o| !o.is_null()) {
			asked.push(("web_search_options".to_string(), Family::OpenAi));
		}
		if let Some(plugins) = body.get("plugins").and_then(Value::as_array) {
			for plugin in plugins.iter().filter(|p| is_web_plugin(p)) {
				let family = if is_plain_web_plugin(plugin) {
					Family::Neutral
				} else {
					Family::OpenRouter
				};
				asked.push(("the web plugin".to_string(), family));
			}
		}
		Self { asked }
	}

	/// For a Responses API call. Its `web_search` is a tool on offer, which Codex sends on every call
	/// whatever the model, so it rules no model out: `responses.rs` leaves it off a model that may not
	/// use it ([`takes_offered_search`]), and that model is still kept off the web.
	pub fn offered_search_is_no_need(mut self) -> Self {
		self.asked.retain(|(what, family)| {
			*family != Family::OpenAi
				|| !what.starts_with("web_search")
				|| what == "web_search_options"
		});
		self
	}

	/// Why `model` may not be sent this request, or `None`. A model whose upstream name itself ends
	/// in `:online` searches on every call, so it needs the level `web` whatever the request says.
	pub fn refused(&self, model: &Model) -> Option<String> {
		let what = self
			.asked
			.first()
			.map(|(what, _)| what.as_str())
			.or(model.upstream.ends_with(":online").then_some(":online"))?;
		if model.internet != Internet::Web {
			return Some(format!(
				"its internet level is {}, which allows no web tools, and this asks for {what}",
				model.internet.as_str()
			));
		}
		let way = WebWay::of(&model.provider);
		self.asked
			.iter()
			.find(|(_, family)| !way.takes(*family))
			.map(|(what, _)| format!("its provider ({}) cannot take {what}", model.provider.name))
	}
}

/// Whether a `web_search` tool offered in a Responses API call goes to `model`.
pub fn takes_offered_search(model: &Model) -> bool {
	model.internet == Internet::Web && WebWay::of(&model.provider).takes(Family::OpenAi)
}

/// The tokens of a prompt whose bound (`budgets::prompt_bound`) is `bytes`: a third of them, the
/// estimate this file's header explains.
pub fn estimated_tokens(bytes: u64) -> u64 {
	bytes.div_ceil(3)
}

/// Whether a message's content holds an image, a tool result's content included.
fn holds_image(content: &Value) -> bool {
	content.as_array().is_some_and(|parts| {
		parts
			.iter()
			.any(|part| match part.get("type").and_then(Value::as_str) {
				Some("image_url" | "input_image" | "image") => true,
				Some("tool_result") => holds_image(&part["content"]),
				_ => false,
			})
	})
}

/// What a request needs from the model that answers it.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Needs {
	/// The prompt's estimated tokens.
	pub prompt: u64,
	/// The reply the caller asked for, when it asked.
	pub reply: Option<u64>,
	pub tools: bool,
	pub images: bool,
	pub json_schema: bool,
	/// The web capabilities it carries.
	pub web: Web,
}

impl Needs {
	/// Reads a chat or completions request. `prompt_bound` is `budgets::prompt_bound` of its bytes.
	pub fn of(body: &Map<String, Value>, prompt_bound: u64, web: Web) -> Self {
		let present = |key: &str| match body.get(key) {
			Some(Value::Array(items)) => !items.is_empty(),
			Some(Value::Null) | None => false,
			Some(_) => true,
		};
		let reply = ["max_completion_tokens", "max_tokens", "max_output_tokens"]
			.iter()
			.find_map(|key| body.get(*key).and_then(Value::as_u64));
		// A Responses API item holds its parts in `content`, or in `output` for a tool's result.
		let images = ["messages", "input"]
			.iter()
			.filter_map(|key| body.get(*key).and_then(Value::as_array))
			.flatten()
			.any(|m| holds_image(&m["content"]) || holds_image(&m["output"]));
		Self {
			prompt: estimated_tokens(prompt_bound),
			reply,
			tools: present("tools") || present("functions"),
			images,
			json_schema: [
				body.get("response_format"),
				body.get("output_format"),
				body.get("output_config").and_then(|c| c.get("format")),
				body.get("text").and_then(|t| t.get("format")),
			]
			.into_iter()
			.flatten()
			.any(|format| format.get("type").and_then(Value::as_str) == Some("json_schema")),
			web,
		}
	}

	/// Why `model` cannot take this request: its internet level, then the facts it wrote; `None`
	/// when it can. A model that wrote no facts is only held to its level.
	pub fn rules_out(&self, model: &Model) -> Option<String> {
		let mut why = Vec::new();
		why.extend(self.web.refused(model));
		if let Some(fits) = model.fits {
			if let Some(context) = fits.context {
				let need = self.prompt + self.reply.or(model.max_output_tokens).unwrap_or(0);
				if need > context {
					why.push(format!("needs about {need} tokens of its {context}"));
				}
			}
			if self.tools && !fits.tools {
				why.push("takes no tools".to_string());
			}
			if self.images && !fits.images {
				why.push("takes no images".to_string());
			}
			if self.json_schema && !fits.json_schema {
				why.push("takes no JSON schema".to_string());
			}
		}
		(!why.is_empty()).then(|| why.join(", "))
	}
}

/// The names a caller may send for automatic choice, and the order each asks for: `auto` the
/// project's own (its `auto` route's sort, or the cheapest first), `auto:floor` the cheapest first
/// and `auto:nitro` the fastest, as OpenRouter's suffixes mean.
pub const AUTO: &str = "auto";
pub const AUTO_FLOOR: &str = "auto:floor";
pub const AUTO_NITRO: &str = "auto:nitro";

/// Whether `name` asks for automatic choice.
pub fn is_auto(name: &str) -> bool {
	matches!(name, AUTO | AUTO_FLOOR | AUTO_NITRO)
}

#[cfg(test)]
mod tests {
	use serde_json::json;

	use super::*;
	use crate::net::Egress;
	use crate::providers::Catalogue;

	fn catalogue() -> Catalogue {
		Catalogue::from_json(
			r#"{"providers": [
				{"name": "a", "kind": "anthropic", "base_url": "https://api.anthropic.com/v1"},
				{"name": "b", "kind": "bedrock", "region": "us-west-2", "credential": "aws"},
				{"name": "o", "kind": "openai", "base_url": "https://api.openai.com/v1"},
				{"name": "r", "kind": "openai", "base_url": "https://openrouter.ai/api/v1"},
				{"name": "g", "kind": "openai", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai"}
			], "models": [
				{"name": "sandboxed", "provider": "a"},
				{"name": "claude", "provider": "a", "internet": "web"},
				{"name": "bedrock", "provider": "b", "internet": "web"},
				{"name": "gpt", "provider": "o", "internet": "web"},
				{"name": "router", "provider": "r", "internet": "web"},
				{"name": "gemini", "provider": "g", "internet": "web"},
				{"name": "always-online", "provider": "r", "upstream": "perplexity/sonar:online"}
			]}"#,
			&std::collections::HashMap::new(),
			Egress::Public,
		)
		.unwrap()
	}

	fn web(body: Value) -> Web {
		Web::of(body.as_object().unwrap(), false)
	}

	fn tool(kind: &str) -> Value {
		json!({"tools": [{"type": kind, "name": "x"}]})
	}

	#[test]
	fn every_form_of_web_capability_is_seen_and_a_function_named_web_search_is_not() {
		for body in [
			tool("web_search_20250305"),
			tool("web_fetch_20250910"),
			tool("web_search"),
			tool("web_search_preview"),
			tool("web_search_2025_08_26"),
			tool("google_search"),
			tool("url_context"),
			tool("openrouter:web_search"),
			json!({"tools": [{"googleSearch": {}}]}),
			json!({"extra_body": {"google": {"tools": [{"url_context": {}}]}}}),
			json!({"web_search_options": {"search_context_size": "low"}}),
			json!({"plugins": [{"id": "web"}]}),
			json!({"plugins": [{"id": "web", "max_results": 3}]}),
		] {
			assert_ne!(web(body.clone()), Web::default(), "{body}");
		}
		for body in [
			json!({}),
			json!({"web_search_options": null}),
			json!({"plugins": [{"id": "file-parser"}]}),
			json!({"tools": [{"type": "function", "function": {"name": "web_search"}}]}),
			json!({"tools": [{"function": {"name": "google_search"}}]}),
		] {
			assert_eq!(web(body.clone()), Web::default(), "{body}");
		}
		assert!(!Web::of(&Map::new(), true).asked.is_empty());
	}

	#[test]
	fn anthropic_s_server_tools_count_and_the_callers_own_do_not() {
		for kind in [
			"web_search_20260318",
			"web_fetch_20260209",
			"code_execution_20260521",
			"tool_search_tool_regex_20251119",
			"mcp_toolset",
		] {
			assert!(is_anthropic_server_tool(kind), "{kind}");
			assert_ne!(web(tool(kind)), Web::default(), "{kind}");
		}
		for kind in [
			"bash_20250124",
			"text_editor_20250728",
			"computer_20251124",
			"computer_toolset_20260801",
			"memory_20250818",
			"web_search_2025_08_26",
			"custom",
		] {
			assert!(!is_anthropic_server_tool(kind), "{kind}");
		}
		assert_eq!(
			web(json!({"tools": [{"name": "run_sql", "input_schema": {"type": "object"}}]})),
			Web::default(),
			"a client tool of Anthropic's shape is the caller's"
		);
		assert_ne!(
			web(json!({"mcp_servers": [{"type": "url", "url": "https://x.example", "name": "x"}]})),
			Web::default()
		);
		assert_eq!(web(json!({"mcp_servers": []})), Web::default());
		let models = catalogue();
		assert!(
			web(tool("code_execution_20260521"))
				.refused(models.model("sandboxed").unwrap())
				.unwrap()
				.contains("code_execution_20260521")
		);
	}

	#[test]
	fn a_request_in_anthropic_s_shape_is_read_for_its_needs() {
		let image = json!({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "x"}});
		for content in [
			json!([image]),
			json!([{"type": "tool_result", "content": [image]}]),
		] {
			let body = json!({"messages": [{"role": "user", "content": content}]});
			assert!(
				Needs::of(body.as_object().unwrap(), 9, Web::default()).images,
				"{body}"
			);
		}
		let body = json!({"messages": [{"role": "user", "content": "hi"}],
			"output_config": {"format": {"type": "json_schema", "schema": {}}}});
		let needs = Needs::of(body.as_object().unwrap(), 9, Web::default());
		assert!(needs.json_schema);
		assert!(!needs.images);
		assert_eq!(needs.prompt, 3);
	}

	#[test]
	fn a_model_below_the_web_level_is_refused_and_says_so() {
		let models = catalogue();
		let why = web(tool("web_search_20250305"))
			.refused(models.model("sandboxed").unwrap())
			.unwrap();
		assert_eq!(
			why,
			"its internet level is provider, which allows no web tools, and this asks for web_search_20250305"
		);
		// Nothing asked, nothing refused, whatever the level says.
		assert_eq!(
			Web::default().refused(models.model("sandboxed").unwrap()),
			None
		);
		// An upstream name that searches by itself needs the level with no help from the caller.
		assert!(
			Web::default()
				.refused(models.model("always-online").unwrap())
				.unwrap()
				.contains("internet level is provider")
		);
		// The level is held even when the model wrote no facts at all.
		let needs = Needs::of(&Map::new(), 10, web(tool("web_search")));
		assert!(
			needs
				.rules_out(models.model("sandboxed").unwrap())
				.is_some()
		);
	}

	#[test]
	fn each_provider_takes_only_the_forms_it_can_run() {
		let models = catalogue();
		let allowed = |model: &str, body: Value| web(body).refused(models.model(model).unwrap());
		let no = |why: Option<String>| assert!(why.unwrap().contains("cannot take"));

		no(allowed("bedrock", tool("web_search_20250305")));
		no(allowed("bedrock", json!({"plugins": [{"id": "web"}]})));
		assert!(
			Web::of(&Map::new(), true)
				.refused(models.model("bedrock").unwrap())
				.unwrap()
				.contains("cannot take :online")
		);

		assert_eq!(allowed("claude", tool("web_fetch_20250910")), None);
		assert_eq!(allowed("claude", json!({"plugins": [{"id": "web"}]})), None);
		no(allowed("claude", tool("web_search_preview")));
		no(allowed("claude", tool("google_search")));
		no(allowed("claude", json!({"web_search_options": {}})));
		no(allowed(
			"claude",
			json!({"plugins": [{"id": "web", "max_results": 3}]}),
		));

		assert_eq!(allowed("gpt", tool("web_search_preview")), None);
		assert_eq!(allowed("gpt", json!({"web_search_options": {}})), None);
		no(allowed("gpt", tool("web_search_20250305")));

		assert_eq!(allowed("gemini", tool("google_search")), None);
		no(allowed("gemini", tool("web_search")));

		for body in [
			tool("web_search_20250305"),
			tool("google_search"),
			tool("openrouter:web_search"),
			json!({"plugins": [{"id": "web", "max_results": 3}]}),
		] {
			assert_eq!(allowed("router", body), None);
		}
	}

	#[test]
	fn a_plain_web_plugin_is_taken_out_and_any_other_stays() {
		let mut body = json!({"plugins": [{"id": "web"}]});
		assert!(take_plain_web_plugin(body.as_object_mut().unwrap()));
		assert_eq!(body, json!({}));
		let mut body = json!({"plugins": [{"id": "web", "max_results": 3}, {"id": "file-parser"}]});
		assert!(!take_plain_web_plugin(body.as_object_mut().unwrap()));
		assert_eq!(body["plugins"].as_array().unwrap().len(), 2);
	}

	#[test]
	fn a_route_s_constraints_leave_out_what_does_not_say_it_meets_them() {
		let models = catalogue();
		let claude = models.model("claude").unwrap();
		let constraints = |value: Value| -> Constraints { serde_json::from_value(value).unwrap() };
		assert_eq!(
			constraints(json!({})).rules_out(claude, 100),
			None,
			"no requirement, nothing left out"
		);
		// The model wrote no facts: a requirement is a promise, so it is not assumed.
		let why = constraints(json!({"tools": true, "context": 1000}))
			.rules_out(claude, 100)
			.unwrap();
		assert!(why.contains("window") && why.contains("tools"), "{why}");
		// A provider with no region is in none.
		assert!(
			constraints(json!({"regions": ["us-west-2"]}))
				.rules_out(claude, 100)
				.is_some()
		);
		assert_eq!(
			constraints(json!({"regions": ["us-west-2"]}))
				.rules_out(models.model("bedrock").unwrap(), 100),
			None
		);
		assert!(
			constraints(json!({"regions": ["eu-west-1"]}))
				.rules_out(models.model("bedrock").unwrap(), 100)
				.is_some()
		);
		// Unpriced, a ceiling cannot be kept.
		assert!(
			constraints(json!({"max_price": {"prompt": 100}}))
				.rules_out(models.model("router").unwrap(), 100)
				.unwrap()
				.contains("no price")
		);
		assert!(serde_json::from_value::<Constraints>(json!({"vision": true})).is_err());
		assert!(
			constraints(json!({"max_price": {"prompt": -1}}))
				.problem()
				.is_some()
		);
	}

	#[test]
	fn a_ceiling_is_checked_at_the_tier_the_prompt_is_charged_at() {
		let spec = r#"{"providers": [{"name": "p", "kind": "openai", "base_url": "https://p.example/v1"}],
			"models": [{"name": "m", "provider": "p", "price": {"input": 3, "output": 15, "request": 0.01,
				"long_context": {"above": 1000, "input": 6, "output": 22.5}}}]}"#;
		let models =
			Catalogue::from_json(spec, &std::collections::HashMap::new(), Egress::Public).unwrap();
		let model = models.model("m").unwrap();
		let ceiling = |value: Value| -> MaxPrice { serde_json::from_value(value).unwrap() };
		assert_eq!(ceiling(json!({"prompt": 4})).over(model, 100), None);
		assert!(ceiling(json!({"prompt": 4})).over(model, 2000).is_some());
		assert!(
			ceiling(json!({"completion": 10}))
				.over(model, 100)
				.is_some()
		);
		assert!(
			ceiling(json!({"request": 0.001}))
				.over(model, 100)
				.is_some()
		);
		assert_eq!(ceiling(json!({"image": 0})).over(model, 100), None);
	}

	#[test]
	fn no_data_collection_is_not_shown_unless_written() {
		assert_eq!(
			Fits::default().to_json(),
			json!({"context": null, "tools": false, "images": false, "json_schema": false})
		);
		let fits = Fits {
			no_data_collection: true,
			..Fits::default()
		};
		assert_eq!(fits.to_json()["no_data_collection"], true);
	}
}
