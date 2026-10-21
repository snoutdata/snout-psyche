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
//! image in any message, a reply held to a JSON schema (`response_format.type: "json_schema"`), and
//! its length: a third of the prompt's bytes (images counted as the budget gate counts them,
//! `budgets::prompt_bound`) plus the reply it asks for (`max_completion_tokens`, `max_tokens`, or the
//! model's own `max_output_tokens`). A third is an estimate, not a tokenizer: English runs nearer a
//! quarter, code and other scripts nearer a third, so it rules a model out a little early rather
//! than sending it a prompt it cannot hold.
//!
//! Everything here is a pure function of the request and the catalogue, so the same request routes
//! the same way, and every model left out is named with its reason in the log row (`routing`).

use serde::Deserialize;
use serde_json::{Map, Value};

use crate::providers::Model;

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
}

impl Fits {
	/// The fields as a JSON object, as the admin API shows them.
	pub fn to_json(self) -> Value {
		serde_json::to_value(self).unwrap_or(Value::Null)
	}
}

/// What a request needs from the model that answers it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Needs {
	/// The prompt's estimated tokens.
	pub prompt: u64,
	/// The reply the caller asked for, when it asked.
	pub reply: Option<u64>,
	pub tools: bool,
	pub images: bool,
	pub json_schema: bool,
}

impl Needs {
	/// Reads a chat or completions request. `prompt_bound` is `budgets::prompt_bound` of its bytes.
	pub fn of(body: &Map<String, Value>, prompt_bound: u64) -> Self {
		let present = |key: &str| match body.get(key) {
			Some(Value::Array(items)) => !items.is_empty(),
			Some(Value::Null) | None => false,
			Some(_) => true,
		};
		let reply = ["max_completion_tokens", "max_tokens"]
			.iter()
			.find_map(|key| body.get(*key).and_then(Value::as_u64));
		let images = body
			.get("messages")
			.and_then(Value::as_array)
			.is_some_and(|messages| {
				messages.iter().any(|m| {
					m.get("content")
						.and_then(Value::as_array)
						.is_some_and(|parts| {
							parts.iter().any(|p| {
								matches!(
									p.get("type").and_then(Value::as_str),
									Some("image_url" | "input_image" | "image")
								)
							})
						})
				})
			});
		Self {
			prompt: prompt_bound.div_ceil(3),
			reply,
			tools: present("tools") || present("functions"),
			images,
			json_schema: body
				.get("response_format")
				.and_then(|f| f.get("type"))
				.and_then(Value::as_str)
				== Some("json_schema"),
		}
	}

	/// Why `model` cannot take this request, by the facts it wrote; `None` when it can, or when it
	/// wrote none.
	pub fn rules_out(&self, model: &Model) -> Option<String> {
		let fits = model.fits?;
		let mut why = Vec::new();
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
