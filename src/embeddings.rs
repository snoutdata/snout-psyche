//! Embeddings: OpenAI's `/embeddings` request and reply, and Bedrock's InvokeModel bodies for the
//! two embedding families it serves.
//!
//! To an OpenAI-style provider the request passes through as the caller wrote it (`proxy.rs`). To
//! Bedrock there is no Converse for embeddings: each model takes its own body at
//! `/model/<id>/invoke`, and which body is a fact the catalogue states (`"embeddings"`), never one
//! read off the id, which an inference-profile ARN hides:
//!
//! - **Titan** (`amazon.titan-embed-text-v2:0`) embeds ONE text per call, so a request of `n`
//!   inputs is `n` calls, sent a few at a time and put back in order. `dimensions` is its own.
//! - **Cohere** (`cohere.embed-english-v3`, `us.cohere.embed-v4:0`) takes up to 96 texts per call
//!   and requires `input_type`, which OpenAI's API has no word for: a caller may send it (Cohere's
//!   values, as other OpenAI-compatible gateways accept it), and it is `search_document`
//!   otherwise, the side an index is built from. `dimensions` becomes `output_dimension`, which v4 takes and v3 refuses itself.
//!
//! Bedrock counts the input in the `X-Amzn-Bedrock-Input-Token-Count` header of every call (checked
//! live 2026-10-18 on all three models), so the usage is the sum of those. Vectors are passed on as
//! the numbers the provider sent, or, for `encoding_format: base64` (the OpenAI SDK's default), as
//! little-endian `f32`s, which is what OpenAI sends.

use base64::Engine;
use serde_json::{Map, Value, json};

use crate::error::{ApiError, Class};
use crate::providers::Embeddings;

/// The most texts one Cohere call takes.
pub const COHERE_BATCH: usize = 96;
/// The most inputs one request may hold, OpenAI's own limit.
pub const MAX_INPUTS: usize = 2048;

/// Fields of OpenAI's request Psyche reads. Anything else is refused by name, never dropped.
const KNOWN: [&str; 6] = [
	"model",
	"input",
	"encoding_format",
	"dimensions",
	"user",
	"input_type",
];

/// A request, read for Bedrock.
#[derive(Debug, Clone, PartialEq)]
pub struct Request {
	pub inputs: Vec<String>,
	pub base64: bool,
	pub dimensions: Option<u64>,
	pub input_type: Option<String>,
}

fn bad(message: impl Into<String>) -> ApiError {
	ApiError::new(Class::BadRequest, message)
}

/// Reads what a caller sent for a Bedrock model of `family`.
pub fn request(body: &Map<String, Value>, family: Embeddings) -> Result<Request, ApiError> {
	if let Some(field) = body.keys().find(|k| !KNOWN.contains(&k.as_str())) {
		return Err(bad(format!(
			"{field} is not a field of an embeddings request."
		)));
	}
	let inputs = match body.get("input") {
		Some(Value::String(text)) => vec![text.clone()],
		Some(Value::Array(items)) if items.iter().all(Value::is_string) => items
			.iter()
			.filter_map(Value::as_str)
			.map(String::from)
			.collect(),
		Some(Value::Array(_)) => {
			return Err(bad(
				"input is a string or a list of strings. Token ids cannot be sent to this model's provider.",
			));
		}
		_ => return Err(bad("input is a string or a list of strings.")),
	};
	if inputs.is_empty() {
		return Err(bad("input holds no text."));
	}
	if inputs.len() > MAX_INPUTS {
		return Err(bad(format!(
			"input holds {} texts; the most one request takes is {MAX_INPUTS}.",
			inputs.len()
		)));
	}
	if inputs.iter().any(String::is_empty) {
		return Err(bad("input holds an empty text, which has no embedding."));
	}
	let base64 = match body.get("encoding_format") {
		None | Some(Value::Null) => false,
		Some(Value::String(f)) if f == "float" => false,
		Some(Value::String(f)) if f == "base64" => true,
		Some(_) => return Err(bad("encoding_format is float or base64.")),
	};
	let dimensions = match body.get("dimensions") {
		None | Some(Value::Null) => None,
		Some(value) => match value.as_u64() {
			Some(n) if n > 0 => Some(n),
			_ => return Err(bad("dimensions is a positive whole number.")),
		},
	};
	let input_type = match (body.get("input_type"), family) {
		(None | Some(Value::Null), _) => None,
		(Some(Value::String(t)), Embeddings::Cohere) => Some(t.clone()),
		(Some(_), Embeddings::Cohere) => return Err(bad("input_type is a string.")),
		(Some(_), Embeddings::Titan) => {
			return Err(bad(
				"input_type is Cohere's; this model is Titan, which takes none.",
			));
		}
	};
	Ok(Request {
		inputs,
		base64,
		dimensions,
		input_type,
	})
}

impl Request {
	/// The InvokeModel bodies, in order: one per input for Titan, one per 96 for Cohere.
	pub fn bodies(&self, family: Embeddings) -> Vec<Value> {
		match family {
			Embeddings::Titan => self
				.inputs
				.iter()
				.map(|text| {
					let mut body = json!({"inputText": text});
					if let Some(n) = self.dimensions {
						body["dimensions"] = json!(n);
					}
					body
				})
				.collect(),
			Embeddings::Cohere => self
				.inputs
				.chunks(COHERE_BATCH)
				.map(|texts| {
					let mut body = json!({
						"texts": texts,
						"input_type": self.input_type.as_deref().unwrap_or("search_document"),
						"embedding_types": ["float"],
					});
					if let Some(n) = self.dimensions {
						body["output_dimension"] = json!(n);
					}
					body
				})
				.collect(),
		}
	}
}

/// The vectors in one InvokeModel reply, in order.
pub fn vectors(family: Embeddings, reply: &Value) -> Result<Vec<Value>, String> {
	let list = match family {
		Embeddings::Titan => {
			return match reply.get("embedding") {
				Some(v @ Value::Array(_)) => Ok(vec![v.clone()]),
				_ => Err("a Titan reply without an embedding".into()),
			};
		}
		// Asked for by type, Cohere answers `{"embeddings": {"float": [...]}}`; asked for nothing,
		// `{"embeddings": [...]}`. Both are read.
		Embeddings::Cohere => match reply.get("embeddings") {
			Some(Value::Object(by_type)) => by_type.get("float"),
			other => other,
		},
	};
	match list {
		Some(Value::Array(vectors)) if vectors.iter().all(Value::is_array) => Ok(vectors.clone()),
		_ => Err("a Cohere reply without float embeddings".into()),
	}
}

/// A vector as OpenAI's base64: its numbers as little-endian `f32`s.
fn base64(vector: &Value) -> Result<String, String> {
	let mut bytes = Vec::new();
	for n in vector.as_array().into_iter().flatten() {
		let n = n
			.as_f64()
			.ok_or("an embedding holds something that is not a number")?;
		bytes.extend_from_slice(&(n as f32).to_le_bytes());
	}
	Ok(base64::engine::general_purpose::STANDARD.encode(bytes))
}

/// OpenAI's reply for `vectors`, under the model name the caller asked for.
pub fn reply(
	vectors: Vec<Value>,
	as_base64: bool,
	model: &str,
	prompt_tokens: Option<u64>,
) -> Result<Value, String> {
	let mut data = Vec::with_capacity(vectors.len());
	for (index, vector) in vectors.into_iter().enumerate() {
		let embedding = if as_base64 {
			Value::String(base64(&vector)?)
		} else {
			vector
		};
		data.push(json!({"object": "embedding", "index": index, "embedding": embedding}));
	}
	let tokens = prompt_tokens.unwrap_or(0);
	Ok(json!({
		"object": "list",
		"data": data,
		"model": model,
		"usage": {"prompt_tokens": tokens, "total_tokens": tokens},
	}))
}

#[cfg(test)]
mod tests {
	use super::*;

	fn body(value: Value) -> Map<String, Value> {
		value.as_object().unwrap().clone()
	}

	#[test]
	fn titan_is_one_call_per_text() {
		let r = request(
			&body(json!({"model": "m", "input": ["a", "b", "c"], "dimensions": 256})),
			Embeddings::Titan,
		)
		.unwrap();
		assert_eq!(
			r.bodies(Embeddings::Titan),
			vec![
				json!({"inputText": "a", "dimensions": 256}),
				json!({"inputText": "b", "dimensions": 256}),
				json!({"inputText": "c", "dimensions": 256}),
			]
		);
	}

	#[test]
	fn cohere_is_batched_and_names_its_input_type() {
		let texts: Vec<String> = (0..100).map(|i| format!("t{i}")).collect();
		let r = request(
			&body(json!({"model": "m", "input": texts})),
			Embeddings::Cohere,
		)
		.unwrap();
		let bodies = r.bodies(Embeddings::Cohere);
		assert_eq!(bodies.len(), 2);
		assert_eq!(bodies[0]["texts"].as_array().unwrap().len(), 96);
		assert_eq!(bodies[1]["texts"], json!(["t96", "t97", "t98", "t99"]));
		assert_eq!(bodies[0]["input_type"], "search_document");
		assert_eq!(bodies[0]["embedding_types"], json!(["float"]));
		assert!(bodies[0].get("output_dimension").is_none());
		let r = request(
			&body(
				json!({"model": "m", "input": "q", "input_type": "search_query", "dimensions": 512}),
			),
			Embeddings::Cohere,
		)
		.unwrap();
		assert_eq!(
			r.bodies(Embeddings::Cohere),
			vec![
				json!({"texts": ["q"], "input_type": "search_query", "embedding_types": ["float"], "output_dimension": 512})
			]
		);
	}

	#[test]
	fn what_cannot_be_sent_is_refused_by_name() {
		for (value, says) in [
			(json!({"input": [1, 2, 3]}), "Token ids"),
			(json!({"input": []}), "no text"),
			(json!({"input": ["a", ""]}), "empty text"),
			(
				json!({"input": "a", "encoding_format": "int8"}),
				"float or base64",
			),
			(json!({"input": "a", "dimensions": 0}), "positive"),
			(
				json!({"input": "a", "truncate": "END"}),
				"truncate is not a field",
			),
			(
				json!({"input": "a", "input_type": "search_query"}),
				"Cohere's",
			),
			(json!({}), "input is"),
		] {
			let error = request(&body(value), Embeddings::Titan).unwrap_err();
			assert!(error.message.contains(says), "{}", error.message);
		}
		let many: Vec<String> = (0..=MAX_INPUTS).map(|i| i.to_string()).collect();
		assert!(request(&body(json!({"input": many})), Embeddings::Cohere).is_err());
	}

	#[test]
	fn replies_are_read_in_both_cohere_shapes() {
		let titan = json!({"embedding": [0.5, 0.25], "inputTextTokenCount": 2});
		assert_eq!(
			vectors(Embeddings::Titan, &titan).unwrap(),
			vec![json!([0.5, 0.25])]
		);
		let by_type =
			json!({"embeddings": {"float": [[1.0], [2.0]]}, "response_type": "embeddings_by_type"});
		let floats = json!({"embeddings": [[1.0], [2.0]], "response_type": "embeddings_floats"});
		for reply in [by_type, floats] {
			assert_eq!(vectors(Embeddings::Cohere, &reply).unwrap().len(), 2);
		}
		assert!(vectors(Embeddings::Cohere, &json!({"embeddings": {"int8": [[1]]}})).is_err());
		assert!(vectors(Embeddings::Titan, &json!({"message": "no"})).is_err());
	}

	#[test]
	fn the_reply_is_openai_s_and_base64_is_little_endian_f32() {
		let out = reply(vec![json!([0.5, -2.0])], false, "embed", Some(7)).unwrap();
		assert_eq!(
			out,
			json!({"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": [0.5, -2.0]}],
				"model": "embed", "usage": {"prompt_tokens": 7, "total_tokens": 7}})
		);
		let out = reply(vec![json!([0.5, -2.0])], true, "embed", None).unwrap();
		let bytes = base64::engine::general_purpose::STANDARD
			.decode(out["data"][0]["embedding"].as_str().unwrap())
			.unwrap();
		let floats: Vec<f32> = bytes
			.chunks(4)
			.map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
			.collect();
		assert_eq!(floats, vec![0.5, -2.0]);
	}
}
