//! What a call used, as the provider reported it.
//!
//! The provider's own count is the truth, because it is what the provider bills. When a provider
//! reports nothing (some OpenAI-compatible servers omit usage from streams), the call is recorded
//! with no usage rather than an invented one; Phase 3 adds an estimate, marked as an estimate.
//!
//! The counts follow OpenAI's meaning whatever the provider: `prompt_tokens` is every token of the
//! prompt, cached or not, and the cache counts are subsets of it. Anthropic counts the three apart
//! (`input_tokens` excludes the cache), so its numbers are added up on the way in.

use serde_json::{Value, json};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
	pub prompt_tokens: u64,
	pub completion_tokens: u64,
	/// Prompt tokens served from the provider's cache, a subset of `prompt_tokens`.
	pub cached_tokens: Option<u64>,
	/// Prompt tokens written to the provider's cache, a subset of `prompt_tokens`. Anthropic bills
	/// these above the plain input price.
	pub cache_write_tokens: Option<u64>,
	/// Completion tokens spent reasoning, a subset of `completion_tokens`.
	pub reasoning_tokens: Option<u64>,
}

impl Usage {
	/// Reads OpenAI's `usage` object (Chat Completions and Completions), or `None` when the value
	/// carries none.
	pub fn from_openai(body: &Value) -> Option<Self> {
		let usage = body.get("usage").filter(|u| u.is_object())?;
		let count = |v: Option<&Value>| v.and_then(Value::as_u64);
		Some(Self {
			prompt_tokens: count(usage.get("prompt_tokens")).unwrap_or(0),
			completion_tokens: count(usage.get("completion_tokens")).unwrap_or(0),
			cached_tokens: count(usage.pointer("/prompt_tokens_details/cached_tokens")),
			cache_write_tokens: None,
			reasoning_tokens: count(usage.pointer("/completion_tokens_details/reasoning_tokens")),
		})
	}

	/// Reads Anthropic's `usage` object (a reply's, or a stream's `message_start` or
	/// `message_delta`), or `None` when it is not one.
	pub fn from_anthropic(usage: &Value) -> Option<Self> {
		let usage = usage.as_object()?;
		let count = |name: &str| usage.get(name).and_then(Value::as_u64);
		let read = count("cache_read_input_tokens");
		let write = count("cache_creation_input_tokens");
		Some(Self {
			prompt_tokens: count("input_tokens").unwrap_or(0)
				+ read.unwrap_or(0)
				+ write.unwrap_or(0),
			completion_tokens: count("output_tokens").unwrap_or(0),
			cached_tokens: read,
			cache_write_tokens: write,
			reasoning_tokens: None,
		})
	}

	/// Reads Bedrock Converse's `usage` (a reply's, or a stream's `metadata`), which counts the cache
	/// apart as Anthropic does (`inputTokens` excludes it).
	pub fn from_bedrock(usage: &Value) -> Option<Self> {
		let usage = usage.as_object()?;
		let count = |name: &str| usage.get(name).and_then(Value::as_u64);
		let read = count("cacheReadInputTokens");
		let write = count("cacheWriteInputTokens");
		Some(Self {
			prompt_tokens: count("inputTokens").unwrap_or(0)
				+ read.unwrap_or(0)
				+ write.unwrap_or(0),
			completion_tokens: count("outputTokens").unwrap_or(0),
			cached_tokens: read,
			cache_write_tokens: write,
			reasoning_tokens: None,
		})
	}

	/// The `usage` object a caller of the OpenAI API gets. The cache counts are given twice: in
	/// OpenAI's place (`prompt_tokens_details.cached_tokens`), and under Anthropic's names,
	/// which Studio reads (`cache_read_input_tokens`, `cache_creation_input_tokens`; PS23).
	pub fn to_openai(&self) -> Value {
		let mut usage = json!({
			"prompt_tokens": self.prompt_tokens,
			"completion_tokens": self.completion_tokens,
			"total_tokens": self.total(),
		});
		if let Some(read) = self.cached_tokens {
			usage["prompt_tokens_details"] = json!({"cached_tokens": read});
			usage["cache_read_input_tokens"] = json!(read);
		}
		if let Some(write) = self.cache_write_tokens {
			usage["cache_creation_input_tokens"] = json!(write);
		}
		if let Some(reasoning) = self.reasoning_tokens {
			usage["completion_tokens_details"] = json!({"reasoning_tokens": reasoning});
		}
		usage
	}

	pub fn total(&self) -> u64 {
		self.prompt_tokens + self.completion_tokens
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use serde_json::json;

	#[test]
	fn openai_usage_with_its_details() {
		let body = json!({"usage": {
			"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
			"prompt_tokens_details": {"cached_tokens": 64},
			"completion_tokens_details": {"reasoning_tokens": 5}
		}});
		let usage = Usage::from_openai(&body).unwrap();
		assert_eq!(usage.total(), 120);
		assert_eq!(usage.cached_tokens, Some(64));
		assert_eq!(usage.reasoning_tokens, Some(5));
		assert_eq!(Usage::from_openai(&json!({"usage": null})), None);
		assert_eq!(Usage::from_openai(&json!({})), None);
	}

	#[test]
	fn anthropic_counts_the_cache_apart_and_openai_does_not() {
		let usage = Usage::from_anthropic(&json!({
			"input_tokens": 9, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 20,
			"output_tokens": 4
		}))
		.unwrap();
		assert_eq!(
			usage.prompt_tokens, 129,
			"the whole prompt, as OpenAI counts it"
		);
		assert_eq!(usage.cached_tokens, Some(100));
		assert_eq!(usage.cache_write_tokens, Some(20));
		let out = usage.to_openai();
		assert_eq!(out["total_tokens"], 133);
		assert_eq!(out["prompt_tokens_details"]["cached_tokens"], 100);
		assert_eq!(out["cache_read_input_tokens"], 100);
		assert_eq!(out["cache_creation_input_tokens"], 20);
		assert_eq!(Usage::from_anthropic(&json!(null)), None);
	}
}
