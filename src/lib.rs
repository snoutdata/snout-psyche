//! snout-psyche: an AI gateway. OpenAI-compatible in, any provider out, with the project's own
//! provider keys, budgets and request log in the project's own Postgres.
//!
//! Built to serve SnoutData AI, and to give every project the same thing for its own apps. Psyche
//! routes to models and is never one (PS3).
//!
//! Built so far: the OpenAI-compatible API over two kinds of provider, OpenAI's (passed through)
//! and Anthropic's (translated, `anthropic.rs`). A request names a model from the catalogue
//! (`providers.rs`), is sent to that model's provider with Psyche's credential, and streams back as
//! it arrives; a caller that goes away cancels the upstream call. Keys, budgets,
//! the request log and routing are Phases 3 and 4.

pub mod anthropic;
pub mod bedrock;
pub mod config;
pub mod embeddings;
pub mod error;
pub mod eventstream;
pub mod jwt;
pub mod layout;
pub mod net;
pub mod providers;
pub mod proxy;
pub mod sigv4;
pub mod sse;
pub mod translate;
pub mod usage;

#[cfg(any(test, feature = "fake"))]
pub mod fake;

/// Entry points for the fuzz targets: each parser of untrusted input, called the way a request
/// or a provider's answer reaches it. Not an API; nothing here is stable.
#[doc(hidden)]
pub mod fuzz {
	/// A provider's stream, in whatever pieces the network delivers it: as server-sent events
	/// through the Anthropic translator, and as AWS event-stream frames through the Bedrock one.
	pub fn sse(input: &[u8]) {
		let mut bedrock = crate::translate::Translator::bedrock(true, "m", "i", 0);
		for piece in input.chunks(7) {
			let _ = bedrock.feed(piece);
		}
		let _ = bedrock.end();
		let mut parser = crate::sse::Parser::new();
		let mut anthropic = crate::anthropic::Stream::new(true, "m", 0);
		for piece in input.chunks(7) {
			let Ok(events) = parser.feed(piece) else {
				return;
			};
			for event in &events {
				let _ = anthropic.event(event);
			}
		}
	}

	/// What a caller sends, and its translation for an Anthropic model.
	pub fn request(input: &[u8]) {
		let Ok(inbound) = crate::proxy::Inbound::parse(input) else {
			return;
		};
		let model = crate::providers::Model {
			name: "m".into(),
			provider: std::sync::Arc::new(crate::providers::Provider {
				name: "p".into(),
				kind: crate::providers::Kind::Anthropic,
				base_url: url::Url::parse("https://api.anthropic.com/v1").expect("a URL"),
				api_key: None,
				aws: None,
			}),
			upstream: "u".into(),
			prompt_caching: true,
			max_output_tokens: None,
			reasoning: Some(crate::providers::Reasoning::Adaptive),
			embeddings: None,
		};
		let _ = crate::anthropic::request(&inbound.body, &model, inbound.stream);
		let _ = crate::bedrock::request(&inbound.body, &model);
		for family in [
			crate::providers::Embeddings::Titan,
			crate::providers::Embeddings::Cohere,
		] {
			if let Ok(request) = crate::embeddings::request(&inbound.body, family) {
				let _ = request.bodies(family);
			}
		}
	}

	/// What a caller presents as its token.
	pub fn token(input: &str) {
		let _ = crate::jwt::verify(input, "a-project-secret-of-at-least-32-chars", 0);
	}
}

pub(crate) fn b64url_decode(text: &str) -> Option<Vec<u8>> {
	use base64::Engine;
	base64::engine::general_purpose::URL_SAFE_NO_PAD
		.decode(text.trim().trim_end_matches('='))
		.ok()
}

#[cfg(any(test, feature = "fake"))]
pub(crate) fn b64url(bytes: &[u8]) -> String {
	use base64::Engine;
	base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}
