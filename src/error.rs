//! One error shape for every refusal and failure: OpenAI's, so any OpenAI client shows it, with a
//! stable `code` that routing and the request log read.
//!
//! A provider's own message is kept, with the provider named, once every credential Psyche holds
//! for it has been scrubbed out of the text: a provider that echoes a key in an error does not get
//! to hand it to the caller or the log.

use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use serde_json::json;

/// What went wrong, as the log and the router count it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
	/// The request is not one Psyche can send (bad JSON, no model, too large).
	BadRequest,
	/// No token, or a token that does not verify.
	Unauthorized,
	/// A verified caller this API does not serve for this call.
	Forbidden,
	/// The model is not in the catalogue.
	ModelNotFound,
	/// The provider refused the request itself (a 4xx other than auth and rate limits).
	Rejected,
	/// The provider is rate-limiting Psyche's credential.
	RateLimited,
	/// The provider refused Psyche's credential: a configuration problem, never the caller's.
	ProviderAuth,
	/// The account behind Psyche's credential has no credit or quota left (OpenAI's
	/// `insufficient_quota`, a 402, Anthropic's low credit balance): a billing problem for the
	/// project's owner, never the caller's, and nothing a retry fixes.
	ProviderBilling,
	/// The provider failed (5xx), or answered with something that is not its API.
	Upstream,
	/// The provider did not answer in time.
	Timeout,
	/// The provider could not be reached.
	Transport,
	/// The egress rules refused the provider's address (`net.rs`).
	EgressRefused,
	/// A hard budget has no room for the call (`budgets.rs`); nothing was sent.
	Budget,
	/// One of Psyche's own rate limits is used up (`limits.rs`), unlike `RateLimited`, which is the
	/// provider's; another target would not help.
	Limited,
}

impl Class {
	pub fn code(self) -> &'static str {
		match self {
			Class::BadRequest => "bad_request",
			Class::Unauthorized => "unauthorized",
			Class::Forbidden => "forbidden",
			Class::ModelNotFound => "model_not_found",
			Class::Rejected => "rejected_by_provider",
			Class::RateLimited => "rate_limited",
			Class::ProviderAuth => "provider_auth",
			Class::ProviderBilling => "provider_billing",
			Class::Upstream => "upstream_error",
			Class::Timeout => "timeout",
			Class::Transport => "transport",
			Class::EgressRefused => "egress_refused",
			Class::Budget => "budget_exceeded",
			Class::Limited => "rate_limit_exceeded",
		}
	}

	/// The OpenAI `type` a client may switch on.
	fn kind(self) -> &'static str {
		match self {
			Class::BadRequest | Class::ModelNotFound | Class::Rejected => "invalid_request_error",
			Class::Unauthorized => "authentication_error",
			Class::Forbidden => "permission_error",
			Class::RateLimited | Class::Limited => "rate_limit_error",
			// OpenAI's name for a spending limit reached, which clients already handle.
			Class::Budget => "insufficient_quota",
			_ => "api_error",
		}
	}

	pub fn status(self) -> StatusCode {
		match self {
			Class::BadRequest | Class::Rejected => StatusCode::BAD_REQUEST,
			Class::Unauthorized => StatusCode::UNAUTHORIZED,
			Class::Forbidden => StatusCode::FORBIDDEN,
			Class::ModelNotFound => StatusCode::NOT_FOUND,
			Class::RateLimited | Class::Budget | Class::Limited => StatusCode::TOO_MANY_REQUESTS,
			Class::Timeout => StatusCode::GATEWAY_TIMEOUT,
			Class::ProviderAuth
			| Class::ProviderBilling
			| Class::Upstream
			| Class::Transport
			| Class::EgressRefused => StatusCode::BAD_GATEWAY,
		}
	}

	/// Whether the same target may succeed if asked again, in a moment.
	pub fn retryable(self) -> bool {
		matches!(
			self,
			Class::RateLimited | Class::Upstream | Class::Timeout | Class::Transport
		)
	}

	/// Whether a route's next target may succeed where this one failed: any failure that was not
	/// the caller's request or the caller's own limits. A budget too, since the next model may be
	/// cheaper.
	pub fn falls_over(self) -> bool {
		self.retryable()
			|| matches!(
				self,
				Class::ProviderAuth | Class::ProviderBilling | Class::EgressRefused | Class::Budget
			)
	}
}

#[derive(Debug, Clone)]
pub struct ApiError {
	pub class: Class,
	pub message: String,
	/// The provider that failed, when one did.
	pub provider: Option<String>,
	/// The status to answer with, when it is not the class's own (a provider's 4xx passed on).
	pub status: Option<StatusCode>,
	/// `retry-after`, passed on from a provider that rate-limited.
	pub retry_after: Option<String>,
	/// For a budget or limit of Psyche's own: whose it is (`project`, `key`, `user`, `group`), so a
	/// client can tell its own budget from the project's.
	pub scope: Option<&'static str>,
}

impl ApiError {
	pub fn new(class: Class, message: impl Into<String>) -> Self {
		Self {
			class,
			message: message.into(),
			provider: None,
			status: None,
			retry_after: None,
			scope: None,
		}
	}

	pub fn scoped(mut self, scope: &'static str) -> Self {
		self.scope = Some(scope);
		self
	}

	pub fn from_provider(mut self, provider: &str) -> Self {
		self.provider = Some(provider.to_string());
		self
	}

	pub fn status(&self) -> StatusCode {
		self.status.unwrap_or_else(|| self.class.status())
	}

	/// The body, as JSON: OpenAI's `{"error": {...}}`.
	pub fn body(&self) -> serde_json::Value {
		let mut body = json!({
			"error": {
				"message": self.message,
				"type": self.class.kind(),
				"code": self.class.code(),
				"param": null,
				"provider": self.provider,
			}
		});
		if let Some(scope) = self.scope {
			body["error"]["scope"] = scope.into();
		}
		body
	}
}

impl IntoResponse for ApiError {
	fn into_response(self) -> Response {
		let status = self.status();
		let mut response = (status, axum::Json(self.body())).into_response();
		if let Some(after) = self
			.retry_after
			.as_deref()
			.and_then(|v| HeaderValue::from_str(v).ok())
		{
			response.headers_mut().insert("retry-after", after);
		}
		response
	}
}

/// Removes every occurrence of each secret from `text`.
pub fn scrub(text: &str, secrets: &[&str]) -> String {
	let mut out = text.to_string();
	for secret in secrets.iter().filter(|s| s.len() >= 8) {
		out = out.replace(secret, "[credential removed]");
	}
	out
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn a_key_echoed_by_a_provider_is_removed() {
		let text = r#"{"error":{"message":"Incorrect API key provided: sk-test-not-a-real-key."}}"#;
		let clean = scrub(text, &["sk-test-not-a-real-key"]);
		assert!(!clean.contains("not-a-real-key"), "{clean}");
		assert!(clean.contains("[credential removed]"));
		// A secret too short to be a key is not used to blank out ordinary words.
		assert_eq!(scrub("an error", &["an"]), "an error");
	}

	#[test]
	fn the_shape_is_openai_s() {
		let error = ApiError::new(Class::ModelNotFound, "No model named x.").from_provider("p");
		let body = error.body();
		assert_eq!(body["error"]["code"], "model_not_found");
		assert_eq!(body["error"]["type"], "invalid_request_error");
		assert_eq!(body["error"]["provider"], "p");
		assert_eq!(error.status(), StatusCode::NOT_FOUND);
	}
}
