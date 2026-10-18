//! The caller's token: HS256, signed with the project's JWT secret, as every SnoutData Cloud
//! project signs them. Verified in constant time before anything else is read.
//!
//! The `role` claim decides what the caller may do, so it is restricted to the three API roles,
//! in either vocabulary (`layout.rs`): a token naming any other role is refused, however well it
//! is signed. Until keys and budgets exist (Phase 3), only the service role may call a model.

use serde_json::{Map, Value};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct JwtError(pub String);

fn refuse(message: &str) -> JwtError {
	JwtError(message.to_string())
}

/// The three API roles, by what they may do rather than by name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
	/// The anon key: an app's code before anyone signed in.
	Anon,
	/// A signed-in user of the project's app.
	Authenticated,
	/// The service key: a server the project owns.
	Service,
}

/// The verified claims, and the role they carry.
#[derive(Debug, Clone, PartialEq)]
pub struct Caller {
	pub role: Role,
	pub claims: Map<String, Value>,
}

/// Verifies `token` and returns the caller. `now` is seconds since the epoch.
pub fn verify(token: &str, secret: &str, now: u64) -> Result<Caller, JwtError> {
	let parts: Vec<&str> = token.split('.').collect();
	let [header, payload, signature] = parts[..] else {
		return Err(refuse("The token is not a JWT."));
	};
	let header: Value = crate::b64url_decode(header)
		.and_then(|b| serde_json::from_slice(&b).ok())
		.ok_or_else(|| refuse("The token's header is not JSON."))?;
	if header.get("alg").and_then(Value::as_str) != Some("HS256") {
		return Err(refuse(
			"The token must be signed HS256 with the project's JWT secret.",
		));
	}
	let signature = crate::b64url_decode(signature)
		.ok_or_else(|| refuse("The token's signature is not base64url."))?;
	let key = ring::hmac::Key::new(ring::hmac::HMAC_SHA256, secret.as_bytes());
	let signed = &token[..token.len() - parts[2].len() - 1];
	ring::hmac::verify(&key, signed.as_bytes(), &signature)
		.map_err(|_| refuse("The token's signature does not verify."))?;

	let claims: Map<String, Value> = crate::b64url_decode(payload)
		.and_then(|b| serde_json::from_slice(&b).ok())
		.ok_or_else(|| refuse("The token's claims are not a JSON object."))?;
	let now = now as f64;
	if claims
		.get("exp")
		.and_then(Value::as_f64)
		.is_some_and(|exp| exp <= now)
	{
		return Err(refuse("The token has expired."));
	}
	if claims
		.get("nbf")
		.and_then(Value::as_f64)
		.is_some_and(|nbf| nbf > now + 30.0)
	{
		return Err(refuse("The token is not valid yet."));
	}
	let claimed = claims.get("role").and_then(Value::as_str).unwrap_or("anon");
	let role = [
		("anon", Role::Anon),
		("authenticated", Role::Authenticated),
		("service_role", Role::Service),
	]
	.into_iter()
	.find(|(name, _)| crate::layout::same_role(name, claimed))
	.map(|(_, role)| role)
	.ok_or_else(|| refuse("The token's role is not one this API serves."))?;
	Ok(Caller { role, claims })
}

#[cfg(any(test, feature = "fake"))]
pub fn sign(claims: &Value, secret: &str) -> String {
	let head = crate::b64url(br#"{"alg":"HS256","typ":"JWT"}"#);
	let body = crate::b64url(claims.to_string().as_bytes());
	let key = ring::hmac::Key::new(ring::hmac::HMAC_SHA256, secret.as_bytes());
	let tag = ring::hmac::sign(&key, format!("{head}.{body}").as_bytes());
	format!("{head}.{body}.{}", crate::b64url(tag.as_ref()))
}

#[cfg(test)]
mod tests {
	use super::*;
	use serde_json::json;

	const SECRET: &str = "a-project-secret-of-at-least-32-chars";

	#[test]
	fn a_good_token_names_its_role() {
		let token = sign(
			&json!({"sub": "u1", "role": "authenticated", "exp": 2000}),
			SECRET,
		);
		let caller = verify(&token, SECRET, 1000).unwrap();
		assert_eq!(caller.role, Role::Authenticated);
		assert_eq!(caller.claims["sub"], "u1");
	}

	#[test]
	fn refusals() {
		let expired = sign(&json!({"role": "anon", "exp": 999}), SECRET);
		assert!(
			verify(&expired, SECRET, 1000)
				.unwrap_err()
				.0
				.contains("expired")
		);
		let other = sign(
			&json!({"role": "anon"}),
			"another-secret-of-at-least-32-chars!",
		);
		assert!(
			verify(&other, SECRET, 0)
				.unwrap_err()
				.0
				.contains("does not verify")
		);
		let superuser = sign(&json!({"role": "postgres"}), SECRET);
		assert!(
			verify(&superuser, SECRET, 0)
				.unwrap_err()
				.0
				.contains("role")
		);
		assert!(verify("a.b", SECRET, 0).is_err());
		// alg: none, however it is dressed up.
		let none = format!(
			"{}.{}.",
			crate::b64url(br#"{"alg":"none"}"#),
			crate::b64url(br#"{"role":"service_role"}"#)
		);
		assert!(verify(&none, SECRET, 0).unwrap_err().0.contains("HS256"));
		// The new role names are the same credentials as the old.
		for (new, role) in [
			("snout_app", Role::Anon),
			("snout_member", Role::Authenticated),
			("snout_server", Role::Service),
		] {
			let token = sign(&json!({ "role": new }), SECRET);
			assert_eq!(verify(&token, SECRET, 0).unwrap().role, role);
		}
		// A token with no role is anon.
		assert_eq!(
			verify(&sign(&json!({}), SECRET), SECRET, 0).unwrap().role,
			Role::Anon
		);
	}
}
