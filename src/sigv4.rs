//! AWS Signature Version 4, for Bedrock: the whole of it Psyche needs, on `ring`, rather than the
//! AWS SDK and its dependency tree.
//!
//! The one trap is the path. A Bedrock model id can hold `:` (`qwen.qwen3-coder-30b-a3b-v1:0`) and
//! an inference-profile ARN holds `:` and `/`, so the path sent is the id percent-encoded once
//! (`%3A`), and the path signed is that path encoded AGAIN (`%253A`), as every AWS service but S3
//! expects. The tests pin the result against signatures botocore produced for the same requests.

use ring::{digest, hmac};

use crate::providers::Secret;

/// What a request is signed with.
#[derive(Clone)]
pub struct Credentials {
	pub access_key_id: String,
	pub secret_access_key: Secret,
	pub session_token: Option<Secret>,
}

impl std::fmt::Debug for Credentials {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("Credentials")
			.field("access_key_id", &self.access_key_id)
			.finish_non_exhaustive()
	}
}

/// The headers a signed request carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Signed {
	pub authorization: String,
	/// `x-amz-date`.
	pub date: String,
	/// `x-amz-security-token`, for temporary credentials.
	pub security_token: Option<String>,
}

/// One path segment, percent-encoded as AWS encodes it: everything but `A-Z a-z 0-9 - _ . ~`.
pub fn encode(segment: &str) -> String {
	let mut out = String::with_capacity(segment.len());
	for byte in segment.bytes() {
		if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
			out.push(byte as char);
		} else {
			out.push_str(&format!("%{byte:02X}"));
		}
	}
	out
}

/// `x-amz-date` for a time in seconds since the epoch: `YYYYMMDDTHHMMSSZ`, in UTC.
pub fn amz_date(unix: u64) -> String {
	let days = (unix / 86_400) as i64;
	let seconds = unix % 86_400;
	// Howard Hinnant's days-to-civil.
	let z = days + 719_468;
	let era = z.div_euclid(146_097);
	let doe = z - era * 146_097;
	let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
	let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
	let mp = (5 * doy + 2) / 153;
	let day = doy - (153 * mp + 2) / 5 + 1;
	let month = if mp < 10 { mp + 3 } else { mp - 9 };
	let year = yoe + era * 400 + i64::from(month <= 2);
	format!(
		"{year:04}{month:02}{day:02}T{:02}{:02}{:02}Z",
		seconds / 3600,
		(seconds / 60) % 60,
		seconds % 60
	)
}

fn hex(bytes: &[u8]) -> String {
	bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn mac(key: &[u8], data: &str) -> Vec<u8> {
	hmac::sign(&hmac::Key::new(hmac::HMAC_SHA256, key), data.as_bytes())
		.as_ref()
		.to_vec()
}

/// One request to sign. `path` is the path as it is SENT (each segment encoded once); `host` is the
/// `Host` header as it is sent, its port included when it is not the scheme's default.
pub struct Request<'a> {
	pub method: &'a str,
	pub host: &'a str,
	pub path: &'a str,
	pub content_type: &'a str,
	pub body: &'a [u8],
}

/// Signs `request` for `service` in `region` at `date` (`x-amz-date`'s form).
pub fn sign(
	credentials: &Credentials,
	region: &str,
	service: &str,
	request: &Request<'_>,
	date: &str,
) -> Signed {
	let day = &date[..8.min(date.len())];
	let token = credentials
		.session_token
		.as_ref()
		.map(|t| t.expose().to_string());
	let canonical_path = request
		.path
		.split('/')
		.map(encode)
		.collect::<Vec<_>>()
		.join("/");
	let mut headers = vec![
		("content-type", request.content_type),
		("host", request.host),
		("x-amz-date", date),
	];
	if let Some(token) = &token {
		headers.push(("x-amz-security-token", token));
	}
	let canonical_headers: String = headers
		.iter()
		.map(|(name, value)| format!("{name}:{}\n", value.trim()))
		.collect();
	let signed_headers = headers
		.iter()
		.map(|(name, _)| *name)
		.collect::<Vec<_>>()
		.join(";");
	let canonical = format!(
		"{}\n{canonical_path}\n\n{canonical_headers}\n{signed_headers}\n{}",
		request.method,
		hex(digest::digest(&digest::SHA256, request.body).as_ref())
	);
	let scope = format!("{day}/{region}/{service}/aws4_request");
	let to_sign = format!(
		"AWS4-HMAC-SHA256\n{date}\n{scope}\n{}",
		hex(digest::digest(&digest::SHA256, canonical.as_bytes()).as_ref())
	);
	let key = format!("AWS4{}", credentials.secret_access_key.expose());
	let key = mac(key.as_bytes(), day);
	let key = mac(&key, region);
	let key = mac(&key, service);
	let key = mac(&key, "aws4_request");
	let signature = hex(&mac(&key, &to_sign));
	Signed {
		authorization: format!(
			"AWS4-HMAC-SHA256 Credential={}/{scope}, SignedHeaders={signed_headers}, Signature={signature}",
			credentials.access_key_id
		),
		date: date.to_string(),
		security_token: token,
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	const BODY: &[u8] = br#"{"messages":[{"role":"user","content":[{"text":"hello there"}]}]}"#;

	fn credentials(token: Option<&str>) -> Credentials {
		Credentials {
			access_key_id: "AKIDEXAMPLE".into(),
			secret_access_key: Secret::new("wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"),
			session_token: token.map(Secret::new),
		}
	}

	/// Each expected header was produced by botocore's own SigV4Auth (the AWS CLI's, 2.38.1) for the
	/// same request at 2026-10-18T12:00:00Z, with the credentials above, which are AWS's documented
	/// example credentials.
	#[test]
	fn signatures_match_botocore() {
		for (model, operation, token, sent, expected) in [
			(
				"qwen.qwen3-coder-30b-a3b-v1:0",
				"converse",
				None,
				"/model/qwen.qwen3-coder-30b-a3b-v1%3A0/converse",
				"SignedHeaders=content-type;host;x-amz-date, Signature=533c724d32168e5b8ff1168ef18a6d629878de2a3e1565daefb7ea9f382d2d40",
			),
			(
				"us.openai.gpt-5.6-luna",
				"converse-stream",
				None,
				"/model/us.openai.gpt-5.6-luna/converse-stream",
				"SignedHeaders=content-type;host;x-amz-date, Signature=9aea1fde9d3ccdd1d8ff2a203c490cf4a4908478bdd6b77d4b7f54848c57c687",
			),
			(
				"arn:aws:bedrock:us-west-2:123456789012:inference-profile/us.anthropic.claude-sonnet-5",
				"converse",
				Some("FwoGZXIvYXdzEXAMPLESESSIONTOKEN"),
				"/model/arn%3Aaws%3Abedrock%3Aus-west-2%3A123456789012%3Ainference-profile%2Fus.anthropic.claude-sonnet-5/converse",
				"SignedHeaders=content-type;host;x-amz-date;x-amz-security-token, Signature=144a02f6dfc06f11d71222280f7a60d87915be699f21896c74329e89fb019f67",
			),
		] {
			let path = format!("/model/{}/{operation}", encode(model));
			assert_eq!(path, sent, "the path sent, encoded once");
			let signed = sign(
				&credentials(token),
				"us-west-2",
				"bedrock",
				&Request {
					method: "POST",
					host: "bedrock-runtime.us-west-2.amazonaws.com",
					path: &path,
					content_type: "application/json",
					body: BODY,
				},
				&amz_date(1_792_324_800),
			);
			assert_eq!(
				signed.authorization,
				format!(
					"AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20261018/us-west-2/bedrock/aws4_request, {expected}"
				),
				"{model}"
			);
			assert_eq!(signed.security_token.as_deref(), token);
		}
	}

	#[test]
	fn dates() {
		assert_eq!(amz_date(1_792_324_800), "20261018T120000Z");
		assert_eq!(amz_date(0), "19700101T000000Z");
		assert_eq!(amz_date(951_782_400), "20000229T000000Z");
		assert_eq!(amz_date(4_102_444_799), "20991231T235959Z");
	}

	#[test]
	fn a_secret_never_prints() {
		let printed = format!("{:?}", credentials(Some("FwoGZXIvYXdzEXAMPLESESSIONTOKEN")));
		assert!(!printed.contains("EXAMPLEKEY") && !printed.contains("SESSIONTOKEN"));
	}
}
