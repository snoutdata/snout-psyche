//! A bearer token, against a project secret.
#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
	if let Ok(text) = std::str::from_utf8(data) {
		snout_psyche::fuzz::token(text);
	}
});
