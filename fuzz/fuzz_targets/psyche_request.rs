//! What a caller sends to /ai/v1.
#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
	snout_psyche::fuzz::request(data);
});
