//! A provider's stream, in whatever pieces the network delivers it.
#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
	snout_psyche::fuzz::sse(data);
});
