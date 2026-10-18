//! A provider's stream, in a protocol that is not OpenAI's, turned into OpenAI's chunks as it
//! arrives: Anthropic's server-sent events (`anthropic.rs`) and Bedrock's event-stream frames
//! (`bedrock.rs`). The proxy hands each network read to [`Translator::feed`] and writes back what
//! comes out, so the translation never holds a reply back (PS9).

use serde_json::Value;

use crate::error::Class;
use crate::eventstream;
use crate::sse;
use crate::usage::Usage;
use crate::{anthropic, bedrock};

/// What a piece of a provider's stream becomes for the caller.
#[derive(Debug, Clone, PartialEq)]
pub enum Out {
	/// An OpenAI chunk.
	Chunk(Value),
	/// `data: [DONE]`.
	Done,
	/// A keep-alive comment, so nothing between here and the caller closes a quiet stream.
	KeepAlive,
	/// The provider reported a failure in the stream, or the stream itself was broken.
	Error(Class, String),
}

pub enum Translator {
	Anthropic {
		parser: Box<sse::Parser>,
		stream: Box<anthropic::Stream>,
	},
	Bedrock {
		decoder: eventstream::Decoder,
		stream: Box<bedrock::Stream>,
	},
}

impl Translator {
	pub fn anthropic(wants_usage: bool, model: &str, created: u64) -> Self {
		Self::Anthropic {
			parser: Box::new(sse::Parser::new()),
			stream: Box::new(anthropic::Stream::new(wants_usage, model, created)),
		}
	}

	pub fn bedrock(wants_usage: bool, model: &str, id: &str, created: u64) -> Self {
		Self::Bedrock {
			decoder: eventstream::Decoder::new(),
			stream: Box::new(bedrock::Stream::new(wants_usage, model, id, created)),
		}
	}

	/// The caller's chunks for the next network read.
	pub fn feed(&mut self, bytes: &[u8]) -> Vec<Out> {
		match self {
			Self::Anthropic { parser, stream } => match parser.feed(bytes) {
				Ok(events) => events.iter().flat_map(|e| stream.event(e)).collect(),
				Err(e) => vec![Out::Error(Class::Upstream, e.to_string())],
			},
			Self::Bedrock { decoder, stream } => match decoder.feed(bytes) {
				Ok(messages) => messages.iter().flat_map(|m| stream.message(m)).collect(),
				Err(e) => vec![Out::Error(Class::Upstream, e.to_string())],
			},
		}
	}

	/// The provider closed its stream: whatever it completed, or an error if the reply was cut.
	pub fn end(&mut self) -> Vec<Out> {
		let cut = || {
			Out::Error(
				Class::Upstream,
				"the stream ended before the reply was complete".into(),
			)
		};
		match self {
			Self::Anthropic { parser, stream } => {
				let mut out: Vec<Out> = std::mem::take(parser.as_mut())
					.finish()
					.iter()
					.flat_map(|e| stream.event(e))
					.collect();
				if !stream.ended() {
					out.push(cut());
				}
				out
			}
			Self::Bedrock { decoder, stream } => {
				if !decoder.is_empty() {
					return vec![cut()];
				}
				stream.end().unwrap_or_else(|| vec![cut()])
			}
		}
	}

	pub fn usage(&self) -> Option<Usage> {
		match self {
			Self::Anthropic { stream, .. } => stream.usage(),
			Self::Bedrock { stream, .. } => stream.usage(),
		}
	}
}
