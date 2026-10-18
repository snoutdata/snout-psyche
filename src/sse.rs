//! Server-sent events, read incrementally from a provider's stream.
//!
//! The network hands a stream over in pieces that respect nothing: an event can arrive split
//! across any number of reads, and a CR can end one read with its LF at the start of the next. The
//! parser keeps each event's raw bytes as well as its fields, so a passthrough forwards exactly
//! what the provider sent (comments and keep-alives included) and drops or rewrites only the
//! events it means to. An event that grows past [`MAX_EVENT`] is refused rather than buffered.

use bytes::Bytes;

/// The largest single event accepted. A streamed chunk is a few hundred bytes; a non-streamed
/// reply never comes through here.
pub const MAX_EVENT: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Event {
	/// The `event:` field, when there was one.
	pub event: Option<String>,
	/// The `data:` lines joined with LF, or `None` for a block of comments only.
	pub data: Option<String>,
	/// The event's bytes as they arrived, its terminating blank line included.
	pub raw: Bytes,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct SseError(pub String);

#[derive(Debug, Default)]
pub struct Parser {
	/// The current line, not yet terminated.
	line: Vec<u8>,
	/// The current event's raw bytes so far.
	raw: Vec<u8>,
	event: Option<String>,
	data: Option<String>,
	/// Whether the current event has any line at all (a field or a comment).
	started: bool,
	/// The last byte seen was a CR, so an LF next belongs with it, to whatever the CR ended.
	after_cr: bool,
	/// An event a CR completed, held until the next byte says whether an LF belongs to it.
	held: Option<Event>,
}

/// What ending a line did.
enum Ended {
	/// A field or comment line: the event goes on.
	Line,
	/// A blank line that completed this event.
	Event(Event),
	/// A blank line between events.
	Nothing,
}

impl Parser {
	pub fn new() -> Self {
		Self::default()
	}

	/// Feeds the next piece of the stream and returns the events it completed.
	pub fn feed(&mut self, piece: &[u8]) -> Result<Vec<Event>, SseError> {
		let mut events = Vec::new();
		for &byte in piece {
			if self.after_cr {
				self.after_cr = false;
				if byte == b'\n' {
					match self.held.take() {
						Some(mut held) => {
							let mut raw = held.raw.to_vec();
							raw.push(byte);
							held.raw = Bytes::from(raw);
							events.push(held);
						}
						// The CR ended a field line (its LF joins the event) or a blank line
						// between events (its LF belongs to nothing).
						None if !self.raw.is_empty() => self.raw.push(byte),
						None => {}
					}
					continue;
				}
				events.extend(self.held.take());
			}
			self.raw.push(byte);
			match byte {
				b'\n' => {
					if let Ended::Event(event) = self.end_line() {
						events.push(event);
					}
				}
				b'\r' => {
					self.after_cr = true;
					if let Ended::Event(event) = self.end_line() {
						self.held = Some(event);
					}
				}
				_ => self.line.push(byte),
			}
			if self.raw.len() > MAX_EVENT {
				return Err(SseError(format!(
					"an event in the provider's stream is larger than {MAX_EVENT} bytes"
				)));
			}
		}
		Ok(events)
	}

	/// The stream ended: an event the provider did not close with a blank line is still an event.
	pub fn finish(mut self) -> Vec<Event> {
		let mut events: Vec<Event> = self.held.take().into_iter().collect();
		if !self.line.is_empty() {
			let line = std::mem::take(&mut self.line);
			self.field(&line);
		}
		events.extend(self.take());
		events
	}

	fn end_line(&mut self) -> Ended {
		let line = std::mem::take(&mut self.line);
		if !line.is_empty() {
			self.field(&line);
			return Ended::Line;
		}
		match self.take() {
			Some(event) => Ended::Event(event),
			None => {
				self.raw.clear();
				Ended::Nothing
			}
		}
	}

	fn field(&mut self, line: &[u8]) {
		self.started = true;
		if line.first() == Some(&b':') {
			return;
		}
		let text = String::from_utf8_lossy(line);
		let (name, value) = match text.split_once(':') {
			Some((name, value)) => (name, value.strip_prefix(' ').unwrap_or(value)),
			None => (text.as_ref(), ""),
		};
		match name {
			"data" => match &mut self.data {
				Some(data) => {
					data.push('\n');
					data.push_str(value);
				}
				None => self.data = Some(value.to_string()),
			},
			"event" => self.event = Some(value.to_string()),
			_ => {}
		}
	}

	fn take(&mut self) -> Option<Event> {
		if !self.started {
			return None;
		}
		self.started = false;
		Some(Event {
			event: self.event.take(),
			data: self.data.take(),
			raw: Bytes::from(std::mem::take(&mut self.raw)),
		})
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn all(pieces: &[&[u8]]) -> Vec<Event> {
		let mut parser = Parser::new();
		let mut events = Vec::new();
		for piece in pieces {
			events.extend(parser.feed(piece).unwrap());
		}
		events.extend(parser.finish());
		events
	}

	#[test]
	fn events_split_anywhere_come_out_whole() {
		let stream = b"data: {\"a\":1}\n\ndata: [DONE]\n\n";
		for cut in 0..stream.len() {
			let events = all(&[&stream[..cut], &stream[cut..]]);
			assert_eq!(events.len(), 2, "cut at {cut}");
			assert_eq!(events[0].data.as_deref(), Some("{\"a\":1}"));
			assert_eq!(&events[0].raw[..], b"data: {\"a\":1}\n\n");
			assert_eq!(events[1].data.as_deref(), Some("[DONE]"));
		}
	}

	#[test]
	fn every_line_ending() {
		for stream in [
			&b"event: delta\r\ndata: x\r\n\r\n"[..],
			b"event: delta\rdata: x\r\r",
			b"event: delta\ndata: x\n\n",
		] {
			for cut in 0..stream.len() {
				let events = all(&[&stream[..cut], &stream[cut..]]);
				assert_eq!(events.len(), 1, "{stream:?} cut at {cut}");
				assert_eq!(events[0].event.as_deref(), Some("delta"));
				assert_eq!(events[0].data.as_deref(), Some("x"));
				assert_eq!(&events[0].raw[..], stream, "raw is what arrived");
			}
		}
	}

	#[test]
	fn comments_multiline_data_and_an_unclosed_last_event() {
		let events = all(&[b": keep-alive\n\ndata: a\ndata: b\n\n\n\ndata:c"]);
		assert_eq!(events.len(), 3);
		assert_eq!(
			events[0].data, None,
			"a comment block passes through as an event"
		);
		assert_eq!(&events[0].raw[..], b": keep-alive\n\n");
		assert_eq!(events[1].data.as_deref(), Some("a\nb"));
		assert_eq!(events[2].data.as_deref(), Some("c"));
	}

	#[test]
	fn an_endless_event_is_refused() {
		let mut parser = Parser::new();
		let line = vec![b'x'; 1024 * 1024];
		let mut refused = false;
		for _ in 0..10 {
			if parser.feed(&line).is_err() {
				refused = true;
				break;
			}
		}
		assert!(refused);
	}
}
