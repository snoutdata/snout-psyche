//! AWS's event-stream framing (`application/vnd.amazon.eventstream`), which Bedrock's
//! ConverseStream speaks instead of server-sent events.
//!
//! A message is a 12-byte prelude (total length, headers length, a CRC32 of those eight bytes),
//! the headers, the payload, and a CRC32 of everything before it. As with SSE, the network hands
//! messages over in pieces that respect nothing, so [`Decoder`] buffers only the message it is in
//! the middle of and refuses one that is larger than [`MAX_MESSAGE`] or whose checksums are wrong:
//! a corrupted frame is the provider's failure, never something to guess past.
//!
//! [`encode`] writes one message, for the fake provider and the tests.

use bytes::Bytes;

/// The largest message accepted. Bedrock's are well under a megabyte.
pub const MAX_MESSAGE: usize = 16 * 1024 * 1024;

const PRELUDE: usize = 12;

/// CRC32 (IEEE), the checksum the framing uses.
const fn table() -> [u32; 256] {
	let mut table = [0u32; 256];
	let mut i = 0;
	while i < 256 {
		let mut c = i as u32;
		let mut k = 0;
		while k < 8 {
			c = if c & 1 != 0 {
				0xEDB8_8320 ^ (c >> 1)
			} else {
				c >> 1
			};
			k += 1;
		}
		table[i] = c;
		i += 1;
	}
	table
}

const TABLE: [u32; 256] = table();

pub fn crc32(bytes: &[u8]) -> u32 {
	let mut c = 0xFFFF_FFFFu32;
	for &b in bytes {
		c = TABLE[((c ^ u32::from(b)) & 0xFF) as usize] ^ (c >> 8);
	}
	!c
}

/// One message: its string headers (the only kind Bedrock sends) and its payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
	pub headers: Vec<(String, String)>,
	pub payload: Bytes,
}

impl Message {
	pub fn header(&self, name: &str) -> Option<&str> {
		self.headers
			.iter()
			.find(|(n, _)| n == name)
			.map(|(_, v)| v.as_str())
	}
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct FrameError(pub String);

#[derive(Debug, Default)]
pub struct Decoder {
	buffer: Vec<u8>,
}

fn be32(bytes: &[u8]) -> u32 {
	u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

impl Decoder {
	pub fn new() -> Self {
		Self::default()
	}

	/// Whether a message was left half-received.
	pub fn is_empty(&self) -> bool {
		self.buffer.is_empty()
	}

	/// Feeds the next piece of the stream and returns the messages it completed.
	pub fn feed(&mut self, piece: &[u8]) -> Result<Vec<Message>, FrameError> {
		self.buffer.extend_from_slice(piece);
		let mut messages = Vec::new();
		loop {
			if self.buffer.len() < PRELUDE {
				break;
			}
			let total = be32(&self.buffer[0..4]) as usize;
			let headers_len = be32(&self.buffer[4..8]) as usize;
			if crc32(&self.buffer[0..8]) != be32(&self.buffer[8..12]) {
				return Err(FrameError(
					"a message in the provider's stream has a bad prelude checksum".into(),
				));
			}
			if !(PRELUDE + 4..=MAX_MESSAGE).contains(&total) || headers_len > total - PRELUDE - 4 {
				return Err(FrameError(format!(
					"a message in the provider's stream claims an impossible length ({total} bytes, {headers_len} of headers)"
				)));
			}
			if self.buffer.len() < total {
				break;
			}
			let frame: Vec<u8> = self.buffer.drain(..total).collect();
			if crc32(&frame[..total - 4]) != be32(&frame[total - 4..]) {
				return Err(FrameError(
					"a message in the provider's stream has a bad checksum".into(),
				));
			}
			let headers = headers(&frame[PRELUDE..PRELUDE + headers_len])?;
			messages.push(Message {
				headers,
				payload: Bytes::copy_from_slice(&frame[PRELUDE + headers_len..total - 4]),
			});
		}
		Ok(messages)
	}
}

fn headers(mut bytes: &[u8]) -> Result<Vec<(String, String)>, FrameError> {
	let broken = || FrameError("a message in the provider's stream has broken headers".into());
	let mut out = Vec::new();
	while !bytes.is_empty() {
		let name_len = *bytes.first().ok_or_else(broken)? as usize;
		let name = bytes.get(1..1 + name_len).ok_or_else(broken)?;
		let name = String::from_utf8_lossy(name).into_owned();
		bytes = &bytes[1 + name_len..];
		let kind = *bytes.first().ok_or_else(broken)?;
		bytes = &bytes[1..];
		// The value's length by type: bool true/false, byte, short, int, long, bytes, string,
		// timestamp, uuid.
		let (skip, value) = match kind {
			0 | 1 => (0, None),
			2 => (1, None),
			3 => (2, None),
			4 => (4, None),
			5 | 8 => (8, None),
			9 => (16, None),
			6 | 7 => {
				let len = bytes.get(0..2).ok_or_else(broken)?;
				let len = u16::from_be_bytes([len[0], len[1]]) as usize;
				let value = bytes.get(2..2 + len).ok_or_else(broken)?;
				(
					2 + len,
					(kind == 7).then(|| String::from_utf8_lossy(value).into_owned()),
				)
			}
			_ => return Err(broken()),
		};
		if bytes.len() < skip {
			return Err(broken());
		}
		bytes = &bytes[skip..];
		if let Some(value) = value {
			out.push((name, value));
		}
	}
	Ok(out)
}

/// One message with string headers.
pub fn encode(headers: &[(&str, &str)], payload: &[u8]) -> Vec<u8> {
	let mut encoded = Vec::new();
	for (name, value) in headers {
		encoded.push(name.len() as u8);
		encoded.extend_from_slice(name.as_bytes());
		encoded.push(7);
		encoded.extend_from_slice(&(value.len() as u16).to_be_bytes());
		encoded.extend_from_slice(value.as_bytes());
	}
	let total = PRELUDE + encoded.len() + payload.len() + 4;
	let mut out = Vec::with_capacity(total);
	out.extend_from_slice(&(total as u32).to_be_bytes());
	out.extend_from_slice(&(encoded.len() as u32).to_be_bytes());
	let prelude_crc = crc32(&out);
	out.extend_from_slice(&prelude_crc.to_be_bytes());
	out.extend_from_slice(&encoded);
	out.extend_from_slice(payload);
	let crc = crc32(&out);
	out.extend_from_slice(&crc.to_be_bytes());
	out
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn the_checksum_is_ieee_crc32() {
		assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
		assert_eq!(crc32(b""), 0);
	}

	fn stream() -> Vec<u8> {
		let mut stream = encode(
			&[
				(":event-type", "messageStart"),
				(":content-type", "application/json"),
				(":message-type", "event"),
			],
			br#"{"role":"assistant"}"#,
		);
		stream.extend(encode(
			&[
				(":event-type", "contentBlockDelta"),
				(":message-type", "event"),
			],
			br#"{"contentBlockIndex":0,"delta":{"text":"Hi"}}"#,
		));
		stream
	}

	#[test]
	fn messages_split_anywhere_come_out_whole() {
		let stream = stream();
		for cut in 0..stream.len() {
			let mut decoder = Decoder::new();
			let mut messages = decoder.feed(&stream[..cut]).unwrap();
			messages.extend(decoder.feed(&stream[cut..]).unwrap());
			assert!(decoder.is_empty(), "cut at {cut}");
			assert_eq!(messages.len(), 2, "cut at {cut}");
			assert_eq!(messages[0].header(":event-type"), Some("messageStart"));
			assert_eq!(&messages[0].payload[..], br#"{"role":"assistant"}"#);
			assert_eq!(messages[1].header(":event-type"), Some("contentBlockDelta"));
		}
	}

	#[test]
	fn a_damaged_frame_is_refused() {
		let mut stream = stream();
		let last = stream.len() - 6;
		stream[last] ^= 0xFF;
		assert!(Decoder::new().feed(&stream).is_err());
		let mut prelude = stream.clone();
		prelude[2] ^= 0x01;
		assert!(Decoder::new().feed(&prelude).is_err());
		let mut huge = encode(&[], b"x");
		huge[0..4].copy_from_slice(&(MAX_MESSAGE as u32 + 1).to_be_bytes());
		let crc = crc32(&huge[0..8]);
		huge[8..12].copy_from_slice(&crc.to_be_bytes());
		assert!(Decoder::new().feed(&huge).is_err());
	}

	#[test]
	fn headers_of_every_type_are_read_past() {
		// A bool, an int and a uuid header before the string one.
		let mut raw = Vec::new();
		raw.extend([1, b'a', 0]);
		raw.extend([1, b'b', 4, 0, 0, 0, 7]);
		raw.push(1);
		raw.push(b'c');
		raw.push(9);
		raw.extend([0u8; 16]);
		raw.extend([1, b'd', 7, 0, 2, b'o', b'k']);
		assert_eq!(headers(&raw).unwrap(), vec![("d".into(), "ok".into())]);
		assert!(headers(&raw[..raw.len() - 1]).is_err());
	}
}
