//! What a call cost: the price catalogue, and the arithmetic.
//!
//! A model's price comes from its entry in the providers file (`price`), or failing that from the
//! catalogue shipped in the binary (`catalogue/models.json`), which is matched on the provider's
//! kind, where it is (a Bedrock region, or another provider's host) and the upstream id. Prices
//! differ on all three: Bedrock charges 10% more for a regional inference profile (`us.…`) than
//! for a global one (`global.…`), and a model's price is not the same in every region. A model
//! with neither still works; its calls are logged with no cost, and Psyche says so at start.
//!
//! Prices are written in US dollars per million tokens, as providers publish them, and held as
//! picodollars (10⁻¹² of a dollar) per token, so any price with up to six decimals per million is
//! exact and a sum never drifts the way floating point does.
//!
//! ```json
//! "price": {"input": 2.2, "output": 11, "cache_read": 0.11, "cache_write": 2.75}
//! ```
//!
//! `cache_read` and `cache_write` default to the input price. `request` is a fee per call, in
//! dollars. `long_context` is the second tier some models have: past `above` prompt tokens, every
//! token of the call is charged at its prices instead.

use serde::Deserialize;

use crate::config::ConfigError;
use crate::providers::Kind;
use crate::usage::Usage;

/// Picodollars per US dollar.
pub const PICO: u64 = 1_000_000_000_000;

/// One tier's prices, in picodollars per token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rates {
	pub input: u64,
	pub output: u64,
	pub cache_read: u64,
	pub cache_write: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Price {
	pub rates: Rates,
	/// A fee per call, in picodollars.
	pub request: u64,
	/// Past this many prompt tokens, the whole call is charged at the second tier.
	pub long_context: Option<(u64, Rates)>,
}

impl Price {
	/// What a call that used `usage` costs, in picodollars.
	pub fn cost(&self, usage: &Usage) -> u64 {
		let rates = match self.long_context {
			Some((above, long)) if usage.prompt_tokens > above => long,
			_ => self.rates,
		};
		let read = usage.cached_tokens.unwrap_or(0);
		let write = usage.cache_write_tokens.unwrap_or(0);
		let plain = usage.prompt_tokens.saturating_sub(read + write);
		let sum = u128::from(plain) * u128::from(rates.input)
			+ u128::from(read) * u128::from(rates.cache_read)
			+ u128::from(write) * u128::from(rates.cache_write)
			+ u128::from(usage.completion_tokens) * u128::from(rates.output)
			+ u128::from(self.request);
		u64::try_from(sum).unwrap_or(u64::MAX)
	}
}

/// How a price is written: dollars per million tokens.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PriceEntry {
	input: f64,
	output: f64,
	#[serde(default)]
	cache_read: Option<f64>,
	#[serde(default)]
	cache_write: Option<f64>,
	/// Dollars per call.
	#[serde(default)]
	request: Option<f64>,
	#[serde(default)]
	long_context: Option<LongContextEntry>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LongContextEntry {
	above: u64,
	input: f64,
	output: f64,
	#[serde(default)]
	cache_read: Option<f64>,
	#[serde(default)]
	cache_write: Option<f64>,
}

/// Dollars per million tokens to picodollars per token, refusing what is not a price.
fn per_token(dollars_per_million: f64, what: &str) -> Result<u64, String> {
	if !dollars_per_million.is_finite() || dollars_per_million < 0.0 {
		return Err(format!("{what} is not a price: {dollars_per_million}"));
	}
	// $1 per million tokens is 10⁶ picodollars per token. A price over $1,000,000 per million
	// tokens is a mistake in the file, not a model.
	if dollars_per_million > 1_000_000.0 {
		return Err(format!(
			"{what} is {dollars_per_million} dollars per million tokens, which is not a price"
		));
	}
	Ok((dollars_per_million * 1_000_000.0).round() as u64)
}

fn rates(
	input: f64,
	output: f64,
	cache_read: Option<f64>,
	cache_write: Option<f64>,
) -> Result<Rates, String> {
	let input = per_token(input, "input")?;
	Ok(Rates {
		input,
		output: per_token(output, "output")?,
		cache_read: cache_read.map_or(Ok(input), |p| per_token(p, "cache_read"))?,
		cache_write: cache_write.map_or(Ok(input), |p| per_token(p, "cache_write"))?,
	})
}

impl PriceEntry {
	pub fn price(&self) -> Result<Price, String> {
		let request = match self.request {
			None => 0,
			Some(d) if d.is_finite() && (0.0..=1_000.0).contains(&d) => {
				(d * PICO as f64).round() as u64
			}
			Some(d) => return Err(format!("request is not a fee per call: {d}")),
		};
		let long_context = match &self.long_context {
			None => None,
			Some(l) => Some((
				l.above,
				rates(l.input, l.output, l.cache_read, l.cache_write)?,
			)),
		};
		Ok(Price {
			rates: rates(self.input, self.output, self.cache_read, self.cache_write)?,
			request,
			long_context,
		})
	}
}

/// The catalogue shipped in the binary.
const SHIPPED: &str = include_str!("../catalogue/models.json");

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Shipped {
	/// When every price in the file was last compared with its source.
	#[allow(dead_code)]
	checked: String,
	#[allow(dead_code)]
	source: String,
	models: Vec<ShippedModel>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ShippedModel {
	provider: Kind,
	/// A Bedrock model's region.
	#[serde(default)]
	region: Option<String>,
	/// Any other provider's API host.
	#[serde(default)]
	host: Option<String>,
	upstream: String,
	price: PriceEntry,
}

/// Where a model is served, for finding its price.
pub struct Place<'a> {
	pub kind: Kind,
	pub region: Option<&'a str>,
	pub host: Option<&'a str>,
	pub upstream: &'a str,
}

/// The shipped catalogue, read once.
pub struct Catalogue {
	models: Vec<ShippedModel>,
}

impl Catalogue {
	pub fn shipped() -> Result<Self, ConfigError> {
		Self::from_json(SHIPPED)
	}

	fn from_json(text: &str) -> Result<Self, ConfigError> {
		let file: Shipped = serde_json::from_str(text)
			.map_err(|e| ConfigError(format!("the price catalogue is not valid: {e}")))?;
		for model in &file.models {
			model.price.price().map_err(|e| {
				ConfigError(format!("the price catalogue's {}: {e}", model.upstream))
			})?;
		}
		Ok(Self {
			models: file.models,
		})
	}

	/// The price of a model served at `place`, if the catalogue has it.
	pub fn find(&self, place: &Place<'_>) -> Option<Price> {
		self.models
			.iter()
			.find(|m| {
				m.provider == place.kind
					&& m.upstream == place.upstream
					&& match place.kind {
						Kind::Bedrock => m.region.as_deref() == place.region,
						_ => m.host.as_deref() == place.host,
					}
			})
			.and_then(|m| m.price.price().ok())
	}
}

/// Picodollars as the decimal Postgres stores in a `numeric` column, in its binary form: twelve
/// places after the point, so a cost is exact to the picodollar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dollars(pub u64);

impl std::fmt::Display for Dollars {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "{}.{:012}", self.0 / PICO, self.0 % PICO)
	}
}

impl tokio_postgres::types::ToSql for Dollars {
	fn to_sql(
		&self,
		_: &tokio_postgres::types::Type,
		out: &mut bytes::BytesMut,
	) -> Result<tokio_postgres::types::IsNull, Box<dyn std::error::Error + Sync + Send>> {
		use bytes::BufMut;
		// numeric's wire form: base-10000 digits around the point, with the point after `weight + 1`
		// of them. Twelve decimal places are three digits exactly. Leading zero digits are allowed;
		// Postgres strips them.
		let whole = self.0 / PICO;
		let fraction = self.0 % PICO;
		let mut digits: Vec<i16> = Vec::new();
		let mut rest = whole;
		while rest > 0 {
			digits.insert(0, (rest % 10_000) as i16);
			rest /= 10_000;
		}
		let weight = digits.len() as i16 - 1;
		for shift in [100_000_000, 10_000, 1] {
			digits.push(((fraction / shift) % 10_000) as i16);
		}
		out.put_i16(digits.len() as i16);
		out.put_i16(weight);
		out.put_u16(0x0000);
		out.put_u16(12);
		for digit in digits {
			out.put_i16(digit);
		}
		Ok(tokio_postgres::types::IsNull::No)
	}

	fn accepts(ty: &tokio_postgres::types::Type) -> bool {
		*ty == tokio_postgres::types::Type::NUMERIC
	}

	tokio_postgres::types::to_sql_checked!();
}

#[cfg(test)]
mod tests {
	use super::*;
	use bytes::BytesMut;
	use tokio_postgres::types::{ToSql, Type};

	fn price(json: &str) -> Price {
		serde_json::from_str::<PriceEntry>(json)
			.unwrap()
			.price()
			.unwrap()
	}

	#[test]
	fn a_call_costs_each_kind_of_token_at_its_price() {
		let p = price(r#"{"input": 2.2, "output": 11, "cache_read": 0.11, "cache_write": 2.75}"#);
		assert_eq!(p.rates.input, 2_200_000);
		let usage = Usage {
			prompt_tokens: 1_000_000 + 2_000_000 + 400_000,
			completion_tokens: 100_000,
			cached_tokens: Some(2_000_000),
			cache_write_tokens: Some(400_000),
			reasoning_tokens: Some(50_000),
		};
		// 1M plain at $2.20, 2M read at $0.11, 0.4M written at $2.75, 0.1M out at $11: reasoning is
		// part of the output, not charged again.
		assert_eq!(Dollars(p.cost(&usage)).to_string(), "4.620000000000");
	}

	#[test]
	fn cache_prices_default_to_the_input_price_and_tiny_calls_are_not_rounded_away() {
		let p = price(r#"{"input": 0.02, "output": 0}"#);
		let usage = Usage {
			prompt_tokens: 7,
			cached_tokens: Some(2),
			..Usage::default()
		};
		assert_eq!(
			p.cost(&usage),
			7 * 20_000,
			"seven tokens at $0.02 per million"
		);
		assert_eq!(Dollars(p.cost(&usage)).to_string(), "0.000000140000");
	}

	#[test]
	fn past_the_threshold_the_whole_call_is_at_the_long_context_price() {
		let p = price(
			r#"{"input": 0.22, "output": 1.32, "long_context": {"above": 272000, "input": 0.44, "output": 1.98}}"#,
		);
		let at = |prompt| {
			p.cost(&Usage {
				prompt_tokens: prompt,
				completion_tokens: 1000,
				..Usage::default()
			})
		};
		assert_eq!(at(272_000), 272_000 * 220_000 + 1000 * 1_320_000);
		assert_eq!(at(272_001), 272_001 * 440_000 + 1000 * 1_980_000);
	}

	#[test]
	fn a_fee_per_call_is_added() {
		let p = price(r#"{"input": 0, "output": 0, "request": 0.001}"#);
		assert_eq!(
			Dollars(p.cost(&Usage::default())).to_string(),
			"0.001000000000"
		);
	}

	#[test]
	fn what_is_not_a_price_is_refused() {
		for bad in [
			r#"{"input": -1, "output": 0}"#,
			r#"{"input": 0, "output": 2000000}"#,
			r#"{"input": 0, "output": 0, "request": -0.5}"#,
		] {
			let entry: PriceEntry = serde_json::from_str(bad).unwrap();
			assert!(entry.price().is_err(), "{bad}");
		}
		assert!(
			serde_json::from_str::<PriceEntry>(r#"{"input": 1, "output": 1, "per_token": 1}"#)
				.is_err()
		);
	}

	#[test]
	fn the_shipped_catalogue_reads_and_tells_regions_and_profiles_apart() {
		let catalogue = Catalogue::shipped().unwrap();
		let find = |region, upstream| {
			catalogue.find(&Place {
				kind: Kind::Bedrock,
				region: Some(region),
				host: None,
				upstream,
			})
		};
		let global = find("us-west-2", "global.anthropic.claude-sonnet-5-5").unwrap();
		let regional = find("us-west-2", "us.anthropic.claude-sonnet-5-5").unwrap();
		assert_eq!(global.rates.input, 2_000_000);
		assert_eq!(regional.rates.input, 2_200_000);
		let luna = find("us-west-2", "us.openai.gpt-5.6-luna").unwrap();
		assert_eq!((luna.rates.input, luna.rates.cache_read), (220_000, 22_000));
		assert_eq!(
			luna.long_context.map(|(above, l)| (above, l.output)),
			Some((272_000, 1_980_000))
		);
		assert_eq!(
			find("us-west-2", "amazon.titan-embed-text-v2:0")
				.unwrap()
				.rates
				.input,
			20_000
		);
		assert!(
			find("ap-southeast-2", "amazon.titan-embed-text-v2:0").is_none(),
			"another region's price is not assumed"
		);
		assert!(
			catalogue
				.find(&Place {
					kind: Kind::Openai,
					region: None,
					host: Some("example.com"),
					upstream: "amazon.titan-embed-text-v2:0",
				})
				.is_none()
		);
	}

	#[test]
	fn dollars_go_to_postgres_as_a_numeric() {
		let mut out = BytesMut::new();
		Dollars(12_345 * PICO + 6_789_012_345)
			.to_sql(&Type::NUMERIC, &mut out)
			.unwrap();
		// 1·10000 + 2345 . 0067 8901 2345: five digits, weight 1, positive, scale 12.
		let words: Vec<i16> = out
			.chunks(2)
			.map(|c| i16::from_be_bytes([c[0], c[1]]))
			.collect();
		assert_eq!(words, vec![5, 1, 0, 12, 1, 2345, 67, 8901, 2345]);
		let mut zero = BytesMut::new();
		Dollars(0).to_sql(&Type::NUMERIC, &mut zero).unwrap();
		let words: Vec<i16> = zero
			.chunks(2)
			.map(|c| i16::from_be_bytes([c[0], c[1]]))
			.collect();
		assert_eq!(words, vec![3, -1, 0, 12, 0, 0, 0]);
	}
}
