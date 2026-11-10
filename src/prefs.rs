//! OpenRouter's request controls (PS15): the `provider` object and the `:floor`, `:nitro` and
//! `:online` suffixes on a model's name, under OpenRouter's own names, so a client written for it
//! works with a change of base URL.
//!
//! They are read out of the request here and never reach a provider, and they only ever narrow or
//! reorder what the route already allows: a target outside the route cannot be named into it.
//!
//! - `only` and `ignore` keep or drop targets; `order` tries the targets it names first and turns
//!   the route's balancing and a conversation's stickiness off. A slug is a provider's name or its
//!   stored credential's, or `name/region` for one region of it, in any case.
//! - `allow_fallbacks: false` keeps only what `order` names, or with no `order` the first target of
//!   each name the caller sent.
//! - `sort` is `price`, `throughput` or `latency`, or `{by, partition}`. `partition` is `model` by
//!   default (each name's targets stay together, in the order the names were sent) and `none`
//!   sorts every target of `model` and `models` as one list. `:floor` is `sort: "price"` and
//!   `:nitro` is `sort: "throughput"`; a `sort` written out wins over either.
//! - `max_price` and `data_collection: "deny"` leave out what cannot be shown to meet them.
//!   `require_parameters` leaves out a target that would refuse or could not be shown to take
//!   something the request sends, instead of the default, where the target decides.
//! - `preferred_min_throughput` (tokens a second) and `preferred_max_latency` (seconds) are
//!   preferences, never filters, as OpenRouter's are: a target measured (`health.rs`) as missing one
//!   goes after those that do not, within each name sent, and stays as a fallback. A number is p50,
//!   an object holds `p50`, `p75`, `p90` and `p99`, all of which must be met. `health.rs` keeps one
//!   moving average of each, not percentiles, so every cutoff written is held against that average,
//!   and the tightest one decides. A target with no measurement is neither preferred nor punished:
//!   it keeps its place. An explicit `order` is the caller's own and they do not move it.
//! - `zdr`, `enforce_distillable_text` and `quantizations` are refused when they ask for anything,
//!   since the catalogue holds no such facts and ignoring one would break a promise silently.
//!
//! When these leave no target the answer is a 404 with a sentence, as OpenRouter's is.

use serde::Deserialize;
use serde_json::Value;

use crate::error::{ApiError, Class};
use crate::fits::MaxPrice;
use crate::providers::{Model, Provider, Sort};

/// What a caller's `provider` object says.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Prefs {
	#[serde(default)]
	order: Option<Vec<String>>,
	#[serde(default)]
	only: Option<Vec<String>>,
	#[serde(default)]
	ignore: Option<Vec<String>>,
	#[serde(default)]
	sort: Option<SortBy>,
	#[serde(default)]
	allow_fallbacks: Option<bool>,
	#[serde(default)]
	max_price: Option<MaxPrice>,
	#[serde(default)]
	data_collection: Option<DataCollection>,
	#[serde(default)]
	require_parameters: Option<bool>,
	#[serde(default)]
	zdr: Option<bool>,
	#[serde(default)]
	enforce_distillable_text: Option<bool>,
	#[serde(default)]
	quantizations: Option<Vec<String>>,
	#[serde(default)]
	preferred_min_throughput: Option<Cutoff>,
	#[serde(default)]
	preferred_max_latency: Option<Cutoff>,
}

/// One of OpenRouter's performance preferences: a number, which is p50, or percentiles.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(untagged)]
enum Cutoff {
	P50(f64),
	Percentiles(Percentiles),
}

#[derive(Debug, Clone, Copy, PartialEq, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct Percentiles {
	#[serde(default)]
	p50: Option<f64>,
	#[serde(default)]
	p75: Option<f64>,
	#[serde(default)]
	p90: Option<f64>,
	#[serde(default)]
	p99: Option<f64>,
}

impl Cutoff {
	/// Every value written.
	fn values(self) -> Vec<f64> {
		match self {
			Self::P50(value) => vec![value],
			Self::Percentiles(p) => [p.p50, p.p75, p.p90, p.p99].into_iter().flatten().collect(),
		}
	}

	fn valid(self) -> bool {
		self.values().iter().all(|v| v.is_finite() && *v >= 0.0)
	}
}

/// The performance a call would rather have (`provider.preferred_*`), in the terms `health.rs`
/// measures.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Preferred {
	/// Tokens a second.
	min_throughput: Option<f64>,
	/// Seconds to the first byte.
	max_latency: Option<f64>,
}

impl Preferred {
	/// Whether a target with these measurements misses a preference. What has not been measured
	/// misses nothing.
	pub fn misses(&self, first_byte_ms: Option<f64>, tokens_per_second: Option<f64>) -> bool {
		self.min_throughput
			.zip(tokens_per_second)
			.is_some_and(|(floor, measured)| measured < floor)
			|| self
				.max_latency
				.zip(first_byte_ms)
				.is_some_and(|(ceiling, ms)| ms / 1000.0 > ceiling)
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum DataCollection {
	Allow,
	Deny,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum SortName {
	Price,
	Throughput,
	Latency,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Partition {
	Model,
	None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
enum SortBy {
	Name(SortName),
	Object {
		by: SortName,
		#[serde(default)]
		partition: Option<Partition>,
	},
}

fn bad(message: impl Into<String>) -> ApiError {
	ApiError::new(Class::BadRequest, message)
}

/// Whether `slug` names this provider.
fn names(slug: &str, provider: &Provider) -> bool {
	slug.eq_ignore_ascii_case(&provider.name)
		|| provider
			.credential
			.as_deref()
			.is_some_and(|c| slug.eq_ignore_ascii_case(c))
		|| provider.region.as_deref().is_some_and(|region| {
			slug.split_once('/').is_some_and(|(name, at)| {
				name.eq_ignore_ascii_case(&provider.name) && at.eq_ignore_ascii_case(region)
			})
		})
}

impl Prefs {
	/// Reads the `provider` object, refusing what could not be kept.
	pub fn parse(value: Value) -> Result<Self, ApiError> {
		let prefs: Self = serde_json::from_value(value).map_err(|e| {
			bad(format!(
				"provider is OpenRouter's routing object (order, only, ignore, sort, allow_fallbacks, max_price, data_collection, require_parameters): {e}"
			))
		})?;
		for (name, asked) in [
			("zdr", prefs.zdr == Some(true)),
			(
				"enforce_distillable_text",
				prefs.enforce_distillable_text == Some(true),
			),
			(
				"quantizations",
				prefs.quantizations.as_ref().is_some_and(|q| !q.is_empty()),
			),
		] {
			if asked {
				return Err(bad(format!(
					"provider.{name} cannot be kept here: the catalogue holds no such fact about a model. Leave it out."
				)));
			}
		}
		for (name, cutoff) in [
			("preferred_min_throughput", prefs.preferred_min_throughput),
			("preferred_max_latency", prefs.preferred_max_latency),
		] {
			if cutoff.is_some_and(|c| !c.valid()) {
				return Err(bad(format!(
					"provider.{name} is a number of 0 or more, or an object of p50, p75, p90 and p99."
				)));
			}
		}
		if let Some(problem) = prefs.max_price.as_ref().and_then(MaxPrice::problem) {
			return Err(bad(format!("provider.{problem}")));
		}
		Ok(prefs)
	}

	/// The performance the caller would rather have, if it said. Every cutoff of one must be met,
	/// so against a single measurement the tightest decides.
	pub fn preferred(&self) -> Option<Preferred> {
		let tightest = |cutoff: Option<Cutoff>, pick: fn(f64, f64) -> f64| {
			cutoff.and_then(|c| c.values().into_iter().reduce(pick))
		};
		let preferred = Preferred {
			min_throughput: tightest(self.preferred_min_throughput, f64::max),
			max_latency: tightest(self.preferred_max_latency, f64::min),
		};
		(preferred != Preferred::default()).then_some(preferred)
	}

	/// The order the caller asked for, if it did.
	pub fn sort(&self) -> Option<Sort> {
		let name = match self.sort? {
			SortBy::Name(name) | SortBy::Object { by: name, .. } => name,
		};
		Some(match name {
			SortName::Price => Sort::Price,
			SortName::Throughput => Sort::Throughput,
			SortName::Latency => Sort::Latency,
		})
	}

	/// Whether every target is sorted as one list, rather than within each name sent.
	pub fn sorted_as_one(&self) -> bool {
		matches!(
			self.sort,
			Some(SortBy::Object {
				partition: Some(Partition::None),
				..
			})
		)
	}

	pub fn has_order(&self) -> bool {
		self.order.as_ref().is_some_and(|o| !o.is_empty())
	}

	pub fn allow_fallbacks(&self) -> bool {
		self.allow_fallbacks != Some(false)
	}

	pub fn require_parameters(&self) -> bool {
		self.require_parameters == Some(true)
	}

	/// Where `order` puts this target, when it names it.
	pub fn rank(&self, model: &Model) -> Option<usize> {
		self.order
			.as_ref()?
			.iter()
			.position(|slug| names(slug, &model.provider))
	}

	/// Why these controls leave `model` out, for a prompt of about `prompt` tokens; `None` when they
	/// do not.
	pub fn rules_out(&self, model: &Model, prompt: u64) -> Option<String> {
		let provider = &model.provider;
		if let Some(only) = &self.only
			&& !only.iter().any(|slug| names(slug, provider))
		{
			return Some(format!("provider.only does not name {}", provider.name));
		}
		if let Some(ignore) = &self.ignore
			&& ignore.iter().any(|slug| names(slug, provider))
		{
			return Some(format!("provider.ignore names {}", provider.name));
		}
		if !self.allow_fallbacks() && self.has_order() && self.rank(model).is_none() {
			return Some(format!(
				"provider.allow_fallbacks is false and provider.order does not name {}",
				provider.name
			));
		}
		if self.data_collection == Some(DataCollection::Deny)
			&& !model.fits.is_some_and(|f| f.no_data_collection)
		{
			return Some(
				"provider.data_collection is deny and it does not say it keeps no data".to_string(),
			);
		}
		self.max_price
			.as_ref()
			.and_then(|ceiling| ceiling.over(model, prompt))
	}
}

#[cfg(test)]
mod tests {
	use std::collections::HashMap;

	use serde_json::json;

	use super::*;
	use crate::net::Egress;
	use crate::providers::Catalogue;

	fn catalogue() -> Catalogue {
		Catalogue::from_json(
			r#"{"providers": [
				{"name": "OpenAI-Main", "kind": "openai", "base_url": "https://api.openai.com/v1"},
				{"name": "west", "kind": "bedrock", "region": "us-west-2", "credential": "acct-a"}
			], "models": [
				{"name": "gpt", "provider": "OpenAI-Main", "price": {"input": 3, "output": 15}},
				{"name": "private", "provider": "west", "fits": {"no_data_collection": true}}
			]}"#,
			&HashMap::new(),
			Egress::Public,
		)
		.unwrap()
	}

	fn prefs(value: Value) -> Prefs {
		Prefs::parse(value).unwrap()
	}

	#[test]
	fn a_slug_is_a_providers_name_its_credential_or_one_of_its_regions() {
		let models = catalogue();
		let west = models.model("private").unwrap();
		let gpt = models.model("gpt").unwrap();
		assert_eq!(prefs(json!({"only": ["WEST"]})).rules_out(west, 0), None);
		assert_eq!(prefs(json!({"only": ["acct-a"]})).rules_out(west, 0), None);
		assert_eq!(
			prefs(json!({"only": ["west/us-west-2"]})).rules_out(west, 0),
			None
		);
		assert!(
			prefs(json!({"only": ["west/eu-west-1"]}))
				.rules_out(west, 0)
				.is_some()
		);
		assert!(prefs(json!({"only": ["west"]})).rules_out(gpt, 0).is_some());
		assert!(
			prefs(json!({"ignore": ["openai-main"]}))
				.rules_out(gpt, 0)
				.is_some()
		);
	}

	#[test]
	fn controls_narrow_by_what_can_be_shown() {
		let models = catalogue();
		let west = models.model("private").unwrap();
		let gpt = models.model("gpt").unwrap();
		let deny = prefs(json!({"data_collection": "deny"}));
		assert_eq!(deny.rules_out(west, 0), None);
		assert!(deny.rules_out(gpt, 0).unwrap().contains("keeps no data"));
		assert_eq!(
			prefs(json!({"data_collection": "allow"})).rules_out(gpt, 0),
			None
		);

		let ceiling = prefs(json!({"max_price": {"prompt": 1}}));
		assert!(ceiling.rules_out(gpt, 0).unwrap().contains("ceiling"));
		assert!(ceiling.rules_out(west, 0).unwrap().contains("no price"));

		let strict = prefs(json!({"order": ["west"], "allow_fallbacks": false}));
		assert_eq!(strict.rules_out(west, 0), None);
		assert!(strict.rules_out(gpt, 0).is_some());
		assert_eq!(strict.rank(west), Some(0));
		assert_eq!(strict.rank(gpt), None);
		assert_eq!(
			prefs(json!({"order": ["west"]})).rules_out(gpt, 0),
			None,
			"others stay as fallbacks"
		);
	}

	#[test]
	fn sort_is_read_in_both_forms() {
		assert_eq!(prefs(json!({"sort": "price"})).sort(), Some(Sort::Price));
		let object = prefs(json!({"sort": {"by": "latency", "partition": "none"}}));
		assert_eq!(object.sort(), Some(Sort::Latency));
		assert!(object.sorted_as_one());
		assert!(!prefs(json!({"sort": {"by": "throughput"}})).sorted_as_one());
		assert_eq!(prefs(json!({})).sort(), None);
		assert!(Prefs::parse(json!({"sort": "random"})).is_err());
	}

	#[test]
	fn a_preference_is_read_in_both_forms_and_the_tightest_cutoff_decides() {
		assert_eq!(prefs(json!({})).preferred(), None);
		assert_eq!(
			prefs(json!({"preferred_max_latency": {}})).preferred(),
			None
		);
		let number = prefs(json!({"preferred_min_throughput": 100, "preferred_max_latency": 2}))
			.preferred()
			.unwrap();
		assert_eq!(number.min_throughput, Some(100.0));
		assert_eq!(number.max_latency, Some(2.0));
		let object = prefs(json!({
			"preferred_min_throughput": {"p50": 100, "p90": 40},
			"preferred_max_latency": {"p50": 1, "p99": 5}
		}))
		.preferred()
		.unwrap();
		assert_eq!(object.min_throughput, Some(100.0));
		assert_eq!(object.max_latency, Some(1.0));
		for wrong in [
			json!({"preferred_min_throughput": -1}),
			json!({"preferred_max_latency": {"p50": -1}}),
			json!({"preferred_max_latency": {"p95": 1}}),
			json!({"preferred_max_latency": "fast"}),
		] {
			assert!(Prefs::parse(wrong.clone()).is_err(), "{wrong}");
		}
	}

	#[test]
	fn what_has_not_been_measured_misses_nothing() {
		let preferred = prefs(json!({"preferred_min_throughput": 100, "preferred_max_latency": 2}))
			.preferred()
			.unwrap();
		assert!(!preferred.misses(None, None));
		assert!(
			!preferred.misses(Some(2000.0), Some(100.0)),
			"the cutoffs are met at the line"
		);
		assert!(preferred.misses(Some(2001.0), None));
		assert!(preferred.misses(None, Some(99.0)));
		assert!(preferred.misses(Some(100.0), Some(50.0)));
	}

	#[test]
	fn what_cannot_be_kept_is_refused_by_name_and_a_no_op_is_not() {
		let refused = |value: Value| Prefs::parse(value).unwrap_err().message;
		assert!(refused(json!({"zdr": true})).contains("provider.zdr"));
		assert!(refused(json!({"quantizations": ["int8"]})).contains("provider.quantizations"));
		assert!(
			refused(json!({"enforce_distillable_text": true})).contains("enforce_distillable_text")
		);
		assert!(refused(json!({"unknown": 1})).contains("provider is OpenRouter's"));
		assert!(refused(json!({"max_price": {"prompt": -1}})).contains("max_price"));
		for fine in [
			json!({"zdr": false, "quantizations": [], "enforce_distillable_text": false}),
			json!({"preferred_min_throughput": 100, "preferred_max_latency": {"p90": 2}}),
			json!({"order": null, "only": null, "ignore": null, "sort": null}),
		] {
			assert_eq!(
				Prefs::parse(fine.clone()).map(|_| ()).ok(),
				Some(()),
				"{fine}"
			);
		}
	}
}
