//! The providers and models as tables (Phase 3, PS11 and PS14): with a database, `psyche.providers`
//! and `psyche.models` are the catalogue Psyche serves.
//!
//! The providers file seeds them at each start ([`seed`]), and the admin API and plain SQL edit
//! them; a change reloads at once through `psyche_changed`. Three rules keep the two sources apart:
//!
//! - **A provider from the file is the file's.** Its key may come from Psyche's environment, so it
//!   is served only exactly as the file describes it: a row that differs (edited in SQL, say) is
//!   left out, never sent the environment's key at an address the file did not name. The admin API
//!   refuses to change one. A provider made through the API takes a stored credential or none.
//! - **A model from the file may be edited**, which marks it `edited`, and the file no longer writes
//!   over it. Removing one through the API disables it, since the next start would put it back.
//! - **A row that cannot be served is left out, with the reason**, and the rest are served: one bad
//!   edit must not take the catalogue down. Lists show the reason beside the row.
//!
//! Without a database the file is the catalogue, as before.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};

use serde_json::{Value, json};
use tokio_postgres::Client;

use crate::keys::KeyId;
use crate::net::Egress;
use crate::price::{self, PriceEntry};
use crate::providers::{
	Catalogue, Embeddings, Kind, Live, ModelSpec, Provider, Reasoning, RouteSpec, Sort, Spec,
	build_model, credential_name_ok, word,
};
use crate::who::{GroupId, UserId, Who};

/// The catalogue as the file describes it, and the one calls use: the rows, with the stored
/// credentials filled in.
#[derive(Clone)]
pub struct Held {
	pub file: Arc<Spec>,
	pub live: Live,
	pub egress: Egress,
	/// What the last reload left out, so each is warned about once, not every minute.
	left_out: Arc<Mutex<Vec<(String, String)>>>,
	/// The models only some may use, and who (`access = 'granted'`, `psyche.model_grants`).
	access: Arc<RwLock<HashMap<String, Grants>>>,
}

/// Who a model that is not everyone's is granted to.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Grants {
	pub users: HashSet<UserId>,
	pub groups: HashSet<GroupId>,
}

impl std::fmt::Debug for Held {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("Held").finish_non_exhaustive()
	}
}

impl Held {
	pub fn new(file: Spec, live: Live, egress: Egress) -> Self {
		Self {
			file: Arc::new(file),
			live,
			egress,
			left_out: Default::default(),
			access: Default::default(),
		}
	}

	/// Whether `who` may use `model`. Only a user, or a key a user owns, is held to a model's
	/// grants; the project's own keys and the service key use every model.
	pub fn allows(&self, model: &str, who: &Who) -> bool {
		let Some(user) = who.user else {
			return true;
		};
		let access = self.access.read().unwrap_or_else(|p| p.into_inner());
		match access.get(model) {
			None => true,
			Some(grants) => {
				grants.users.contains(&user)
					|| who.group.is_some_and(|g| grants.groups.contains(&g))
			}
		}
	}

	/// Reads the providers, models and credentials, and serves what they describe.
	pub async fn reload(&self, client: &Client) -> Result<(), tokio_postgres::Error> {
		let (catalogue, left_out) = self.read(client).await?;
		let stored = crate::credentials::load(client).await?;
		let access = load_access(client).await?;
		self.live.set(catalogue.with_credentials(&stored));
		*self.access.write().unwrap_or_else(|p| p.into_inner()) = access;
		let mut last = self.left_out.lock().unwrap_or_else(|p| p.into_inner());
		for (name, why) in &left_out {
			if !last.contains(&(name.clone(), why.clone())) {
				tracing::warn!(name, why, "not served");
			}
		}
		*last = left_out;
		Ok(())
	}

	/// The catalogue the rows describe, before the credentials, and what it left out and why.
	async fn read(
		&self,
		client: &Client,
	) -> Result<(Catalogue, Vec<(String, String)>), tokio_postgres::Error> {
		let (spec, mut left_out) = load(client, &self.file, self.egress).await?;
		let (catalogue, skipped) = match Catalogue::build_lenient(&spec) {
			Ok(built) => built,
			// Only the shipped price catalogue can fail here, and it was read at start.
			Err(error) => (Catalogue::default(), vec![("*".into(), error.0)]),
		};
		left_out.extend(skipped);
		Ok((catalogue, left_out))
	}
}

/// The models only some may use, with their grants.
async fn load_access(client: &Client) -> Result<HashMap<String, Grants>, tokio_postgres::Error> {
	let mut access: HashMap<String, Grants> = HashMap::new();
	for row in client
		.query(
			"SELECT name FROM psyche.models WHERE access = 'granted'",
			&[],
		)
		.await?
	{
		access.insert(row.get(0), Grants::default());
	}
	for row in client
		.query(
			"SELECT model, user_id::text, group_id::text FROM psyche.model_grants",
			&[],
		)
		.await?
	{
		let Some(grants) = access.get_mut(row.get::<_, &str>(0)) else {
			continue;
		};
		if let Some(user) = row.get::<_, Option<&str>>(1).and_then(KeyId::parse) {
			grants.users.insert(user);
		}
		if let Some(group) = row.get::<_, Option<&str>>(2).and_then(KeyId::parse) {
			grants.groups.insert(group);
		}
	}
	Ok(access)
}

/// Writes the file's providers and models into the tables: new ones added, the file's own
/// rewritten (a model only while it is not `edited`), and the file's own that it no longer names
/// removed when nothing depends on them. A row the project made under the same name is kept, and
/// named in what is returned.
pub async fn seed(client: &mut Client, file: &Spec) -> Result<Vec<String>, tokio_postgres::Error> {
	let tx = client.transaction().await?;
	tx.batch_execute("SET LOCAL psyche.seeding = 'on'").await?;
	let mut kept = Vec::new();
	for provider in &file.providers {
		let written = tx
			.execute(
				"INSERT INTO psyche.providers (name, kind, base_url, region, credential, auth_header, \
				 origin) VALUES ($1, $2, $3, $4, $5, $6, 'file') ON CONFLICT (name) DO UPDATE SET \
				 kind = excluded.kind, base_url = excluded.base_url, region = excluded.region, \
				 credential = excluded.credential, auth_header = excluded.auth_header, \
				 updated_at = clock_timestamp() WHERE psyche.providers.origin = 'file'",
				&[
					&provider.name,
					&provider.kind.as_str(),
					&provider.base_url.as_str(),
					&provider.region,
					&provider.credential,
					&provider.auth_header,
				],
			)
			.await?;
		if written == 0 {
			kept.push(format!(
				"the provider {} made through the API is kept; the file's is not used",
				provider.name
			));
		}
	}
	let names: Vec<&str> = file.models.iter().map(|m| m.name.as_str()).collect();
	for model in &file.models {
		let max = model
			.max_output_tokens
			.map(|n| i32::try_from(n).unwrap_or(i32::MAX));
		let written = tx
			.execute(
				"INSERT INTO psyche.models (name, provider, upstream, prompt_caching, \
				 max_output_tokens, reasoning, embeddings, price, origin) \
				 VALUES ($1, $2, $3, $4, $5, $6, $7, $8::text::jsonb, 'file') \
				 ON CONFLICT (name) DO UPDATE SET provider = excluded.provider, \
				 upstream = excluded.upstream, prompt_caching = excluded.prompt_caching, \
				 max_output_tokens = excluded.max_output_tokens, reasoning = excluded.reasoning, \
				 embeddings = excluded.embeddings, price = excluded.price, \
				 updated_at = clock_timestamp() \
				 WHERE psyche.models.origin = 'file' AND NOT psyche.models.edited",
				&[
					&model.name,
					&model.provider,
					&model.upstream,
					&model.prompt_caching,
					&max,
					&model.reasoning.map(Reasoning::as_str),
					&model.embeddings.map(Embeddings::as_str),
					&model.price.as_ref().map(Value::to_string),
				],
			)
			.await?;
		if written == 0 {
			let edited: bool = tx
				.query_one(
					"SELECT origin = 'file' FROM psyche.models WHERE name = $1",
					&[&model.name],
				)
				.await?
				.get(0);
			if !edited {
				kept.push(format!(
					"the model {} made through the API is kept; the file's is not used",
					model.name
				));
			}
		}
	}
	tx.execute(
		"DELETE FROM psyche.models WHERE origin = 'file' AND NOT edited AND NOT (name = ANY($1))",
		&[&names],
	)
	.await?;
	let providers: Vec<&str> = file.providers.iter().map(|p| p.name.as_str()).collect();
	tx.execute(
		"DELETE FROM psyche.providers p WHERE origin = 'file' AND NOT (name = ANY($1)) \
		 AND NOT EXISTS (SELECT 1 FROM psyche.models m WHERE m.provider = p.name)",
		&[&providers],
	)
	.await?;
	for route in &file.routes {
		let written = tx
			.execute(
				"INSERT INTO psyche.routes (name, targets, sort, origin) VALUES ($1, $2, $3, 'file') \
				 ON CONFLICT (name) DO UPDATE SET targets = excluded.targets, sort = excluded.sort, \
				 updated_at = clock_timestamp() WHERE psyche.routes.origin = 'file'",
				&[&route.name, &route.targets, &route.sort.as_str()],
			)
			.await?;
		if written == 0 {
			kept.push(format!(
				"the route {} made through the API is kept; the file's is not used",
				route.name
			));
		}
	}
	let routes: Vec<&str> = file.routes.iter().map(|r| r.name.as_str()).collect();
	tx.execute(
		"DELETE FROM psyche.routes WHERE origin = 'file' AND NOT (name = ANY($1))",
		&[&routes],
	)
	.await?;
	tx.commit().await?;
	Ok(kept)
}

/// The rows, as a [`Spec`], and the providers left out with the reason. A provider from the file
/// takes the file's resolved provider (its environment key included) only when the row says
/// exactly what the file does.
pub async fn load(
	client: &Client,
	file: &Spec,
	egress: Egress,
) -> Result<(Spec, Vec<(String, String)>), tokio_postgres::Error> {
	let mut spec = Spec::default();
	let mut left_out = Vec::new();
	for row in client
		.query(
			"SELECT name, kind, base_url, region, credential, origin, auth_header \
			 FROM psyche.providers ORDER BY created_at, name",
			&[],
		)
		.await?
	{
		let name: String = row.get(0);
		let base_url: String = row.get(2);
		let region: Option<String> = row.get(3);
		let credential: Option<String> = row.get(4);
		let auth_header: Option<String> = row.get(6);
		let Some(kind) = word::<Kind>(row.get(1)) else {
			continue;
		};
		if row.get::<_, &str>(5) == "file" {
			match file.provider(&name) {
				Some(theirs)
					if theirs.kind == kind
						&& theirs.base_url.as_str() == base_url
						&& theirs.region == region
						&& theirs.credential == credential
						&& theirs.auth_header == auth_header =>
				{
					spec.providers.push(theirs.clone());
				}
				Some(_) => left_out.push((
					name,
					"this provider is defined in Psyche's providers file, and the row no longer says what the file does; change it in the file".into(),
				)),
				None => left_out.push((
					name,
					"this provider came from Psyche's providers file, which no longer defines it".into(),
				)),
			}
			continue;
		}
		match Provider::from_row(
			&name,
			kind,
			Some(&base_url),
			region.as_deref(),
			credential.as_deref(),
			auth_header.as_deref(),
			egress,
		) {
			Ok(provider) => spec.providers.push(Arc::new(provider)),
			Err(why) => left_out.push((name, why)),
		}
	}
	for row in client
		.query(
			"SELECT name, provider, upstream, prompt_caching, max_output_tokens, reasoning, \
			 embeddings, price::text FROM psyche.models WHERE enabled ORDER BY created_at, name",
			&[],
		)
		.await?
	{
		let name: String = row.get(0);
		let price = match row.get::<_, Option<String>>(7) {
			None => None,
			Some(text) => match serde_json::from_str(&text) {
				Ok(price) => Some(price),
				Err(_) => continue,
			},
		};
		spec.models.push(ModelSpec {
			name,
			provider: row.get(1),
			upstream: row.get(2),
			prompt_caching: row.get(3),
			max_output_tokens: row
				.get::<_, Option<i32>>(4)
				.and_then(|n| u64::try_from(n).ok()),
			reasoning: row.get::<_, Option<&str>>(5).and_then(word),
			embeddings: row.get::<_, Option<&str>>(6).and_then(word),
			price,
		});
	}
	for row in client
		.query(
			"SELECT name, targets, sort FROM psyche.routes ORDER BY created_at, name",
			&[],
		)
		.await?
	{
		let Some(sort) = word::<Sort>(row.get(2)) else {
			continue;
		};
		spec.routes.push(RouteSpec {
			name: row.get(0),
			targets: row.get(1),
			sort,
		});
	}
	Ok((spec, left_out))
}

/// Why the admin API did not make a change.
#[derive(Debug)]
pub enum Refused {
	/// What was sent is not a provider or model that can be served.
	Invalid(String),
	/// It clashes with what is there (the file's, or a model that depends on it).
	Conflict(String),
	NotFound(String),
	Database(tokio_postgres::Error),
}

impl From<tokio_postgres::Error> for Refused {
	fn from(error: tokio_postgres::Error) -> Self {
		match error.as_db_error() {
			Some(db) if db.code().code().starts_with("23") => Refused::Invalid(db.message().into()),
			_ => Refused::Database(error),
		}
	}
}

/// Checks a provider's or a model's name from a URL path.
pub fn name_ok(text: &str, longest: usize) -> Result<&str, Refused> {
	if text.is_empty() || text.chars().count() > longest || text.chars().any(char::is_whitespace) {
		return Err(Refused::Invalid(format!(
			"A name is 1 to {longest} characters with no spaces."
		)));
	}
	Ok(text)
}

/// What the admin API makes a provider from.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NewProvider {
	pub kind: Kind,
	#[serde(default)]
	pub base_url: Option<String>,
	#[serde(default)]
	pub region: Option<String>,
	#[serde(default)]
	pub credential: Option<String>,
	/// The header the key is sent in, instead of the provider's own way.
	#[serde(default)]
	pub auth_header: Option<String>,
}

/// Makes or replaces a provider through the admin API. One from the providers file is refused.
pub async fn put_provider(
	client: &Client,
	held: &Held,
	name: &str,
	new: NewProvider,
) -> Result<Value, Refused> {
	let name = name_ok(name, 100)?;
	if name.contains('/') {
		return Err(Refused::Invalid("A provider's name has no /.".into()));
	}
	if let Some(credential) = &new.credential
		&& !credential_name_ok(credential)
	{
		return Err(Refused::Invalid(
			"A credential's name is lowercase letters, digits, - and _, up to 63.".into(),
		));
	}
	if new.kind == Kind::Bedrock && new.credential.is_none() {
		return Err(Refused::Invalid(
			"A Bedrock provider made here signs with a stored credential: name one with \"credential\"."
				.into(),
		));
	}
	from_file_refused(client, "providers", name).await?;
	let provider = Provider::from_row(
		name,
		new.kind,
		new.base_url.as_deref(),
		new.region.as_deref(),
		new.credential.as_deref(),
		new.auth_header.as_deref(),
		held.egress,
	)
	.map_err(Refused::Invalid)?;
	client
		.execute(
			"INSERT INTO psyche.providers (name, kind, base_url, region, credential, auth_header) \
			 VALUES ($1, $2, $3, $4, $5, $6) ON CONFLICT (name) DO UPDATE SET kind = excluded.kind, \
			 base_url = excluded.base_url, region = excluded.region, \
			 credential = excluded.credential, auth_header = excluded.auth_header, \
			 updated_at = clock_timestamp()",
			&[
				&name,
				&provider.kind.as_str(),
				&provider.base_url.as_str(),
				&provider.region,
				&provider.credential,
				&provider.auth_header,
			],
		)
		.await?;
	held.reload(client).await?;
	Ok(json!({
		"name": name,
		"kind": provider.kind.as_str(),
		"base_url": provider.base_url.as_str(),
		"region": provider.region,
		"credential": provider.credential,
		"auth_header": provider.auth_header,
		"origin": "api",
	}))
}

/// Removes a provider made through the API, once no model uses it.
pub async fn delete_provider(client: &Client, held: &Held, name: &str) -> Result<(), Refused> {
	from_file_refused(client, "providers", name).await?;
	let users: Vec<String> = client
		.query(
			"SELECT name FROM psyche.models WHERE provider = $1 ORDER BY name",
			&[&name],
		)
		.await?
		.iter()
		.map(|r| r.get(0))
		.collect();
	if !users.is_empty() {
		return Err(Refused::Conflict(format!(
			"{name} serves {}; remove those models or move them to another provider first.",
			users.join(", ")
		)));
	}
	let deleted = client
		.execute("DELETE FROM psyche.providers WHERE name = $1", &[&name])
		.await?;
	if deleted == 0 {
		return Err(Refused::NotFound(
			"There is no provider by that name.".into(),
		));
	}
	held.reload(client).await?;
	Ok(())
}

async fn from_file_refused(client: &Client, table: &str, name: &str) -> Result<(), Refused> {
	let origin = client
		.query_opt(
			&format!("SELECT origin FROM psyche.{table} WHERE name = $1"),
			&[&name],
		)
		.await?;
	if origin.is_some_and(|row| row.get::<_, &str>(0) == "file") {
		let why = if table == "providers" {
			": its key may be Psyche's own"
		} else {
			""
		};
		return Err(Refused::Conflict(format!(
			"{name} is defined in Psyche's providers file, which is where it is changed{why}."
		)));
	}
	Ok(())
}

/// What the admin API makes a model from: the providers file's fields, and `enabled`.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NewModel {
	pub provider: String,
	#[serde(default)]
	pub upstream: Option<String>,
	#[serde(default)]
	pub prompt_caching: bool,
	#[serde(default)]
	pub max_output_tokens: Option<u64>,
	#[serde(default)]
	pub reasoning: Option<Reasoning>,
	#[serde(default)]
	pub embeddings: Option<Embeddings>,
	#[serde(default)]
	pub price: Option<Value>,
	#[serde(default = "yes")]
	pub enabled: bool,
	/// `everyone` (the default), or `granted`: only the users and groups in `psyche.model_grants`.
	#[serde(default)]
	pub access: Option<String>,
}

fn yes() -> bool {
	true
}

/// Makes or replaces a model. It is built first against its provider as served, so one that could
/// not be served is refused with the reason and nothing is written. A model from the file is
/// changed in place and marked `edited`.
pub async fn put_model(
	client: &Client,
	held: &Held,
	name: &str,
	new: NewModel,
) -> Result<Value, Refused> {
	let name = name_ok(name, 200)?;
	if client
		.query_opt("SELECT 1 FROM psyche.routes WHERE name = $1", &[&name])
		.await?
		.is_some()
	{
		return Err(Refused::Conflict(format!(
			"{name} is a route. A model and a route never share a name."
		)));
	}
	let access = new.access.as_deref().unwrap_or("everyone");
	if !matches!(access, "everyone" | "granted") {
		return Err(Refused::Invalid(
			"access is everyone, or granted (only the users and groups in psyche.model_grants)."
				.into(),
		));
	}
	if new
		.max_output_tokens
		.is_some_and(|n| n == 0 || n > i32::MAX as u64)
	{
		return Err(Refused::Invalid(
			"max_output_tokens is a number of tokens above 0.".into(),
		));
	}
	if let Some(price) = &new.price {
		serde_json::from_value::<PriceEntry>(price.clone())
			.map_err(|e| e.to_string())
			.and_then(|p| p.price())
			.map_err(|e| Refused::Invalid(format!("price: {e}")))?;
	}
	let spec = ModelSpec {
		name: name.to_string(),
		provider: new.provider.clone(),
		upstream: new.upstream.clone().filter(|u| !u.is_empty()),
		prompt_caching: new.prompt_caching,
		max_output_tokens: new.max_output_tokens,
		reasoning: new.reasoning,
		embeddings: new.embeddings,
		price: new.price.clone(),
	};
	let (rows, left_out) = load(client, &held.file, held.egress).await?;
	let provider = rows.provider(&spec.provider).ok_or_else(|| {
		match left_out.iter().find(|(n, _)| *n == spec.provider) {
			Some((_, why)) => Refused::Invalid(format!(
				"The provider {} is not served: {why}",
				spec.provider
			)),
			None => Refused::Invalid(format!(
				"There is no provider named {}. GET /ai/v1/admin/providers lists them.",
				spec.provider
			)),
		}
	})?;
	let prices = price::Catalogue::shipped().map_err(|e| Refused::Invalid(e.0))?;
	let model = build_model(&spec, provider, &prices).map_err(Refused::Invalid)?;
	client
		.execute(
			"INSERT INTO psyche.models (name, provider, upstream, prompt_caching, max_output_tokens, \
			 reasoning, embeddings, price, enabled, access) \
			 VALUES ($1, $2, $3, $4, $5, $6, $7, $8::text::jsonb, $9, $10) \
			 ON CONFLICT (name) DO UPDATE SET provider = excluded.provider, \
			 upstream = excluded.upstream, prompt_caching = excluded.prompt_caching, \
			 max_output_tokens = excluded.max_output_tokens, reasoning = excluded.reasoning, \
			 embeddings = excluded.embeddings, price = excluded.price, enabled = excluded.enabled, \
			 access = excluded.access",
			&[
				&name,
				&spec.provider,
				&spec.upstream,
				&spec.prompt_caching,
				&spec.max_output_tokens.map(|n| n as i32),
				&spec.reasoning.map(Reasoning::as_str),
				&spec.embeddings.map(Embeddings::as_str),
				&spec.price.as_ref().map(Value::to_string),
				&new.enabled,
				&access,
			],
		)
		.await?;
	held.reload(client).await?;
	Ok(json!({
		"name": name,
		"provider": spec.provider,
		"upstream": model.upstream,
		"enabled": new.enabled,
		"access": access,
		"price": model.price.map(|p| p.to_json()),
		"price_from": price_from(spec.price.is_some(), model.price.is_some()),
	}))
}

/// Removes a model made through the API; one from the file is disabled instead, since the next
/// start would put it back. Says which it did.
pub async fn delete_model(client: &Client, held: &Held, name: &str) -> Result<Value, Refused> {
	let row = client
		.query_opt("SELECT origin FROM psyche.models WHERE name = $1", &[&name])
		.await?
		.ok_or_else(|| Refused::NotFound("There is no model by that name.".into()))?;
	let from_file = row.get::<_, &str>(0) == "file";
	if from_file {
		client
			.execute(
				"UPDATE psyche.models SET enabled = false WHERE name = $1",
				&[&name],
			)
			.await?;
	} else {
		client
			.execute("DELETE FROM psyche.models WHERE name = $1", &[&name])
			.await?;
	}
	held.reload(client).await?;
	Ok(json!({"name": name, "deleted": !from_file, "disabled": from_file}))
}

/// What the admin API makes a route from.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NewRoute {
	pub targets: Vec<String>,
	#[serde(default)]
	pub sort: Sort,
}

/// The most models a route may name.
pub const MOST_TARGETS: usize = 20;

/// Makes or replaces a route. Its name is not a model's, and each model it names exists (enabled or
/// not); a route from the file is the file's.
pub async fn put_route(
	client: &Client,
	held: &Held,
	name: &str,
	new: NewRoute,
) -> Result<Value, Refused> {
	let name = name_ok(name, 200)?;
	if new.targets.is_empty() || new.targets.len() > MOST_TARGETS {
		return Err(Refused::Invalid(format!(
			"A route names 1 to {MOST_TARGETS} models, in targets."
		)));
	}
	let mut seen = HashSet::new();
	if let Some(twice) = new.targets.iter().find(|t| !seen.insert(t.as_str())) {
		return Err(Refused::Invalid(format!("The route names {twice} twice.")));
	}
	if client
		.query_opt("SELECT 1 FROM psyche.models WHERE name = $1", &[&name])
		.await?
		.is_some()
	{
		return Err(Refused::Conflict(format!(
			"{name} is a model. A route and a model never share a name."
		)));
	}
	let known: Vec<String> = client
		.query(
			"SELECT name FROM psyche.models WHERE name = ANY($1)",
			&[&new.targets],
		)
		.await?
		.iter()
		.map(|r| r.get(0))
		.collect();
	if let Some(missing) = new.targets.iter().find(|t| !known.contains(t)) {
		return Err(Refused::Invalid(format!(
			"There is no model named {missing}. GET /ai/v1/admin/models lists them."
		)));
	}
	from_file_refused(client, "routes", name).await?;
	client
		.execute(
			"INSERT INTO psyche.routes (name, targets, sort) VALUES ($1, $2, $3) \
			 ON CONFLICT (name) DO UPDATE SET targets = excluded.targets, sort = excluded.sort, \
			 updated_at = clock_timestamp()",
			&[&name, &new.targets, &new.sort.as_str()],
		)
		.await?;
	held.reload(client).await?;
	Ok(json!({"name": name, "targets": new.targets, "sort": new.sort.as_str(), "origin": "api"}))
}

/// Removes a route made through the API.
pub async fn delete_route(client: &Client, held: &Held, name: &str) -> Result<(), Refused> {
	from_file_refused(client, "routes", name).await?;
	let deleted = client
		.execute("DELETE FROM psyche.routes WHERE name = $1", &[&name])
		.await?;
	if deleted == 0 {
		return Err(Refused::NotFound("There is no route by that name.".into()));
	}
	held.reload(client).await?;
	Ok(())
}

/// Every route, with the models it skips because they are not served (disabled, removed, or left
/// out with a reason) and whether it is served at all.
pub async fn list_routes(client: &Client, held: &Held) -> Result<Vec<Value>, Refused> {
	let served = held.live.get();
	let rows = client
		.query(
			"SELECT name, targets, sort, origin, created_at::text, updated_at::text \
			 FROM psyche.routes ORDER BY created_at, name",
			&[],
		)
		.await?;
	Ok(rows
		.iter()
		.map(|row| {
			let name: String = row.get(0);
			let targets: Vec<String> = row.get(1);
			let skipped: Vec<&String> = targets
				.iter()
				.filter(|t| served.model(t).is_none())
				.collect();
			let route = served.route(&name);
			let tried: Vec<String> = route
				.map(|r| served.targets(r).iter().map(|m| m.name.clone()).collect())
				.unwrap_or_default();
			json!({
				"name": name,
				"targets": targets,
				"sort": row.get::<_, &str>(2),
				"origin": row.get::<_, &str>(3),
				"created_at": row.get::<_, String>(4),
				"updated_at": row.get::<_, String>(5),
				"tried": tried,
				"skipped": skipped,
				"served": route.is_some() && !tried.is_empty(),
			})
		})
		.collect())
}

fn price_from(own: bool, priced: bool) -> Option<&'static str> {
	match (own, priced) {
		(true, _) => Some("model"),
		(false, true) => Some("catalogue"),
		(false, false) => None,
	}
}

/// Every provider, with where its key comes from and the models it serves, and the reason when it
/// is not served.
pub async fn list_providers(client: &Client, held: &Held) -> Result<Vec<Value>, Refused> {
	let (_, left_out) = held.read(client).await?;
	let stored: Vec<String> = client
		.query("SELECT name FROM psyche.credentials", &[])
		.await?
		.iter()
		.map(|r| r.get(0))
		.collect();
	let rows = client
		.query(
			"SELECT p.name, p.kind, p.base_url, p.region, p.credential, p.origin, \
			 coalesce(array_agg(m.name ORDER BY m.name) FILTER (WHERE m.name IS NOT NULL), '{}'), \
			 p.auth_header FROM psyche.providers p LEFT JOIN psyche.models m ON m.provider = p.name \
			 GROUP BY p.name ORDER BY p.created_at, p.name",
			&[],
		)
		.await?;
	Ok(rows
		.iter()
		.map(|row| {
			let name: String = row.get(0);
			let credential: Option<String> = row.get(4);
			let origin: String = row.get(5);
			let from_environment = origin == "file"
				&& held
					.file
					.provider(&name)
					.is_some_and(|p| p.api_key.is_some() || p.aws.is_some());
			let key = match (&credential, from_environment) {
				(Some(_), _) => "credential",
				(None, true) => "environment",
				(None, false) => "none",
			};
			json!({
				"name": name,
				"kind": row.get::<_, String>(1),
				"base_url": row.get::<_, String>(2),
				"region": row.get::<_, Option<String>>(3),
				"credential": credential,
				"credential_stored": credential.as_ref().map(|c| stored.contains(c)),
				"key": key,
				"origin": origin,
				"models": row.get::<_, Vec<String>>(6),
				"auth_header": row.get::<_, Option<String>>(7),
				"problem": left_out.iter().find(|(n, _)| *n == name).map(|(_, why)| why),
			})
		})
		.collect())
}

/// Every model, with the price it is charged at and where that price comes from, and the reason
/// when it is not served.
pub async fn list_models(client: &Client, held: &Held) -> Result<Vec<Value>, Refused> {
	let (served, left_out) = held.read(client).await?;
	let rows = client
		.query(
			"SELECT name, provider, upstream, prompt_caching, max_output_tokens, reasoning, \
			 embeddings, price::text, enabled, origin, edited, access, \
			 (SELECT count(*) FROM psyche.model_grants g WHERE g.model = m.name) \
			 FROM psyche.models m ORDER BY created_at, name",
			&[],
		)
		.await?;
	let mut problems: HashMap<&str, &str> = HashMap::new();
	for (name, why) in &left_out {
		problems.insert(name, why);
	}
	Ok(rows
		.iter()
		.map(|row| {
			let name: String = row.get(0);
			let provider: String = row.get(1);
			let own = row.get::<_, Option<String>>(7);
			let model = served.model(&name);
			let enabled: bool = row.get(8);
			let problem = problems
				.get(name.as_str())
				.or_else(|| problems.get(provider.as_str()))
				.map(|why| why.to_string());
			json!({
				"name": name,
				"provider": provider,
				"upstream": row.get::<_, Option<String>>(2).unwrap_or_else(|| name.clone()),
				"prompt_caching": row.get::<_, bool>(3),
				"max_output_tokens": row.get::<_, Option<i32>>(4),
				"reasoning": row.get::<_, Option<String>>(5),
				"embeddings": row.get::<_, Option<String>>(6),
				"own_price": own.as_deref().and_then(|t| serde_json::from_str::<Value>(t).ok()),
				"price": model.and_then(|m| m.price).map(|p| p.to_json()),
				"price_from": model.and_then(|m| price_from(own.is_some(), m.price.is_some())),
				"enabled": enabled,
				"served": model.is_some(),
				"origin": row.get::<_, String>(9),
				"edited": row.get::<_, bool>(10),
				"access": row.get::<_, String>(11),
				"grants": row.get::<_, i64>(12),
				"problem": if enabled { problem } else { None },
			})
		})
		.collect())
}
