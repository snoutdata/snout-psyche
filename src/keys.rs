//! Keys: how a person, an app or an agent calls Psyche without the project's service key.
//!
//! A key is `psy_` and 256 random bits in base64url. Only its SHA-256 is stored (`psyche.keys`), so
//! it is shown once, when it is made. Psyche holds every key that is not revoked in memory, so a
//! call is checked with one hash and one lookup and never waits for the database; the set is
//! reloaded whenever the table changes (its trigger sends `psyche_changed`, which [`watch`] listens
//! for) and every [`RELOAD_EVERY`] in case a notification was missed. Expiry is checked at each
//! call, so an expired key stops at its second, not at the next reload.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};

use futures_util::StreamExt;
use tokio_postgres::{AsyncMessage, Client, NoTls};

/// What every key starts with, so a caller (and a secret scanner) can tell one from a JWT.
pub const PREFIX: &str = "psy_";
/// How often the keys are reloaded even when no change was announced.
pub const RELOAD_EVERY: Duration = Duration::from_secs(60);
/// How long a lost connection waits before the next attempt.
const RECONNECT_AFTER: Duration = Duration::from_secs(5);

/// A key's id, a uuid.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct KeyId(pub [u8; 16]);

impl KeyId {
	/// Reads `8-4-4-4-12` hex, as Postgres prints a uuid.
	pub fn parse(text: &str) -> Option<Self> {
		let hex: Vec<u8> = text.bytes().filter(|b| *b != b'-').collect();
		if hex.len() != 32 || text.len() != 36 {
			return None;
		}
		let mut id = [0u8; 16];
		for (i, pair) in hex.chunks(2).enumerate() {
			let pair = std::str::from_utf8(pair).ok()?;
			id[i] = u8::from_str_radix(pair, 16).ok()?;
		}
		Some(Self(id))
	}
}

impl std::fmt::Display for KeyId {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		for (i, b) in self.0.iter().enumerate() {
			if matches!(i, 4 | 6 | 8 | 10) {
				f.write_str("-")?;
			}
			write!(f, "{b:02x}")?;
		}
		Ok(())
	}
}

impl std::fmt::Debug for KeyId {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "KeyId({self})")
	}
}

impl tokio_postgres::types::ToSql for KeyId {
	fn to_sql(
		&self,
		_: &tokio_postgres::types::Type,
		out: &mut bytes::BytesMut,
	) -> Result<tokio_postgres::types::IsNull, Box<dyn std::error::Error + Sync + Send>> {
		out.extend_from_slice(&self.0);
		Ok(tokio_postgres::types::IsNull::No)
	}

	fn accepts(ty: &tokio_postgres::types::Type) -> bool {
		*ty == tokio_postgres::types::Type::UUID
	}

	tokio_postgres::types::to_sql_checked!();
}

/// A key as Psyche holds it: never the key itself.
#[derive(Debug, Clone, PartialEq)]
pub struct Key {
	pub id: KeyId,
	pub name: String,
	/// The model names it may use; `None` for every model.
	pub models: Option<Vec<String>>,
	pub expires_at: Option<SystemTime>,
	/// The user it belongs to, whose budgets and limits it counts against too.
	pub owner: Option<KeyId>,
}

impl Key {
	pub fn allows(&self, model: &str) -> bool {
		self.models
			.as_ref()
			.is_none_or(|models| models.iter().any(|m| m == model))
	}
}

pub type Hash = [u8; 32];

pub fn hash(key: &str) -> Hash {
	let digest = ring::digest::digest(&ring::digest::SHA256, key.as_bytes());
	let mut out = [0u8; 32];
	out.copy_from_slice(digest.as_ref());
	out
}

/// A new key.
pub fn generate() -> String {
	use ring::rand::SecureRandom;
	let mut bytes = [0u8; 32];
	ring::rand::SystemRandom::new()
		.fill(&mut bytes)
		.expect("the system's random number generator");
	format!("{PREFIX}{}", crate::b64url(&bytes))
}

/// What a list shows of a key: enough to recognise it, too little to use it.
pub fn hint(key: &str) -> String {
	let chars: Vec<char> = key.chars().collect();
	let head: String = chars.iter().take(8).collect();
	let tail: String = chars[chars.len().saturating_sub(4)..].iter().collect();
	format!("{head}…{tail}")
}

/// Why a key was not accepted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
	/// Not a key Psyche knows: never made, or revoked.
	Unknown,
	Expired,
}

/// The keys in memory, shared by every request and the task that reloads them.
#[derive(Clone, Default)]
pub struct Store(Arc<RwLock<HashMap<Hash, Arc<Key>>>>);

impl Store {
	pub fn find(&self, key: &str, now: SystemTime) -> Result<Arc<Key>, Refusal> {
		let found = self
			.0
			.read()
			.unwrap_or_else(|e| e.into_inner())
			.get(&hash(key))
			.cloned()
			.ok_or(Refusal::Unknown)?;
		match found.expires_at {
			Some(at) if at <= now => Err(Refusal::Expired),
			_ => Ok(found),
		}
	}

	fn replace(&self, keys: HashMap<Hash, Arc<Key>>) {
		*self.0.write().unwrap_or_else(|e| e.into_inner()) = keys;
	}

	fn insert(&self, hash: Hash, key: Key) {
		self.0
			.write()
			.unwrap_or_else(|e| e.into_inner())
			.insert(hash, Arc::new(key));
	}

	fn remove(&self, id: KeyId) {
		self.0
			.write()
			.unwrap_or_else(|e| e.into_inner())
			.retain(|_, key| key.id != id);
	}

	pub fn len(&self) -> usize {
		self.0.read().unwrap_or_else(|e| e.into_inner()).len()
	}

	pub fn is_empty(&self) -> bool {
		self.len() == 0
	}
}

/// The column is a uuid read as text, so this cannot fail on a row Postgres gave us.
fn key_id(text: &str) -> KeyId {
	KeyId::parse(text).unwrap_or(KeyId([0; 16]))
}

/// Every key that is not revoked.
pub async fn load(client: &Client) -> Result<HashMap<Hash, Arc<Key>>, tokio_postgres::Error> {
	let rows = client
		.query(
			"SELECT id::text, name, hash, models, expires_at, owner::text FROM psyche.keys \
			 WHERE revoked_at IS NULL",
			&[],
		)
		.await?;
	let mut keys = HashMap::with_capacity(rows.len());
	for row in rows {
		let stored: Vec<u8> = row.get(2);
		let Ok(hash) = Hash::try_from(stored.as_slice()) else {
			continue;
		};
		keys.insert(
			hash,
			Arc::new(Key {
				id: key_id(row.get(0)),
				name: row.get(1),
				models: row.get(3),
				expires_at: row.get(4),
				owner: row.get::<_, Option<&str>>(5).and_then(KeyId::parse),
			}),
		);
	}
	Ok(keys)
}

/// What Psyche holds in memory from its configuration tables.
#[derive(Clone, Default)]
pub struct Held {
	pub keys: Store,
	pub budgets: Option<crate::budgets::Budgets>,
	pub limits: Option<crate::limits::Limits>,
	pub members: crate::who::Members,
	pub credentials: Option<crate::credentials::Held>,
}

/// Which tables to read again.
#[derive(Clone, Copy, Default)]
struct Stale {
	keys: bool,
	budgets: bool,
	limits: bool,
	members: bool,
	credentials: bool,
}

impl Stale {
	const ALL: Stale = Stale {
		keys: true,
		budgets: true,
		limits: true,
		members: true,
		credentials: true,
	};

	/// From a notification's payload, the table that changed. A group deleted takes its budgets,
	/// limits and members with it.
	fn table(&mut self, name: &str) {
		match name {
			"keys" => {
				// A key deleted takes its budgets and limits with it.
				self.keys = true;
				self.budgets = true;
				self.limits = true;
			}
			"budgets" => self.budgets = true,
			"limits" => self.limits = true,
			"members" => self.members = true,
			"credentials" => self.credentials = true,
			"groups" => {
				self.budgets = true;
				self.limits = true;
				self.members = true;
			}
			_ => *self = Stale::ALL,
		}
	}
}

/// Keeps `held` in step with `psyche.keys`, `budgets`, `limits`, `members` and `credentials` for as long as the
/// process runs: loads them, then reloads the table each `psyche_changed` names and all of them
/// every [`RELOAD_EVERY`], reconnecting when the connection is lost. `ready` is told after the first
/// load.
pub async fn watch(url: String, held: Held, ready: Option<tokio::sync::oneshot::Sender<()>>) {
	let mut ready = ready;
	loop {
		if let Err(error) = listen(&url, &held, &mut ready).await {
			tracing::warn!(error = %crate::db_message(&error), "the keys' connection failed; the keys held are kept until it is back");
		}
		tokio::time::sleep(RECONNECT_AFTER).await;
	}
}

async fn listen(
	url: &str,
	held: &Held,
	ready: &mut Option<tokio::sync::oneshot::Sender<()>>,
) -> Result<(), tokio_postgres::Error> {
	// One notification name serves every table; its payload names the table, so only that one is
	// read again (members can be many).
	let reload = async |client: &Client, stale: Stale| -> Result<(), tokio_postgres::Error> {
		if stale.keys {
			held.keys.replace(load(client).await?);
		}
		if stale.members {
			held.members.replace(crate::who::load(client).await?);
		}
		if let Some(budgets) = held.budgets.as_ref().filter(|_| stale.budgets) {
			budgets.replace(crate::budgets::load(client).await?);
		}
		if let Some(limits) = held.limits.as_ref().filter(|_| stale.limits) {
			limits.replace(crate::limits::load(client).await?);
		}
		if let Some(credentials) = held.credentials.as_ref().filter(|_| stale.credentials) {
			credentials.replace(&crate::credentials::load(client).await?);
		}
		Ok(())
	};
	let mut config: tokio_postgres::Config = url.parse()?;
	config.application_name("snout-psyche keys");
	let (client, mut connection) = config.connect(NoTls).await?;
	let (changed, mut changes) = tokio::sync::mpsc::unbounded_channel();
	let driver = tokio::spawn(async move {
		let mut messages = futures_util::stream::poll_fn(move |cx| connection.poll_message(cx));
		while let Some(message) = messages.next().await {
			match message {
				Ok(AsyncMessage::Notification(note)) => {
					if changed.send(note.payload().to_string()).is_err() {
						return;
					}
				}
				Ok(_) => {}
				Err(error) => {
					tracing::debug!(%error, "the keys' connection closed");
					return;
				}
			}
		}
	});
	let result = async {
		client.batch_execute("LISTEN psyche_changed").await?;
		reload(&client, Stale::ALL).await?;
		if let Some(ready) = ready.take() {
			let _ = ready.send(());
		}
		let mut tick = tokio::time::interval(RELOAD_EVERY);
		tick.tick().await;
		loop {
			let mut stale = Stale::default();
			tokio::select! {
				change = changes.recv() => {
					let Some(table) = change else {
						return Ok(());
					};
					stale.table(&table);
					// A burst of changes is one reload.
					while let Ok(table) = changes.try_recv() {
						stale.table(&table);
					}
				}
				_ = tick.tick() => stale = Stale::ALL,
			}
			reload(&client, stale).await?;
		}
	}
	.await;
	driver.abort();
	result
}

/// What the admin API makes a key from.
#[derive(Debug, Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NewKey {
	pub name: String,
	#[serde(default)]
	pub models: Option<Vec<String>>,
	/// Seconds from now until it stops working.
	#[serde(default)]
	pub expires_in: Option<u64>,
	/// A user of the project's app, by id.
	#[serde(default)]
	pub owner: Option<String>,
}

/// Makes a key, holds it at once (so it works before the notification comes back) and returns its
/// row together with the key, which is never shown again.
pub async fn create(
	client: &Client,
	store: &Store,
	new: NewKey,
) -> Result<serde_json::Value, CreateError> {
	let name = new.name.trim().to_string();
	if name.is_empty() || name.len() > 200 {
		return Err(CreateError::Invalid(
			"name is 1 to 200 characters.".to_string(),
		));
	}
	if let Some(models) = &new.models
		&& (models.is_empty() || models.iter().any(|m| m.is_empty()))
	{
		return Err(CreateError::Invalid(
			"models is a list of model names, or left out for every model.".to_string(),
		));
	}
	let owner = match &new.owner {
		None => None,
		Some(owner) => Some(
			KeyId::parse(owner)
				.ok_or_else(|| CreateError::Invalid("owner is a user's id, a uuid.".to_string()))?,
		),
	};
	let expires_at = match new.expires_in {
		None => None,
		Some(0) => {
			return Err(CreateError::Invalid(
				"expires_in is a number of seconds above zero.".to_string(),
			));
		}
		Some(seconds) => {
			Some(SystemTime::now() + Duration::from_secs(seconds.min(100 * 365 * 86_400)))
		}
	};
	let key = generate();
	let digest = hash(&key);
	let row = client
		.query_one(
			"INSERT INTO psyche.keys (name, hash, hint, owner, models, expires_at) \
			 VALUES ($1, $2, $3, $4, $5, $6) RETURNING id::text, created_at",
			&[
				&name,
				&digest.as_slice(),
				&hint(&key),
				&owner,
				&new.models,
				&expires_at,
			],
		)
		.await?;
	let id = key_id(row.get(0));
	let created_at: SystemTime = row.get(1);
	store.insert(
		digest,
		Key {
			id,
			name: name.clone(),
			models: new.models.clone(),
			expires_at,
			owner,
		},
	);
	Ok(serde_json::json!({
		"id": id.to_string(),
		"key": key,
		"name": name,
		"hint": hint(&key),
		"owner": owner.map(|o| o.to_string()),
		"models": new.models,
		"created_at": unix(created_at),
		"expires_at": expires_at.map(unix),
	}))
}

#[derive(Debug, thiserror::Error)]
pub enum CreateError {
	#[error("{0}")]
	Invalid(String),
	#[error("database: {}", crate::db_message(.0))]
	Database(#[from] tokio_postgres::Error),
}

fn unix(at: SystemTime) -> u64 {
	at.duration_since(SystemTime::UNIX_EPOCH)
		.map_or(0, |d| d.as_secs())
}

/// Every key, revoked ones included, newest first, as lists show them.
pub async fn list(client: &Client) -> Result<Vec<serde_json::Value>, tokio_postgres::Error> {
	let rows = client
		.query(
			"SELECT id::text, name, hint, owner::text, models, created_at, expires_at, revoked_at \
			 FROM psyche.keys ORDER BY created_at DESC, id",
			&[],
		)
		.await?;
	Ok(rows
		.iter()
		.map(|row| {
			serde_json::json!({
				"id": row.get::<_, String>(0),
				"name": row.get::<_, String>(1),
				"hint": row.get::<_, String>(2),
				"owner": row.get::<_, Option<String>>(3),
				"models": row.get::<_, Option<Vec<String>>>(4),
				"created_at": unix(row.get(5)),
				"expires_at": row.get::<_, Option<SystemTime>>(6).map(unix),
				"revoked_at": row.get::<_, Option<SystemTime>>(7).map(unix),
			})
		})
		.collect())
}

/// Revokes a key, here at once and everywhere else when the change is announced. Returns whether
/// there was a key to revoke.
pub async fn revoke(
	client: &Client,
	store: &Store,
	id: KeyId,
) -> Result<bool, tokio_postgres::Error> {
	let revoked = client
		.execute(
			"UPDATE psyche.keys SET revoked_at = now() WHERE id = $1 AND revoked_at IS NULL",
			&[&id],
		)
		.await?;
	store.remove(id);
	Ok(revoked == 1)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn a_key_is_256_random_bits_behind_its_prefix() {
		let a = generate();
		let b = generate();
		assert!(a.starts_with("psy_"));
		assert_eq!(a.len(), 4 + 43);
		assert_ne!(a, b);
		assert_ne!(hash(&a), hash(&b));
		let shown = hint(&a);
		assert!(shown.starts_with(&a[..8]) && shown.ends_with(&a[a.len() - 4..]));
		assert!(shown.chars().count() < 14, "{shown}");
	}

	#[test]
	fn ids_read_and_print_as_postgres_does() {
		let text = "0190f1c2-7a3b-4c5d-8e9f-a0b1c2d3e4f5";
		let id = KeyId::parse(text).unwrap();
		assert_eq!(id.to_string(), text);
		assert_eq!(id.0[0], 0x01);
		for bad in [
			"",
			"0190f1c27a3b4c5d8e9fa0b1c2d3e4f5",
			"0190f1c2-7a3b-4c5d-8e9f-a0b1c2d3e4fz",
		] {
			assert!(KeyId::parse(bad).is_none(), "{bad}");
		}
	}

	#[test]
	fn a_key_is_found_by_its_hash_and_refused_once_expired() {
		let store = Store::default();
		let now = SystemTime::now();
		let key = generate();
		store.insert(
			hash(&key),
			Key {
				id: KeyId([1; 16]),
				name: "ci".into(),
				models: Some(vec!["luna".into()]),
				expires_at: Some(now + Duration::from_secs(60)),
				owner: None,
			},
		);
		let found = store.find(&key, now).unwrap();
		assert!(found.allows("luna") && !found.allows("sonnet"));
		assert_eq!(
			store.find(&key, now + Duration::from_secs(60)),
			Err(Refusal::Expired)
		);
		assert_eq!(store.find(&generate(), now), Err(Refusal::Unknown));
		store.remove(KeyId([1; 16]));
		assert_eq!(store.find(&key, now), Err(Refusal::Unknown));
	}
}
