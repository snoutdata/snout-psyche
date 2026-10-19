//! The `psyche` schema: run what has not run, refuse what has changed.
//!
//! The same ledger rule as snout-push's and snout-storage's runners: a `migrations` table in the
//! schema holding each file's SHA-256, one transaction per file together with its ledger row, and an
//! advisory lock around the whole run so two Psyches starting at once do not both migrate. Psyche's
//! files name no role and know no layout, so this copy is the plain one.

use tokio_postgres::Client;

pub struct Migration {
	pub id: i32,
	pub name: &'static str,
	pub sql: &'static str,
}

impl Migration {
	pub fn hash(&self) -> String {
		let digest = ring::digest::digest(&ring::digest::SHA256, self.sql.as_bytes());
		digest.as_ref().iter().map(|b| format!("{b:02x}")).collect()
	}
}

/// A migration's statements are never edited once released: a change is a new file with the next id.
pub const MIGRATIONS: &[Migration] = &[
	Migration {
		id: 1,
		name: "request-log",
		sql: include_str!("../migrations/0001-request-log.sql"),
	},
	Migration {
		id: 2,
		name: "cost",
		sql: include_str!("../migrations/0002-cost.sql"),
	},
	Migration {
		id: 3,
		name: "keys",
		sql: include_str!("../migrations/0003-keys.sql"),
	},
	Migration {
		id: 4,
		name: "budgets",
		sql: include_str!("../migrations/0004-budgets.sql"),
	},
	Migration {
		id: 5,
		name: "limits",
		sql: include_str!("../migrations/0005-limits.sql"),
	},
	Migration {
		id: 6,
		name: "users",
		sql: include_str!("../migrations/0006-users.sql"),
	},
	Migration {
		id: 7,
		name: "credentials",
		sql: include_str!("../migrations/0007-credentials.sql"),
	},
	Migration {
		id: 8,
		name: "models",
		sql: include_str!("../migrations/0008-models.sql"),
	},
	Migration {
		id: 9,
		name: "accounts",
		sql: include_str!("../migrations/0009-accounts.sql"),
	},
];

/// "snout-ps", so it collides with no other advisory lock by accident.
const LOCK_KEY: i64 = 0x736e_6f75_742d_7073;

#[derive(Debug, thiserror::Error)]
pub enum MigrationError {
	#[error("database: {}", .0.as_db_error().map_or_else(|| .0.to_string(), |e| e.message().to_string()))]
	Database(#[from] tokio_postgres::Error),
	#[error(
		"migration {id} was applied as '{applied_name}' with hash {applied}, and this server's migration {id} is '{name}' with hash {expected}: the schema was made by something else, or a migration file changed after it ran"
	)]
	Mismatch {
		id: i32,
		name: String,
		applied_name: String,
		applied: String,
		expected: String,
	},
	#[error("migration {id} ({name}) failed and was rolled back; nothing after it ran: {reason}")]
	Failed {
		id: i32,
		name: String,
		reason: String,
	},
}

/// Runs whatever of `migrations` this database has not run, and returns the ids it ran.
pub async fn migrate(
	client: &mut Client,
	migrations: &[Migration],
) -> Result<Vec<i32>, MigrationError> {
	loop {
		let row = client
			.query_one("SELECT pg_try_advisory_lock($1)", &[&LOCK_KEY])
			.await?;
		if row.get::<_, bool>(0) {
			break;
		}
		tokio::time::sleep(std::time::Duration::from_secs(1)).await;
	}
	let outcome = run_locked(client, migrations).await;
	let _ = client
		.execute("SELECT pg_advisory_unlock($1)", &[&LOCK_KEY])
		.await;
	outcome
}

async fn run_locked(
	client: &mut Client,
	migrations: &[Migration],
) -> Result<Vec<i32>, MigrationError> {
	// Looked up first: `CREATE SCHEMA IF NOT EXISTS` checks CREATE on the database before it checks
	// existence, and an operator may make the schema in advance, owned by the role that connects.
	let exists: bool = client
		.query_one(
			"SELECT EXISTS (SELECT FROM pg_namespace WHERE nspname = 'psyche')",
			&[],
		)
		.await?
		.get(0);
	if !exists {
		client.batch_execute("CREATE SCHEMA psyche").await?;
	}
	client
		.batch_execute(
			"CREATE TABLE IF NOT EXISTS psyche.migrations (\
			 id integer PRIMARY KEY, name text NOT NULL, hash text NOT NULL, \
			 executed_at timestamptz NOT NULL DEFAULT now())",
		)
		.await?;

	let mut applied: Vec<(i32, String, String)> = Vec::new();
	for row in client
		.query(
			"SELECT id, name, hash FROM psyche.migrations ORDER BY id",
			&[],
		)
		.await?
	{
		applied.push((row.get(0), row.get(1), row.get(2)));
	}
	for migration in migrations {
		if let Some((id, name, hash)) = applied.iter().find(|(id, _, _)| *id == migration.id) {
			let expected = migration.hash();
			if *hash != expected || name != migration.name {
				return Err(MigrationError::Mismatch {
					id: *id,
					name: migration.name.to_string(),
					applied_name: name.clone(),
					applied: hash.clone(),
					expected,
				});
			}
		}
	}

	let mut ran = Vec::new();
	for migration in migrations
		.iter()
		.filter(|m| !applied.iter().any(|(id, _, _)| *id == m.id))
	{
		let result: Result<(), tokio_postgres::Error> = async {
			client.batch_execute("START TRANSACTION").await?;
			client.batch_execute(migration.sql).await?;
			client
				.execute(
					"INSERT INTO psyche.migrations (id, name, hash) VALUES ($1, $2, $3)",
					&[&migration.id, &migration.name, &migration.hash()],
				)
				.await?;
			client.batch_execute("COMMIT").await?;
			Ok(())
		}
		.await;
		if let Err(error) = result {
			let _ = client.batch_execute("ROLLBACK").await;
			let reason = error
				.as_db_error()
				.map_or_else(|| error.to_string(), |e| e.message().to_string());
			return Err(MigrationError::Failed {
				id: migration.id,
				name: migration.name.to_string(),
				reason,
			});
		}
		ran.push(migration.id);
	}
	Ok(ran)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn the_list_is_ordered_and_unique_and_lf() {
		assert!(MIGRATIONS.windows(2).all(|w| w[0].id < w[1].id));
		// The ledger hashes the bytes: a CRLF checkout would make every database look changed.
		assert!(MIGRATIONS.iter().all(|m| !m.sql.contains('\r')));
		assert_eq!(MIGRATIONS[0].hash().len(), 64);
	}
}
