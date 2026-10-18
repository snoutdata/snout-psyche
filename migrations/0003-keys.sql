-- snout-psyche, migration 3: keys.
--
-- A key is how a person, an app or an agent calls Psyche without the project's service key:
-- `Authorization: Bearer psy_…`. Only its SHA-256 is kept, so the key itself is shown once, when it
-- is made, and never again; `hint` (its first eight characters and its last four) is what lists show.
-- A key is 256 random bits, so a plain hash is enough: there is nothing to guess.
--
-- `models` limits the key to those model names; null allows every model. `owner` is whose key it is
-- (a user of the project's app, by id), null for the project's own. A revoked key keeps its row, so
-- the request log's `key_id` still names something.
--
-- Psyche holds every live key in memory and reloads them when this table changes: the trigger
-- below sends `psyche_changed`, which Psyche listens for, so a revoke takes effect at once. Anything
-- that edits the table directly (SQL, the admin API, another Psyche) is seen the same way.

CREATE TABLE psyche.keys (
	id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
	name text NOT NULL CHECK (length(name) BETWEEN 1 AND 200),
	hash bytea NOT NULL UNIQUE CHECK (length(hash) = 32),
	hint text NOT NULL,
	owner uuid,
	models text[],
	created_at timestamptz NOT NULL DEFAULT now(),
	expires_at timestamptz,
	revoked_at timestamptz
);

CREATE INDEX keys_owner ON psyche.keys (owner) WHERE owner IS NOT NULL;

ALTER TABLE psyche.keys ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON psyche.keys FROM PUBLIC;

CREATE FUNCTION psyche.changed() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
	PERFORM pg_notify('psyche_changed', TG_TABLE_NAME);
	RETURN NULL;
END
$$;

CREATE TRIGGER keys_changed AFTER INSERT OR UPDATE OR DELETE OR TRUNCATE ON psyche.keys
	FOR EACH STATEMENT EXECUTE FUNCTION psyche.changed();

-- Which key made each call; null for the service key.
ALTER TABLE psyche.requests ADD COLUMN key_id uuid;
