-- snout-psyche, migration 8: the providers and models as tables.
--
-- With a database, these rows are the catalogue Psyche serves. The providers file seeds them at
-- each start (`origin = 'file'`), and the admin API and plain SQL add and change them
-- (`origin = 'api'`); a change reloads at once (the triggers), so a new model needs no restart.
--
-- A provider the file defines is the file's: the file's version is written over it at each start,
-- and Psyche serves it only as the file says, since its key may come from Psyche's environment and
-- must never be sent anywhere the file did not name. A provider made here takes a stored credential
-- (`psyche.credentials`) or none.
--
-- A model the file defines may be edited here (its price, its limits, `enabled`): the edit marks
-- it `edited`, and the file no longer writes over it. `price` is the model's own price, in dollars
-- per million tokens as the providers file writes it; null takes the price catalogue shipped with
-- Psyche, so a corrected catalogue reaches every model that did not set its own.
--
-- Only Psyche's own role reads them.

CREATE TABLE psyche.providers (
	name text PRIMARY KEY CHECK (length(name) BETWEEN 1 AND 100 AND name !~ '[[:space:]/]'),
	kind text NOT NULL CHECK (kind IN ('openai', 'anthropic', 'bedrock')),
	base_url text NOT NULL CHECK (length(base_url) BETWEEN 1 AND 2000),
	region text CHECK (region ~ '^[a-z0-9-]{1,40}$'),
	-- A stored credential, by name; it may be set after the provider is made.
	credential text CHECK (credential ~ '^[a-z0-9_-]{1,63}$'),
	origin text NOT NULL DEFAULT 'api' CHECK (origin IN ('file', 'api')),
	created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
	updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
	CHECK ((kind = 'bedrock') = (region IS NOT NULL)),
	-- Made here, Bedrock has no other way to sign.
	CHECK (origin = 'file' OR kind <> 'bedrock' OR credential IS NOT NULL)
);

CREATE TABLE psyche.models (
	name text PRIMARY KEY CHECK (length(name) BETWEEN 1 AND 200 AND name !~ '[[:space:]]'),
	provider text NOT NULL REFERENCES psyche.providers (name) ON UPDATE CASCADE,
	-- What the provider calls it; null is the model's own name.
	upstream text CHECK (length(upstream) BETWEEN 1 AND 300),
	prompt_caching boolean NOT NULL DEFAULT false,
	max_output_tokens integer CHECK (max_output_tokens > 0),
	reasoning text CHECK (reasoning IN ('budget', 'adaptive', 'effort')),
	embeddings text CHECK (embeddings IN ('titan', 'cohere')),
	price jsonb CHECK (jsonb_typeof(price) = 'object'),
	enabled boolean NOT NULL DEFAULT true,
	origin text NOT NULL DEFAULT 'api' CHECK (origin IN ('file', 'api')),
	-- A model from the file changed here, which the file no longer writes over.
	edited boolean NOT NULL DEFAULT false,
	created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
	updated_at timestamptz NOT NULL DEFAULT clock_timestamp()
);

CREATE INDEX models_provider ON psyche.models (provider);

-- Any change to a model from the file, except the seeding itself (which sets psyche.seeding for its
-- transaction), marks it edited.
CREATE FUNCTION psyche.model_edited() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
	IF coalesce(current_setting('psyche.seeding', true), '') <> 'on' THEN
		NEW.updated_at := clock_timestamp();
		IF NEW.origin = 'file' THEN
			NEW.edited := true;
		END IF;
	END IF;
	RETURN NEW;
END
$$;

CREATE TRIGGER models_edited BEFORE UPDATE ON psyche.models
	FOR EACH ROW EXECUTE FUNCTION psyche.model_edited();

ALTER TABLE psyche.providers ENABLE ROW LEVEL SECURITY;
ALTER TABLE psyche.models ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON psyche.providers, psyche.models FROM PUBLIC;

CREATE TRIGGER providers_changed AFTER INSERT OR UPDATE OR DELETE OR TRUNCATE ON psyche.providers
	FOR EACH STATEMENT EXECUTE FUNCTION psyche.changed();
CREATE TRIGGER models_changed AFTER INSERT OR UPDATE OR DELETE OR TRUNCATE ON psyche.models
	FOR EACH STATEMENT EXECUTE FUNCTION psyche.changed();
