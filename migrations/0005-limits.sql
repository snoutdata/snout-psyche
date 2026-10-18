-- snout-psyche, migration 5: rate limits.
--
-- Requests and tokens a minute, for the project (key_id null), which every call counts against,
-- or for one key. Psyche keeps them as token buckets in memory and reloads this table on
-- `psyche_changed`; a limit that is not changed keeps its bucket through a reload.

CREATE TABLE psyche.limits (
	id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
	name text NOT NULL UNIQUE CHECK (length(name) BETWEEN 1 AND 200),
	key_id uuid REFERENCES psyche.keys (id) ON DELETE CASCADE,
	requests_per_minute integer CHECK (requests_per_minute > 0),
	tokens_per_minute integer CHECK (tokens_per_minute > 0),
	created_at timestamptz NOT NULL DEFAULT now(),
	CHECK (requests_per_minute IS NOT NULL OR tokens_per_minute IS NOT NULL)
);

ALTER TABLE psyche.limits ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON psyche.limits FROM PUBLIC;

CREATE TRIGGER limits_changed AFTER INSERT OR UPDATE OR DELETE OR TRUNCATE ON psyche.limits
	FOR EACH STATEMENT EXECUTE FUNCTION psyche.changed();
