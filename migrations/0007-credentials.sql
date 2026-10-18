-- snout-psyche, migration 7: stored provider credentials.
--
-- A provider in the providers file may name a credential (`"credential": "openai"`) instead of the
-- environment variable that holds its key. The credential is a row here, set with the service key
-- through `PUT /ai/v1/admin/credentials/<name>`, proven against the provider before it is stored,
-- and never read back: lists show its kind, a fingerprint and its last four characters.
--
-- Only Psyche's own role, which owns the table, reads it. No role is granted it: not the signed-in
-- or anon roles, and not the project's service role, which bypasses RLS but still needs a grant.
-- A superuser can read any table, so this is the boundary the database draws, not encryption: the
-- same one push's credentials sit behind.
--
-- A change reloads at once (the trigger), so a new key reaches the next call with no restart.

CREATE TABLE psyche.credentials (
	name text PRIMARY KEY CHECK (name ~ '^[a-z0-9_-]{1,63}$'),
	kind text NOT NULL CHECK (kind IN ('api_key', 'aws')),
	-- A provider's API key (OpenAI-style, Anthropic, or a Bedrock API key).
	api_key text,
	-- AWS credentials, which sign Bedrock requests.
	access_key_id text,
	secret_access_key text,
	session_token text,
	-- What lists show: `sha256:` and the first 12 hex digits of the secret's hash, and its last four.
	fingerprint text NOT NULL,
	last_four text NOT NULL,
	created_at timestamptz NOT NULL DEFAULT now(),
	updated_at timestamptz NOT NULL DEFAULT now(),
	CHECK ((kind = 'api_key') = (api_key IS NOT NULL)),
	CHECK ((kind = 'aws') = (access_key_id IS NOT NULL AND secret_access_key IS NOT NULL)),
	CHECK (kind = 'aws' OR session_token IS NULL)
);

ALTER TABLE psyche.credentials ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON psyche.credentials FROM PUBLIC;

CREATE TRIGGER credentials_changed AFTER INSERT OR UPDATE OR DELETE OR TRUNCATE ON psyche.credentials
	FOR EACH STATEMENT EXECUTE FUNCTION psyche.changed();
