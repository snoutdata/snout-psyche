-- snout-psyche, migration 6: signed-in users and groups.
--
-- An app's signed-in users call Psyche with their own token (the project's JWT, role
-- `authenticated`, the user's id in `sub`). They are budgeted and limited three ways besides the
-- project and keys: each user by a default (`every_user`, which every user has a budget of their
-- own under), one user by id (which replaces the default for that user), and a group (a team inside
-- the app, which its members spend from together). A key with an `owner` counts against its
-- owner's budgets and limits too.
--
-- A user is in at most one group (`members`, keyed by the user), so the daily totals can name the
-- group a call was made under and a group's spend reads back exactly after a restart.
--
-- Owners see their own rows through RLS: their keys (never the hash), their calls, their daily
-- totals, the budgets and limits that apply to them, and their group. The policies read the
-- caller from `request.jwt.claims`, which the data API sets, so they work on any Postgres. The
-- grants are made only where the signed-in role exists.

CREATE TABLE psyche.groups (
	id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
	name text NOT NULL UNIQUE CHECK (length(name) BETWEEN 1 AND 200),
	created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE psyche.members (
	user_id uuid PRIMARY KEY,
	group_id uuid NOT NULL REFERENCES psyche.groups (id) ON DELETE CASCADE,
	created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX members_group ON psyche.members (group_id);

ALTER TABLE psyche.groups ENABLE ROW LEVEL SECURITY;
ALTER TABLE psyche.members ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON psyche.groups, psyche.members FROM PUBLIC;

CREATE TRIGGER groups_changed AFTER INSERT OR UPDATE OR DELETE OR TRUNCATE ON psyche.groups
	FOR EACH STATEMENT EXECUTE FUNCTION psyche.changed();
CREATE TRIGGER members_changed AFTER INSERT OR UPDATE OR DELETE OR TRUNCATE ON psyche.members
	FOR EACH STATEMENT EXECUTE FUNCTION psyche.changed();

-- Whom a budget or a limit is for: the project (none set), a key, every user each, one user, or a
-- group.
ALTER TABLE psyche.budgets
	ADD COLUMN every_user boolean NOT NULL DEFAULT false,
	ADD COLUMN user_id uuid,
	ADD COLUMN group_id uuid REFERENCES psyche.groups (id) ON DELETE CASCADE,
	ADD CONSTRAINT budgets_one_subject
		CHECK (num_nonnulls(key_id, user_id, group_id) + every_user::int <= 1);
ALTER TABLE psyche.limits
	ADD COLUMN every_user boolean NOT NULL DEFAULT false,
	ADD COLUMN user_id uuid,
	ADD COLUMN group_id uuid REFERENCES psyche.groups (id) ON DELETE CASCADE,
	ADD CONSTRAINT limits_one_subject
		CHECK (num_nonnulls(key_id, user_id, group_id) + every_user::int <= 1);

-- Which user made each call, and under which group.
ALTER TABLE psyche.requests ADD COLUMN user_id uuid, ADD COLUMN group_id uuid;
ALTER TABLE psyche.usage_daily ADD COLUMN user_id uuid, ADD COLUMN group_id uuid;
DROP INDEX psyche.usage_daily_row;
CREATE UNIQUE INDEX usage_daily_row ON psyche.usage_daily (day, operation, model, provider,
	(coalesce(key_id, '00000000-0000-0000-0000-000000000000'::uuid)),
	(coalesce(user_id, '00000000-0000-0000-0000-000000000000'::uuid)),
	(coalesce(group_id, '00000000-0000-0000-0000-000000000000'::uuid)));
CREATE INDEX usage_daily_user ON psyche.usage_daily (user_id, day) WHERE user_id IS NOT NULL;
CREATE INDEX budgets_user ON psyche.budgets (user_id) WHERE user_id IS NOT NULL;
CREATE INDEX limits_user ON psyche.limits (user_id) WHERE user_id IS NOT NULL;

-- The caller, as the data API names it; null outside a request.
CREATE FUNCTION psyche.uid() RETURNS uuid LANGUAGE sql STABLE AS $$
	SELECT nullif(nullif(current_setting('request.jwt.claims', true), '')::jsonb ->> 'sub', '')::uuid
$$;

CREATE POLICY own_keys ON psyche.keys FOR SELECT USING (owner = psyche.uid());
CREATE POLICY own_requests ON psyche.requests FOR SELECT USING (user_id = psyche.uid());
CREATE POLICY own_usage ON psyche.usage_daily FOR SELECT USING (user_id = psyche.uid());
CREATE POLICY own_membership ON psyche.members FOR SELECT USING (user_id = psyche.uid());
CREATE POLICY own_group ON psyche.groups FOR SELECT
	USING (id IN (SELECT group_id FROM psyche.members WHERE user_id = psyche.uid()));
CREATE POLICY own_budgets ON psyche.budgets FOR SELECT USING (
	every_user OR user_id = psyche.uid()
	OR group_id IN (SELECT group_id FROM psyche.members WHERE user_id = psyche.uid()));
CREATE POLICY own_limits ON psyche.limits FOR SELECT USING (
	every_user OR user_id = psyche.uid()
	OR group_id IN (SELECT group_id FROM psyche.members WHERE user_id = psyche.uid()));

-- The signed-in role has two names (`authenticated`, and `snout_member` in the new layout); each
-- that exists is granted.
DO $$
DECLARE
	member text;
BEGIN
	FOR member IN SELECT rolname FROM pg_roles WHERE rolname IN ('authenticated', 'snout_member') LOOP
		EXECUTE format('GRANT USAGE ON SCHEMA psyche TO %I', member);
		EXECUTE format('GRANT EXECUTE ON FUNCTION psyche.uid() TO %I', member);
		-- Never the hash.
		EXECUTE format('GRANT SELECT (id, name, hint, owner, models, created_at, expires_at, revoked_at) '
			'ON psyche.keys TO %I', member);
		EXECUTE format('GRANT SELECT ON psyche.requests, psyche.usage_daily, psyche.members, '
			'psyche.groups, psyche.budgets, psyche.limits TO %I', member);
	END LOOP;
END
$$;
