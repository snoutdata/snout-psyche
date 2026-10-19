-- snout-psyche, migration 9: what an account may use and how it pays.
--
-- Four things a service that sells AI to its own users needs, and a project's app too:
--
-- - **Model access.** A model is open to everyone (the default) or only to the users and groups
--   granted it (`access = 'granted'`, `psyche.model_grants`): a model being tried out by a few, or
--   one a plan includes. The project's own keys and the service key are not held to it.
-- - **A window that starts on a date** (`period = 'since'`, `starts_on`): a budget that counts from
--   a billing period's first day, which is neither a calendar month nor the last N days.
-- - **Credits.** A user's balance (`psyche.credits.granted`, US dollars, never resetting), spent only
--   when the user's own budgets (theirs, every user's, their group's) have no room and the credits
--   are switched on. A call is paid wholly by the budgets or wholly by credits. Credits never stand
--   in for the project's budget or a key's. What credits paid is marked on the request log
--   (`credits`) and counted apart in the daily totals (`credits_cost`), so the budgets do not count
--   it and the balance can be read back after a restart.
-- - **A provider that takes its key in a header of its own** (`auth_header`), for a server behind a
--   gateway that checks one: the stored credential is sent as that header instead of a Bearer
--   token.
--
-- And the soft budgets a call went over, on its log row (`budget_warning`), as the response's
-- header already says them.

ALTER TABLE psyche.models
	ADD COLUMN access text NOT NULL DEFAULT 'everyone' CHECK (access IN ('everyone', 'granted'));

CREATE TABLE psyche.model_grants (
	id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
	model text NOT NULL REFERENCES psyche.models (name) ON DELETE CASCADE ON UPDATE CASCADE,
	user_id uuid,
	group_id uuid REFERENCES psyche.groups (id) ON DELETE CASCADE,
	created_at timestamptz NOT NULL DEFAULT now(),
	CHECK (num_nonnulls(user_id, group_id) = 1)
);

CREATE UNIQUE INDEX model_grants_row ON psyche.model_grants (model,
	(coalesce(user_id, '00000000-0000-0000-0000-000000000000'::uuid)),
	(coalesce(group_id, '00000000-0000-0000-0000-000000000000'::uuid)));
CREATE INDEX model_grants_user ON psyche.model_grants (user_id) WHERE user_id IS NOT NULL;

ALTER TABLE psyche.budgets
	ADD COLUMN starts_on date,
	DROP CONSTRAINT budgets_period_check,
	ADD CONSTRAINT budgets_period_check CHECK (period IN ('day', 'month', 'rolling', 'since')),
	ADD CONSTRAINT budgets_since CHECK ((period = 'since') = (starts_on IS NOT NULL));

CREATE TABLE psyche.credits (
	user_id uuid PRIMARY KEY,
	-- US dollars granted in all, ever: what was bought, less nothing. What is left is this less
	-- the daily totals' credits_cost for the user.
	granted numeric NOT NULL DEFAULT 0 CHECK (granted >= 0 AND granted < 1000000),
	enabled boolean NOT NULL DEFAULT true,
	updated_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE psyche.requests
	ADD COLUMN credits boolean NOT NULL DEFAULT false,
	ADD COLUMN budget_warning text;
ALTER TABLE psyche.usage_daily ADD COLUMN credits_cost numeric NOT NULL DEFAULT 0;

ALTER TABLE psyche.providers
	ADD COLUMN auth_header text CHECK (auth_header ~ '^[a-z0-9-]{1,64}$'
		AND auth_header NOT IN ('host', 'content-type', 'content-length', 'transfer-encoding', 'connection'));

ALTER TABLE psyche.model_grants ENABLE ROW LEVEL SECURITY;
ALTER TABLE psyche.credits ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON psyche.model_grants, psyche.credits FROM PUBLIC;

CREATE TRIGGER model_grants_changed AFTER INSERT OR UPDATE OR DELETE OR TRUNCATE ON psyche.model_grants
	FOR EACH STATEMENT EXECUTE FUNCTION psyche.changed();
CREATE TRIGGER credits_changed AFTER INSERT OR UPDATE OR DELETE OR TRUNCATE ON psyche.credits
	FOR EACH STATEMENT EXECUTE FUNCTION psyche.changed();

-- Owners read their own grants and their own balance.
CREATE POLICY own_model_grants ON psyche.model_grants FOR SELECT USING (
	user_id = psyche.uid()
	OR group_id IN (SELECT group_id FROM psyche.members WHERE user_id = psyche.uid()));
CREATE POLICY own_credits ON psyche.credits FOR SELECT USING (user_id = psyche.uid());

DO $$
DECLARE
	member text;
BEGIN
	FOR member IN SELECT rolname FROM pg_roles WHERE rolname IN ('authenticated', 'snout_member') LOOP
		EXECUTE format('GRANT SELECT ON psyche.model_grants, psyche.credits TO %I', member);
	END LOOP;
END
$$;
