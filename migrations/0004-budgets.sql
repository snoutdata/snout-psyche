-- snout-psyche, migration 4: budgets.
--
-- A budget caps what the project (key_id null) or one key spends in a window: a calendar day or
-- month in UTC, or the last `days` days. Psyche reserves a call's worst-case cost against every
-- budget in its scope before sending it, refuses one that a hard budget has no room for, and
-- settles to the provider's count when it ends. A soft budget only warns.
--
-- The daily totals gain the key, so a key's spend outlives the request log's retention and is
-- what a budget reads back at start. A row per key per day is also each key's usage, ready to show.

CREATE TABLE psyche.budgets (
	id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
	name text NOT NULL UNIQUE CHECK (length(name) BETWEEN 1 AND 200),
	key_id uuid REFERENCES psyche.keys (id) ON DELETE CASCADE,
	-- US dollars.
	amount numeric NOT NULL CHECK (amount >= 0 AND amount < 1000000),
	period text NOT NULL CHECK (period IN ('day', 'month', 'rolling')),
	days integer CHECK (days BETWEEN 1 AND 366),
	hard boolean NOT NULL DEFAULT true,
	created_at timestamptz NOT NULL DEFAULT now(),
	CHECK ((period = 'rolling') = (days IS NOT NULL))
);

ALTER TABLE psyche.budgets ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON psyche.budgets FROM PUBLIC;

CREATE TRIGGER budgets_changed AFTER INSERT OR UPDATE OR DELETE OR TRUNCATE ON psyche.budgets
	FOR EACH STATEMENT EXECUTE FUNCTION psyche.changed();

ALTER TABLE psyche.usage_daily ADD COLUMN key_id uuid;
ALTER TABLE psyche.usage_daily DROP CONSTRAINT usage_daily_pkey;
-- The service key's rows (key_id null) are one row per day like a key's: a unique index over the
-- key with null read as the nil uuid, which the log's upsert names.
CREATE UNIQUE INDEX usage_daily_row ON psyche.usage_daily
	(day, operation, model, provider, (coalesce(key_id, '00000000-0000-0000-0000-000000000000'::uuid)));
