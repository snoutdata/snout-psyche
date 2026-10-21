-- snout-psyche, migration 13: each target's health, and routes sorted by it.
--
-- A target is a model of the catalogue, by its provider and its name. Psyche measures each from
-- its own calls (PS16): the time to the first byte and tokens a second as moving averages, and the
-- requests, failures and provider rate limits of the last five minutes. Three failures in a row
-- cool a target down, so a route tries it last until a probe call answers; each cooldown in a row
-- is twice the last, up to five minutes. The statistics live in memory and are written here every
-- few seconds, so the dashboard can show them and a restart does not start blind: the averages and
-- a cooldown still running are read back at start.
--
-- `remaining_requests` and `remaining_tokens` are the provider's own reading of what is left on
-- Psyche's credential, from its rate-limit headers, when it sends them.

CREATE TABLE psyche.provider_stats (
	provider text NOT NULL,
	model text NOT NULL,
	state text NOT NULL CHECK (state IN ('ok', 'cooling', 'probing')),
	requests integer NOT NULL DEFAULT 0,
	failed integer NOT NULL DEFAULT 0,
	rate_limited integer NOT NULL DEFAULT 0,
	first_byte_ms double precision,
	tokens_per_second double precision,
	failures_in_a_row integer NOT NULL DEFAULT 0,
	trips integer NOT NULL DEFAULT 0,
	cooling_until timestamptz,
	last_failure text,
	remaining_requests bigint,
	remaining_tokens bigint,
	updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
	PRIMARY KEY (provider, model)
);

ALTER TABLE psyche.provider_stats ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON psyche.provider_stats FROM PUBLIC;

ALTER TABLE psyche.routes
	DROP CONSTRAINT routes_sort_check,
	ADD CONSTRAINT routes_sort_check CHECK (sort IN ('order', 'price', 'latency', 'throughput'));
