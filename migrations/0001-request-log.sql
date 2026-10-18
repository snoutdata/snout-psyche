-- snout-psyche, migration 1: the request log.
--
-- One row per call, and only what describes the call: which model and provider, how it ended, the
-- tokens the provider reported and how long it took. Never a prompt or a reply.
--
-- Rows arrive in batches by COPY from a task of their own, never by an INSERT on a request's path,
-- so the log costs a call nothing. What a crash can lose is the batch not yet written: at most half
-- a second of calls, or 1,000 of them.
--
-- The table is partitioned by day (UTC) so that an expired day is dropped whole rather than deleted
-- row by row. Psyche makes each day's partition ahead of time and drops those older than its
-- retention (PSYCHE_LOG_RETENTION_DAYS, 30 by default). usage_daily keeps each day's totals after
-- its rows are gone; it is written in the same transaction as the rows it counts.
--
-- Neither table is granted to anyone: only the role that owns the schema, which is the one Psyche
-- connects as, reads or writes them. Row-level security is on for both, with no policy, so a grant
-- made by mistake still shows no row.

CREATE TABLE psyche.requests (
	at timestamptz NOT NULL,
	-- chat, completions or embeddings.
	operation text NOT NULL,
	-- The catalogue's name for the model, as the caller asked for it.
	model text NOT NULL,
	provider text NOT NULL,
	streamed boolean NOT NULL,
	-- completed, cancelled (the caller went away first), or the error's code.
	outcome text NOT NULL,
	-- As the provider reported them; null when it reported nothing.
	prompt_tokens integer,
	completion_tokens integer,
	cached_tokens integer,
	cache_write_tokens integer,
	reasoning_tokens integer,
	first_byte_ms integer NOT NULL,
	total_ms integer NOT NULL
) PARTITION BY RANGE (at);

CREATE INDEX requests_at ON psyche.requests USING brin (at);

CREATE TABLE psyche.usage_daily (
	day date NOT NULL,
	operation text NOT NULL,
	model text NOT NULL,
	provider text NOT NULL,
	requests bigint NOT NULL DEFAULT 0,
	failed bigint NOT NULL DEFAULT 0,
	cancelled bigint NOT NULL DEFAULT 0,
	-- Calls whose provider reported no usage, so their tokens are missing from the sums below.
	uncounted bigint NOT NULL DEFAULT 0,
	prompt_tokens bigint NOT NULL DEFAULT 0,
	completion_tokens bigint NOT NULL DEFAULT 0,
	cached_tokens bigint NOT NULL DEFAULT 0,
	cache_write_tokens bigint NOT NULL DEFAULT 0,
	reasoning_tokens bigint NOT NULL DEFAULT 0,
	PRIMARY KEY (day, operation, model, provider)
);

ALTER TABLE psyche.requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE psyche.usage_daily ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON SCHEMA psyche FROM PUBLIC;
REVOKE ALL ON ALL TABLES IN SCHEMA psyche FROM PUBLIC;
