-- snout-psyche, migration 11: routes, and on each logged call the route it was made for.
--
-- A route is a name a caller may ask for that stands for several models (PS15), tried in turn: the
-- next only when the one before failed for a reason that was not the caller's. Its targets are
-- model names, not references, so a model disabled or removed is skipped rather than taking the
-- route with it; the admin API names those it skips. A route and a model never share a name.
--
-- The providers file may define routes too (`origin = 'file'`), written over at each start and
-- not changed through the admin API, as its providers are.
--
-- Each attempt a call made is its own row in the request log: `route` is what the caller named when
-- it was a route (or `models`, for OpenRouter's fallback list), and `attempt` counts from 1, so the
-- rows of one call read as the routing's explanation.

CREATE TABLE psyche.routes (
	name text PRIMARY KEY CHECK (length(name) BETWEEN 1 AND 200 AND name !~ '[[:space:]]'),
	targets text[] NOT NULL CHECK (cardinality(targets) BETWEEN 1 AND 20),
	sort text NOT NULL DEFAULT 'order' CHECK (sort IN ('order', 'price')),
	origin text NOT NULL DEFAULT 'api' CHECK (origin IN ('file', 'api')),
	created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
	updated_at timestamptz NOT NULL DEFAULT clock_timestamp()
);

ALTER TABLE psyche.routes ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON psyche.routes FROM PUBLIC;

CREATE TRIGGER routes_changed AFTER INSERT OR UPDATE OR DELETE OR TRUNCATE ON psyche.routes
	FOR EACH STATEMENT EXECUTE FUNCTION psyche.changed();

ALTER TABLE psyche.requests
	ADD COLUMN route text,
	ADD COLUMN attempt smallint;
