-- snout-psyche, migration 10: when a key was last used.
--
-- Read, never written: the request log's newest row for the key, and once that day has passed the
-- log's retention, the newest day in the daily totals, which are kept for good. A column on
-- psyche.keys updated by every call would announce a change, and so reload every key, on every call.
-- These two indexes make both reads one index probe per key.

CREATE INDEX requests_key ON psyche.requests (key_id, at) WHERE key_id IS NOT NULL;
CREATE INDEX usage_daily_key ON psyche.usage_daily (key_id, day) WHERE key_id IS NOT NULL;
