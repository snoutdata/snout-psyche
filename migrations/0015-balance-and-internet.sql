-- snout-psyche, migration 15: a model's internet level and quota, and routes that balance.
--
-- `internet` says how far a model's calls may reach. `provider`, every existing model's level: to
-- its provider, and a request carrying a capability the provider runs on the open web (`fits.rs`,
-- `Web`) is not sent to it. `web`: such a request goes through. `none`: to no public address at
-- all, so a call goes only to a provider on a private network, which needs PSYCHE_EGRESS=private.
--
-- `quota` is what the owner knows a model on its account may be sent in a minute (requests, tokens),
-- for a provider that does not say so in its headers, as Bedrock does not. Routing spreads traffic
-- by what is left of it (PS17).
--
-- A route may be `weighted` (its `weights`, by model name) or `balanced` (price, health, speed and
-- room together), and may hold `constraints` every target must meet (PS15).

ALTER TABLE psyche.models
	ADD COLUMN internet text NOT NULL DEFAULT 'provider'
		CHECK (internet IN ('none', 'provider', 'web')),
	ADD COLUMN quota jsonb CHECK (jsonb_typeof(quota) = 'object');

ALTER TABLE psyche.routes
	DROP CONSTRAINT routes_sort_check,
	ADD CONSTRAINT routes_sort_check
		CHECK (sort IN ('order', 'price', 'latency', 'throughput', 'weighted', 'balanced')),
	ADD COLUMN weights jsonb CHECK (jsonb_typeof(weights) = 'object'),
	ADD COLUMN constraints jsonb CHECK (jsonb_typeof(constraints) = 'object');
