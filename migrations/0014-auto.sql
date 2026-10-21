-- snout-psyche, migration 14: what a model can take, and on each logged call why its model was
-- the one tried.
--
-- `fits` is the model's own facts as its owner writes them (`fits.rs`): its window in tokens and
-- whether it takes tools, images and a JSON schema. `auto` chooses only among models that wrote
-- them, and every route skips a model whose facts rule a request out before anything is sent.
--
-- `routing` says, on each attempt of a call made for `auto`, a route or a fallback list, how its
-- models were ordered and which were left out and why, so the routing is explained in the log.

ALTER TABLE psyche.models
	ADD COLUMN fits jsonb CHECK (jsonb_typeof(fits) = 'object');

ALTER TABLE psyche.requests
	ADD COLUMN routing text;
