-- snout-psyche, migration 2: what each call cost.
--
-- In US dollars, exact to the picodollar (10^-12 of a dollar), from the model's price (the
-- providers file's, or the catalogue shipped in the binary) and the tokens the provider reported.
-- Null when the provider reported no usage or nothing prices the model; usage_daily counts the
-- second kind in `unpriced`, so a day's cost never looks complete when it is not.

ALTER TABLE psyche.requests ADD COLUMN cost numeric;

ALTER TABLE psyche.usage_daily
	ADD COLUMN cost numeric NOT NULL DEFAULT 0,
	-- Calls counted whose model has no price, so they are missing from `cost`.
	ADD COLUMN unpriced bigint NOT NULL DEFAULT 0;
