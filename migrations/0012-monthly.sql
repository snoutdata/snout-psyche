-- snout-psyche, migration 12: a budget that starts again every month on its own day.
--
-- `period = 'monthly'` with `starts_on`: the window is the month that began on the latest
-- anniversary of `starts_on` (the 31st reads as a shorter month's last day). It is a billing
-- period that renews by itself, so a budget kept to a subscription's period starts again on time
-- even when nothing writes the new period's first day.

ALTER TABLE psyche.budgets
	DROP CONSTRAINT budgets_period_check,
	ADD CONSTRAINT budgets_period_check CHECK (period IN ('day', 'month', 'rolling', 'since', 'monthly')),
	DROP CONSTRAINT budgets_since,
	ADD CONSTRAINT budgets_since CHECK ((period IN ('since', 'monthly')) = (starts_on IS NOT NULL));
