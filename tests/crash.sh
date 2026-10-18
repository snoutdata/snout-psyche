#!/usr/bin/env bash
# What the request log keeps when Psyche stops, cleanly and by `kill -9`, as a test with a verdict.
#
#   bash tests/crash.sh
#
# A throwaway Postgres 17 on its own network, and the stack's dev container joined to it, running
# tests/crash-run.sh: a release Psyche in front of the fake provider, held under load by
# `psyche-bench load`, and then stopped. Each run has a database of its own, and once they are done
# the rows each one wrote are counted here and set against the calls the load saw answered.
#
#   clean      64 in flight for 3 s, then SIGTERM: every answered call is written.
#   kill-64    64 in flight, SIGKILL 2 s in: what is lost is at most what the log's task held since
#   kill-4     4 in flight, the same       its last write, which is two batches (`FLUSH_ROWS`, one
#                                          being written and one filling) and the calls in flight.
#
# A row can also be written for a call whose answer the load had not finished reading when Psyche
# died, so the rows may exceed the answered calls by up to the calls in flight.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
engine="${STACK_ENGINE:-docker}"
net="psyche-crash-$$"
pg="psyche-crash-pg-$$"
out="$(mktemp)"
flush_rows=1000

"$engine" network create "$net" >/dev/null
cleanup() {
	"$engine" rm -f "$pg" >/dev/null 2>&1 || true
	"$engine" network rm "$net" >/dev/null 2>&1 || true
	rm -f "$out"
}
trap cleanup EXIT
"$engine" run -d --rm --name "$pg" --network "$net" -e POSTGRES_PASSWORD=test postgres:17-alpine >/dev/null
until "$engine" exec "$pg" psql -h 127.0.0.1 -U postgres -tAc 'select 1' >/dev/null 2>&1; do sleep 0.5; done
sql() { "$engine" exec "$pg" psql -h 127.0.0.1 -U postgres -tAq "$@"; }
for run in clean kill_64 kill_4; do
	sql -c "CREATE DATABASE crash_$run"
done

dev="$here/../../scripts/dev.sh"
[ -f "$dev" ] || dev="$here/../scripts/dev.sh"
STACK_DEV_ARGS="--network $net -e CRASH_DATABASE=postgres://postgres:test@$pg:5432" \
	bash "$dev" bash psyche/tests/crash-run.sh | tee "$out"

failed=0
while read -r run in_flight answered; do
	rows="$(sql -d "crash_$run" -c 'SELECT count(*) FROM psyche.requests')"
	lost=$((answered - rows))
	case "$run" in
	clean) allowed=0 ;;
	*) allowed=$((2 * flush_rows + in_flight)) ;;
	esac
	verdict=ok
	if [ "$lost" -gt "$allowed" ] || [ "$((rows - answered))" -gt "$in_flight" ]; then
		verdict=FAILED
		failed=1
	fi
	printf '%-8s %3s in flight: %7s answered, %7s written, %5s lost (at most %s): %s\n' \
		"$run" "$in_flight" "$answered" "$rows" "$lost" "$allowed" "$verdict"
done < <(sed -n 's/^result //p' "$out")
exit "$failed"
