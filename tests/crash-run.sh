#!/usr/bin/env bash
# The inside of tests/crash.sh, run in the stack's dev container with CRASH_DATABASE set to a
# Postgres URL without a database name. Prints one `result <run> <in flight> <answered>` line a run.
set -euo pipefail
cd /work
cargo build --release -q -p snout-psyche --features fake --bins
bin="${CARGO_TARGET_DIR:-/work/target}/release"
work="$(mktemp -d)"
cat >"$work/providers.json" <<'EOF'
{"providers": [{"name": "fake", "kind": "openai", "base_url": "http://127.0.0.1:5399/v1", "api_key_env": "FAKE_KEY"}],
 "models": [{"name": "ok", "provider": "fake", "upstream": "fake-ok"}]}
EOF
secret="a-crash-test-secret-of-at-least-32-chars"
token="$("$bin/psyche-bench" sign --secret "$secret")"

"$bin/psyche-bench" fake --addr 127.0.0.1:5399 2>/dev/null &
fake=$!
trap 'kill "$fake" 2>/dev/null || true' EXIT

# run <name> <in flight> <signal> <seconds of load> <seconds before the signal>
run() {
	local name="$1" in_flight="$2" signal="$3" seconds="$4" after="$5"
	PSYCHE_JWT_SECRET="$secret" PSYCHE_PROVIDERS_FILE="$work/providers.json" PSYCHE_EGRESS=private \
		PSYCHE_ADDR=127.0.0.1:5300 FAKE_KEY=sk-fake-provider-key-0123456789 PSYCHE_LOG=error \
		DATABASE_URL="$CRASH_DATABASE/crash_$name" "$bin/snout-psyche" &
	local psyche=$!
	"$bin/psyche-bench" ready --url http://127.0.0.1:5300/ai/v1/models --seconds 60
	"$bin/psyche-bench" load --url http://127.0.0.1:5300/ai/v1/chat/completions --token "$token" \
		--model ok --concurrency "$in_flight" --seconds "$seconds" --warmup 0 >"$work/$name.json" &
	local load=$!
	sleep "$after"
	kill -"$signal" "$psyche"
	wait "$psyche" 2>/dev/null || true
	wait "$load"
	local answered
	answered="$(grep -o '"requests":[0-9]*' "$work/$name.json" | cut -d: -f2)"
	echo "result $name $in_flight $answered"
}

# A clean stop, once the load is over, so every call has ended.
run clean 64 TERM 3 4
run kill_64 64 KILL 4 2
run kill_4 4 KILL 4 2
