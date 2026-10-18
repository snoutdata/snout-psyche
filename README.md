# snout-psyche

An AI gateway for Postgres-backed apps. Your app, your users and your tools call one
OpenAI-compatible endpoint; Psyche sends each call to the right provider with **your own provider
keys**, streams the answer back as it arrives, and counts what every call used. One static binary,
written in Rust. On SnoutData Cloud one runs beside each project's database, in that project's own
pod, and keeps its keys, budgets and request log in the project's own Postgres.

**Status: early, not production-ready.** What works today is the OpenAI-compatible API over
OpenAI-style providers, Anthropic and Amazon Bedrock, and the request log in Postgres (below).
Keys and budgets, more providers and automatic routing are being built.

## What works today

- `POST /ai/v1/chat/completions` and `POST /ai/v1/completions`, streamed or not, to any provider
  that speaks OpenAI's API (OpenAI, vLLM, Groq, Together, DeepSeek, LM Studio, Ollama's `/v1`).
  Each is also served at `/v1/…`, for clients whose base URL ends in `/v1`.
- **Anthropic models through the same API.** A chat completion request to a model whose provider
  is `anthropic` is translated to Anthropic's Messages API and the answer translated back, streamed
  or not: system messages, images, tools and tool results (parallel calls become one turn),
  `tool_choice`, `stop`, `user`, and usage with the cache reads and writes counted. What has no
  Anthropic equivalent and would change the answer (`n` above 1, `response_format`, log
  probabilities, a field Psyche does not know) is refused with a 400 that names it, never dropped
  in silence; what only tunes OpenAI's own service (`seed`, `store`, the penalties) is dropped.
  `/completions` is refused for an Anthropic model, which has no such API.
- **Amazon Bedrock through the same API**, on its Converse API, which every chat model there
  answers (Claude, OpenAI's models on Bedrock, Qwen, …): the same translation as Anthropic's, with
  Converse's cache points, its binary event stream turned into OpenAI's chunks as it arrives, and a
  model id or inference-profile ARN in the path as Bedrock needs it (`qwen.qwen3-coder-30b-a3b-v1:0`
  works). It signs with AWS Signature Version 4 from an access key (with a session token, for
  temporary credentials), or sends a Bedrock API key instead. Throttling and "no capacity"
  (`ThrottlingException`, `ServiceUnavailableException`, as a status or in the stream) come back as
  `rate_limited`. An image must be inline (a base64 data URL), and `tool_choice: none` and
  `parallel_tool_calls: false` are refused, since Converse cannot say them.
- **Reasoning.** `reasoning_effort` (`none`, `minimal`, `low`, `medium`, `high`) is sent in the
  form the model takes, which its catalogue entry names with `"reasoning"`: `budget` (Claude before
  Sonnet 5: a thinking budget of 1024, 2048 or 4096 tokens, added to `max_tokens`), `adaptive`
  (Claude Sonnet 5 and after: adaptive thinking with that effort) or `effort` (OpenAI's models on
  Bedrock). A model whose entry does not say has `reasoning_effort` refused, never guessed. The
  reply carries `reasoning_content` (the thinking a person can read, when the model shows it) and
  `thinking_blocks` (the blocks as the model wrote them, signatures included); send
  `thinking_blocks` back on the assistant turn and they reach the model byte for byte, which a tool
  loop with thinking on needs.
- **Embeddings** (`POST /ai/v1/embeddings`), in OpenAI's shape. To an OpenAI-style provider the
  request passes through. To Bedrock it becomes InvokeModel calls in the body the model's catalogue
  entry names with `"embeddings"`: `titan` (Amazon Titan, one text per call, so a list of texts is
  sent a few calls at a time and put back in order) or `cohere` (Cohere v3 and v4, up to 96 texts a
  call; `input_type` is Cohere's and may be sent, `search_document` otherwise). `dimensions` maps to
  the model's own field, `encoding_format: base64` returns little-endian `f32`s as OpenAI does, and
  usage is Bedrock's own input count. Token-id input is refused for Bedrock, which takes text only.
- **Prompt-caching breakpoints reach only a model that takes them.** A `cache_control` breakpoint
  (on a message, a content part or a tool) is sent where you put it to a model whose catalogue
  entry says `"prompt_caching": true`, and removed for every other model, which can answer the
  whole request with an error otherwise. Off unless written. Psyche never adds one.
- `GET /ai/v1/models`: the models this server answers to.
- To an OpenAI-style provider the request goes as you wrote it, with three changes: the model
  becomes the provider's own name for it, a streamed call asks the provider for usage so every call
  is counted (if you did not ask for the usage chunk yourself, you do not get it), and breakpoints
  are removed as above. The reply carries the model name **you** asked for; the
  `x-psyche-provider` header says which provider answered.
- Streams are forwarded as they arrive, never buffered. If the caller goes away, the provider's
  call is cancelled, so nobody pays for tokens no one reads.
- A failure after the stream started ends it with an error event, OpenAI's way. Before that it is
  a status: the provider's own status when your request was at fault (a 400), 429 with
  `retry-after` when the provider is rate-limiting, 502 when the provider refused **Psyche's**
  credential (never passed on as if your key were wrong), 504 when it did not answer in time.
- A provider error that echoes the key Psyche sent never reaches the caller or the log.
- Every address is checked before a connection is made: no private ranges, no metadata service,
  no redirects, https only, unless you run it with `PSYCHE_EGRESS=private` beside your own model
  servers.

Callers authenticate as a Bearer token or an `apikey` header, with the project's service key (an
HS256 JWT with `role: service_role`) or, when there is a database, a Psyche key (below). Budgets
come next.

### Keys

A Psyche key (`psy_` and 256 random bits) is for a person, an app or an agent: give each its own,
limit it to the models it needs, and revoke it alone. The service key manages them:

```sh
curl -X POST $PSYCHE/ai/v1/admin/keys -H "Authorization: Bearer $SERVICE_KEY" \
  -d '{"name": "ci", "models": ["sonnet"], "expires_in": 2592000}'
# {"id": "…", "key": "psy_…", "hint": "psy_Ab3x…9QzT", …}   the key is shown this once
curl $PSYCHE/ai/v1/admin/keys -H "Authorization: Bearer $SERVICE_KEY"            # list
curl -X DELETE $PSYCHE/ai/v1/admin/keys/<id> -H "Authorization: Bearer $SERVICE_KEY"  # revoke
```

Only a key's SHA-256 is stored (`psyche.keys`), so a lost key is replaced, not recovered. `models`
left out allows every model; `expires_in` is seconds; `owner` is a user's id, for a key that belongs to
someone. `GET /ai/v1/models` with a key lists only the models it may use, and a call to any other is
refused (403). Each row of the request log names the key that made it (`key_id`).

Psyche holds the live keys in memory, so checking one costs a hash and a lookup, never a query. It
listens for `psyche_changed`, which a trigger on the table sends on every change, so a key made or
revoked anywhere (this API, SQL, another Psyche) takes effect at once; it also reloads every minute
in case a notification was missed while its connection was down. Without `DATABASE_URL` there are no
keys, and only the service key works.

## Configuration

Every setting is an environment variable. Nothing secret has a default: without its JWT secret and
its providers file the server refuses to start, and says which one is missing.

| Variable | Default | Secret | What it is |
| --- | --- | --- | --- |
| `PSYCHE_JWT_SECRET` | (required) | yes | The project's JWT secret, which callers' tokens are verified with. At least 32 characters. |
| `PSYCHE_PROVIDERS_FILE` | (required) | no | The providers and models file (below). |
| `PSYCHE_ADDR` | `0.0.0.0:5300` | no | Where the API listens. |
| `PSYCHE_EGRESS` | `public` | no | `public`: https to public addresses only. `private`: anywhere, http included, for a self-hosted server beside its own model servers. |
| `PSYCHE_FIRST_BYTE_TIMEOUT_S` | `300` | no | How long to wait for a provider's response to begin. |
| `PSYCHE_IDLE_TIMEOUT_S` | `120` | no | The longest silence allowed in the middle of a stream. |
| `PSYCHE_TOTAL_TIMEOUT_S` | `600` | no | A whole non-streamed reply. |
| `PSYCHE_MAX_REQUEST_MB` | `32` | no | The largest request accepted (1 to 256). |
| `PSYCHE_LOG` | `info` | no | Log filter, `tracing` syntax. |
| `PSYCHE_LOG_FORMAT` | text | no | `json` for one JSON object per line. |
| `DATABASE_URL` | (none) | yes | The Postgres the request log is kept in (below). Without it no call is stored. |
| `PSYCHE_LOG_RETENTION_DAYS` | `30` | no | Days of request rows kept (1 to 3650). The daily totals are kept for good. |
| each provider's `api_key_env` | (as named) | yes | The provider's key, in the variable the providers file names. |

### The providers file

The file holds no keys: each provider names the environment variable its key is in, so the file can
be committed.

```json
{
  "providers": [
    {"name": "openai", "kind": "openai", "base_url": "https://api.openai.com/v1",
     "api_key_env": "OPENAI_API_KEY"},
    {"name": "local", "kind": "openai", "base_url": "http://ollama:11434/v1"}
  ],
  "models": [
    {"name": "fast", "provider": "openai", "upstream": "gpt-4o-mini"},
    {"name": "qwen2.5-coder", "provider": "local"}
  ]
}
```

`kind` is `openai` (OpenAI, or any server that speaks its API), `anthropic` (base URL
`https://api.anthropic.com/v1`) or `bedrock`. A Bedrock provider names its `region` instead of a
base URL (`base_url` overrides the endpoint, for a VPC endpoint) and either a Bedrock API key in
`api_key_env` or AWS credentials, each part named by the variable that holds it:

```json
{"name": "bedrock", "kind": "bedrock", "region": "us-west-2",
 "aws": {"access_key_id_env": "AWS_ACCESS_KEY_ID", "secret_access_key_env": "AWS_SECRET_ACCESS_KEY",
         "session_token_env": "AWS_SESSION_TOKEN"}}
```

Its models' `upstream` is the Bedrock model id or inference profile (`us.anthropic.claude-sonnet-5`,
`us.openai.gpt-5.6-luna`). Mark `"prompt_caching": true` only on the models that take cache points
(Claude does; OpenAI's models on Bedrock cache by themselves and refuse a request that has one). `upstream` is the provider's name for the model, and defaults to
`name`. A model may also say `"prompt_caching": true` (above) and `"max_output_tokens"`, the reply
length asked for when the caller names none: Anthropic requires one on every request, and without
either Psyche asks for 4096. A provider with no `api_key_env` is sent no key. A mistake (an unknown provider, a name used twice, a key variable
that is not set, an http URL under `public` egress) stops the server at start with its name.

Each finished call is logged under the `psyche::call` target: operation, model, provider, outcome
(completed, cancelled, or failed with a class), tokens, cost, time to first byte and total time.

### Budgets

A budget caps what the project, or one key, spends in a calendar day or month (UTC) or the last N
days. A hard budget refuses a call that would not fit, before anything is sent; a soft one lets it
through and names itself in the `x-psyche-budget-warning` header.

```sh
curl -X POST $PSYCHE/ai/v1/admin/budgets -H "Authorization: Bearer $SERVICE_KEY" \
  -d '{"name": "ci daily", "amount": 5, "period": "day", "key_id": "<key id>"}'
# period: day, month, or rolling with "days": 1-366; "hard": false for a soft budget;
# no key_id for the project's own. GET lists them, DELETE /ai/v1/admin/budgets/<id> removes one.
```

A budget is kept by **reserving, then settling**. Before a call goes to its provider, its worst
case is reserved against every budget in its scope: the prompt's upper bound (its bytes, since a
token is at least a byte; an inline image counts 2,000 tokens) and the reply's cap (`max_tokens`, or
the model's `max_output_tokens`), at the model's price. When it ends, the reservation becomes what
the provider reported. So two calls that each fit cannot both be sent into room for one, and a long
reply cannot run through a budget: its room was taken before it started. A refusal is a 429 with
the code `budget_exceeded` (OpenAI's type `insufficient_quota`) and a sentence naming the budget,
what it has left and what the call could cost. A model with no price cannot be budgeted, so a call
to one that a budget covers is refused.

Spend is held in memory, per day, and read at start from `psyche.usage_daily` (which keeps each
key's totals for good), so a restart resumes where it was. What it can miss, stated rather than
hidden: the rows a crash did not write (see the request log), and a reply with no cap at all (an
OpenAI-style provider called without `max_tokens`, for a model with no `max_output_tokens`), which
is reserved at 4,096 tokens and can run past that once. Measured against the fake provider, 300
calls at once on a key with a $0.001 day: 18 answered, 282 refused, $0.000342 spent. The bound is
generous where prompts are short, so near its end a budget refuses some calls that would have fit.

### Rate limits

Requests and tokens a minute, for the project (every call counts) or one key:

```sh
curl -X POST $PSYCHE/ai/v1/admin/limits -H "Authorization: Bearer $SERVICE_KEY" \
  -d '{"name": "ci", "key_id": "<key id>", "requests_per_minute": 60, "tokens_per_minute": 100000}'
```

They are token buckets in memory, refilled continuously. A call takes one request and its token
bound (the same worst case a budget reserves) from every limit on it, and gives back what it did
not use when it ends. Over a limit, it is refused with 429, the code `rate_limit_exceeded`, a
`retry-after` and a sentence saying which limit; a call whose bound is more than a limit's whole
minute is refused without a `retry-after`, since waiting would not help. Every response to a call
a limit applies to carries OpenAI's `x-ratelimit-limit-*`, `-remaining-*` and `-reset-*` headers
for requests and tokens, from the tightest limit. A limit that is not changed keeps its bucket
when the table is reloaded; a restart starts every bucket full.

### Prices

A call's cost is the tokens the provider reported at the model's price. The price comes from the
model's `price` in the providers file, in US dollars per million tokens as providers publish them:

```json
{"name": "sonnet", "provider": "bedrock", "upstream": "us.anthropic.claude-sonnet-5-5",
 "price": {"input": 2.2, "output": 11, "cache_read": 0.11, "cache_write": 2.75}}
```

`cache_read` and `cache_write` default to the input price; `request` adds a fee per call in dollars;
`long_context: {"above": 272000, "input": …, "output": …}` is the second tier some models have,
charged for the whole call once the prompt is longer than `above`. Without a `price`, Psyche looks the
model up in the catalogue it ships ([`catalogue/models.json`](catalogue/models.json)), matched on the
provider's kind, its region (Bedrock) or host (the others), and the upstream id, since all three
change the price: a `us.` inference profile costs 10% more than a `global.` one. Today the catalogue
holds the Bedrock models Psyche has been run against, in `us-west-2`, from the AWS Price List API,
with the date they were checked. A model priced by neither still works: its calls are logged with
no cost, and Psyche names it in a warning at start. Costs are kept exact to the picodollar, so a
sum of a million small calls does not drift.

### The request log

With `DATABASE_URL` set, Psyche makes a `psyche` schema in that database at start (migrations in
`migrations/`, each recorded with its hash, so a changed file is refused rather than run again) and
keeps one row per call in `psyche.requests`: the same fields as above, the cost as a `numeric` in
dollars (null when the call was not counted or its model has no price), never a prompt or a reply.
`psyche.usage_daily` holds each day's totals per operation, model and provider, cost included, and
`unpriced`, the calls missing from that cost. Neither is granted
to any other role, and row-level security is on for both.

No call waits for the database. A finished call is handed to a task of its own, which writes what
it holds as one `COPY` every 500 ms or every 1,000 rows, whichever comes first, with the day's
totals in the same transaction. **What a crash can lose is what that task held**: under load about
a thousand rows (killed with `kill -9` at 70,000 calls a second, 1,268 of 159,550 were lost), at low
traffic half a second of calls. A clean stop (SIGTERM) writes everything. If the database cannot be
reached the rows are held and written once it can, up to 100,000; past that the oldest are dropped
and the number is logged.

The rows are partitioned by day (UTC). Psyche makes the partitions from yesterday to two days ahead
and drops those older than `PSYCHE_LOG_RETENTION_DAYS`, so an expired day is one `DROP TABLE`. Tested
on Postgres 17.

## Building and testing

`bash scripts/dev.sh cargo test -p snout-psyche` (in the stack's container) or
`cargo test -p snout-psyche`. The tests run against a fake provider on 127.0.0.1
(`src/fake.rs`, OpenAI's API, Anthropic's and Bedrock's, which checks each SigV4 signature
against the request as it arrived) and never spend a token. The signer is pinned against
signatures botocore made for the same requests. `bash tests/db.sh` runs the same suite with a
throwaway Postgres 17 beside it, which the request log's tests need (they skip without it).

## License

Apache-2.0.
