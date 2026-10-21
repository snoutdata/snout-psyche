//! The Psyche bench's helper (the bench runs it in a container):
//!
//!   psyche-bench fake --addr 0.0.0.0:5399
//!       the fake provider (`src/fake.rs`) as a server, keeping no record of what it receives
//!   psyche-bench ready --url <url> [--seconds 180]
//!       waits until the URL answers with anything below 500
//!   psyche-bench sign --secret <jwt secret>
//!       prints a service-role token, for calling Psyche
//!   psyche-bench load --url <url> --token <bearer> --model <name> [--stream]
//!                     [--concurrency 1] [--seconds 10] [--warmup 2]
//!       holds `concurrency` chat completion requests in flight for `seconds`, after `warmup`
//!       seconds that are not counted, and prints one JSON object
//!   psyche-bench chaos [--seconds 60] [--concurrency 16] [--targets 3] [--seed 1]
//!                      [--sort order|latency|throughput] [--first-byte-ms 2000]
//!       routing's chaos gate: Psyche and the fake in this process, a route over
//!       `targets` fake models, and a driver that every 0.2 to 1.2 s turns one of them to answering,
//!       slow, 500, 503, 429 with a `retry-after` of 1 to 4 s, or hanging, at random but never all
//!       unhealthy at once. Half the load streams. A request FAILS THE GATE when it failed although
//!       some target was healthy from its start to its end; the run prints one JSON object and
//!       exits 1 if any did
//!
//! Each request's latency is from sending it to the last byte of its reply; for a stream, its time
//! to first byte is to the first piece of the body, the first chunk a caller could show. A
//! response other than 200, or one that breaks off, is an error, counted and never thrown: a server
//! that refuses under load is a result.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use futures_util::StreamExt;
use serde_json::{Value, json};
use snout_psyche::fake::{Chaos, Fake};

fn main() {
	let args: Vec<String> = std::env::args().skip(1).collect();
	let runtime = tokio::runtime::Runtime::new().expect("a runtime");
	let code = runtime.block_on(async move {
		match args.first().map(String::as_str) {
			Some("fake") => fake(&args[1..]).await,
			Some("ready") => ready(&args[1..]).await,
			Some("sign") => sign(&args[1..]),
			Some("load") => load(&args[1..]).await,
			Some("chaos") => chaos(&args[1..]).await,
			_ => {
				eprintln!(
					"usage: psyche-bench fake|ready|sign|load|chaos …  (the file's header says how)"
				);
				2
			}
		}
	});
	std::process::exit(code);
}

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
	args.iter()
		.position(|a| a == name)
		.and_then(|i| args.get(i + 1))
		.map(String::as_str)
}

fn number(args: &[String], name: &str, default: u64) -> u64 {
	flag(args, name)
		.map(|v| {
			v.parse()
				.unwrap_or_else(|_| panic!("{name} is a number, not {v}"))
		})
		.unwrap_or(default)
}

async fn fake(args: &[String]) -> i32 {
	let addr = flag(args, "--addr").unwrap_or("0.0.0.0:5399");
	let fake = Fake::serve(addr, true).await;
	eprintln!("the fake provider is listening on {}", fake.origin);
	let _ = tokio::signal::ctrl_c().await;
	0
}

async fn ready(args: &[String]) -> i32 {
	let Some(url) = flag(args, "--url") else {
		eprintln!("ready needs --url");
		return 2;
	};
	let deadline = Instant::now() + Duration::from_secs(number(args, "--seconds", 180));
	let client = reqwest::Client::new();
	while Instant::now() < deadline {
		if let Ok(response) = client.get(url).send().await
			&& response.status().as_u16() < 500
		{
			return 0;
		}
		tokio::time::sleep(Duration::from_millis(250)).await;
	}
	eprintln!("{url} did not answer in time");
	1
}

fn sign(args: &[String]) -> i32 {
	let Some(secret) = flag(args, "--secret") else {
		eprintln!("sign needs --secret");
		return 2;
	};
	println!(
		"{}",
		snout_psyche::jwt::sign(&json!({"role": "service_role"}), secret)
	);
	0
}

/// One request's outcome.
enum Outcome {
	Done { total: Duration, first: Duration },
	Failed(String),
}

async fn once(client: &reqwest::Client, url: &str, token: &str, body: &[u8]) -> Outcome {
	let started = Instant::now();
	let response = match client
		.post(url)
		.bearer_auth(token)
		.header("content-type", "application/json")
		.body(body.to_vec())
		.send()
		.await
	{
		Ok(response) => response,
		Err(e) => return Outcome::Failed(format!("send: {e}")),
	};
	if response.status() != 200 {
		let status = response.status();
		let text = response.text().await.unwrap_or_default();
		return Outcome::Failed(format!(
			"{status}: {}",
			text.chars().take(200).collect::<String>()
		));
	}
	let mut stream = response.bytes_stream();
	let mut first = None;
	while let Some(piece) = stream.next().await {
		match piece {
			Ok(bytes) if !bytes.is_empty() => {
				first.get_or_insert_with(|| started.elapsed());
			}
			Ok(_) => {}
			Err(e) => return Outcome::Failed(format!("body: {e}")),
		}
	}
	let total = started.elapsed();
	Outcome::Done {
		total,
		first: first.unwrap_or(total),
	}
}

fn percentile(sorted: &[f64], p: f64) -> Option<f64> {
	if sorted.is_empty() {
		return None;
	}
	let at = ((p / 100.0) * sorted.len() as f64).floor() as usize;
	Some((sorted[at.min(sorted.len() - 1)] * 1000.0).round() / 1000.0)
}

async fn load(args: &[String]) -> i32 {
	let (Some(url), Some(token), Some(model)) = (
		flag(args, "--url"),
		flag(args, "--token"),
		flag(args, "--model"),
	) else {
		eprintln!("load needs --url, --token and --model");
		return 2;
	};
	let stream = args.iter().any(|a| a == "--stream");
	let concurrency = number(args, "--concurrency", 1).max(1) as usize;
	let seconds = number(args, "--seconds", 10);
	let warmup = number(args, "--warmup", 2);
	let body = serde_json::to_vec(&json!({
		"model": model, "stream": stream, "max_tokens": 16,
		"messages": [{"role": "user", "content": "hello there"}],
	}))
	.expect("a body");
	let client = reqwest::Client::builder()
		.pool_max_idle_per_host(concurrency)
		.http1_only()
		.timeout(Duration::from_secs(60))
		.build()
		.expect("a client");

	let started = Instant::now();
	let counted_from = started + Duration::from_secs(warmup);
	let deadline = counted_from + Duration::from_secs(seconds);
	let mut workers = Vec::new();
	for _ in 0..concurrency {
		let (client, url, token, body) = (
			client.clone(),
			url.to_string(),
			token.to_string(),
			body.clone(),
		);
		workers.push(tokio::spawn(async move {
			let mut totals = Vec::new();
			let mut firsts = Vec::new();
			let mut errors: Vec<String> = Vec::new();
			while Instant::now() < deadline {
				let sent = Instant::now();
				let outcome = once(&client, &url, &token, &body).await;
				if sent < counted_from {
					continue;
				}
				match outcome {
					Outcome::Done { total, first } => {
						totals.push(total.as_secs_f64() * 1000.0);
						firsts.push(first.as_secs_f64() * 1000.0);
					}
					Outcome::Failed(why) => errors.push(why),
				}
			}
			(totals, firsts, errors)
		}));
	}
	let mut totals = Vec::new();
	let mut firsts = Vec::new();
	let mut errors = Vec::new();
	for worker in workers {
		if let Ok((t, f, e)) = worker.await {
			totals.extend(t);
			firsts.extend(f);
			errors.extend(e);
		}
	}
	totals.sort_by(f64::total_cmp);
	firsts.sort_by(f64::total_cmp);
	let mut kinds: Vec<String> = errors.clone();
	kinds.sort();
	kinds.dedup();
	println!(
		"{}",
		json!({
			"url": url, "model": model, "stream": stream, "concurrency": concurrency,
			"seconds": seconds, "requests": totals.len(), "errors": errors.len(),
			"perSecond": (totals.len() as f64 / seconds as f64).round(),
			"p50Ms": percentile(&totals, 50.0), "p99Ms": percentile(&totals, 99.0),
			"firstByteP50Ms": percentile(&firsts, 50.0), "firstByteP99Ms": percentile(&firsts, 99.0),
			"errorKinds": kinds.into_iter().take(3).collect::<Vec<_>>(),
		})
	);
	0
}

/// xorshift64: the chaos run is replayed by its seed, and needs no crate for it.
struct Rng(u64);

impl Rng {
	fn new(seed: u64) -> Self {
		Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
	}

	fn below(&mut self, n: u64) -> u64 {
		let mut x = self.0;
		x ^= x << 13;
		x ^= x >> 7;
		x ^= x << 17;
		self.0 = x;
		x % n
	}
}

/// One request of the chaos run: when it was sent and finished, and the model that answered or why
/// it failed.
struct Sent {
	start: Instant,
	end: Instant,
	stream: bool,
	answer: Result<String, String>,
}

/// One chat completion through the route, read to its end. A stream counts only if it reached
/// `[DONE]` with no error event on the way.
async fn through(
	client: &reqwest::Client,
	url: &str,
	token: &str,
	stream: bool,
) -> Result<String, String> {
	let mut body = json!({
		"model": "chaos", "stream": stream, "max_tokens": 16,
		"messages": [{"role": "user", "content": "hello there"}],
	});
	if stream {
		body["stream_options"] = json!({"include_usage": true});
	}
	let response = client
		.post(url)
		.bearer_auth(token)
		.json(&body)
		.send()
		.await
		.map_err(|e| format!("send: {e}"))?;
	let status = response.status();
	let model = response
		.headers()
		.get("x-psyche-model")
		.and_then(|v| v.to_str().ok())
		.unwrap_or("")
		.to_string();
	let text = response.text().await.map_err(|e| format!("body: {e}"))?;
	if status != 200 {
		return Err(format!(
			"{status}: {}",
			text.chars().take(160).collect::<String>()
		));
	}
	if stream && (!text.contains("[DONE]") || text.contains("\"error\"")) {
		return Err(format!(
			"the stream broke: {}",
			text.chars()
				.rev()
				.take(160)
				.collect::<String>()
				.chars()
				.rev()
				.collect::<String>()
		));
	}
	Ok(model)
}

async fn chaos(args: &[String]) -> i32 {
	use snout_psyche::config::Timeouts;
	use snout_psyche::net::Egress;
	use snout_psyche::providers::{Catalogue, Live, Spec};
	use snout_psyche::proxy::{Proxy, router};

	let seconds = number(args, "--seconds", 60);
	let concurrency = number(args, "--concurrency", 16).max(1) as usize;
	let count = number(args, "--targets", 3).max(2) as usize;
	let seed = number(args, "--seed", 1);
	let sort = flag(args, "--sort").unwrap_or("order").to_string();
	let first_byte = Duration::from_millis(number(args, "--first-byte-ms", 2000));

	let fake = Arc::new(Fake::serve("127.0.0.1:0", true).await);
	let names: Vec<String> = (1..=count).map(|i| format!("chaos-{i}")).collect();
	let file = json!({
		"providers": [{"name": "fake", "kind": "openai", "base_url": format!("{}/v1", fake.origin), "api_key_env": "FAKE_KEY"}],
		"models": names.iter().map(|n| json!({"name": n, "provider": "fake", "upstream": n})).collect::<Vec<_>>(),
		"routes": [{"name": "chaos", "targets": names, "sort": sort}],
	});
	let spec = match Spec::from_json(
		&file.to_string(),
		&HashMap::from([("FAKE_KEY".to_string(), Fake::KEY.to_string())]),
		Egress::Private,
	)
	.and_then(|spec| Catalogue::build(&spec))
	{
		Ok(catalogue) => catalogue,
		Err(error) => {
			eprintln!("the chaos route could not be built: {error}");
			return 2;
		}
	};
	let secret = "the-chaos-run-signs-its-own-token-0123";
	let health = snout_psyche::health::Health::default();
	let proxy = Arc::new(Proxy {
		catalogue: Live::new(spec),
		tables: None,
		http: snout_psyche::net::client(Egress::Private).expect("a client"),
		jwt_secret: secret.into(),
		timeouts: Timeouts {
			first_byte,
			idle: Duration::from_secs(5),
			total: Duration::from_secs(30),
		},
		max_request_bytes: 1024 * 1024,
		calls: None,
		keys: None,
		budgets: None,
		limits: None,
		health: health.clone(),
	});
	let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
		.await
		.expect("a port");
	let url = format!(
		"http://{}/v1/chat/completions",
		listener.local_addr().expect("its address")
	);
	tokio::spawn(async move {
		let _ = axum::serve(listener, router(proxy)).await;
	});
	let token = snout_psyche::jwt::sign(&json!({"role": "service_role"}), secret);

	// Each target's states, from the moment the fake took each one.
	let all_ok: Vec<(String, Chaos)> = names.iter().map(|n| (n.clone(), Chaos::Ok)).collect();
	let began = fake.set_chaos(&all_ok);
	let timeline = Arc::new(Mutex::new(vec![vec![(began, true)]; count]));
	let deadline = began + Duration::from_secs(seconds);

	let driver = {
		let (fake, timeline, names) = (fake.clone(), timeline.clone(), names.clone());
		tokio::spawn(async move {
			let mut rng = Rng::new(seed);
			let mut modes = vec![Chaos::Ok; count];
			let mut flips = 0u64;
			loop {
				tokio::time::sleep(Duration::from_millis(200 + rng.below(1000))).await;
				if Instant::now() >= deadline {
					break;
				}
				let i = rng.below(count as u64) as usize;
				modes[i] = match rng.below(11) {
					0..=2 => Chaos::Ok,
					3 | 4 => Chaos::Slow,
					5 | 6 => Chaos::Fail,
					7 => Chaos::Unavailable,
					8 | 9 => Chaos::RateLimited(1 + rng.below(4)),
					_ => Chaos::Hang,
				};
				let mut changed = vec![i];
				if !modes.iter().any(|m| m.healthy()) {
					let j = rng.below(count as u64) as usize;
					modes[j] = Chaos::Ok;
					changed.push(j);
				}
				let set: Vec<(String, Chaos)> = changed
					.iter()
					.map(|&k| (names[k].clone(), modes[k]))
					.collect();
				let at = fake.set_chaos(&set);
				let mut timeline = timeline.lock().expect("the timeline");
				for &k in &changed {
					timeline[k].push((at, modes[k].healthy()));
				}
				flips += 1;
			}
			flips
		})
	};

	let client = reqwest::Client::builder()
		.pool_max_idle_per_host(concurrency)
		.timeout(Duration::from_secs(60))
		.build()
		.expect("a client");
	let mut workers = Vec::new();
	for w in 0..concurrency {
		let (client, url, token) = (client.clone(), url.clone(), token.clone());
		let stream = w % 2 == 1;
		workers.push(tokio::spawn(async move {
			let mut sent = Vec::new();
			while Instant::now() < deadline {
				let start = Instant::now();
				let answer = through(&client, &url, &token, stream).await;
				sent.push(Sent {
					start,
					end: Instant::now(),
					stream,
					answer,
				});
			}
			sent
		}));
	}
	let mut sent: Vec<Sent> = Vec::new();
	for worker in workers {
		if let Ok(mine) = worker.await {
			sent.extend(mine);
		}
	}
	let flips = driver.await.unwrap_or(0);

	let timeline = timeline.lock().expect("the timeline");
	// Healthy from `start` to `end`: healthy as the request began, and never turned otherwise
	// before it ended.
	let healthy_through = |k: usize, start: Instant, end: Instant| {
		let states = &timeline[k];
		let at_start = states
			.iter()
			.rev()
			.find(|(t, _)| *t <= start)
			.is_none_or(|(_, h)| *h);
		at_start && !states.iter().any(|(t, h)| *t > start && *t <= end && !h)
	};
	let mut gate = Vec::new();
	let mut excused = Vec::new();
	let mut answered_by: HashMap<String, u64> = HashMap::new();
	let ms = |stream: bool| -> Vec<f64> {
		let mut v: Vec<f64> = sent
			.iter()
			.filter(|s| s.stream == stream && s.answer.is_ok())
			.map(|s| (s.end - s.start).as_secs_f64() * 1000.0)
			.collect();
		v.sort_by(f64::total_cmp);
		v
	};
	let (whole_ms, stream_ms) = (ms(false), ms(true));
	for s in &sent {
		match &s.answer {
			Ok(model) => *answered_by.entry(model.clone()).or_default() += 1,
			Err(why) => {
				let seen = json!({
					"atMs": (s.start - began).as_millis() as u64,
					"tookMs": (s.end - s.start).as_millis() as u64,
					"stream": s.stream, "error": why,
				});
				if (0..count).any(|k| healthy_through(k, s.start, s.end)) {
					gate.push(seen);
				} else {
					// Each target's state as it began, and every change while it ran (ms from the start).
					let mut seen = seen;
					seen["targets"] = (0..count)
						.map(|k| {
							let states = &timeline[k];
							let first = states
								.iter()
								.rev()
								.find(|(t, _)| *t <= s.start)
								.is_none_or(|(_, h)| *h);
							let changes: Vec<Value> = states
								.iter()
								.filter(|(t, _)| *t > s.start && *t <= s.end)
								.map(|(t, h)| json!([(*t - s.start).as_millis() as u64, h]))
								.collect();
							json!({"healthyAtStart": first, "changes": changes})
						})
						.collect();
					excused.push(seen);
				}
			}
		}
	}
	let served = fake.served();
	let upstream: u64 = served.values().sum();
	let wasted: u64 = served
		.iter()
		.filter(|((_, mode), _)| !matches!(*mode, "ok" | "slow"))
		.map(|(_, n)| n)
		.sum();
	let mut by_mode: HashMap<&str, u64> = HashMap::new();
	for ((_, mode), n) in &served {
		*by_mode.entry(mode).or_default() += n;
	}
	let failed = sent.iter().filter(|s| s.answer.is_err()).count();
	let result = json!({
		"seconds": seconds, "concurrency": concurrency, "targets": count, "seed": seed, "sort": sort,
		"firstByteTimeoutMs": first_byte.as_millis() as u64, "flips": flips,
		"requests": sent.len(), "answered": sent.len() - failed, "failed": failed,
		"failedWithAHealthyTarget": gate.len(), "failedWithNoHealthyTarget": excused.len(),
		"answeredBy": answered_by,
		"wholeP50Ms": percentile(&whole_ms, 50.0), "wholeP99Ms": percentile(&whole_ms, 99.0),
		"streamP50Ms": percentile(&stream_ms, 50.0), "streamP99Ms": percentile(&stream_ms, 99.0),
		"upstreamAttempts": upstream,
		"attemptsPerRequest": (upstream as f64 / sent.len().max(1) as f64 * 1000.0).round() / 1000.0,
		"attemptsOnAnUnhealthyTarget": wasted, "upstreamByState": by_mode,
		"gateFailures": gate.iter().take(10).collect::<Vec<&Value>>(),
		"excusedFailures": excused.iter().take(10).collect::<Vec<&Value>>(),
		"health": health.report(SystemTime::now()),
	});
	println!("{}", serde_json::to_string_pretty(&result).expect("json"));
	i32::from(!gate.is_empty())
}
