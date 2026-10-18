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
//!
//! Each request's latency is from sending it to the last byte of its reply; for a stream, its time
//! to first byte is to the first piece of the body, the first chunk a caller could show. A
//! response other than 200, or one that breaks off, is an error, counted and never thrown: a server
//! that refuses under load is a result.

use std::time::{Duration, Instant};

use futures_util::StreamExt;
use serde_json::json;
use snout_psyche::fake::Fake;

fn main() {
	let args: Vec<String> = std::env::args().skip(1).collect();
	let runtime = tokio::runtime::Runtime::new().expect("a runtime");
	let code = runtime.block_on(async move {
		match args.first().map(String::as_str) {
			Some("fake") => fake(&args[1..]).await,
			Some("ready") => ready(&args[1..]).await,
			Some("sign") => sign(&args[1..]),
			Some("load") => load(&args[1..]).await,
			_ => {
				eprintln!(
					"usage: psyche-bench fake|ready|sign|load …  (the file's header says how)"
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
