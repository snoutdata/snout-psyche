//! snout-psyche: the server. Configuration is the environment (`config.rs` lists every setting).

use std::sync::Arc;

use axum::serve::ListenerExt;
use snout_psyche::config::Config;
use snout_psyche::providers::Catalogue;
use snout_psyche::proxy::{Keys, Proxy, router};

#[tokio::main]
async fn main() {
	let json = std::env::var("PSYCHE_LOG_FORMAT").is_ok_and(|v| v == "json");
	let filter =
		tracing_subscriber::EnvFilter::try_from_env("PSYCHE_LOG").unwrap_or_else(|_| "info".into());
	if json {
		tracing_subscriber::fmt()
			.json()
			.with_env_filter(filter)
			.init();
	} else {
		tracing_subscriber::fmt().with_env_filter(filter).init();
	}

	let config = Config::from_env().unwrap_or_else(|error| fail(2, &error.to_string()));
	let text = std::fs::read_to_string(&config.providers_file).unwrap_or_else(|error| {
		fail(
			2,
			&format!(
				"cannot read PSYCHE_PROVIDERS_FILE ({}): {error}",
				config.providers_file
			),
		)
	});
	let catalogue = Catalogue::from_json(&text, &std::env::vars().collect(), config.egress)
		.unwrap_or_else(|error| fail(2, &error.to_string()));
	let http = snout_psyche::net::client(config.egress).unwrap_or_else(|error| fail(1, &error));
	let models = catalogue.models().len();
	for model in catalogue.unpriced() {
		// PS14: served, and logged with no cost. A price in the providers file silences it.
		tracing::warn!(
			model,
			"no price for this model: its calls are logged with no cost; give it a price in the providers file"
		);
	}
	// What calls use: the file's catalogue, with stored credentials filled in once they are loaded.
	let live = snout_psyche::providers::Live::new(catalogue.clone());
	let credentials = snout_psyche::credentials::Held {
		base: catalogue,
		live: live.clone(),
	};
	let held_credentials = config.log.is_some().then(|| credentials.clone());
	// The request log: the schema migrated before the first call, then a task of its own that the
	// calls are handed to. A database that cannot be migrated stops the start, since a log that
	// silently is not kept is worse than none.
	let (calls, log, keys, budgets, limits) = match &config.log {
		None => (None, None, None, None, None),
		Some(settings) => {
			let mut client = snout_psyche::log::connect(&settings.database_url)
				.await
				.unwrap_or_else(|error| fail(1, &format!("cannot reach DATABASE_URL: {error}")));
			let ran = snout_psyche::migrations::migrate(
				&mut client,
				snout_psyche::migrations::MIGRATIONS,
			)
			.await
			.unwrap_or_else(|error| fail(1, &format!("cannot migrate the psyche schema: {error}")));
			if !ran.is_empty() {
				tracing::info!(?ran, "migrated the psyche schema");
			}
			// What was spent before this start, which every budget counts from.
			let budgets = snout_psyche::budgets::Budgets::default();
			let limits = snout_psyche::limits::Limits::default();
			snout_psyche::budgets::load_spent(&client, &budgets, std::time::SystemTime::now())
				.await
				.unwrap_or_else(|error| fail(1, &format!("cannot read what was spent: {error}")));
			drop(client);
			let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
			let task = tokio::spawn(snout_psyche::log::run(settings.clone(), rx));
			// The keys and budgets are loaded before the first call is taken, so a key that works works
			// at once and no call slips past a budget that exists.
			let keys = Keys {
				store: Default::default(),
				members: Default::default(),
				database_url: settings.database_url.clone(),
			};
			let (ready, loaded) = tokio::sync::oneshot::channel();
			tokio::spawn(snout_psyche::keys::watch(
				settings.database_url.clone(),
				snout_psyche::keys::Held {
					keys: keys.store.clone(),
					budgets: Some(budgets.clone()),
					limits: Some(limits.clone()),
					members: keys.members.clone(),
					credentials: Some(credentials.clone()),
				},
				Some(ready),
			));
			if loaded.await.is_err() {
				fail(1, "the keys could not be loaded");
			}
			tracing::info!(keys = keys.store.len(), "loaded the keys");
			for provider in live.get().credentialed() {
				if provider.api_key.is_none() && provider.aws.is_none() {
					tracing::warn!(
						provider = provider.name,
						credential = provider.credential,
						"this provider's stored credential is not set, so its calls are refused until it is"
					);
				}
			}
			(
				Some(tx),
				Some(task),
				Some(keys),
				Some(budgets),
				Some(limits),
			)
		}
	};
	let proxy = Arc::new(Proxy {
		catalogue: live,
		credentials: held_credentials,
		http,
		jwt_secret: config.jwt_secret.clone(),
		timeouts: config.timeouts,
		max_request_bytes: config.max_request_bytes,
		calls,
		keys,
		budgets,
		limits,
	});
	let listener = tokio::net::TcpListener::bind(config.addr)
		.await
		.unwrap_or_else(|error| fail(1, &format!("cannot listen on {}: {error}", config.addr)))
		// A stream is many small writes; Nagle's algorithm would hold each one back for the ACK of
		// the last, which the bench measured as most of a millisecond on every chunk.
		.tap_io(|stream| {
			let _ = stream.set_nodelay(true);
		});
	tracing::info!(addr = %config.addr, models, version = env!("CARGO_PKG_VERSION"), "snout-psyche listening");
	if let Err(error) = axum::serve(listener, router(proxy))
		.with_graceful_shutdown(shutdown())
		.await
	{
		fail(1, &error.to_string());
	}
	// Every connection has ended and the router with it, so the log's channel is closed: it writes
	// what it holds and stops.
	if let Some(log) = log
		&& tokio::time::timeout(std::time::Duration::from_secs(10), log)
			.await
			.is_err()
	{
		tracing::error!("the request log did not finish writing within 10 seconds");
	}
}

fn fail(code: i32, message: &str) -> ! {
	eprintln!("snout-psyche: {message}");
	std::process::exit(code);
}

async fn shutdown() {
	let interrupt = tokio::signal::ctrl_c();
	#[cfg(unix)]
	{
		let mut terminate =
			tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
				.expect("a SIGTERM handler");
		tokio::select! {
			_ = interrupt => {}
			_ = terminate.recv() => {}
		}
	}
	#[cfg(not(unix))]
	{
		let _ = interrupt.await;
	}
}
