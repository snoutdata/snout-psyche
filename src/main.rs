//! snout-psyche: the server. Configuration is the environment (`config.rs` lists every setting).

use std::sync::Arc;

use axum::serve::ListenerExt;
use snout_psyche::config::Config;
use snout_psyche::providers::Catalogue;
use snout_psyche::proxy::{Proxy, router};

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
	let proxy = Arc::new(Proxy {
		catalogue,
		http,
		jwt_secret: config.jwt_secret.clone(),
		timeouts: config.timeouts,
		max_request_bytes: config.max_request_bytes,
		calls: None,
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
