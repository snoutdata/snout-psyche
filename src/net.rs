//! The only way Psyche makes a request to a provider, and the rules every request keeps.
//!
//! A provider's base URL is configuration a project's owner writes, so in the Cloud it is untrusted
//! input: pointed at the metadata service or the pod network, it would be a way in. The checks live
//! here, in process, as in snout-push (whose address rules are carried over unchanged):
//!
//!  - **Every address is checked before a connection is made.** Names resolve through
//!    [`SafeResolver`], which drops anything private, loopback, link-local, carrier-grade NAT,
//!    multicast, reserved or documentation-only, in IPv4 and IPv6. The connection goes to an
//!    address that passed, so DNS rebinding never reaches a private one: there is no second lookup.
//!  - **A base URL naming an address directly is refused**, since it would skip the resolver.
//!  - **No redirects, no proxies, https only.**
//!
//! [`Egress::Private`] relaxes all three, for a self-hosted Psyche whose models are on its own
//! network (Ollama on a LAN, vLLM beside it). It is a setting a self-hoster chooses
//! (`PSYCHE_EGRESS=private`); the Cloud never sets it.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Where Psyche may send.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Egress {
	/// The public internet over https, and nothing else. The default, and the Cloud's only setting.
	Public,
	/// Anywhere, http included: a self-hosted Psyche beside its own model servers.
	Private,
}

/// Whether an address is somewhere on the public internet this server may send to.
pub fn is_public(ip: IpAddr) -> bool {
	match ip {
		IpAddr::V4(v4) => is_public_v4(v4),
		IpAddr::V6(v6) => is_public_v6(v6),
	}
}

fn is_public_v4(ip: Ipv4Addr) -> bool {
	let [a, b, c, _] = ip.octets();
	!(ip.is_unspecified()
		|| ip.is_loopback()
		|| ip.is_private()
		|| ip.is_link_local()
		|| ip.is_broadcast()
		|| ip.is_multicast()
		|| ip.is_documentation()
		|| a == 0
		// Carrier-grade NAT (100.64/10).
		|| (a == 100 && (64..=127).contains(&b))
		// IETF protocol assignments (192.0.0/24) and 6to4 relay anycast (192.88.99/24).
		|| (a == 192 && b == 0 && c == 0)
		|| (a == 192 && b == 88 && c == 99)
		// Benchmarking (198.18/15).
		|| (a == 198 && (b == 18 || b == 19))
		// Reserved (240/4).
		|| a >= 240)
}

fn is_public_v6(ip: Ipv6Addr) -> bool {
	let s = ip.segments();
	// IPv4 carried in IPv6 is judged as the IPv4 it carries.
	if let Some(v4) = ip.to_ipv4_mapped() {
		return is_public_v4(v4);
	}
	// NAT64 (64:ff9b::/96): the last 32 bits are an IPv4 address.
	if s[0] == 0x64 && s[1] == 0xff9b && s[2..6] == [0, 0, 0, 0] {
		return is_public_v4(Ipv4Addr::new(
			(s[6] >> 8) as u8,
			s[6] as u8,
			(s[7] >> 8) as u8,
			s[7] as u8,
		));
	}
	// 6to4 (2002::/16): bits 16-47 are an IPv4 address.
	if s[0] == 0x2002 {
		return is_public_v4(Ipv4Addr::new(
			(s[1] >> 8) as u8,
			s[1] as u8,
			(s[2] >> 8) as u8,
			s[2] as u8,
		));
	}
	!(ip.is_unspecified()
		|| ip.is_loopback()
		|| ip.is_multicast()
		// Unique local (fc00::/7) and link-local (fe80::/10).
		|| (s[0] & 0xfe00) == 0xfc00
		|| (s[0] & 0xffc0) == 0xfe80
		// Site-local, deprecated but still routed somewhere (fec0::/10).
		|| (s[0] & 0xffc0) == 0xfec0
		// Documentation (2001:db8::/32) and Teredo (2001::/32, which tunnels to anywhere).
		|| (s[0] == 0x2001 && s[1] == 0x0db8)
		|| (s[0] == 0x2001 && s[1] == 0)
		// The old IPv4-compatible form (::a.b.c.d) and discard-only (100::/64).
		|| (s[0..6] == [0, 0, 0, 0, 0, 0])
		|| (s[0] == 0x100 && s[1..4] == [0, 0, 0]))
}

/// Resolves a host name to the addresses that pass [`is_public`], or refuses.
#[derive(Debug, Clone, Copy)]
pub struct SafeResolver {
	allow_private: bool,
}

/// What the resolver says when a name resolves only to addresses it will not send to; the proxy
/// looks for it to tell a refusal from a network failure.
pub const REFUSED: &str = "resolves only to addresses Psyche does not send to";

impl reqwest::dns::Resolve for SafeResolver {
	fn resolve(&self, name: reqwest::dns::Name) -> reqwest::dns::Resolving {
		let allow_private = self.allow_private;
		let host = name.as_str().to_string();
		Box::pin(async move {
			let found: Vec<SocketAddr> =
				tokio::net::lookup_host((host.as_str(), 0)).await?.collect();
			let kept: Vec<SocketAddr> = found
				.iter()
				.copied()
				.filter(|addr| allow_private || is_public(addr.ip()))
				.collect();
			if kept.is_empty() {
				let error: Box<dyn std::error::Error + Send + Sync> = if found.is_empty() {
					format!("{host} does not resolve").into()
				} else {
					format!("{host} {REFUSED}").into()
				};
				return Err(error);
			}
			let addrs: reqwest::dns::Addrs = Box::new(kept.into_iter());
			Ok(addrs)
		})
	}
}

/// Whether a provider's base URL may be used at all under `egress`. Checked when the catalogue is
/// loaded, so a bad one stops the server at start with its name, rather than failing every call.
pub fn check_base_url(url: &url::Url, egress: Egress) -> Result<(), String> {
	match egress {
		Egress::Private => match url.scheme() {
			"http" | "https" => Ok(()),
			other => Err(format!("{url} is {other}:, not http or https")),
		},
		Egress::Public => {
			if url.scheme() != "https" {
				return Err(format!("{url} is not https"));
			}
			match url.host() {
				Some(url::Host::Domain(_)) => Ok(()),
				_ => Err(format!(
					"{url} names an address, not a host; Psyche sends to host names only"
				)),
			}
		}
	}
}

/// The one client every provider call goes through. No overall timeout: a stream may run for
/// minutes, so the proxy times the first byte and the gaps between chunks itself.
pub fn client(egress: Egress) -> Result<reqwest::Client, String> {
	let private = egress == Egress::Private;
	reqwest::Client::builder()
		.dns_resolver(Arc::new(SafeResolver {
			allow_private: private,
		}))
		.redirect(reqwest::redirect::Policy::none())
		.no_proxy()
		.https_only(!private)
		.connect_timeout(CONNECT_TIMEOUT)
		.pool_idle_timeout(Duration::from_secs(90))
		.tcp_keepalive(Duration::from_secs(60))
		.user_agent(concat!("snout-psyche/", env!("CARGO_PKG_VERSION")))
		.build()
		.map_err(|e| format!("could not build an HTTP client: {e}"))
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn what_counts_as_public() {
		for public in [
			"17.188.143.34",
			"142.250.72.106",
			"2620:149:a44::1",
			"2a00:1450:4001::200e",
			"1.1.1.1",
		] {
			assert!(is_public(public.parse().unwrap()), "{public}");
		}
		for private in [
			"127.0.0.1",
			"10.92.0.5",
			"172.16.0.1",
			"192.168.1.1",
			"169.254.169.254",
			"100.64.0.1",
			"0.0.0.0",
			"0.1.2.3",
			"255.255.255.255",
			"224.0.0.1",
			"240.0.0.1",
			"192.0.2.1",
			"198.18.0.1",
			"192.0.0.8",
			"::1",
			"::",
			"fc00::1",
			"fd12:3456::1",
			"fe80::1",
			"fec0::1",
			"ff02::1",
			"2001:db8::1",
			"2001::1",
			"::ffff:127.0.0.1",
			"::ffff:169.254.169.254",
			"64:ff9b::a9fe:a9fe",
			"2002:a9fe:a9fe::1",
			"::127.0.0.1",
			"100::1",
		] {
			assert!(!is_public(private.parse().unwrap()), "{private}");
		}
		// Public IPv4 inside NAT64 and 6to4 is public.
		assert!(is_public("64:ff9b::808:808".parse().unwrap()));
		assert!(is_public("2002:808:808::1".parse().unwrap()));
	}

	#[test]
	fn base_urls_under_each_policy() {
		let url = |s: &str| url::Url::parse(s).unwrap();
		assert!(check_base_url(&url("https://api.openai.com/v1"), Egress::Public).is_ok());
		for refused in [
			"http://api.openai.com/v1",
			"https://127.0.0.1:11434",
			"https://[::1]/v1",
			"https://169.254.169.254/latest",
		] {
			assert!(
				check_base_url(&url(refused), Egress::Public).is_err(),
				"{refused}"
			);
		}
		assert!(check_base_url(&url("http://192.168.1.20:11434"), Egress::Private).is_ok());
		assert!(check_base_url(&url("ftp://example.com"), Egress::Private).is_err());
	}

	#[tokio::test]
	async fn the_public_client_refuses_a_name_that_resolves_here() {
		let client = client(Egress::Public).unwrap();
		let error = client
			.get("https://localhost/v1/models")
			.send()
			.await
			.unwrap_err();
		let chain = std::iter::successors(std::error::Error::source(&error), |s| s.source())
			.map(ToString::to_string)
			.collect::<Vec<_>>()
			.join(": ");
		assert!(chain.contains(REFUSED), "{chain}");
	}
}
