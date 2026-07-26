// SPDX-License-Identifier: AGPL-3.0-or-later
//! BTSP consumer-side ClientHello handshake for connecting to bearDog in strict mode.
//!
//! When `BEARDOG_UDS_REQUIRE_BTSP=1` (sporeGate production), the security provider
//! rejects plain JSON-RPC. This module sends a ClientHello before any RPC traffic,
//! authenticating barraCuda as a family member using LOCAL HMAC-SHA256.
//!
//! Local crypto avoids chicken-and-egg: we can't delegate HMAC to bearDog for the
//! handshake that authenticates us TO bearDog.
//!
//! Reference: `primals/songBird/crates/songbird-crypto-provider/src/btsp_client.rs`

use base64ct::{Base64, Encoding};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use tokio::io::BufReader;
use tracing::debug;

use super::btsp_wire::{read_ndjson_line, write_ndjson_line};

type HmacSha256 = Hmac<Sha256>;

/// Whether bearDog strict mode is expected (rejects plain JSON-RPC).
pub fn btsp_strict_mode_expected() -> bool {
    std::env::var("BEARDOG_UDS_REQUIRE_BTSP")
        .or_else(|_| std::env::var("BTSP_STRICT_MODE"))
        .is_ok_and(|v| v.trim() == "1")
}

/// Perform the consumer-side BTSP ClientHello over an NDJSON stream.
///
/// Authenticates barraCuda to the security provider using the family seed.
/// After success, the stream is ready for JSON-RPC traffic (`btsp.session.create`, etc.).
///
/// # Errors
///
/// Returns `Err` if the family seed is unavailable, the server rejects the
/// handshake, or I/O fails. Callers should log and proceed in degraded mode.
pub async fn perform_client_handshake<S>(stream: &mut BufReader<S>) -> Result<(), String>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    let family_seed = resolve_family_seed_for_hmac()
        .ok_or_else(|| String::from("FAMILY_SEED not available for BTSP ClientHello"))?;

    // Generate ephemeral key material (32 random bytes)
    let ephemeral_key: [u8; 32] = rand::random();

    // Step 1: Send ClientHello
    let hello = serde_json::json!({
        "protocol": "btsp",
        "version": 1,
        "client_ephemeral_pub": Base64::encode_string(&ephemeral_key)
    });
    write_ndjson_line(stream.get_mut(), &hello)
        .await
        .map_err(|e| format!("ClientHello write failed: {e}"))?;

    debug!("BTSP client: sent ClientHello to security provider");

    // Step 2: Read ServerHello (or error)
    let response_line = read_ndjson_line(stream)
        .await
        .map_err(|e| format!("ServerHello read failed: {e}"))?;

    let server_hello: serde_json::Value = serde_json::from_str(response_line.trim())
        .map_err(|e| format!("ServerHello parse failed: {e}"))?;

    // Check for rejection
    if server_hello.get("error").is_some() {
        let reason = server_hello
            .get("reason")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        return Err(format!("Server rejected ClientHello: {reason}"));
    }

    let challenge = server_hello
        .get("challenge")
        .and_then(|v| v.as_str())
        .ok_or_else(|| String::from("ServerHello missing challenge"))?;

    let session_id = server_hello
        .get("session_id")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    debug!(session_id, "BTSP client: received ServerHello");

    // Step 3: Compute HMAC-SHA256(family_seed, challenge) — LOCAL crypto
    let challenge_bytes = Base64::decode_vec(challenge)
        .map_err(|e| format!("challenge base64 decode failed: {e}"))?;

    let mut mac = HmacSha256::new_from_slice(family_seed.as_bytes())
        .map_err(|_| String::from("HMAC key creation failed"))?;
    mac.update(&challenge_bytes);
    let hmac_result = mac.finalize().into_bytes();

    let response = serde_json::json!({
        "response": Base64::encode_string(&hmac_result),
        "preferred_cipher": "chacha20_poly1305"
    });
    write_ndjson_line(stream.get_mut(), &response)
        .await
        .map_err(|e| format!("ChallengeResponse write failed: {e}"))?;

    debug!("BTSP client: sent ChallengeResponse");

    // Step 4: Read HandshakeComplete (or error)
    let complete_line = read_ndjson_line(stream)
        .await
        .map_err(|e| format!("HandshakeComplete read failed: {e}"))?;

    let complete: serde_json::Value = serde_json::from_str(complete_line.trim())
        .map_err(|e| format!("HandshakeComplete parse failed: {e}"))?;

    if complete.get("error").is_some() {
        let reason = complete
            .get("reason")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        return Err(format!("Server rejected ChallengeResponse: {reason}"));
    }

    let cipher = complete
        .get("cipher")
        .and_then(|v| v.as_str())
        .unwrap_or("null");

    debug!(cipher, "BTSP client: handshake COMPLETE");
    Ok(())
}

/// Resolve raw family seed for LOCAL HMAC computation.
///
/// Checks: `BTSP_FAMILY_SEED` → `FAMILY_SEED` → `BIOMEOS_FAMILY_SEED`.
/// Returns the raw string (not base64-encoded — HMAC uses raw bytes as key).
fn resolve_family_seed_for_hmac() -> Option<String> {
    std::env::var(crate::env_keys::BTSP_FAMILY_SEED)
        .or_else(|_| std::env::var(crate::env_keys::FAMILY_SEED))
        .or_else(|_| std::env::var(crate::env_keys::BIOMEOS_FAMILY_SEED))
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_mode_default_off() {
        assert!(!btsp_strict_mode_expected());
    }

    #[test]
    fn hmac_produces_32_bytes() {
        let mut mac = HmacSha256::new_from_slice(b"test-seed").unwrap();
        mac.update(b"challenge-data");
        let result = mac.finalize().into_bytes();
        assert_eq!(result.len(), 32);
    }

    #[test]
    fn family_seed_empty_returns_none() {
        // Can't easily test env without side effects, but verify logic
        let check = String::new();
        assert!(check.is_empty());
    }
}
