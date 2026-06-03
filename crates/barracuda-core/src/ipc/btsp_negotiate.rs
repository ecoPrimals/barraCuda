// SPDX-License-Identifier: AGPL-3.0-or-later
//! BTSP Phase 3: cipher negotiation and HKDF key derivation.
//!
//! After the Phase 2 handshake succeeds (session_id + session_key established),
//! Phase 3 upgrades the connection to encrypted framing. The client sends
//! `btsp.negotiate` with its preferred cipher and a random nonce; the server
//! derives stream cipher keys via HKDF-SHA256 and returns its own nonce.

use super::btsp::{BtspCipher, BtspSession};

/// Error during `btsp.negotiate` Phase 3 cipher upgrade.
#[derive(Debug, thiserror::Error)]
pub enum NegotiateError {
    /// The `session_id` in the request does not match the connection's session.
    #[error("Invalid or unknown session_id")]
    InvalidSession,
    /// HKDF key derivation failed (structurally unreachable with valid inputs).
    #[error("HKDF key derivation failed: {0}")]
    KeyDerivation(String),
}

/// Outcome of a successful `btsp.negotiate` cipher upgrade.
#[derive(Debug)]
pub struct NegotiateResult {
    /// Upgraded session with derived key material for the negotiated cipher.
    pub session: BtspSession,
    /// JSON-RPC result value to return to the client.
    pub response: serde_json::Value,
}

/// Handle a `btsp.negotiate` Phase 3 cipher-upgrade request.
///
/// Validates the caller's `session_id` against the connection's authenticated
/// session, generates a 12-byte random server nonce, derives stream cipher
/// keys via HKDF-SHA256, and returns the negotiated cipher + nonce. The
/// caller is responsible for switching to encrypted framing after sending
/// the JSON-RPC response.
///
/// Returns `Ok(NegotiateResult)` on success (including graceful fallback to
/// NULL cipher when key material is unavailable). Returns `Err` only for
/// hard failures (mismatched session_id or key derivation failure).
pub fn negotiate_phase3(
    session: &BtspSession,
    params: &serde_json::Value,
) -> std::result::Result<NegotiateResult, NegotiateError> {
    let requested_id = params
        .get("session_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if requested_id.is_empty() || requested_id != session.session_id {
        return Err(NegotiateError::InvalidSession);
    }

    let preferred = params
        .get("preferred_cipher")
        .and_then(|v| v.as_str())
        .unwrap_or("chacha20-poly1305");

    let client_nonce_hex = params
        .get("client_nonce")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    if session.session_key.is_empty() {
        tracing::debug!("btsp.negotiate: no key material from Phase 1 — staying on NULL cipher");
        return Ok(NegotiateResult {
            session: session.clone(),
            response: serde_json::json!({ "cipher": "null" }),
        });
    }

    let cipher = BtspCipher::from_wire(preferred);
    if !cipher.requires_key() {
        return Ok(NegotiateResult {
            session: session.clone(),
            response: serde_json::json!({ "cipher": "null" }),
        });
    }

    let mut server_nonce = [0u8; 12];
    rand::Fill::fill(&mut server_nonce, &mut rand::rng());

    let client_nonce = hex_decode(client_nonce_hex);

    use hkdf::Hkdf;
    use sha2::Sha256;

    let mut salt = Vec::with_capacity(client_nonce.len() + server_nonce.len());
    salt.extend_from_slice(&client_nonce);
    salt.extend_from_slice(&server_nonce);

    let hkdf = Hkdf::<Sha256>::new(Some(&salt), &session.session_key);
    let mut derived_key = [0u8; 32];
    hkdf.expand(b"btsp-v1-phase3", &mut derived_key)
        .map_err(|e| NegotiateError::KeyDerivation(e.to_string()))?;

    let nonce_hex = hex_encode(&server_nonce);
    tracing::info!(
        cipher = cipher.wire_name(),
        session_id = %session.session_id,
        "btsp.negotiate: upgraded to encrypted framing"
    );

    Ok(NegotiateResult {
        session: BtspSession {
            session_id: session.session_id.clone(),
            cipher,
            session_key: derived_key.to_vec(),
        },
        response: serde_json::json!({
            "cipher": cipher.wire_name(),
            "server_nonce": nonce_hex,
        }),
    })
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(s, "{b:02x}");
    }
    s
}

fn hex_decode(hex: &str) -> Vec<u8> {
    let hex = hex.trim();
    if hex.is_empty() {
        return Vec::new();
    }
    (0..hex.len())
        .step_by(2)
        .filter_map(|i| {
            hex.get(i..i + 2)
                .and_then(|pair| u8::from_str_radix(pair, 16).ok())
        })
        .collect()
}
