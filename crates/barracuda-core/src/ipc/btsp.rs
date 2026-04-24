// SPDX-License-Identifier: AGPL-3.0-or-later
//! BTSP Phase 2+3: connection authentication guard with handshake relay and
//! post-handshake stream encryption.
//!
//! Per `BTSP_PROTOCOL_STANDARD.md`: when `FAMILY_ID` is set, incoming
//! connections prove family membership via the security-domain provider's
//! handshake-as-a-service RPC. Phase 3 adds real cipher negotiation and
//! encrypted length-prefixed framing after the handshake succeeds.
//!
//! ## Architecture
//!
//! Consumer primals (barraCuda) relay the handshake to whichever primal
//! provides the `crypto` security domain (discovered at runtime via
//! capability-based socket resolution — zero hardcoded primal names):
//!
//! 1. Client connects to barraCuda UDS/TCP listener.
//! 2. barraCuda reads `ClientHello` (with X25519 ephemeral pub) from client.
//! 3. barraCuda calls security provider `btsp.session.create` with
//!    `client_ephemeral_pub`.
//! 4. Provider returns `ServerHello` (server ephemeral pub + HMAC challenge).
//! 5. barraCuda relays `ServerHello` to client.
//! 6. Client computes `X25519(client_priv, server_pub)` → shared secret,
//!    then `HMAC-SHA256(shared_secret, challenge || "btsp-v1")`.
//! 7. Client sends `ChallengeResponse` (HMAC proof) to barraCuda.
//! 8. barraCuda calls security provider `btsp.session.verify` with the proof.
//! 9. Provider verifies → returns session_id, cipher, session_key.
//! 10. barraCuda relays `HandshakeComplete` to client.
//! 11. **Phase 3**: both sides switch to length-prefixed encrypted frames
//!     using the negotiated cipher and session key.
//!
//! ## Degraded Mode
//!
//! When `FAMILY_ID` is set but the security-domain provider is unreachable,
//! or the client does not send a `ClientHello` (legacy client), the guard
//! logs a warning and accepts in degraded mode. This prevents a hard
//! dependency on provider availability during the Phase 2 rollout.

use super::transport::{resolve_family_id, resolve_socket_dir};

/// Negotiated cipher suite per `BTSP_PROTOCOL_STANDARD.md` §Cipher Suites.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BtspCipher {
    /// `BTSP_NULL` — raw plaintext frames. Authentication only (handshake).
    /// Only valid for Covalent bonds.
    Null,
    /// `BTSP_HMAC_PLAIN` — HMAC-SHA256 tag appended per frame. Integrity
    /// and authentication without confidentiality.
    HmacPlain,
    /// `BTSP_CHACHA20_POLY1305` — full AEAD encryption. Default for all bonds.
    ChaCha20Poly1305,
}

impl BtspCipher {
    /// Parse the cipher name from the provider's handshake response.
    fn from_wire(s: &str) -> Self {
        match s {
            "chacha20_poly1305" | "chacha20" | "BTSP_CHACHA20_POLY1305" => Self::ChaCha20Poly1305,
            "hmac_plain" | "hmac" | "BTSP_HMAC_PLAIN" => Self::HmacPlain,
            _ => Self::Null,
        }
    }

    /// Whether this cipher requires the session key for frame I/O.
    pub fn requires_key(self) -> bool {
        !matches!(self, Self::Null)
    }
}

/// Session state after a successful BTSP handshake.
#[derive(Debug, Clone)]
pub struct BtspSession {
    /// Security-provider-issued session identifier.
    pub session_id: String,
    /// Negotiated cipher suite for post-handshake frames.
    pub cipher: BtspCipher,
    /// Session key material (base64-decoded from provider response).
    /// Empty when cipher is `Null`.
    pub session_key: Vec<u8>,
}

/// Result of a BTSP handshake attempt on an incoming connection.
#[derive(Debug)]
pub enum BtspOutcome {
    /// No `FAMILY_ID` set — development mode, no handshake required.
    DevMode,
    /// `FAMILY_ID` set, full handshake succeeded. Contains the negotiated
    /// cipher and session key for Phase 3 encrypted framing.
    Authenticated(BtspSession),
    /// `FAMILY_ID` set, security provider unreachable or handshake incomplete.
    /// Connection accepted with warning — operators see actionable log.
    Degraded {
        /// Human-readable explanation for monitoring/alerting.
        reason: String,
        /// First NDJSON line consumed during the handshake attempt.
        /// Non-None when a legacy (non-BTSP) client's request was read by
        /// the guard before determining it wasn't a `ClientHello`. Must be
        /// replayed to the JSON-RPC handler so the request isn't silently
        /// dropped (LD-10 fix).
        consumed_line: Option<String>,
    },
    /// `FAMILY_ID` set, handshake explicitly failed — connection refused.
    Rejected {
        /// Why the handshake was rejected.
        reason: String,
    },
}

impl BtspOutcome {
    /// Whether the incoming connection should be accepted.
    pub fn should_accept(&self) -> bool {
        matches!(
            self,
            Self::DevMode | Self::Authenticated(_) | Self::Degraded { .. }
        )
    }

    /// Return any NDJSON line consumed during the handshake that must be
    /// replayed to the JSON-RPC handler.
    pub fn consumed_line(&self) -> Option<&str> {
        match self {
            Self::Degraded {
                consumed_line: Some(line),
                ..
            } => Some(line),
            _ => None,
        }
    }

    /// Extract the session if the handshake was authenticated.
    pub fn session(&self) -> Option<&BtspSession> {
        match self {
            Self::Authenticated(session) => Some(session),
            _ => None,
        }
    }
}

/// Detect whether a parsed JSON message is a BTSP `ClientHello`.
///
/// Accepts both the legacy `{"type":"ClientHello",...}` format and
/// primalSpring's JSON-line format `{"protocol":"btsp","version":1,...}`.
fn is_btsp_client_hello(msg: &serde_json::Value) -> bool {
    let is_type_hello = msg.get("type").and_then(|v| v.as_str()) == Some("ClientHello");
    let is_protocol_btsp = msg.get("protocol").and_then(|v| v.as_str()) == Some("btsp");
    is_type_hello || is_protocol_btsp
}

/// Attempt full BTSP Phase 2 handshake on an incoming connection stream.
///
/// Called once per accepted connection in the UDS/TCP accept loop,
/// before routing the stream to `handle_connection`. When `FAMILY_ID`
/// is unset, returns immediately (`DevMode`). When set, discovers the
/// security-domain provider and orchestrates the full X25519+HMAC
/// challenge-response handshake as a relay between client and provider.
pub async fn guard_connection<S>(stream: &mut S) -> BtspOutcome
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    let Some(family_id) = resolve_family_id() else {
        return BtspOutcome::DevMode;
    };

    let Some(provider_sock) = discover_security_provider() else {
        let socket_dir = resolve_socket_dir();
        let reason = format!(
            "FAMILY_ID={family_id} but security-domain provider not discoverable \
             at {}. Accepting in degraded mode.",
            socket_dir.display()
        );
        tracing::warn!("{reason}");
        return BtspOutcome::Degraded {
            reason,
            consumed_line: None,
        };
    };

    match perform_handshake_relay(stream, &provider_sock, &family_id).await {
        Ok(session) => {
            tracing::debug!(
                session_id = session.session_id,
                cipher = ?session.cipher,
                "BTSP handshake succeeded"
            );
            BtspOutcome::Authenticated(session)
        }
        Err(HandshakeError::ClientLegacy {
            reason,
            consumed_line,
        }) => {
            tracing::warn!("{reason}");
            BtspOutcome::Degraded {
                reason,
                consumed_line,
            }
        }
        Err(HandshakeError::ProviderUnavailable(reason)) => {
            tracing::warn!("{reason}");
            BtspOutcome::Degraded {
                reason,
                consumed_line: None,
            }
        }
        Err(HandshakeError::Rejected(reason)) => {
            tracing::warn!("BTSP handshake rejected: {reason}");
            BtspOutcome::Rejected { reason }
        }
        Err(HandshakeError::Protocol(reason)) => {
            tracing::warn!("BTSP protocol error: {reason}");
            BtspOutcome::Degraded {
                reason,
                consumed_line: None,
            }
        }
    }
}

/// Internal error type for the handshake relay — not exposed to callers.
enum HandshakeError {
    /// Client didn't send ClientHello (legacy/plain JSON-RPC client).
    ClientLegacy {
        reason: String,
        /// The first line read from the stream, if any. Must be replayed
        /// to the JSON-RPC handler to avoid silently dropping the request.
        consumed_line: Option<String>,
    },
    /// Security-domain provider unreachable or RPC failed.
    ProviderUnavailable(String),
    /// Provider explicitly rejected the handshake (bad HMAC, etc.).
    Rejected(String),
    /// Wire protocol error (malformed JSON, unexpected message type).
    Protocol(String),
}

/// Time limit for the client to send ClientHello before we fall back
/// to treating the connection as a legacy (non-BTSP) client.
const CLIENT_HELLO_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

/// Full handshake relay between client stream and security-domain provider.
///
/// Per `BTSP_PROTOCOL_STANDARD.md` §Phase 2+3:
/// 1. Read ClientHello from client (with timeout for legacy fallback)
/// 2. Forward to provider via `btsp.session.create`
/// 3. Relay ServerHello to client
/// 4. Read ChallengeResponse from client
/// 5. Forward to provider via `btsp.session.verify`
/// 6. Relay `HandshakeComplete` to client
/// 7. Return `BtspSession` with cipher + session_key for Phase 3 framing
#[cfg(unix)]
async fn perform_handshake_relay<S>(
    stream: &mut S,
    provider_sock: &std::path::Path,
    family_id: &str,
) -> std::result::Result<BtspSession, HandshakeError>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    use tokio::io::BufReader;

    // Single BufReader for the entire handshake — avoids the edge-case where
    // a second BufReader silently discards data buffered by the first.
    // Writes go through `get_mut()` to access the underlying stream.
    let mut buf = BufReader::new(&mut *stream);

    // Step 1: Read ClientHello from client (with timeout for legacy clients)
    let client_hello_line =
        match tokio::time::timeout(CLIENT_HELLO_TIMEOUT, read_ndjson_line(&mut buf)).await {
            Ok(Ok(line)) => line,
            Ok(Err(e)) => {
                return Err(HandshakeError::ClientLegacy {
                    reason: format!(
                        "Client stream error before ClientHello: {e}. Treating as legacy client."
                    ),
                    consumed_line: None,
                });
            }
            Err(_) => {
                return Err(HandshakeError::ClientLegacy {
                    reason: "Client did not send ClientHello within timeout. \
                             Treating as legacy (non-BTSP) client."
                        .to_string(),
                    consumed_line: None,
                });
            }
        };

    let client_hello: serde_json::Value = match serde_json::from_str(&client_hello_line) {
        Ok(v) => v,
        Err(e) => {
            return Err(HandshakeError::ClientLegacy {
                reason: format!("First line is not valid JSON ({e}). Treating as legacy client."),
                consumed_line: Some(client_hello_line),
            });
        }
    };

    if !is_btsp_client_hello(&client_hello) {
        return Err(HandshakeError::ClientLegacy {
            reason: "First message is not a BTSP ClientHello. \
                     Treating as legacy client."
                .to_string(),
            consumed_line: Some(client_hello_line),
        });
    }

    let client_ephemeral_pub = client_hello
        .get("client_ephemeral_pub")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            HandshakeError::Protocol("ClientHello missing client_ephemeral_pub field".to_string())
        })?;

    // Step 2: Open ONE connection to BearDog for the entire handshake.
    // Per SOURDOUGH_BTSP_RELAY_PATTERN: both create and verify must use the
    // same socket connection — BearDog associates session state with it.
    let family_seed = resolve_family_seed_raw().ok_or_else(|| {
        HandshakeError::Protocol(
            "BTSP handshake requires BEARDOG_FAMILY_SEED or FAMILY_SEED env var".to_string(),
        )
    })?;

    let beardog_stream = tokio::net::UnixStream::connect(provider_sock)
        .await
        .map_err(|e| {
            HandshakeError::ProviderUnavailable(format!(
                "Cannot connect to security-domain provider at {}: {e}. \
                 Accepting in degraded mode.",
                provider_sock.display()
            ))
        })?;
    let mut beardog = tokio::io::BufReader::new(beardog_stream);

    // Step 2a: btsp.session.create on BearDog connection
    let create_request = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "btsp.session.create",
        "params": {
            "family_id": family_id,
            "client_ephemeral_pub": client_ephemeral_pub,
            "family_seed": family_seed,
        },
        "id": 1
    });
    let create_result = beardog_rpc(&mut beardog, &create_request).await?;

    let session_id = create_result
        .get("session_token")
        .or_else(|| create_result.get("session_id"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            HandshakeError::Protocol(
                "session.create response missing session_token/session_id".to_string(),
            )
        })?
        .to_string();

    // Step 3: Relay ServerHello to client (write through BufReader::get_mut)
    let server_hello = serde_json::json!({
        "type": "ServerHello",
        "version": 1,
        "server_ephemeral_pub": create_result.get("server_ephemeral_pub"),
        "challenge": create_result.get("challenge"),
    });
    write_ndjson_line(buf.get_mut(), &server_hello)
        .await
        .map_err(|e| HandshakeError::Protocol(format!("Failed to write ServerHello: {e}")))?;

    // Step 4: Read ChallengeResponse from client (same BufReader preserves buffered data)
    let challenge_line = read_ndjson_line(&mut buf)
        .await
        .map_err(|e| HandshakeError::Protocol(format!("Failed to read ChallengeResponse: {e}")))?;

    let challenge_resp: serde_json::Value = serde_json::from_str(&challenge_line)
        .map_err(|e| HandshakeError::Protocol(format!("Malformed ChallengeResponse JSON: {e}")))?;

    let client_response = challenge_resp
        .get("response")
        .or_else(|| challenge_resp.get("hmac"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            HandshakeError::Protocol("ChallengeResponse missing response/hmac field".to_string())
        })?;

    let preferred_cipher = challenge_resp
        .get("preferred_cipher")
        .and_then(|v| v.as_str())
        .unwrap_or("chacha20_poly1305");

    // Step 5: btsp.session.verify on SAME BearDog connection
    let verify_request = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "btsp.session.verify",
        "params": {
            "session_token": session_id,
            "response": client_response,
            "client_ephemeral_pub": client_ephemeral_pub,
            "preferred_cipher": preferred_cipher,
        },
        "id": 2
    });
    let verify_result = beardog_rpc(&mut beardog, &verify_request).await?;

    let verified = verify_result
        .get("verified")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if !verified {
        let reason = verify_result
            .get("reason")
            .and_then(|v| v.as_str())
            .unwrap_or("HMAC verification failed");
        return Err(HandshakeError::Rejected(reason.to_string()));
    }

    // Step 6: Extract cipher and session key from provider response
    let cipher_name = verify_result
        .get("cipher")
        .and_then(|v| v.as_str())
        .unwrap_or("none");
    let cipher = BtspCipher::from_wire(cipher_name);

    let session_key = if cipher.requires_key() {
        let key_b64 = verify_result
            .get("session_key")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                HandshakeError::Protocol(format!(
                    "Provider returned cipher '{cipher_name}' but no session_key"
                ))
            })?;
        use base64ct::{Base64, Encoding};
        Base64::decode_vec(key_b64)
            .map_err(|e| HandshakeError::Protocol(format!("Invalid base64 session_key: {e}")))?
    } else {
        Vec::new()
    };

    // Step 7: Relay HandshakeComplete to client
    let complete = serde_json::json!({
        "type": "HandshakeComplete",
        "session_id": session_id,
        "cipher": cipher_name,
    });
    write_ndjson_line(buf.get_mut(), &complete)
        .await
        .map_err(|e| HandshakeError::Protocol(format!("Failed to write HandshakeComplete: {e}")))?;

    Ok(BtspSession {
        session_id,
        cipher,
        session_key,
    })
}

#[cfg(not(unix))]
async fn perform_handshake_relay<S>(
    _stream: &mut S,
    _provider_sock: &std::path::Path,
    _family_id: &str,
) -> std::result::Result<BtspSession, HandshakeError>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    Err(HandshakeError::Protocol(
        "BTSP handshake requires Unix domain sockets".to_string(),
    ))
}

// ── NDJSON wire helpers ─────────────────────────────────────────────

async fn read_ndjson_line<R>(reader: &mut R) -> std::io::Result<String>
where
    R: tokio::io::AsyncBufRead + Unpin,
{
    use tokio::io::AsyncBufReadExt;
    let mut line = String::new();
    reader.read_line(&mut line).await?;
    if line.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "connection closed before NDJSON line",
        ));
    }
    Ok(line)
}

async fn write_ndjson_line<W>(writer: &mut W, value: &serde_json::Value) -> std::io::Result<()>
where
    W: tokio::io::AsyncWrite + Unpin,
{
    use tokio::io::AsyncWriteExt;
    let mut line = serde_json::to_string(value)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    line.push('\n');
    writer.write_all(line.as_bytes()).await?;
    writer.flush().await?;
    Ok(())
}

// ── BearDog RPC helper ──────────────────────────────────────────────

/// Send a JSON-RPC request to BearDog over an existing connection and return
/// the `result` field. Uses NDJSON framing (write line + flush, read line).
///
/// Per `SOURDOUGH_BTSP_RELAY_PATTERN.md`: both `session.create` and
/// `session.verify` MUST use the same connection — BearDog associates
/// session state with it.
#[cfg(unix)]
async fn beardog_rpc(
    stream: &mut tokio::io::BufReader<tokio::net::UnixStream>,
    request: &serde_json::Value,
) -> std::result::Result<serde_json::Value, HandshakeError> {
    write_ndjson_line(stream.get_mut(), request)
        .await
        .map_err(|e| {
            HandshakeError::ProviderUnavailable(format!("Write to security provider failed: {e}"))
        })?;

    let response_line = read_ndjson_line(stream).await.map_err(|e| {
        HandshakeError::ProviderUnavailable(format!("Read from security provider failed: {e}"))
    })?;

    let response: serde_json::Value = serde_json::from_str(&response_line)
        .map_err(|e| HandshakeError::Protocol(format!("Malformed provider response: {e}")))?;

    if let Some(error) = response.get("error") {
        return Err(HandshakeError::ProviderUnavailable(format!(
            "Security provider RPC error: {error}"
        )));
    }

    response
        .get("result")
        .cloned()
        .ok_or_else(|| HandshakeError::Protocol("Provider response missing result".to_string()))
}

// ── Family seed resolution ───────────────────────────────────────────

/// Load the BTSP family seed as a raw string for BearDog.
///
/// Reads `FAMILY_SEED` → `BEARDOG_FAMILY_SEED` → `BIOMEOS_FAMILY_SEED`
/// from environment. Per `SOURDOUGH_BTSP_RELAY_PATTERN.md`: pass the env
/// value as-is (trimmed). Do NOT hex-decode or base64-encode — BearDog
/// handles its own key material decoding.
fn resolve_family_seed_raw() -> Option<String> {
    let raw = std::env::var("FAMILY_SEED")
        .or_else(|_| std::env::var("BEARDOG_FAMILY_SEED"))
        .or_else(|_| std::env::var("BIOMEOS_FAMILY_SEED"))
        .ok()
        .filter(|s| !s.is_empty())?;
    Some(raw.trim().to_string())
}

/// Decode a hex string to bytes. Returns `None` if the input is not valid hex.
#[cfg(test)]
fn hex_to_bytes(hex: &str) -> Option<Vec<u8>> {
    let hex = hex.trim();
    if !hex.len().is_multiple_of(2) {
        return None;
    }
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).ok())
        .collect()
}

// ── Discovery ───────────────────────────────────────────────────────

/// Domain stem for the security capability provider (BTSP handshake).
const SECURITY_DOMAIN: &str = "crypto";

/// Discover the security-domain socket for BTSP handshake delegation.
///
/// Resolution chain (capability-based, not primal-name-based):
/// 1. `$BIOMEOS_SOCKET_DIR/{SECURITY_DOMAIN}-{family_id}.sock`
/// 2. `$BIOMEOS_SOCKET_DIR/{SECURITY_DOMAIN}.sock`
/// 3. Discovery files in `$BIOMEOS_SOCKET_DIR/*.json` advertising
///    `btsp.session.create` capability
fn discover_security_provider() -> Option<std::path::PathBuf> {
    let sock_dir = resolve_socket_dir();

    if let Some(fid) = resolve_family_id() {
        let scoped = sock_dir.join(format!("{SECURITY_DOMAIN}-{fid}.sock"));
        if scoped.exists() {
            return Some(scoped);
        }
    }
    let unscoped = sock_dir.join(format!("{SECURITY_DOMAIN}.sock"));
    if unscoped.exists() {
        return Some(unscoped);
    }

    discover_by_capability(&sock_dir, "btsp.session.create")
}

fn discover_by_capability(sock_dir: &std::path::Path, method: &str) -> Option<std::path::PathBuf> {
    let entries = std::fs::read_dir(sock_dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "json") {
            if let Some(sock) = check_discovery_file_for_method(&path, method) {
                return Some(sock);
            }
        }
    }
    None
}

fn check_discovery_file_for_method(
    path: &std::path::Path,
    method: &str,
) -> Option<std::path::PathBuf> {
    let content = std::fs::read_to_string(path).ok()?;
    let info: serde_json::Value = serde_json::from_str(&content).ok()?;
    let methods = info.get("methods")?.as_array()?;
    let has_method = methods
        .iter()
        .any(|m| m.as_str().is_some_and(|s| s == method));
    if !has_method {
        return None;
    }
    let unix_addr = info
        .get("transports")
        .and_then(|t| t.get("unix"))
        .and_then(|v| v.as_str())
        .and_then(|s| s.strip_prefix("unix://"))?;
    let sock = std::path::PathBuf::from(unix_addr);
    sock.exists().then_some(sock)
}

#[cfg(test)]
#[path = "btsp_tests.rs"]
mod tests;
