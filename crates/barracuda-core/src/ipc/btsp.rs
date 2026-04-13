// SPDX-License-Identifier: AGPL-3.0-or-later
//! BTSP Phase 2: connection authentication guard with full handshake relay.
//!
//! Per `BTSP_PROTOCOL_STANDARD.md` §Phase 2: when `FAMILY_ID` is set,
//! incoming connections must prove family membership via the security-domain
//! provider's handshake-as-a-service RPC before accessing JSON-RPC methods.
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
//! 9. Provider verifies and returns `HandshakeComplete` (session_id + cipher).
//! 10. barraCuda relays `HandshakeComplete` to client.
//!
//! ## Degraded Mode
//!
//! When `FAMILY_ID` is set but the security-domain provider is unreachable,
//! or the client does not send a `ClientHello` (legacy client), the guard
//! logs a warning and accepts in degraded mode. This prevents a hard
//! dependency on provider availability during the Phase 2 rollout.

use super::transport::{resolve_family_id, resolve_socket_dir};

/// Result of a BTSP handshake attempt on an incoming connection.
#[derive(Debug)]
pub enum BtspOutcome {
    /// No `FAMILY_ID` set — development mode, no handshake required.
    DevMode,
    /// `FAMILY_ID` set, full handshake succeeded.
    Authenticated {
        /// Security-provider-issued session identifier.
        session_id: String,
    },
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
            Self::DevMode | Self::Authenticated { .. } | Self::Degraded { .. }
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
        Ok(session_id) => {
            tracing::debug!(session_id, "BTSP handshake succeeded");
            BtspOutcome::Authenticated { session_id }
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
/// Per BTSP_PROTOCOL_STANDARD §Phase 2:
/// 1. Read ClientHello from client (with timeout for legacy fallback)
/// 2. Forward to provider via btsp.session.create
/// 3. Relay ServerHello to client
/// 4. Read ChallengeResponse from client
/// 5. Forward to provider via btsp.session.verify
/// 6. Relay HandshakeComplete to client
#[cfg(unix)]
async fn perform_handshake_relay<S>(
    stream: &mut S,
    provider_sock: &std::path::Path,
    family_id: &str,
) -> std::result::Result<String, HandshakeError>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    use tokio::io::BufReader;

    // Step 1: Read ClientHello from client (with timeout for legacy clients)
    let mut buf_reader = BufReader::new(&mut *stream);
    let client_hello_line =
        match tokio::time::timeout(CLIENT_HELLO_TIMEOUT, read_ndjson_line(&mut buf_reader)).await {
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
                reason: format!(
                    "First line is not valid JSON ({e}). Treating as legacy client."
                ),
                consumed_line: Some(client_hello_line),
            });
        }
    };

    let msg_type = client_hello
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if msg_type != "ClientHello" {
        return Err(HandshakeError::ClientLegacy {
            reason: format!(
                "Expected ClientHello, got message type '{msg_type}'. \
                 Treating as legacy client."
            ),
            consumed_line: Some(client_hello_line),
        });
    }

    let client_ephemeral_pub = client_hello
        .get("client_ephemeral_pub")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            HandshakeError::Protocol("ClientHello missing client_ephemeral_pub field".to_string())
        })?;

    // Step 2: Call security provider btsp.session.create
    let create_result = session_create_rpc(provider_sock, family_id, client_ephemeral_pub).await?;

    let session_id = create_result
        .get("session_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            HandshakeError::Protocol("session.create response missing session_id".to_string())
        })?
        .to_string();

    // Step 3: Relay ServerHello to client
    let server_hello = serde_json::json!({
        "type": "ServerHello",
        "version": 1,
        "server_ephemeral_pub": create_result.get("server_ephemeral_pub"),
        "challenge": create_result.get("challenge"),
    });
    write_ndjson_line(stream, &server_hello)
        .await
        .map_err(|e| HandshakeError::Protocol(format!("Failed to write ServerHello: {e}")))?;

    // Step 4: Read ChallengeResponse from client
    let mut buf_reader = BufReader::new(&mut *stream);
    let challenge_line = read_ndjson_line(&mut buf_reader)
        .await
        .map_err(|e| HandshakeError::Protocol(format!("Failed to read ChallengeResponse: {e}")))?;

    let challenge_resp: serde_json::Value = serde_json::from_str(&challenge_line)
        .map_err(|e| HandshakeError::Protocol(format!("Malformed ChallengeResponse JSON: {e}")))?;

    let hmac_proof = challenge_resp
        .get("hmac")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            HandshakeError::Protocol("ChallengeResponse missing hmac field".to_string())
        })?;

    // Step 5: Call security provider btsp.session.verify
    let verify_result = session_verify_rpc(provider_sock, &session_id, hmac_proof).await?;

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

    // Step 6: Relay HandshakeComplete to client
    let complete = serde_json::json!({
        "type": "HandshakeComplete",
        "session_id": session_id,
        "cipher": verify_result.get("cipher").cloned().unwrap_or_else(|| serde_json::json!("none")),
    });
    write_ndjson_line(stream, &complete)
        .await
        .map_err(|e| HandshakeError::Protocol(format!("Failed to write HandshakeComplete: {e}")))?;

    Ok(session_id)
}

#[cfg(not(unix))]
async fn perform_handshake_relay<S>(
    _stream: &mut S,
    _provider_sock: &std::path::Path,
    _family_id: &str,
) -> std::result::Result<String, HandshakeError>
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

// ── Security-domain provider RPC helpers ────────────────────────────

/// Call `btsp.session.create` on the security-domain provider.
#[cfg(unix)]
async fn session_create_rpc(
    provider_sock: &std::path::Path,
    family_id: &str,
    client_ephemeral_pub: &str,
) -> std::result::Result<serde_json::Value, HandshakeError> {
    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "btsp.session.create",
        "params": {
            "family_id": family_id,
            "client_ephemeral_pub": client_ephemeral_pub,
        },
        "id": 1
    });
    security_provider_rpc(provider_sock, &request).await
}

/// Call `btsp.session.verify` on the security-domain provider.
#[cfg(unix)]
async fn session_verify_rpc(
    provider_sock: &std::path::Path,
    session_id: &str,
    hmac: &str,
) -> std::result::Result<serde_json::Value, HandshakeError> {
    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "btsp.session.verify",
        "params": {
            "session_id": session_id,
            "hmac": hmac,
        },
        "id": 2
    });
    security_provider_rpc(provider_sock, &request).await
}

/// Send a JSON-RPC request to the security-domain provider and return the
/// `result` field. The provider is discovered at runtime by capability —
/// no primal names are hardcoded.
#[cfg(unix)]
async fn security_provider_rpc(
    provider_sock: &std::path::Path,
    request: &serde_json::Value,
) -> std::result::Result<serde_json::Value, HandshakeError> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    let stream = tokio::net::UnixStream::connect(provider_sock)
        .await
        .map_err(|e| {
            HandshakeError::ProviderUnavailable(format!(
                "Cannot connect to security-domain provider at {}: {e}. \
                 Accepting in degraded mode.",
                provider_sock.display()
            ))
        })?;
    let (reader, mut writer) = stream.into_split();

    let mut line = serde_json::to_string(request)
        .map_err(|e| HandshakeError::Protocol(format!("Failed to serialize provider RPC: {e}")))?;
    line.push('\n');
    writer.write_all(line.as_bytes()).await.map_err(|e| {
        HandshakeError::ProviderUnavailable(format!("Write to security provider failed: {e}"))
    })?;
    writer.shutdown().await.map_err(|e| {
        HandshakeError::ProviderUnavailable(format!("Provider write shutdown: {e}"))
    })?;

    let mut lines = BufReader::new(reader).lines();
    let response_line = lines
        .next_line()
        .await
        .map_err(|e| {
            HandshakeError::ProviderUnavailable(format!("Read from security provider failed: {e}"))
        })?
        .ok_or_else(|| {
            HandshakeError::ProviderUnavailable(
                "No response from security-domain provider".to_string(),
            )
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
mod tests {
    use super::*;

    #[test]
    fn btsp_outcome_dev_mode_accepts() {
        let outcome = BtspOutcome::DevMode;
        assert!(outcome.should_accept());
    }

    #[test]
    fn btsp_outcome_authenticated_accepts() {
        let outcome = BtspOutcome::Authenticated {
            session_id: "test-session".to_string(),
        };
        assert!(outcome.should_accept());
    }

    #[test]
    fn btsp_outcome_degraded_accepts() {
        let outcome = BtspOutcome::Degraded {
            reason: "security provider not found".to_string(),
            consumed_line: None,
        };
        assert!(outcome.should_accept());
        assert!(outcome.consumed_line().is_none());
    }

    #[test]
    fn btsp_outcome_degraded_with_consumed_line() {
        let outcome = BtspOutcome::Degraded {
            reason: "legacy client".to_string(),
            consumed_line: Some(r#"{"jsonrpc":"2.0","method":"tensor.dot","id":1}"#.to_string()),
        };
        assert!(outcome.should_accept());
        assert!(outcome.consumed_line().unwrap().contains("tensor.dot"));
    }

    #[test]
    fn btsp_outcome_rejected_refuses() {
        let outcome = BtspOutcome::Rejected {
            reason: "bad challenge response".to_string(),
        };
        assert!(!outcome.should_accept());
    }

    #[test]
    fn discover_security_provider_returns_none_when_no_socket() {
        assert!(discover_security_provider().is_none());
    }

    #[test]
    fn btsp_outcome_debug_impl() {
        let outcome = BtspOutcome::Authenticated {
            session_id: "abc".into(),
        };
        let debug = format!("{outcome:?}");
        assert!(debug.contains("Authenticated"));
        assert!(debug.contains("abc"));
    }
}
