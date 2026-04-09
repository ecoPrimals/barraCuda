// SPDX-License-Identifier: AGPL-3.0-or-later
//! BTSP Phase 2: connection authentication guard.
//!
//! Per `BTSP_PROTOCOL_STANDARD.md` §Phase 2: when `FAMILY_ID` is set,
//! incoming connections must prove family membership via BearDog's
//! handshake-as-a-service RPC before accessing JSON-RPC methods.
//!
//! ## Architecture
//!
//! Consumer primals (barraCuda) delegate handshake to BearDog:
//! 1. Client connects to barraCuda UDS/TCP listener.
//! 2. barraCuda calls BearDog `btsp.session.create` → gets challenge.
//! 3. barraCuda sends challenge to connecting client.
//! 4. Client responds with proof (X25519 + HMAC-SHA256).
//! 5. barraCuda calls BearDog `btsp.session.verify` → accept or reject.
//!
//! ## Degraded Mode
//!
//! When `FAMILY_ID` is set but BearDog is unreachable or its session
//! layer is incomplete (stub-style RPCs), the guard logs a warning
//! and accepts the connection. This prevents a hard dependency on
//! BearDog availability during the Phase 2 rollout.

use super::transport::{resolve_family_id, resolve_socket_dir};

/// Result of a BTSP handshake attempt on an incoming connection.
#[derive(Debug)]
pub enum BtspOutcome {
    /// No `FAMILY_ID` set — development mode, no handshake required.
    DevMode,
    /// `FAMILY_ID` set, handshake succeeded.
    Authenticated {
        /// BearDog-issued session identifier.
        session_id: String,
    },
    /// `FAMILY_ID` set, BearDog unreachable or session RPC incomplete.
    /// Connection accepted with warning — operators see actionable log.
    Degraded {
        /// Human-readable explanation for monitoring/alerting.
        reason: String,
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
}

/// Attempt BTSP handshake guard for an incoming connection.
///
/// Called once per accepted connection in the UDS/TCP accept loop,
/// before routing the stream to `handle_connection`. When `FAMILY_ID`
/// is unset, returns immediately (`DevMode`). When set, discovers
/// BearDog and attempts session creation.
///
/// Integration with the full BTSP handshake (X25519 challenge-response
/// over the client stream) will be added when BearDog completes its
/// `btsp.session.*` RPC layer.
pub async fn guard_connection() -> BtspOutcome {
    let Some(family_id) = resolve_family_id() else {
        return BtspOutcome::DevMode;
    };

    let Some(beardog_sock) = discover_beardog_socket() else {
        let socket_dir = resolve_socket_dir();
        let reason = format!(
            "FAMILY_ID={family_id} but BearDog not discoverable at {}. \
             BTSP handshake cannot be enforced — accepting in degraded mode. \
             Deploy BearDog to enable BTSP authentication.",
            socket_dir.display()
        );
        tracing::warn!("{reason}");
        return BtspOutcome::Degraded { reason };
    };

    match create_btsp_session(&beardog_sock, &family_id).await {
        Ok(session_id) => {
            tracing::debug!(session_id, "BTSP handshake succeeded");
            BtspOutcome::Authenticated { session_id }
        }
        Err(e) => {
            let reason = format!(
                "BTSP session creation failed (BearDog at {}): {e}. \
                 Accepting in degraded mode.",
                beardog_sock.display()
            );
            tracing::warn!("{reason}");
            BtspOutcome::Degraded { reason }
        }
    }
}

/// Domain stem for the security capability provider (BTSP handshake).
///
/// Per `PRIMAL_SELF_KNOWLEDGE_STANDARD.md` §2: primals discover peers
/// by capability domain, not by primal name. The "crypto" domain owns
/// encryption, signing, and BTSP handshake — whichever primal provides
/// that capability is the one we connect to.
const SECURITY_DOMAIN: &str = "crypto";

/// Discover the security-domain socket for BTSP handshake delegation.
///
/// Resolution chain (capability-based, not primal-name-based):
/// 1. `$BIOMEOS_SOCKET_DIR/{SECURITY_DOMAIN}-{family_id}.sock`
/// 2. `$BIOMEOS_SOCKET_DIR/{SECURITY_DOMAIN}.sock`
/// 3. Discovery files in `$BIOMEOS_SOCKET_DIR/*.json` advertising
///    `btsp.session.create` capability
///
/// Returns `None` if no security-domain provider is discoverable.
fn discover_beardog_socket() -> Option<std::path::PathBuf> {
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

    // Capability-based fallback: scan discovery files for btsp.session.create
    discover_by_capability(&sock_dir, "btsp.session.create")
}

/// Scan discovery files for a primal advertising a specific method.
///
/// Per wateringHole capability-based discovery: each running primal writes
/// a `{namespace}-core.json` discovery file listing its methods. We scan
/// for a primal that provides the requested method and return its Unix
/// socket path. No primal names are hardcoded.
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

/// Check a single discovery file for a primal advertising a given method.
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

/// Create a BTSP session via BearDog's `btsp.session.create` RPC.
///
/// Connects to BearDog over UDS, sends a `btsp.session.create` request,
/// and returns the session ID on success. This is the client-side
/// integration point for the full BTSP handshake flow.
///
/// # Phase 2 Evolution Path
///
/// Currently calls `btsp.session.create` only. The full flow will add:
/// 1. Parse BearDog's challenge from the `session.create` response
/// 2. Forward challenge to the connecting client over its stream
/// 3. Receive client's X25519 proof
/// 4. Call `btsp.session.verify` with the proof
/// 5. Return cipher parameters for encrypted framing
#[cfg(unix)]
async fn create_btsp_session(
    beardog_sock: &std::path::Path,
    family_id: &str,
) -> crate::error::Result<String> {
    use crate::error::BarracudaCoreError;
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    let stream = tokio::net::UnixStream::connect(beardog_sock)
        .await
        .map_err(|e| BarracudaCoreError::ipc(format!("connect to BearDog: {e}")))?;
    let (reader, mut writer) = stream.into_split();

    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "btsp.session.create",
        "params": { "family_id": family_id },
        "id": 1
    });

    let mut line = serde_json::to_string(&request)?;
    line.push('\n');
    writer.write_all(line.as_bytes()).await?;
    writer.shutdown().await?;

    let mut lines = BufReader::new(reader).lines();
    let response_line = lines
        .next_line()
        .await?
        .ok_or_else(|| BarracudaCoreError::ipc("no response from BearDog"))?;

    let response: serde_json::Value = serde_json::from_str(&response_line)?;

    if let Some(error) = response.get("error") {
        return Err(BarracudaCoreError::ipc(format!(
            "btsp.session.create: {error}"
        )));
    }

    response
        .get("result")
        .and_then(|r| r.get("session_id"))
        .and_then(|s| s.as_str())
        .map(str::to_string)
        .ok_or_else(|| {
            BarracudaCoreError::ipc("missing session_id in btsp.session.create response")
        })
}

#[cfg(not(unix))]
async fn create_btsp_session(
    _beardog_sock: &std::path::Path,
    _family_id: &str,
) -> crate::error::Result<String> {
    Err(crate::error::BarracudaCoreError::ipc(
        "BTSP handshake requires Unix domain sockets",
    ))
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
            reason: "BearDog not found".to_string(),
        };
        assert!(outcome.should_accept());
    }

    #[test]
    fn btsp_outcome_rejected_refuses() {
        let outcome = BtspOutcome::Rejected {
            reason: "bad challenge response".to_string(),
        };
        assert!(!outcome.should_accept());
    }

    #[test]
    fn discover_beardog_returns_none_when_no_socket() {
        // With no BearDog running, discovery should return None
        assert!(discover_beardog_socket().is_none());
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
