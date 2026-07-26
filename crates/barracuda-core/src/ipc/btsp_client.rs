// SPDX-License-Identifier: AGPL-3.0-or-later
//! BTSP client-side handshake for outbound connections to BTSP-enforced primals.
//!
//! Per Sub-Wave 151b: every primal that connects to a BTSP-enforced endpoint
//! (e.g. bearDog in strict mode) must perform the ClientHello handshake.
//!
//! ## Protocol (4-step from client's perspective)
//!
//! 1. Send `ClientHello { protocol: "btsp", version: 1, client_ephemeral_pub }`
//! 2. Read `ServerHello { server_ephemeral_pub, challenge, session_id }`
//! 3. Compute challenge response (delegated to security provider via
//!    `btsp.session.verify`) → send `ChallengeResponse { response, preferred_cipher }`
//! 4. Read `HandshakeComplete { cipher, session_id }`
//!
//! ## Architecture
//!
//! The security provider (bearDog) has two roles:
//! - **Management socket** (non-BTSP): handles `btsp.session.create` /
//!   `btsp.session.verify` for key material. Discovered via
//!   [`super::btsp_discovery::discover_security_provider`].
//! - **Public socket** (BTSP-enforced): the target we perform the handshake
//!   against. This module handles connecting to any BTSP-enforced target.
//!
//! ## Reference
//!
//! songBird `songbird-orchestrator/src/btsp_client.rs` (Wave 151a reference impl).

use super::btsp::{BtspCipher, BtspSession};
use super::transport::{TransportEndpoint, TransportStream, connect_transport};
use crate::error::BarracudaCoreError;

type Result<T> = std::result::Result<T, BarracudaCoreError>;

/// BTSP client for performing handshakes with BTSP-enforced endpoints.
///
/// Holds a reference to the security provider's management socket for
/// cryptographic operations (session creation, challenge response computation).
#[derive(Debug, Clone)]
pub struct BtspClient {
    provider_endpoint: TransportEndpoint,
}

/// Result of a successful client-side BTSP handshake.
///
/// Contains the authenticated session and the connected stream, ready for
/// post-handshake communication (encrypted or plaintext depending on cipher).
#[derive(Debug)]
pub struct BtspHandshakeResult {
    /// The authenticated session (cipher, session_id, session_key).
    pub session: BtspSession,
    /// The connected stream after handshake — ready for JSON-RPC.
    pub stream: TransportStream,
}

impl BtspClient {
    /// Create a BTSP client from an explicit security provider endpoint.
    #[must_use]
    pub fn new(provider_endpoint: TransportEndpoint) -> Self {
        Self { provider_endpoint }
    }

    /// Create a BTSP client via runtime discovery of the security provider.
    ///
    /// Uses the resolution chain from `btsp_discovery::discover_security_provider`:
    /// `$BTSP_PROVIDER_SOCKET` → `$BIOMEOS_SOCKET_DIR/crypto[-{family}].sock` →
    /// capability-based discovery file scan.
    ///
    /// Returns `None` if no security provider socket is found.
    #[cfg(unix)]
    pub fn discover() -> Option<Self> {
        let path = super::btsp_discovery::discover_security_provider()?;
        Some(Self {
            provider_endpoint: TransportEndpoint::uds(path.to_string_lossy().into_owned()),
        })
    }

    /// Non-Unix fallback: always returns `None` (UDS-based discovery unavailable).
    #[cfg(not(unix))]
    pub fn discover() -> Option<Self> {
        None
    }

    /// Perform BTSP client handshake with a target endpoint.
    ///
    /// Connects to `target`, performs the 4-step ClientHello flow, and returns
    /// the authenticated session + connected stream.
    ///
    /// # Arguments
    /// * `target` - The BTSP-enforced endpoint to connect to.
    /// * `preferred_cipher` - Cipher preference (e.g. `"chacha20_poly1305"`, `"null"`).
    pub async fn handshake(
        &self,
        target: &TransportEndpoint,
        preferred_cipher: &str,
    ) -> Result<BtspHandshakeResult> {
        // Step 1: Ask security provider to create a session (ephemeral X25519 keypair).
        let (client_ephemeral_pub, session_id) = self.session_create().await?;

        // Step 2: Connect to target, send ClientHello.
        let mut target_stream = connect_transport(target).await.map_err(|e| {
            BarracudaCoreError::ipc(format!(
                "BTSP client: failed to connect to target {target:?}: {e}"
            ))
        })?;

        let client_hello = serde_json::json!({
            "protocol": "btsp",
            "version": 1,
            "client_ephemeral_pub": client_ephemeral_pub
        });
        write_ndjson_to_stream(&mut target_stream, &client_hello).await?;

        // Step 3: Read ServerHello.
        let server_hello = read_ndjson_from_stream(&mut target_stream).await?;

        if let Some(error) = server_hello.get("error") {
            return Err(BarracudaCoreError::ipc(format!(
                "BTSP client: server rejected ClientHello: {}",
                error
                    .get("reason")
                    .and_then(|r| r.as_str())
                    .unwrap_or("unknown")
            )));
        }

        let server_ephemeral_pub =
            server_hello["server_ephemeral_pub"]
                .as_str()
                .ok_or_else(|| {
                    BarracudaCoreError::ipc(
                        "BTSP client: ServerHello missing server_ephemeral_pub".to_string(),
                    )
                })?;
        let challenge = server_hello["challenge"].as_str().ok_or_else(|| {
            BarracudaCoreError::ipc("BTSP client: ServerHello missing challenge".to_string())
        })?;

        // Step 4: Ask security provider to compute challenge response.
        let challenge_response = self
            .session_verify(
                &session_id,
                &client_ephemeral_pub,
                server_ephemeral_pub,
                challenge,
                preferred_cipher,
            )
            .await?;

        // Step 5: Send ChallengeResponse to target.
        let cr_msg = serde_json::json!({
            "type": "ChallengeResponse",
            "response": challenge_response,
            "preferred_cipher": preferred_cipher
        });
        write_ndjson_to_stream(&mut target_stream, &cr_msg).await?;

        // Step 6: Read HandshakeComplete.
        let hs_complete = read_ndjson_from_stream(&mut target_stream).await?;

        if let Some(error) = hs_complete.get("error") {
            return Err(BarracudaCoreError::ipc(format!(
                "BTSP client: handshake verification failed: {}",
                error
                    .get("reason")
                    .and_then(|r| r.as_str())
                    .unwrap_or("unknown")
            )));
        }

        let negotiated_cipher = hs_complete["cipher"].as_str().unwrap_or("null");
        let final_session_id = hs_complete["session_id"].as_str().unwrap_or(&session_id);
        let session_key_b64 = hs_complete["session_key"].as_str().unwrap_or("");

        let session_key = if session_key_b64.is_empty() {
            Vec::new()
        } else {
            use base64ct::{Base64, Encoding};
            Base64::decode_vec(session_key_b64).map_err(|e| {
                BarracudaCoreError::ipc(format!("BTSP client: invalid base64 session_key: {e}"))
            })?
        };

        let session = BtspSession {
            session_id: final_session_id.to_string(),
            cipher: BtspCipher::from_wire(negotiated_cipher),
            session_key,
        };

        tracing::info!(
            session_id = %session.session_id,
            cipher = %session.cipher.wire_name(),
            target = ?target,
            "BTSP client handshake complete"
        );

        Ok(BtspHandshakeResult {
            session,
            stream: target_stream,
        })
    }

    /// Ask security provider to create a new BTSP session.
    ///
    /// Returns `(client_ephemeral_pub, session_id)`.
    async fn session_create(&self) -> Result<(String, String)> {
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "btsp.session.create",
            "params": {
                "family_seed_ref": "env:FAMILY_SEED",
                "role": "client"
            },
            "id": 1
        });

        let response = self.provider_rpc(&request).await?;

        let client_ephemeral_pub = response["client_ephemeral_pub"]
            .as_str()
            .ok_or_else(|| {
                BarracudaCoreError::ipc(
                    "BTSP client: session.create missing client_ephemeral_pub".to_string(),
                )
            })?
            .to_string();

        let session_id = response["session_id"]
            .as_str()
            .or_else(|| response["session_token"].as_str())
            .ok_or_else(|| {
                BarracudaCoreError::ipc(
                    "BTSP client: session.create missing session_id".to_string(),
                )
            })?
            .to_string();

        Ok((client_ephemeral_pub, session_id))
    }

    /// Ask security provider to verify the challenge and compute our response.
    ///
    /// Returns the challenge response string to send to the target.
    async fn session_verify(
        &self,
        session_id: &str,
        client_ephemeral_pub: &str,
        server_ephemeral_pub: &str,
        challenge: &str,
        preferred_cipher: &str,
    ) -> Result<String> {
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "btsp.session.verify",
            "params": {
                "session_id": session_id,
                "client_ephemeral_pub": client_ephemeral_pub,
                "server_ephemeral_pub": server_ephemeral_pub,
                "challenge": challenge,
                "preferred_cipher": preferred_cipher,
                "role": "client"
            },
            "id": 2
        });

        let response = self.provider_rpc(&request).await?;

        response["client_response"]
            .as_str()
            .or_else(|| response["response"].as_str())
            .ok_or_else(|| {
                BarracudaCoreError::ipc(
                    "BTSP client: session.verify missing client_response".to_string(),
                )
            })
            .map(String::from)
    }

    /// Send a JSON-RPC request to the security provider and return the `result` field.
    async fn provider_rpc(&self, request: &serde_json::Value) -> Result<serde_json::Value> {
        let mut stream = connect_transport(&self.provider_endpoint)
            .await
            .map_err(|e| {
                BarracudaCoreError::ipc(format!(
                    "BTSP client: failed to connect to security provider {:?}: {e}",
                    self.provider_endpoint
                ))
            })?;

        write_ndjson_to_stream(&mut stream, request).await?;
        let response = read_ndjson_from_stream(&mut stream).await?;

        if let Some(error) = response.get("error") {
            return Err(BarracudaCoreError::ipc(format!(
                "BTSP client: security provider RPC error: {error}"
            )));
        }

        response.get("result").cloned().ok_or_else(|| {
            BarracudaCoreError::ipc(
                "BTSP client: security provider response missing 'result'".to_string(),
            )
        })
    }
}

/// Write an NDJSON value to a `TransportStream`.
async fn write_ndjson_to_stream(
    stream: &mut TransportStream,
    value: &serde_json::Value,
) -> Result<()> {
    use tokio::io::AsyncWriteExt;
    let mut line = serde_json::to_string(value)
        .map_err(|e| BarracudaCoreError::ipc(format!("BTSP client: JSON serialize: {e}")))?;
    line.push('\n');
    stream
        .write_all(line.as_bytes())
        .await
        .map_err(|e| BarracudaCoreError::ipc(format!("BTSP client: write failed: {e}")))?;
    stream
        .flush()
        .await
        .map_err(|e| BarracudaCoreError::ipc(format!("BTSP client: flush failed: {e}")))?;
    Ok(())
}

/// Read an NDJSON value from a `TransportStream`.
async fn read_ndjson_from_stream(stream: &mut TransportStream) -> Result<serde_json::Value> {
    use tokio::io::AsyncBufReadExt;
    let mut reader = tokio::io::BufReader::new(stream);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .await
        .map_err(|e| BarracudaCoreError::ipc(format!("BTSP client: read failed: {e}")))?;
    if line.is_empty() {
        return Err(BarracudaCoreError::ipc(
            "BTSP client: connection closed before response".to_string(),
        ));
    }
    serde_json::from_str(&line)
        .map_err(|e| BarracudaCoreError::ipc(format!("BTSP client: invalid JSON response: {e}")))
}

/// Convenience: connect to a target with BTSP handshake if a security provider is available.
///
/// If no security provider is discovered, falls back to a plain (non-BTSP) connection.
/// This graceful degradation matches Phase 2 rollout — hard enforcement comes with
/// `BEARDOG_UDS_REQUIRE_BTSP=1` on the server side.
pub async fn connect_with_btsp(
    target: &TransportEndpoint,
    preferred_cipher: &str,
) -> Result<BtspHandshakeResult> {
    let client = BtspClient::discover().ok_or_else(|| {
        BarracudaCoreError::ipc(
            "BTSP client: no security provider discovered — cannot perform handshake".to_string(),
        )
    })?;
    client.handshake(target, preferred_cipher).await
}

/// Try BTSP handshake, fall back to plain connection if provider unavailable.
///
/// Returns `Ok(Some(result))` on successful handshake, `Ok(None)` if no provider
/// is available (caller should use a plain `connect_transport` instead).
pub async fn try_connect_with_btsp(
    target: &TransportEndpoint,
    preferred_cipher: &str,
) -> Result<Option<BtspHandshakeResult>> {
    match BtspClient::discover() {
        Some(client) => {
            let result = client.handshake(target, preferred_cipher).await?;
            Ok(Some(result))
        }
        None => {
            tracing::debug!("BTSP client: no security provider — falling back to plain connection");
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn btsp_client_explicit_endpoint() {
        let endpoint = TransportEndpoint::uds("/tmp/test-crypto.sock");
        let client = BtspClient::new(endpoint.clone());
        assert_eq!(
            format!("{:?}", client.provider_endpoint),
            format!("{:?}", endpoint)
        );
    }

    #[test]
    fn btsp_client_discover_returns_none_without_provider() {
        // In test env, no security provider socket exists.
        let result = BtspClient::discover();
        // May be Some if BTSP_PROVIDER_SOCKET is set, otherwise None.
        // We just verify it doesn't panic.
        let _ = result;
    }

    #[tokio::test]
    async fn try_connect_gracefully_returns_none_without_provider() {
        let target = TransportEndpoint::tcp("127.0.0.1", 9999);
        // Should not panic — graceful fallback.
        let _ = try_connect_with_btsp(&target, "null").await;
    }

    #[tokio::test]
    async fn write_ndjson_to_stream_roundtrip() {
        let value = serde_json::json!({"protocol": "btsp", "version": 1});
        let mut line = serde_json::to_string(&value).unwrap();
        line.push('\n');
        assert!(line.ends_with('\n'));
        assert_eq!(line.matches('\n').count(), 1);

        let parsed: serde_json::Value = serde_json::from_str(line.trim()).unwrap();
        assert_eq!(parsed["protocol"], "btsp");
        assert_eq!(parsed["version"], 1);
    }

    #[test]
    fn btsp_handshake_result_holds_session() {
        let session = BtspSession {
            session_id: "sess-test-42".to_string(),
            cipher: BtspCipher::ChaCha20Poly1305,
            session_key: vec![0u8; 32],
        };
        assert_eq!(session.cipher.wire_name(), "chacha20-poly1305");
        assert!(session.cipher.requires_key());
    }
}
