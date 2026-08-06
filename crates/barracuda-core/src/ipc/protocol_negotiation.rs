// SPDX-License-Identifier: AGPL-3.0-or-later
//! G65 Protocol Negotiation — single-socket protocol selection.
//!
//! Enables tarpc and JSON-RPC on the same UDS socket. The client sends
//! `PROTOCOLS: tarpc,jsonrpc\n`, the server responds `PROTOCOL: tarpc\n`,
//! and the connection proceeds in the selected protocol.
//!
//! If no negotiation header is received (e.g. a legacy JSON-RPC client),
//! the connection defaults to JSON-RPC — full backward compatibility.
//!
//! Per G65 standard: each primal implements independently, no shared crate.
//! Reference: `specs/PROTOCOL_NEGOTIATION_SPEC.md`.

use super::ipc_protocol::IpcProtocol;
use super::transport_endpoint::TransportStream;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Timeout for detecting a `PROTOCOLS:` negotiation line.
///
/// If the client doesn't send any data within this window, we assume
/// JSON-RPC (backward compatible with legacy clients).
const NEGOTIATION_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(100);

/// Maximum length of a negotiation line to prevent abuse.
const MAX_NEGOTIATION_LINE: usize = 512;

/// The `PROTOCOLS:` prefix that clients send for negotiation.
const PROTOCOLS_PREFIX: &str = "PROTOCOLS: ";

/// The `PROTOCOL:` prefix that servers respond with.
const PROTOCOL_PREFIX: &str = "PROTOCOL: ";

/// Select the best protocol from the client's preference list.
///
/// Client preference wins: the first protocol in `client_supported` that
/// the server also supports is selected. Falls back to JSON-RPC.
#[must_use]
pub fn select_protocol(
    client_supported: &[IpcProtocol],
    server_supported: &[IpcProtocol],
) -> IpcProtocol {
    for client_proto in client_supported {
        if server_supported.contains(client_proto) {
            return *client_proto;
        }
    }
    IpcProtocol::JsonRpc
}

/// Read a negotiation line byte-by-byte from the stream.
///
/// Avoids `BufReader` to prevent over-reading past the negotiation line,
/// which would consume bytes belonging to the subsequent protocol.
async fn read_negotiation_line<S>(stream: &mut S) -> std::io::Result<String>
where
    S: tokio::io::AsyncRead + Unpin,
{
    let mut line = Vec::with_capacity(64);
    let mut byte = [0u8; 1];

    loop {
        match stream.read(&mut byte).await? {
            0 => break,
            _ => {
                if byte[0] == b'\n' {
                    break;
                }
                line.push(byte[0]);
                if line.len() >= MAX_NEGOTIATION_LINE {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "negotiation line too long",
                    ));
                }
            }
        }
    }

    String::from_utf8(line).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

/// Parse a `PROTOCOLS: tarpc,jsonrpc` line into a list of protocols.
fn parse_protocol_request(line: &str) -> Option<Vec<IpcProtocol>> {
    let trimmed = line.trim();
    let proto_list = trimmed.strip_prefix(PROTOCOLS_PREFIX)?;
    let protocols: Vec<IpcProtocol> = proto_list
        .split(',')
        .filter_map(|name| IpcProtocol::from_negotiation_name(name.trim()))
        .collect();

    if protocols.is_empty() {
        return None;
    }
    Some(protocols)
}

/// Format the server's `PROTOCOL: selected\n` response.
fn format_protocol_response(selected: IpcProtocol) -> String {
    format!("{PROTOCOL_PREFIX}{}\n", selected.negotiation_name())
}

/// Attempt G65 protocol negotiation on a new connection.
///
/// Peeks at the first byte with a 100ms timeout:
/// - If `P` (0x50): reads the `PROTOCOLS:` line, selects best match,
///   responds with `PROTOCOL:`, returns `Some(selected)`.
/// - If any other byte or timeout: returns `None` — the connection
///   should proceed through the existing BTSP guard / JSON-RPC path.
///   The peeked byte is NOT consumed (uses `peek`, not `read`).
pub async fn try_negotiate(stream: &mut TransportStream) -> Option<IpcProtocol> {
    let mut peek_buf = [0u8; 1];

    let peek_result = tokio::time::timeout(NEGOTIATION_TIMEOUT, stream.peek(&mut peek_buf)).await;

    match peek_result {
        Ok(Ok(1)) if peek_buf[0] == b'P' => {
            // Looks like a PROTOCOLS: line — read it fully
        }
        _ => return None,
    }

    let line = match tokio::time::timeout(NEGOTIATION_TIMEOUT, read_negotiation_line(stream)).await
    {
        Ok(Ok(line)) => line,
        Ok(Err(e)) => {
            tracing::debug!("G65 negotiation line read error: {e}");
            return None;
        }
        Err(_) => {
            tracing::debug!("G65 negotiation line read timeout");
            return None;
        }
    };

    let client_protocols = match parse_protocol_request(&line) {
        Some(p) => p,
        None => {
            tracing::debug!("G65 line not a valid PROTOCOLS request: {line:?}");
            return None;
        }
    };

    let server_supported = IpcProtocol::supported();
    let selected = select_protocol(&client_protocols, &server_supported);

    let response = format_protocol_response(selected);
    if let Err(e) = stream.write_all(response.as_bytes()).await {
        tracing::warn!("G65 negotiation response write error: {e}");
        return None;
    }
    if let Err(e) = stream.flush().await {
        tracing::warn!("G65 negotiation response flush error: {e}");
        return None;
    }

    tracing::info!(
        protocol = selected.negotiation_name(),
        client_requested = ?client_protocols.iter().map(|p| p.negotiation_name()).collect::<Vec<_>>(),
        "G65 protocol negotiated"
    );

    Some(selected)
}

/// Negotiate protocol from the client side.
///
/// Sends `PROTOCOLS: tarpc,jsonrpc\n`, reads `PROTOCOL: selected\n`,
/// returns the selected protocol.
pub async fn negotiate_client(
    stream: &mut TransportStream,
    preferred: &[IpcProtocol],
) -> Result<IpcProtocol, crate::error::BarracudaCoreError> {
    let names: Vec<&str> = preferred.iter().map(|p| p.negotiation_name()).collect();
    let request_line = format!("{PROTOCOLS_PREFIX}{}\n", names.join(","));

    stream
        .write_all(request_line.as_bytes())
        .await
        .map_err(|e| crate::error::BarracudaCoreError::ipc(format!("G65 request write: {e}")))?;
    stream
        .flush()
        .await
        .map_err(|e| crate::error::BarracudaCoreError::ipc(format!("G65 request flush: {e}")))?;

    let response_line = read_negotiation_line(stream)
        .await
        .map_err(|e| crate::error::BarracudaCoreError::ipc(format!("G65 response read: {e}")))?;

    let trimmed = response_line.trim();
    let proto_name = trimmed.strip_prefix(PROTOCOL_PREFIX).ok_or_else(|| {
        crate::error::BarracudaCoreError::ipc(format!(
            "G65 response missing PROTOCOL: prefix: {trimmed:?}"
        ))
    })?;

    IpcProtocol::from_negotiation_name(proto_name).ok_or_else(|| {
        crate::error::BarracudaCoreError::ipc(format!("G65 unknown protocol: {proto_name:?}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_protocol_client_preference_wins() {
        #[cfg(feature = "tarpc-transport")]
        {
            let client = vec![IpcProtocol::Tarpc, IpcProtocol::JsonRpc];
            let server = vec![IpcProtocol::Tarpc, IpcProtocol::JsonRpc];
            assert_eq!(select_protocol(&client, &server), IpcProtocol::Tarpc);
        }
    }

    #[test]
    fn select_protocol_falls_back_to_jsonrpc() {
        let client = vec![IpcProtocol::JsonRpc];
        let server = vec![IpcProtocol::JsonRpc];
        assert_eq!(select_protocol(&client, &server), IpcProtocol::JsonRpc);
    }

    #[cfg(feature = "tarpc-transport")]
    #[test]
    fn select_protocol_server_lacks_tarpc() {
        let client = vec![IpcProtocol::Tarpc, IpcProtocol::JsonRpc];
        let server = vec![IpcProtocol::JsonRpc];
        assert_eq!(select_protocol(&client, &server), IpcProtocol::JsonRpc);
    }

    #[test]
    fn parse_protocol_request_valid() {
        let result = parse_protocol_request("PROTOCOLS: jsonrpc");
        assert_eq!(result, Some(vec![IpcProtocol::JsonRpc]));
    }

    #[cfg(feature = "tarpc-transport")]
    #[test]
    fn parse_protocol_request_multi() {
        let result = parse_protocol_request("PROTOCOLS: tarpc,jsonrpc");
        assert_eq!(result, Some(vec![IpcProtocol::Tarpc, IpcProtocol::JsonRpc]));
    }

    #[test]
    fn parse_protocol_request_invalid_prefix() {
        assert!(parse_protocol_request("PROTO: jsonrpc").is_none());
    }

    #[test]
    fn parse_protocol_request_no_valid_protocols() {
        assert!(parse_protocol_request("PROTOCOLS: grpc,quic").is_none());
    }

    #[test]
    fn format_response() {
        assert_eq!(
            format_protocol_response(IpcProtocol::JsonRpc),
            "PROTOCOL: jsonrpc\n"
        );
    }

    #[cfg(feature = "tarpc-transport")]
    #[test]
    fn format_response_tarpc() {
        assert_eq!(
            format_protocol_response(IpcProtocol::Tarpc),
            "PROTOCOL: tarpc\n"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn negotiate_uds_jsonrpc() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let sock = dir.path().join("test_neg.sock");
        let listener = tokio::net::UnixListener::bind(&sock).expect("bind");

        let sock2 = sock.clone();
        let client_task = tokio::spawn(async move {
            let stream = tokio::net::UnixStream::connect(&sock2)
                .await
                .expect("connect");
            let mut ts = TransportStream::Unix(stream);
            negotiate_client(&mut ts, &[IpcProtocol::JsonRpc]).await
        });

        let (stream, _) = listener.accept().await.expect("accept");
        let mut ts = TransportStream::Unix(stream);
        let server_result = try_negotiate(&mut ts).await;

        let client_result = client_task.await.expect("client task");
        assert_eq!(client_result.expect("negotiation"), IpcProtocol::JsonRpc);
        assert_eq!(server_result, Some(IpcProtocol::JsonRpc));
    }

    #[cfg(all(unix, feature = "tarpc-transport"))]
    #[tokio::test]
    async fn negotiate_uds_tarpc_preferred() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let sock = dir.path().join("test_neg_tarpc.sock");
        let listener = tokio::net::UnixListener::bind(&sock).expect("bind");

        let sock2 = sock.clone();
        let client_task = tokio::spawn(async move {
            let stream = tokio::net::UnixStream::connect(&sock2)
                .await
                .expect("connect");
            let mut ts = TransportStream::Unix(stream);
            negotiate_client(&mut ts, &[IpcProtocol::Tarpc, IpcProtocol::JsonRpc]).await
        });

        let (stream, _) = listener.accept().await.expect("accept");
        let mut ts = TransportStream::Unix(stream);
        let server_result = try_negotiate(&mut ts).await;

        let client_result = client_task.await.expect("client task");
        assert_eq!(client_result.expect("negotiation"), IpcProtocol::Tarpc);
        assert_eq!(server_result, Some(IpcProtocol::Tarpc));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn no_negotiation_returns_none() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let sock = dir.path().join("test_neg_legacy.sock");
        let listener = tokio::net::UnixListener::bind(&sock).expect("bind");

        let sock2 = sock.clone();
        tokio::spawn(async move {
            let mut stream = tokio::net::UnixStream::connect(&sock2)
                .await
                .expect("connect");
            stream
                .write_all(b"{\"jsonrpc\":\"2.0\"}\n")
                .await
                .expect("write");
            stream.flush().await.expect("flush");
        });

        let (stream, _) = listener.accept().await.expect("accept");
        let mut ts = TransportStream::Unix(stream);
        let result = try_negotiate(&mut ts).await;
        assert!(result.is_none());
    }
}
