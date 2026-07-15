// SPDX-License-Identifier: AGPL-3.0-or-later
//! Transport endpoint types and outbound connection logic.
//!
//! Implements the `TransportEndpoint` wire format locally (identical to the
//! sourDough canonical format) per the primal self-knowledge principle:
//! a primal only knows itself. The JSON wire format is the contract.
//!
//! See `sourDough/crates/sourdough-core/src/transport.rs` as spec reference.

use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};

/// Structured transport endpoint for launcher-injected transport.
///
/// Wire format (JSON, serde tagged):
/// ```json
/// { "transport": "uds", "path": "/run/user/1000/biomeos/beardog.sock" }
/// { "transport": "tcp", "host": "127.0.0.1", "port": 9100 }
/// { "transport": "mesh_relay", "peer_id": "strandgate", "capability": "security" }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "transport")]
pub enum TransportEndpoint {
    /// Unix Domain Socket.
    #[serde(rename = "uds")]
    Uds {
        /// Filesystem path to the socket.
        path: String,
    },
    /// TCP socket.
    #[serde(rename = "tcp")]
    Tcp {
        /// Bind or connect host address.
        host: String,
        /// TCP port.
        port: u16,
    },
    /// Mesh relay via Songbird (not directly connectable).
    #[serde(rename = "mesh_relay")]
    MeshRelay {
        /// Remote gate identity.
        peer_id: String,
        /// Capability being requested.
        capability: String,
    },
}

impl TransportEndpoint {
    /// Create a UDS endpoint.
    #[must_use]
    pub fn uds(path: impl Into<String>) -> Self {
        Self::Uds { path: path.into() }
    }

    /// Create a TCP endpoint.
    #[must_use]
    pub fn tcp(host: impl Into<String>, port: u16) -> Self {
        Self::Tcp {
            host: host.into(),
            port,
        }
    }
}

/// Transport-agnostic connected stream.
#[derive(Debug)]
pub enum TransportStream {
    /// Connected Unix domain socket.
    #[cfg(unix)]
    Unix(tokio::net::UnixStream),
    /// Connected TCP stream.
    Tcp(tokio::net::TcpStream),
}

impl AsyncRead for TransportStream {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        match self.get_mut() {
            #[cfg(unix)]
            Self::Unix(s) => Pin::new(s).poll_read(cx, buf),
            Self::Tcp(s) => Pin::new(s).poll_read(cx, buf),
        }
    }
}

impl AsyncWrite for TransportStream {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        match self.get_mut() {
            #[cfg(unix)]
            Self::Unix(s) => Pin::new(s).poll_write(cx, buf),
            Self::Tcp(s) => Pin::new(s).poll_write(cx, buf),
        }
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        match self.get_mut() {
            #[cfg(unix)]
            Self::Unix(s) => Pin::new(s).poll_flush(cx),
            Self::Tcp(s) => Pin::new(s).poll_flush(cx),
        }
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        match self.get_mut() {
            #[cfg(unix)]
            Self::Unix(s) => Pin::new(s).poll_shutdown(cx),
            Self::Tcp(s) => Pin::new(s).poll_shutdown(cx),
        }
    }
}

/// Connect to a service via its resolved [`TransportEndpoint`].
///
/// Returns a [`TransportStream`] ready for JSON-RPC or binary framing.
pub async fn connect_transport(endpoint: &TransportEndpoint) -> std::io::Result<TransportStream> {
    match endpoint {
        #[cfg(unix)]
        TransportEndpoint::Uds { path } => {
            let stream = tokio::net::UnixStream::connect(path).await?;
            Ok(TransportStream::Unix(stream))
        }
        #[cfg(not(unix))]
        TransportEndpoint::Uds { path } => Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            format!("UDS not available on this platform for {path}"),
        )),
        TransportEndpoint::Tcp { host, port } => {
            let stream = tokio::net::TcpStream::connect(format!("{host}:{port}")).await?;
            Ok(TransportStream::Tcp(stream))
        }
        TransportEndpoint::MeshRelay { peer_id, .. } => Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            format!("mesh_relay ({peer_id}) requires mesh relay routing, not direct connect"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uds_roundtrip() {
        let ep = TransportEndpoint::uds("/tmp/test.sock");
        let json = serde_json::to_string(&ep).expect("serialize");
        let parsed: TransportEndpoint = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(ep, parsed);
        assert!(json.contains(r#""transport":"uds""#));
        assert!(json.contains(r#""path":"/tmp/test.sock""#));
    }

    #[test]
    fn tcp_roundtrip() {
        let ep = TransportEndpoint::tcp("192.168.1.1", 7700);
        let json = serde_json::to_string(&ep).expect("serialize");
        let parsed: TransportEndpoint = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(ep, parsed);
        assert!(json.contains(r#""transport":"tcp""#));
        assert!(json.contains(r#""port":7700"#));
    }

    #[test]
    fn mesh_relay_roundtrip() {
        let ep = TransportEndpoint::MeshRelay {
            peer_id: "eastgate".into(),
            capability: "security".into(),
        };
        let json = serde_json::to_string(&ep).expect("serialize");
        let parsed: TransportEndpoint = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(ep, parsed);
        assert!(json.contains(r#""transport":"mesh_relay""#));
    }

    #[test]
    fn wire_compat_from_raw_json() {
        let raw = r#"{"transport":"uds","path":"/run/user/1000/biomeos/beardog.sock"}"#;
        let ep: TransportEndpoint = serde_json::from_str(raw).expect("parse raw");
        assert_eq!(
            ep,
            TransportEndpoint::Uds {
                path: "/run/user/1000/biomeos/beardog.sock".into()
            }
        );
    }

    #[test]
    fn tcp_wire_compat() {
        let raw = r#"{"transport":"tcp","host":"0.0.0.0","port":9100}"#;
        let ep: TransportEndpoint = serde_json::from_str(raw).expect("parse raw");
        assert_eq!(
            ep,
            TransportEndpoint::Tcp {
                host: "0.0.0.0".into(),
                port: 9100
            }
        );
    }

    #[test]
    fn unknown_transport_errors() {
        let raw = r#"{"transport":"quic","addr":"example.com"}"#;
        assert!(serde_json::from_str::<TransportEndpoint>(raw).is_err());
    }

    #[tokio::test]
    async fn connect_mesh_relay_returns_unsupported() {
        let ep = TransportEndpoint::MeshRelay {
            peer_id: "test".into(),
            capability: "compute".into(),
        };
        let err = connect_transport(&ep).await.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
    }

    #[tokio::test]
    async fn connect_tcp_refuses_bad_addr() {
        let ep = TransportEndpoint::tcp("127.0.0.1", 1);
        let err = connect_transport(&ep).await.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::ConnectionRefused);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn connect_uds_refuses_nonexistent() {
        let ep = TransportEndpoint::uds("/tmp/nonexistent_barracuda_test_39dj3.sock");
        let err = connect_transport(&ep).await.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    }
}
