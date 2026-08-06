// SPDX-License-Identifier: AGPL-3.0-or-later
//! IPC protocol enumeration for G65 protocol negotiation.
//!
//! Defines the supported wire protocols that barraCuda can serve on a
//! single socket. Each primal implements this enum independently per
//! the G65 standard — no shared crate (primal violation).

/// Supported IPC protocols for G65 single-socket negotiation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IpcProtocol {
    /// JSON-RPC 2.0 — default, always supported.
    JsonRpc,
    /// tarpc binary RPC — high-throughput intra-gate composition.
    #[cfg(feature = "tarpc-transport")]
    Tarpc,
}

impl IpcProtocol {
    /// Wire name used in `PROTOCOLS:` / `PROTOCOL:` negotiation lines.
    #[must_use]
    pub const fn negotiation_name(self) -> &'static str {
        match self {
            Self::JsonRpc => "jsonrpc",
            #[cfg(feature = "tarpc-transport")]
            Self::Tarpc => "tarpc",
        }
    }

    /// Parse from a wire negotiation name (case-insensitive).
    #[must_use]
    pub fn from_negotiation_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "jsonrpc" | "json-rpc" | "json_rpc" => Some(Self::JsonRpc),
            #[cfg(feature = "tarpc-transport")]
            "tarpc" => Some(Self::Tarpc),
            _ => None,
        }
    }

    /// All protocols this primal supports, in preference order (tarpc first).
    #[must_use]
    pub fn supported() -> Vec<Self> {
        vec![
            #[cfg(feature = "tarpc-transport")]
            Self::Tarpc,
            Self::JsonRpc,
        ]
    }
}

impl std::fmt::Display for IpcProtocol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.negotiation_name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jsonrpc_roundtrip() {
        let proto = IpcProtocol::JsonRpc;
        assert_eq!(proto.negotiation_name(), "jsonrpc");
        assert_eq!(
            IpcProtocol::from_negotiation_name("jsonrpc"),
            Some(IpcProtocol::JsonRpc)
        );
        assert_eq!(
            IpcProtocol::from_negotiation_name("JSONRPC"),
            Some(IpcProtocol::JsonRpc)
        );
        assert_eq!(
            IpcProtocol::from_negotiation_name("json-rpc"),
            Some(IpcProtocol::JsonRpc)
        );
    }

    #[cfg(feature = "tarpc-transport")]
    #[test]
    fn tarpc_roundtrip() {
        let proto = IpcProtocol::Tarpc;
        assert_eq!(proto.negotiation_name(), "tarpc");
        assert_eq!(
            IpcProtocol::from_negotiation_name("tarpc"),
            Some(IpcProtocol::Tarpc)
        );
    }

    #[test]
    fn unknown_returns_none() {
        assert_eq!(IpcProtocol::from_negotiation_name("grpc"), None);
        assert_eq!(IpcProtocol::from_negotiation_name(""), None);
    }

    #[test]
    fn supported_always_includes_jsonrpc() {
        let supported = IpcProtocol::supported();
        assert!(supported.contains(&IpcProtocol::JsonRpc));
    }

    #[cfg(feature = "tarpc-transport")]
    #[test]
    fn tarpc_is_preferred() {
        let supported = IpcProtocol::supported();
        assert_eq!(supported[0], IpcProtocol::Tarpc);
    }
}
