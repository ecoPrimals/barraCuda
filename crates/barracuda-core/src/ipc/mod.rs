// SPDX-License-Identifier: AGPL-3.0-only
//! Inter-Primal Communication (IPC) for barraCuda.
//!
//! Implements JSON-RPC 2.0 per wateringHole `PRIMAL_IPC_PROTOCOL.md` and
//! `UNIVERSAL_IPC_STANDARD_V3.md`. Each primal implements the protocol
//! itself — no shared IPC crate.
//!
//! ## Transport
//!
//! Primary: Unix domain socket at `$XDG_RUNTIME_DIR/{PRIMAL_NAMESPACE}/{PRIMAL_NAMESPACE}.sock`
//! Fallback: TCP, resolved via [`transport::resolve_bind_address()`](crate::ipc::transport::resolve_bind_address) —
//! `BARRACUDA_IPC_BIND` or `BARRACUDA_IPC_HOST`:`BARRACUDA_IPC_PORT`,
//! default `localhost:0` (ephemeral)
//!
//! ## Endpoints
//!
//! All methods follow the semantic naming standard: `{domain}.{operation}`.
//! See `specs/BARRACUDA_SPECIFICATION.md` for the full contract.

pub mod jsonrpc;
pub mod methods;
pub mod transport;

pub use jsonrpc::{JsonRpcError, JsonRpcRequest, JsonRpcResponse};
pub use transport::IpcServer;
