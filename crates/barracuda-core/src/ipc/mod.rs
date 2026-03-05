// SPDX-License-Identifier: AGPL-3.0-or-later
//! Inter-Primal Communication (IPC) for barraCuda.
//!
//! Implements JSON-RPC 2.0 per wateringHole `PRIMAL_IPC_PROTOCOL.md` and
//! `UNIVERSAL_IPC_STANDARD_V3.md`. Each primal implements the protocol
//! itself — no shared IPC crate.
//!
//! ## Transport
//!
//! Primary: Unix domain socket at `$XDG_RUNTIME_DIR/barracuda/barracuda.sock`
//! Fallback: TCP on `127.0.0.1:{BARRACUDA_IPC_PORT}` if set, else `127.0.0.1:0` (ephemeral)
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
