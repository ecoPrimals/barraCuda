// SPDX-License-Identifier: AGPL-3.0-or-later
//! Transport layer for barraCuda IPC.
//!
//! Transport-agnostic JSON-RPC 2.0 handler over any `AsyncRead + AsyncWrite`.
//! Supports Unix domain sockets (primary) and TCP (fallback), per
//! wateringHole `UNIVERSAL_IPC_STANDARD_V3.md` and `ECOBIN_ARCHITECTURE_STANDARD.md`.
//!
//! Configuration utilities (bind address resolution, socket paths, env vars) live
//! in the sibling [`super::transport_config`] module.

mod connection;
mod dispatch;
mod server;

pub use super::transport_config::{
    DEFAULT_BIND_HOST, DEFAULT_ECOSYSTEM_SOCKET_DIR, discovery_socket_path, resolve_bind_address,
    resolve_bind_host, resolve_family_id, resolve_socket_dir, validate_insecure_guard,
};
pub use super::transport_endpoint::{
    TransportEndpoint, TransportListener, TransportStream, connect_transport,
};
pub use server::IpcServer;

#[cfg(test)]
use crate::BarraCudaPrimal;
#[cfg(test)]
use std::sync::Arc;
#[cfg(test)]
use tokio::io::AsyncWriteExt;

#[cfg(test)]
use connection::{handle_connection, strip_genetics_prefix};
#[cfg(test)]
use dispatch::{handle_batch, handle_line};
#[cfg(test)]
use server::{max_connections, max_frame_bytes};

#[cfg(test)]
#[path = "../transport_tests/mod.rs"]
mod tests;
