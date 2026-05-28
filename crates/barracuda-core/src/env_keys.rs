// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2025-2026 ecoPrimals Collective
//! Named constants for environment variable keys used in production code.
//!
//! Centralizes `std::env::var` string literals for IPC transport, discovery,
//! and lifecycle configuration.

/// Skip GPU probe at startup (`1`, `true`, `yes`).
pub const BARRACUDA_NO_GPU_PROBE: &str = "BARRACUDA_NO_GPU_PROBE";

/// Discovery service Unix socket path.
pub const DISCOVERY_SOCKET: &str = "DISCOVERY_SOCKET";

/// Maximum JSON-RPC frame size in bytes.
pub const BARRACUDA_MAX_FRAME_BYTES: &str = "BARRACUDA_MAX_FRAME_BYTES";

/// Maximum concurrent IPC connections.
pub const BARRACUDA_MAX_CONNECTIONS: &str = "BARRACUDA_MAX_CONNECTIONS";

/// TCP bind host for IPC (default `127.0.0.1`).
pub const BARRACUDA_IPC_HOST: &str = "BARRACUDA_IPC_HOST";

/// Explicit TCP bind address override (`host:port`).
pub const BARRACUDA_IPC_BIND: &str = "BARRACUDA_IPC_BIND";

/// TCP bind port for IPC (used with [`BARRACUDA_IPC_HOST`]).
pub const BARRACUDA_IPC_PORT: &str = "BARRACUDA_IPC_PORT";

/// Canonical TCP port per `plasmidBin/ports.env`.
pub const BARRACUDA_PORT: &str = "BARRACUDA_PORT";

/// barraCuda-specific family ID override.
pub const BARRACUDA_FAMILY_ID: &str = "BARRACUDA_FAMILY_ID";

/// Generic ecosystem family ID.
pub const FAMILY_ID: &str = "FAMILY_ID";

/// Legacy biomeOS family ID.
pub const BIOMEOS_FAMILY_ID: &str = "BIOMEOS_FAMILY_ID";

/// Ecosystem socket directory override.
pub const BIOMEOS_SOCKET_DIR: &str = "BIOMEOS_SOCKET_DIR";

/// Skip BTSP authentication when set to `1` or `true`.
pub const BIOMEOS_INSECURE: &str = "BIOMEOS_INSECURE";

/// XDG runtime directory for Unix socket placement.
pub const XDG_RUNTIME_DIR: &str = "XDG_RUNTIME_DIR";

/// BTSP security-provider socket (composition-injected).
pub const BTSP_PROVIDER_SOCKET: &str = "BTSP_PROVIDER_SOCKET";

/// Legacy beardog security-provider socket path.
pub const BEARDOG_SOCKET: &str = "BEARDOG_SOCKET";

/// BTSP family seed (preferred).
pub const BTSP_FAMILY_SEED: &str = "BTSP_FAMILY_SEED";

/// Generic ecosystem family seed.
pub const FAMILY_SEED: &str = "FAMILY_SEED";

/// Legacy biomeOS family seed.
pub const BIOMEOS_FAMILY_SEED: &str = "BIOMEOS_FAMILY_SEED";

/// Legacy beardog family seed.
pub const BEARDOG_FAMILY_SEED: &str = "BEARDOG_FAMILY_SEED";

/// IPC authentication enforcement mode (`permissive` or `enforced`).
pub const BARRACUDA_AUTH_MODE: &str = "BARRACUDA_AUTH_MODE";

/// systemd `Type=notify` socket for readiness signaling.
pub const NOTIFY_SOCKET: &str = "NOTIFY_SOCKET";
