// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2025-2026 ecoPrimals Collective
//! Named constants for environment variable keys used in production code.
//!
//! Centralizes `std::env::var` string literals for device discovery and
//! runtime configuration.

/// XDG runtime directory for Unix socket and manifest discovery.
pub const XDG_RUNTIME_DIR: &str = "XDG_RUNTIME_DIR";

/// Ecosystem socket directory override.
pub const BIOMEOS_SOCKET_DIR: &str = "BIOMEOS_SOCKET_DIR";

/// Default ecosystem socket namespace per wateringHole `PRIMAL_IPC_PROTOCOL` v3.0.
///
/// All primals place Unix sockets under `$XDG_RUNTIME_DIR/{namespace}/`.
/// Override at runtime with the [`BIOMEOS_SOCKET_DIR`] environment variable.
pub const DEFAULT_ECOSYSTEM_SOCKET_NAMESPACE: &str = "biomeos";

/// JSON manifest discovery subdirectory under [`XDG_RUNTIME_DIR`].
pub const ECOPRIMALS_DISCOVERY_DIR: &str = "ECOPRIMALS_DISCOVERY_DIR";

/// Sovereign dispatch GPR register count override.
pub const BARRACUDA_GPR_COUNT: &str = "BARRACUDA_GPR_COUNT";

/// Sovereign dispatch default workgroup X dimension.
pub const BARRACUDA_DEFAULT_WORKGROUP_X: &str = "BARRACUDA_DEFAULT_WORKGROUP_X";

/// Explicit GPU architecture target for sovereign compile path.
pub const BARRACUDA_TARGET_ARCH: &str = "BARRACUDA_TARGET_ARCH";

/// wgpu poll timeout in seconds (0 disables timeout).
pub const BARRACUDA_POLL_TIMEOUT_SECS: &str = "BARRACUDA_POLL_TIMEOUT_SECS";

/// Require GPU adapter in validation binaries (`1` or `true`).
pub const BARRACUDA_REQUIRE_GPU: &str = "BARRACUDA_REQUIRE_GPU";
