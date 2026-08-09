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

/// GPU adapter selector override (index, substring match, or `"auto"`).
pub const BARRACUDA_GPU_ADAPTER: &str = "BARRACUDA_GPU_ADAPTER";

/// Concurrency budget override for GPU command submission.
pub const BARRACUDA_CONCURRENCY_BUDGET: &str = "BARRACUDA_CONCURRENCY_BUDGET";

/// Small matmul CPU-only threshold (element count).
pub const BARRACUDA_MATMUL_SMALL_THRESHOLD: &str = "BARRACUDA_MATMUL_SMALL_THRESHOLD";

/// GPU matmul threshold — matrices above this size dispatch to GPU.
pub const BARRACUDA_MATMUL_GPU_THRESHOLD: &str = "BARRACUDA_MATMUL_GPU_THRESHOLD";

/// Explicit shader compiler address override (`unix:/path` or `host:port`).
pub const BARRACUDA_SHADER_COMPILER_ADDR: &str = "BARRACUDA_SHADER_COMPILER_ADDR";

/// Shader compiler port for localhost fallback probe.
pub const BARRACUDA_SHADER_COMPILER_PORT: &str = "BARRACUDA_SHADER_COMPILER_PORT";

/// XDG cache home for autotune and NCBI caches.
pub const XDG_CACHE_HOME: &str = "XDG_CACHE_HOME";

/// XDG data home for hardware-specific SDK paths.
pub const XDG_DATA_HOME: &str = "XDG_DATA_HOME";

/// User home directory fallback for XDG paths.
pub const HOME: &str = "HOME";

/// Default loopback address for localhost-only IPC and discovery probes.
pub const DEFAULT_LOOPBACK: &str = "127.0.0.1";

/// Test backend selector (`gpu` runs tests against GPU workload path).
pub const BARRACUDA_TEST_BACKEND: &str = "BARRACUDA_TEST_BACKEND";

/// Akida SDK home directory for version and model discovery.
pub const AKIDA_HOME: &str = "AKIDA_HOME";

/// Akida SDK directory override for version discovery.
pub const AKIDA_SDK_DIR: &str = "AKIDA_SDK_DIR";

/// Explicit Akida model file path for NPU model discovery.
pub const AKIDA_MODEL_PATH: &str = "AKIDA_MODEL_PATH";

/// Akida models directory (backward-compatible discovery path).
pub const AKIDA_MODELS_DIR: &str = "AKIDA_MODELS_DIR";
