// SPDX-License-Identifier: AGPL-3.0-only
//! Wire types for the coralReef IPC protocol.

use bytes::Bytes;
use serde::{Deserialize, Serialize};

/// Cached native binary produced by coralReef.
#[derive(Debug, Clone)]
pub struct CoralBinary {
    /// Raw GPU binary (SM70+ native code).
    pub binary: Bytes,
    /// Target architecture (e.g. `sm_70`).
    pub arch: String,
}

/// SPIR-V compile request — mirrors `coralreef-core::service::CompileRequest`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct CompileRequest {
    pub spirv_words: Vec<u32>,
    pub arch: String,
    pub opt_level: u32,
    pub fp64_software: bool,
}

/// WGSL direct compile request (Phase 10) — avoids local naga SPIR-V step.
///
/// coralReef Phase 10 (`shader.compile.wgsl`) accepts raw WGSL and handles
/// the full WGSL → IR → native binary pipeline server-side.
///
/// `fp64_strategy` is the Phase 2 field aligned with coralReef's `Fp64Strategy`
/// enum. `fp64_software` is kept for backward compatibility with Phase 1 servers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct CompileWgslRequest {
    pub wgsl_source: String,
    pub arch: String,
    pub opt_level: u32,
    pub fp64_software: bool,
    /// Precision strategy hint for coralReef (Phase 2).
    /// Maps to coralReef's `Fp64Strategy`: `"native"`, `"double_float"`, `"f32_only"`.
    /// Ignored by Phase 1 servers that only read `fp64_software`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fp64_strategy: Option<String>,
}

/// Map a barraCuda `Precision` tier to coralReef's `Fp64Strategy` string.
///
/// This is the interface between barraCuda's precision model and coralReef's
/// compilation pipeline. barraCuda decides WHICH precision; coralReef decides
/// HOW to compile it to hardware.
#[must_use]
pub fn precision_to_coral_strategy(
    precision: &crate::shaders::precision::Precision,
) -> &'static str {
    use crate::shaders::precision::Precision;
    match precision {
        Precision::F32 => "f32_only",
        Precision::F64 => "native",
        Precision::Df64 => "double_float",
    }
}

/// Compile response — mirrors `coralreef-core::service::CompileResponse`.
///
/// `binary` deserializes as `Vec<u8>` from JSON but is immediately converted
/// to `bytes::Bytes` at the call site for zero-copy sharing downstream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct CompileResponse {
    pub binary: Vec<u8>,
    pub size: usize,
}

impl CompileResponse {
    /// Convert the owned binary to a shared `bytes::Bytes` (zero-copy freeze).
    pub fn into_bytes(self) -> Bytes {
        Bytes::from(self.binary)
    }
}

/// Health response from coralReef.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Primal name (e.g. `"coralReef"`)
    pub name: String,
    /// Version string
    pub version: String,
    /// Health status
    pub status: String,
    /// Supported GPU architectures (e.g. `["sm_70", "sm_75", "sm_80", "sm_89"]`)
    pub supported_archs: Vec<String>,
}

/// Map a barraCuda `GpuArch` to coralReef's arch string.
///
/// Supports NVIDIA (SM70+) and AMD RDNA2+ architectures per coralReef Phase 10.
/// Returns `None` for architectures that coralReef cannot compile for
/// (Intel Arc, Apple M, software rasterizers, unknowns).
#[must_use]
pub fn arch_to_coral(arch: &crate::device::driver_profile::GpuArch) -> Option<&'static str> {
    use crate::device::driver_profile::GpuArch;
    match arch {
        GpuArch::Volta => Some("sm_70"),
        GpuArch::Turing => Some("sm_75"),
        GpuArch::Ampere => Some("sm_80"),
        GpuArch::Ada => Some("sm_89"),
        GpuArch::Rdna2 => Some("gfx1030"),
        GpuArch::Rdna3 => Some("gfx1100"),
        GpuArch::Cdna2 => Some("gfx90a"),
        _ => None,
    }
}
