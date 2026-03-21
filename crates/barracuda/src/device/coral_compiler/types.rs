// SPDX-License-Identifier: AGPL-3.0-or-later
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
    /// Adapter descriptor for arch-agnostic compilation (Phase 2).
    /// When present, coralReef determines the ISA target from adapter info
    /// rather than requiring barraCuda to specify `arch` explicitly.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adapter: Option<AdapterDescriptor>,
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
        Precision::F16 => "f16_fast",
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

/// Adapter descriptor for IPC compile requests.
///
/// Carries adapter identification to `coralReef` so it can determine the correct
/// ISA compilation target. `barraCuda` does not embed per-generation ISA knowledge
/// — `coralReef` owns the adapter-to-ISA mapping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterDescriptor {
    /// PCI vendor ID (e.g. `0x10DE` for NVIDIA, `0x1002` for AMD).
    pub vendor_id: u32,
    /// Adapter name as reported by wgpu (e.g. "NVIDIA `GeForce` RTX 3090").
    pub device_name: String,
    /// Device type as string (`DiscreteGpu`, `IntegratedGpu`, `Cpu`, etc.).
    pub device_type: String,
}

impl AdapterDescriptor {
    /// Build an adapter descriptor from wgpu adapter info.
    #[must_use]
    pub fn from_adapter_info(info: &wgpu::AdapterInfo) -> Self {
        Self {
            vendor_id: info.vendor,
            device_name: info.name.clone(),
            device_type: format!("{:?}", info.device_type),
        }
    }

    /// Cache key derived from this adapter descriptor.
    #[must_use]
    pub fn cache_key(&self) -> String {
        format!("adapter:{:04x}:{}", self.vendor_id, self.device_name)
    }
}
