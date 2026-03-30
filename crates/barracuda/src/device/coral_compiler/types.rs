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
    /// Precision routing advice from barraCuda's `PrecisionBrain`.
    /// Tells coralReef which precision tier was selected and whether
    /// f64 transcendental lowering is needed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub precision_advice: Option<PrecisionAdvice>,
}

/// Precision routing advice carried in compile requests.
///
/// Enables coralReef to make informed compilation decisions based on
/// barraCuda's hardware probe results and domain requirements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionAdvice {
    /// The precision tier selected by `PrecisionBrain` (e.g. "F64", "DF64", "F32").
    pub tier: String,
    /// Whether hardware native f64 transcendentals are broken (probed).
    pub needs_transcendental_lowering: bool,
    /// Whether DF64 (f32-pair) transcendentals are poisoned by naga.
    pub df64_naga_poisoned: bool,
    /// Physics domain that motivated this compilation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain: Option<String>,
}

/// Map a barraCuda `Precision` tier to coralReef's compilation strategy string.
///
/// This is the interface between barraCuda's precision model and coralReef's
/// compilation pipeline. barraCuda decides WHICH precision; coralReef decides
/// HOW to compile it to hardware. toadStool routes to silicon.
///
/// Strategy strings are the wire-protocol contract with coralReef.
/// New tiers added here require corresponding support in coralReef's
/// `CompileStrategy` dispatcher.
#[must_use]
pub fn precision_to_coral_strategy(
    precision: &crate::shaders::precision::Precision,
) -> &'static str {
    use crate::shaders::precision::Precision;
    match precision {
        Precision::Binary => "binary",
        Precision::Int2 => "int2",
        Precision::Q4 => "q4_block",
        Precision::Q8 => "q8_block",
        Precision::Fp8E5M2 => "fp8_e5m2",
        Precision::Fp8E4M3 => "fp8_e4m3",
        Precision::Bf16 => "bf16_emulated",
        Precision::F16 => "f16_fast",
        Precision::F32 => "f32_only",
        Precision::F64 => "native",
        Precision::Df64 => "double_float",
        Precision::Qf128 => "quad_float",
        Precision::Df128 => "double_double_f64",
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

/// f64 transcendental polyfill capabilities reported by coralReef.
///
/// Mirrors `coralreef-core::service::F64TranscendentalCapabilities`.
/// Each field indicates whether coralReef can provide a software polyfill
/// for that operation via its sovereign compilation pipeline.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CoralF64Capabilities {
    /// Software sin(f64) lowering available.
    pub sin: bool,
    /// Software cos(f64) lowering available.
    pub cos: bool,
    /// Software sqrt(f64) lowering available (Newton-Raphson).
    pub sqrt: bool,
    /// Software exp2(f64) lowering available.
    pub exp2: bool,
    /// Software log2(f64) lowering available.
    pub log2: bool,
    /// Software 1/x (reciprocal) lowering available.
    pub rcp: bool,
    /// Software exp(f64) lowering available.
    pub exp: bool,
    /// Software log(f64) lowering available.
    pub log: bool,
    /// Full composite lowering available (all ops can be combined in one shader).
    pub composite_lowering: bool,
}

impl CoralF64Capabilities {
    /// Whether coralReef can lower all f64 transcendentals (full polyfill).
    #[must_use]
    pub fn has_full_lowering(&self) -> bool {
        self.sin
            && self.cos
            && self.sqrt
            && self.exp2
            && self.log2
            && self.exp
            && self.log
            && self.composite_lowering
    }
}

/// Structured capabilities response from coralReef.
///
/// Mirrors `coralreef-core::service::CompileCapabilitiesResponse`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoralCapabilitiesResponse {
    /// Supported GPU ISA architectures (e.g. `["sm_70", "sm_80", "gfx1030"]`).
    pub supported_archs: Vec<String>,
    /// Per-operation f64 transcendental polyfill availability.
    pub f64_transcendental_capabilities: CoralF64Capabilities,
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
