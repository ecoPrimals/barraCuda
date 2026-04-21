// SPDX-License-Identifier: AGPL-3.0-or-later
//! Wire types for the shader compiler JSON-RPC protocol.
//!
//! Any primal discovered at runtime via the `shader.compile` capability (see
//! [`crate::device::coral_compiler::discovery`]) may implement this contract;
//! **coralReef** is the reference implementation and historically defined these payloads (`coralreef-core`).

use bytes::Bytes;
use serde::{Deserialize, Serialize};

/// Cached native binary from a shader compiler primal (discovered by the
/// `shader.compile` capability).
#[derive(Debug, Clone)]
pub struct CoralBinary {
    /// Raw GPU binary (SM70+ native code).
    pub binary: Bytes,
    /// Target architecture (e.g. `sm_70`).
    pub arch: String,
    /// Compiler-reported GPR count (from register allocation).
    /// `None` when the compiler doesn't report metadata (legacy servers).
    pub gpr_count: Option<u32>,
    /// Compiler-reported workgroup dimensions.
    /// `None` when the compiler doesn't report metadata.
    pub workgroup: Option<[u32; 3]>,
    /// Shared memory in bytes (from shader analysis).
    pub shared_mem_bytes: Option<u32>,
    /// Barrier count used by the shader.
    pub barrier_count: Option<u32>,
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
/// Phase 10 (`shader.compile.wgsl`) accepts raw WGSL and handles the full
/// WGSL → IR → native binary pipeline server-side (coralReef reference naming).
///
/// `fp64_strategy` is the Phase 2 field aligned with the reference compiler's
/// `Fp64Strategy` enum. `fp64_software` is kept for backward compatibility with legacy servers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct CompileWgslRequest {
    pub wgsl_source: String,
    pub arch: String,
    pub opt_level: u32,
    pub fp64_software: bool,
    /// Precision strategy hint for the remote compiler (Phase 2).
    /// Maps to the wire `Fp64Strategy`: `"native"`, `"double_float"`, `"f32_only"`.
    /// Ignored by legacy servers that only read `fp64_software`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fp64_strategy: Option<String>,
    /// Adapter descriptor for arch-agnostic compilation (Phase 2).
    /// When present, the shader compiler primal determines the ISA target from adapter info
    /// rather than requiring barraCuda to specify `arch` explicitly.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adapter: Option<AdapterDescriptor>,
    /// Precision routing advice from barraCuda's `PrecisionBrain`.
    /// Tells the remote shader compiler which precision tier was selected and whether
    /// f64 transcendental lowering is needed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub precision_advice: Option<PrecisionAdvice>,
}

/// Precision routing advice carried in compile requests.
///
/// Enables the shader compiler primal to make informed compilation decisions based on
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

/// Map a barraCuda `Precision` tier to the remote compiler's strategy string.
///
/// This is the interface between barraCuda's precision model and a shader compiler primal's
/// compilation pipeline (discovered via `shader.compile`). barraCuda decides WHICH precision;
/// the compiler decides HOW to compile it to hardware. toadStool routes to silicon.
///
/// Strategy strings are the wire-protocol contract with the compiler.
/// New tiers added here require corresponding support in the reference
/// `CompileStrategy` dispatcher (coralReef).
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
///
/// The real coralReef server nests metadata inside `info: CompilationInfoResponse`.
/// Legacy flat fields are kept for backward compatibility with servers
/// that send metadata at the top level instead of nested in `info`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct CompileResponse {
    pub binary: Vec<u8>,
    pub size: usize,
    /// Nested compilation metadata from the server (coralReef reference format).
    #[serde(default)]
    pub info: Option<CompilationInfoResponse>,
    /// Legacy flat GPR count (servers without nested `info`).
    #[serde(default)]
    pub gpr_count: Option<u32>,
    /// Legacy flat workgroup dimensions (servers without nested `info`).
    #[serde(default)]
    pub workgroup: Option<[u32; 3]>,
    /// Legacy flat shared memory (servers without nested `info`).
    #[serde(default)]
    pub shared_mem_bytes: Option<u32>,
    /// Legacy flat barrier count (servers without nested `info`).
    #[serde(default)]
    pub barrier_count: Option<u32>,
}

/// Compilation metadata nested inside `CompileResponse.info`.
///
/// Mirrors `coralreef-core::service::types::CompilationInfoResponse`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(super) struct CompilationInfoResponse {
    #[serde(default)]
    pub gpr_count: u32,
    #[serde(default)]
    pub instr_count: u32,
    #[serde(default)]
    pub shared_mem_bytes: u32,
    #[serde(default)]
    pub barrier_count: u32,
    #[serde(default)]
    pub workgroup_size: [u32; 3],
}

impl CompileResponse {
    /// Convert to a `CoralBinary`, preferring nested `info` over legacy flat fields.
    pub fn into_coral_binary(self, arch: String) -> CoralBinary {
        if let Some(info) = self.info {
            CoralBinary {
                binary: Bytes::from(self.binary),
                arch,
                gpr_count: Some(info.gpr_count),
                workgroup: Some(info.workgroup_size),
                shared_mem_bytes: Some(info.shared_mem_bytes),
                barrier_count: Some(info.barrier_count),
            }
        } else {
            CoralBinary {
                binary: Bytes::from(self.binary),
                arch,
                gpr_count: self.gpr_count,
                workgroup: self.workgroup,
                shared_mem_bytes: self.shared_mem_bytes,
                barrier_count: self.barrier_count,
            }
        }
    }
}

/// Health response from `shader.compile.status`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Primal name as reported by the endpoint (e.g. a reference implementation may use `"coralReef"`).
    pub name: String,
    /// Version string
    pub version: String,
    /// Health status
    pub status: String,
    /// Supported GPU architectures (e.g. `["sm_70", "sm_75", "sm_80", "sm_89"]`)
    pub supported_archs: Vec<String>,
}

/// f64 transcendental polyfill capabilities from `shader.compile.capabilities`.
///
/// Mirrors `coralreef-core::service::F64TranscendentalCapabilities`.
/// Each field indicates whether the shader compiler primal can provide a software polyfill
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
    /// Whether the compiler can lower all f64 transcendentals (full polyfill).
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

/// Structured capabilities response from `shader.compile.capabilities`.
///
/// Mirrors `coralreef-core::service::CompileCapabilitiesResponse`.
/// Phase 3 adds CPU compilation and validation capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoralCapabilitiesResponse {
    /// Supported GPU ISA architectures (e.g. `["sm_70", "sm_80", "gfx1030"]`).
    pub supported_archs: Vec<String>,
    /// Per-operation f64 transcendental polyfill availability.
    pub f64_transcendental_capabilities: CoralF64Capabilities,
    /// Supported CPU architectures for `shader.compile.cpu` (Phase 3).
    /// Empty if CPU compilation is not available.
    #[serde(default)]
    pub cpu_archs: Vec<String>,
    /// Whether `shader.execute.cpu` is available (Phase 3).
    #[serde(default)]
    pub supports_cpu_execution: bool,
    /// Whether `shader.validate` is available (Phase 3).
    #[serde(default)]
    pub supports_validation: bool,
}

// ============================================================================
// Phase 3: CPU compilation, execution, and validation wire types
// ============================================================================

/// CPU compilation request (`shader.compile.cpu`).
///
/// Compiles a WGSL compute shader to a native CPU binary via Cranelift.
/// The binary can be loaded and executed directly — no GPU driver needed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileCpuRequest {
    /// WGSL source code.
    pub wgsl_source: String,
    /// Target CPU architecture: `"x86_64"`, `"aarch64"`, or `"auto"` (host).
    pub arch: String,
    /// Optimization level (0 = debug, 1 = default, 2 = aggressive).
    pub opt_level: u32,
    /// Compute shader entry point name.
    pub entry_point: String,
}

/// CPU compilation response (`shader.compile.cpu`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileCpuResponse {
    /// Native CPU binary.
    pub binary: Vec<u8>,
    /// Target architecture the binary was compiled for.
    pub arch: String,
    /// Binary size in bytes.
    pub size: usize,
}

/// CPU execution request (`shader.execute.cpu`).
///
/// Compile and execute a WGSL compute shader on the CPU in one IPC call.
/// The shader compiler primal compiles the shader to native code, simulates the dispatch,
/// and returns the modified buffer contents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecuteCpuRequest {
    /// WGSL source code.
    pub wgsl_source: String,
    /// Compute shader entry point name.
    pub entry_point: String,
    /// Workgroup dispatch dimensions `[x, y, z]`.
    pub workgroups: [u32; 3],
    /// Storage and read-only storage buffer bindings.
    pub bindings: Vec<BufferBinding>,
    /// Uniform buffer bindings.
    #[serde(default)]
    pub uniforms: Vec<BufferBinding>,
}

/// CPU execution response (`shader.execute.cpu`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecuteCpuResponse {
    /// Modified buffer contents after execution.
    pub bindings: Vec<BufferBinding>,
    /// Execution wall-clock time in nanoseconds.
    pub execution_time_ns: u64,
}

/// Shader validation request (`shader.validate`).
///
/// Execute a shader on CPU and compare results against expected values
/// with per-element tolerances. Used for automated correctness testing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidateRequest {
    /// WGSL source code.
    pub wgsl_source: String,
    /// Compute shader entry point name.
    pub entry_point: String,
    /// Workgroup dispatch dimensions `[x, y, z]`.
    pub workgroups: [u32; 3],
    /// Input buffer bindings (storage + read-only storage).
    pub bindings: Vec<BufferBinding>,
    /// Uniform buffer bindings.
    #[serde(default)]
    pub uniforms: Vec<BufferBinding>,
    /// Expected output values with tolerances.
    pub expected: Vec<ExpectedBinding>,
}

/// Shader validation response (`shader.validate`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidateResponse {
    /// Whether all expected values matched within tolerance.
    pub passed: bool,
    /// Per-element mismatches (empty if `passed` is true).
    #[serde(default)]
    pub mismatches: Vec<ValidationMismatch>,
}

/// A GPU buffer binding for CPU execution requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferBinding {
    /// Bind group index.
    pub group: u32,
    /// Binding index within the group.
    pub binding: u32,
    /// Buffer contents as base64-encoded bytes.
    pub data: String,
    /// Buffer usage: `"storage"`, `"storage_read"`, or `"uniform"`.
    pub usage: String,
}

/// Expected output for shader validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedBinding {
    /// Bind group index.
    pub group: u32,
    /// Binding index within the group.
    pub binding: u32,
    /// Expected buffer contents as base64-encoded bytes.
    pub data: String,
    /// Tolerance for comparison.
    pub tolerance: ValidationTolerance,
}

/// Per-element tolerance for shader validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationTolerance {
    /// Absolute tolerance (element-wise).
    pub abs: f64,
    /// Relative tolerance (element-wise).
    pub rel: f64,
}

/// A single element mismatch in shader validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMismatch {
    /// Bind group of the mismatched buffer.
    pub group: u32,
    /// Binding index of the mismatched buffer.
    pub binding: u32,
    /// Element index within the buffer.
    pub index: usize,
    /// Actual value produced by the shader.
    pub got: f64,
    /// Expected value.
    pub expected: f64,
    /// Absolute error `|got - expected|`.
    pub abs_error: f64,
    /// Relative error `|got - expected| / |expected|`.
    pub rel_error: f64,
}

/// Adapter descriptor for IPC compile requests.
///
/// Carries adapter identification to the shader compiler primal so it can determine the correct
/// ISA compilation target. `barraCuda` does not embed per-generation ISA knowledge
/// — the discovered compiler owns the adapter-to-ISA mapping.
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
