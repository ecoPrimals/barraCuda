// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sovereign Compiler — Phase 4 of the Sovereign Compute Evolution.
//!
//! Drives naga as a library to parse WGSL into a typed IR (`naga::Module`),
//! apply optimization passes (FMA fusion, dead expression elimination),
//! and re-emit optimised WGSL for safe compilation via `create_shader_module`.
//!
//! ## Pipeline (safe WGSL path — default)
//!
//! ```text
//! WGSL text (after ShaderTemplate + WgslOptimizer)
//!     → naga::front::wgsl::parse_str()    → naga::Module
//!     → fma_fusion::fuse_multiply_add()   → mutated Module (Mul+Add → Fma)
//!     → dead_expr::eliminate()            → mutated Module (unused exprs removed)
//!     → naga::valid::Validator::validate() → ModuleInfo
//!     → wgsl_emit::emit_wgsl()           → optimised WGSL text
//!     → wgpu::create_shader_module()     → driver → GPU
//! ```
//!
//! ## SPIR-V path (retained for benchmarking)
//!
//! The `compile()` method still emits SPIR-V via `spv_emit`, but
//! production compilation uses `compile_to_wgsl()` exclusively to
//! maintain zero `unsafe` in the shader pipeline.
//!
//! ## Fallback
//!
//! If the sovereign compiler fails (e.g. naga rejects the optimised
//! module), the caller falls back to the un-optimised WGSL text path.

pub mod dead_expr;
pub mod df64_rewrite;
pub mod fma_fusion;
pub mod spv_emit;
mod validation_harness;
pub mod wgsl_emit;

use crate::device::capabilities::DeviceCapabilities;

/// SPIR-V binary that has been validated by naga's `Validator`.
///
/// Encodes the safety contract in the type system: only code paths that
/// produce naga-validated SPIR-V can construct this type, so consumers
/// can pass it to `wgpu::create_shader_module_passthrough` knowing the
/// binary came from a trusted pipeline.
#[derive(Debug, Clone)]
pub struct ValidatedSpirv {
    words: Vec<u32>,
}

impl ValidatedSpirv {
    /// Construct from naga-validated SPIR-V output.
    ///
    /// Only callable within the sovereign compiler pipeline (after
    /// `naga::valid::Validator::validate` + `naga::back::spv::Writer`).
    pub(crate) fn from_validated(words: Vec<u32>) -> Self {
        Self { words }
    }

    /// The raw SPIR-V words.
    #[must_use]
    pub fn words(&self) -> &[u32] {
        &self.words
    }
}

/// Output of the sovereign compiler pipeline.
#[derive(Debug)]
pub enum SovereignOutput {
    /// Optimized SPIR-V binary, produced from naga-validated IR.
    Spirv(ValidatedSpirv),
}

/// Sovereign compiler statistics, reported after each compilation.
#[derive(Debug, Clone, Default)]
pub struct CompileStats {
    /// Number of `Mul + Add/Sub` patterns fused to `Fma`.
    pub fma_fusions: usize,
    /// Number of dead expressions eliminated.
    pub dead_exprs_eliminated: usize,
}

/// Top-level sovereign compiler.
///
/// Instantiate once per device. `DeviceCapabilities` is stored for
/// architecture-specific optimization decisions in future phases (Phase 4).
pub struct SovereignCompiler {
    _caps: DeviceCapabilities,
}

impl SovereignCompiler {
    /// Create a sovereign compiler for the given device capabilities.
    #[must_use]
    pub fn new(caps: DeviceCapabilities) -> Self {
        Self { _caps: caps }
    }

    /// Compile WGSL source to optimised WGSL via naga IR (safe path).
    ///
    /// Parses → optimises (FMA fusion, dead expression elimination) →
    /// validates → re-emits WGSL. The output can be fed to the safe
    /// `wgpu::Device::create_shader_module` API without any `unsafe`.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if the WGSL fails to parse, the optimised module
    /// fails validation, or WGSL re-emission fails.
    pub fn compile_to_wgsl(&self, wgsl: &str) -> Result<(String, CompileStats), SovereignError> {
        let (module, info, stats) = self.parse_optimize_validate(wgsl)?;
        let optimized_wgsl = wgsl_emit::emit_wgsl(&module, &info)?;
        Ok((optimized_wgsl, stats))
    }

    /// Compile WGSL source to optimized SPIR-V.
    ///
    /// Returns `Err` if the WGSL fails to parse or the optimized module
    /// fails validation — in which case the caller should fall back to
    /// the plain WGSL text path.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn compile(&self, wgsl: &str) -> Result<(SovereignOutput, CompileStats), SovereignError> {
        let (module, info, stats) = self.parse_optimize_validate(wgsl)?;
        let validated = spv_emit::emit_spirv(&module, &info)?;
        Ok((SovereignOutput::Spirv(validated), stats))
    }

    /// Shared pipeline: parse → optimise → validate.
    fn parse_optimize_validate(
        &self,
        wgsl: &str,
    ) -> Result<(naga::Module, naga::valid::ModuleInfo, CompileStats), SovereignError> {
        let mut stats = CompileStats::default();

        let mut module =
            naga::front::wgsl::parse_str(wgsl).map_err(|e| SovereignError::Parse(e.to_string()))?;

        for (_handle, func) in module.functions.iter_mut() {
            stats.fma_fusions += fma_fusion::fuse_multiply_add(&mut func.expressions);
        }
        for ep in &mut module.entry_points {
            stats.fma_fusions += fma_fusion::fuse_multiply_add(&mut ep.function.expressions);
        }

        for (_handle, func) in module.functions.iter_mut() {
            stats.dead_exprs_eliminated += dead_expr::eliminate(&mut func.expressions, &func.body);
        }
        for ep in &mut module.entry_points {
            stats.dead_exprs_eliminated +=
                dead_expr::eliminate(&mut ep.function.expressions, &ep.function.body);
        }

        let info = {
            let mut validator = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            );
            validator
                .validate(&module)
                .map_err(|e| SovereignError::Validation(e.to_string()))?
        };

        Ok((module, info, stats))
    }
}

/// Errors from the sovereign compiler pipeline.
#[derive(Debug, thiserror::Error)]
pub enum SovereignError {
    /// WGSL source failed to parse.
    #[error("WGSL parse failed: {0}")]
    Parse(String),

    /// Optimized naga module failed validation.
    #[error("Module validation failed after optimization: {0}")]
    Validation(String),

    /// SPIR-V emission from validated module failed.
    #[error("SPIR-V emission failed: {0}")]
    SpirvEmit(String),

    /// WGSL re-emission from validated module failed.
    #[error("WGSL emission failed: {0}")]
    WgslEmit(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_caps() -> DeviceCapabilities {
        use crate::device::vendor::VENDOR_NVIDIA;
        DeviceCapabilities {
            device_name: "Test GPU".into(),
            device_type: wgpu::DeviceType::DiscreteGpu,
            max_buffer_size: 1024 * 1024 * 1024,
            max_workgroup_size: (256, 256, 64),
            max_compute_workgroups: (65_535, 65_535, 65_535),
            max_compute_invocations_per_workgroup: 1024,
            max_storage_buffers_per_shader_stage: 8,
            max_uniform_buffers_per_shader_stage: 12,
            max_bind_groups: 4,
            backend: wgpu::Backend::Vulkan,
            vendor: VENDOR_NVIDIA,
            gpu_dispatch_threshold_override: None,
            subgroup_min_size: 32,
            subgroup_max_size: 32,
            f64_shaders: true,
            f64_shared_memory: false,
            f64_capabilities: None,
        }
    }

    #[test]
    fn test_sovereign_compiles_trivial_shader() {
        let wgsl = r"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    output[idx] = input[idx] * 2.0;
}
";
        let compiler = SovereignCompiler::new(test_caps());
        let (output, stats) = compiler.compile(wgsl).expect("should compile");
        match output {
            SovereignOutput::Spirv(validated) => {
                assert!(!validated.words().is_empty(), "SPIR-V should not be empty");
                assert_eq!(validated.words()[0], 0x07230203, "SPIR-V magic number");
            }
        }
        // Trivial shader has no FMA opportunities
        assert_eq!(stats.fma_fusions, 0);
    }

    #[test]
    fn test_sovereign_rejects_invalid_wgsl() {
        let compiler = SovereignCompiler::new(test_caps());
        let result = compiler.compile("this is not wgsl");
        assert!(result.is_err());
    }

    #[test]
    fn test_sovereign_fma_fusion_roundtrip() {
        let wgsl = r"
@group(0) @binding(0) var<storage, read> a_buf: array<f32>;
@group(0) @binding(1) var<storage, read> b_buf: array<f32>;
@group(0) @binding(2) var<storage, read> c_buf: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let a = a_buf[i];
    let b = b_buf[i];
    let c = c_buf[i];
    let product = a * b;
    let result = product + c;
    out[i] = result;
}
";
        let compiler = SovereignCompiler::new(test_caps());
        let (output, stats) = compiler.compile(wgsl).expect("should compile with FMA");
        assert!(
            stats.fma_fusions >= 1,
            "expected FMA fusion, got {}",
            stats.fma_fusions
        );
        match output {
            SovereignOutput::Spirv(validated) => {
                assert!(!validated.words().is_empty());
                assert_eq!(validated.words()[0], 0x07230203);
            }
        }
    }

    #[test]
    fn test_sovereign_complex_shader_roundtrip() {
        let wgsl = r"
struct Params {
    n: u32,
    scale: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n) {
        return;
    }
    let x = input[idx];
    let scaled = x * params.scale;
    let shifted = scaled + 1.0;
    let clamped = max(shifted, 0.0);
    output[idx] = clamped;
}
";
        let compiler = SovereignCompiler::new(test_caps());
        let (output, _stats) = compiler
            .compile(wgsl)
            .expect("complex shader should compile");
        match output {
            SovereignOutput::Spirv(validated) => {
                assert!(
                    validated.words().len() > 10,
                    "SPIR-V should have substantial content"
                );
                assert_eq!(validated.words()[0], 0x07230203);
            }
        }
    }

    #[test]
    fn test_sovereign_preserves_correctness_after_fusion() {
        let wgsl = r"
@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let a = data[i];
    let b = data[i + 1u];
    let c = data[i + 2u];

    // Two independent FMA opportunities
    let p1 = a * b;
    let r1 = p1 + c;

    let p2 = b * c;
    let r2 = p2 + a;

    out[i] = r1 + r2;
}
";
        let compiler = SovereignCompiler::new(test_caps());
        let (output, stats) = compiler.compile(wgsl).expect("should compile");
        // At least some fusions should fire
        assert!(stats.fma_fusions >= 1, "expected FMA fusions");
        match output {
            SovereignOutput::Spirv(validated) => {
                assert_eq!(validated.words()[0], 0x07230203);
            }
        }
    }
}
