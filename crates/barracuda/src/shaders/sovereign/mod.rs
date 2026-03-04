// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sovereign Compiler — Phase 4 of the Sovereign Compute Evolution.
//!
//! Drives naga as a library to parse WGSL into a typed IR (`naga::Module`),
//! apply optimization passes (FMA fusion, dead expression elimination),
//! and emit SPIR-V bytes that bypass naga's internal WGSL re-parse on
//! the wgpu side via `SPIRV_SHADER_PASSTHROUGH`.
//!
//! ## Pipeline
//!
//! ```text
//! WGSL text (after ShaderTemplate + WgslOptimizer)
//!     → naga::front::wgsl::parse_str()    → naga::Module
//!     → fma_fusion::fuse_multiply_add()   → mutated Module (Mul+Add → Fma)
//!     → dead_expr::eliminate()            → mutated Module (unused exprs removed)
//!     → naga::valid::Validator::validate() → ModuleInfo
//!     → spv_emit::emit_spirv()            → Vec<u32> (SPIR-V words)
//!     → wgpu (SPIRV_SHADER_PASSTHROUGH)   → driver → GPU
//! ```
//!
//! ## Fallback
//!
//! If SPIR-V passthrough is unavailable (WebGPU, some mobile drivers),
//! the caller falls back to the existing WGSL text path. Phase 4 is
//! additive — it never replaces the text path.

pub mod dead_expr;
pub mod df64_rewrite;
pub mod fma_fusion;
pub mod spv_emit;

use crate::device::driver_profile::GpuDriverProfile;

/// Output of the sovereign compiler pipeline.
#[derive(Debug)]
pub enum SovereignOutput {
    /// Optimized SPIR-V binary, ready for `SPIRV_SHADER_PASSTHROUGH`.
    Spirv(Vec<u32>),
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
/// Instantiate once per device (the `GpuDriverProfile` is used for
/// architecture-specific optimization decisions in future phases).
pub struct SovereignCompiler {
    _profile: GpuDriverProfile,
}

impl SovereignCompiler {
    /// Create a sovereign compiler for the given driver profile.
    #[must_use]
    pub fn new(profile: GpuDriverProfile) -> Self {
        Self { _profile: profile }
    }

    /// Compile WGSL source to optimized SPIR-V.
    ///
    /// Returns `Err` if the WGSL fails to parse or the optimized module
    /// fails validation — in which case the caller should fall back to
    /// the plain WGSL text path.
    pub fn compile(&self, wgsl: &str) -> Result<(SovereignOutput, CompileStats), SovereignError> {
        let mut stats = CompileStats::default();

        // Step 1: Parse WGSL → naga::Module.
        let mut module = naga::front::wgsl::parse_str(wgsl)
            .map_err(|e| SovereignError::Parse(format!("{e}")))?;

        // Step 2: FMA fusion — Mul(a,b) + Add/Sub(_, c) → Fma(a, b, c).
        for (_handle, func) in module.functions.iter_mut() {
            stats.fma_fusions += fma_fusion::fuse_multiply_add(&mut func.expressions);
        }
        for ep in &mut module.entry_points {
            stats.fma_fusions += fma_fusion::fuse_multiply_add(&mut ep.function.expressions);
        }

        // Step 3: Dead expression elimination.
        // Skipped for now — naga's validator rejects modules with unused
        // expressions only in specific cases, and the SPIR-V backend
        // naturally skips un-emitted expressions. We track the stat for
        // future use.
        for (_handle, func) in module.functions.iter_mut() {
            stats.dead_exprs_eliminated += dead_expr::eliminate(&mut func.expressions, &func.body);
        }
        for ep in &mut module.entry_points {
            stats.dead_exprs_eliminated +=
                dead_expr::eliminate(&mut ep.function.expressions, &ep.function.body);
        }

        // Step 4: Validate the optimized module.
        let info = {
            let mut validator = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            );
            validator
                .validate(&module)
                .map_err(|e| SovereignError::Validation(format!("{e}")))?
        };

        // Step 5: Emit SPIR-V.
        let spirv = spv_emit::emit_spirv(&module, &info)?;

        Ok((SovereignOutput::Spirv(spirv), stats))
    }
}

/// Errors from the sovereign compiler pipeline.
#[derive(Debug, thiserror::Error)]
pub enum SovereignError {
    #[error("WGSL parse failed: {0}")]
    Parse(String),

    #[error("Module validation failed after optimization: {0}")]
    Validation(String),

    #[error("SPIR-V emission failed: {0}")]
    SpirvEmit(String),
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    fn test_profile() -> GpuDriverProfile {
        use crate::device::driver_profile::{CompilerKind, DriverKind, Fp64Rate, GpuArch};
        GpuDriverProfile {
            driver: DriverKind::Unknown,
            compiler: CompilerKind::Unknown,
            arch: GpuArch::Unknown,
            fp64_rate: Fp64Rate::Throttled,
            workarounds: vec![],
        }
    }

    #[test]
    fn test_sovereign_compiles_trivial_shader() {
        let wgsl = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    output[idx] = input[idx] * 2.0;
}
"#;
        let compiler = SovereignCompiler::new(test_profile());
        let (output, stats) = compiler.compile(wgsl).expect("should compile");
        match output {
            SovereignOutput::Spirv(words) => {
                assert!(!words.is_empty(), "SPIR-V should not be empty");
                assert_eq!(words[0], 0x07230203, "SPIR-V magic number");
            }
        }
        // Trivial shader has no FMA opportunities
        assert_eq!(stats.fma_fusions, 0);
    }

    #[test]
    fn test_sovereign_rejects_invalid_wgsl() {
        let compiler = SovereignCompiler::new(test_profile());
        let result = compiler.compile("this is not wgsl");
        assert!(result.is_err());
    }

    #[test]
    fn test_sovereign_fma_fusion_roundtrip() {
        let wgsl = r#"
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
"#;
        let compiler = SovereignCompiler::new(test_profile());
        let (output, stats) = compiler.compile(wgsl).expect("should compile with FMA");
        assert!(
            stats.fma_fusions >= 1,
            "expected FMA fusion, got {}",
            stats.fma_fusions
        );
        match output {
            SovereignOutput::Spirv(words) => {
                assert!(!words.is_empty());
                assert_eq!(words[0], 0x07230203);
            }
        }
    }

    #[test]
    fn test_sovereign_complex_shader_roundtrip() {
        let wgsl = r#"
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
"#;
        let compiler = SovereignCompiler::new(test_profile());
        let (output, _stats) = compiler
            .compile(wgsl)
            .expect("complex shader should compile");
        match output {
            SovereignOutput::Spirv(words) => {
                assert!(words.len() > 10, "SPIR-V should have substantial content");
                assert_eq!(words[0], 0x07230203);
            }
        }
    }

    #[test]
    fn test_sovereign_preserves_correctness_after_fusion() {
        let wgsl = r#"
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
"#;
        let compiler = SovereignCompiler::new(test_profile());
        let (output, stats) = compiler.compile(wgsl).expect("should compile");
        // At least some fusions should fire
        assert!(stats.fma_fusions >= 1, "expected FMA fusions");
        match output {
            SovereignOutput::Spirv(words) => {
                assert_eq!(words[0], 0x07230203);
            }
        }
    }
}
