// SPDX-License-Identifier: AGPL-3.0-or-later
//! Generic Precision Shader System
//!
//! Provides compile-time and runtime shader generation for any precision type.
//! ONE template → shaders for f16, f32, f64, and CPU implementations.

pub mod compiler;
pub mod cpu;
pub mod eps;
mod math_f64;
pub mod polyfill;

// Re-export public downcast/compiler API for external callers
pub use compiler::{
    downcast_f64_to_df64, downcast_f64_to_f32, downcast_f64_to_f32_with_transcendentals,
};

/// Hardware precision tiers.
///
/// Math is written in f64-canonical WGSL — pure math, conceptually infinite
/// precision. The compilation pipeline then targets one of three hardware
/// tiers. This maps directly to coralReef's `Fp64Strategy`:
///
/// | Tier | coralReef strategy | Mantissa | Throughput (RTX 3090) |
/// |------|-------------------|----------|----------------------|
/// | F32  | `F32Only`         | 24 bits  | ~29,770 GFLOPS       |
/// | Df64 | `DoubleFloat`     | ~48 bits | ~7,000–10,000 GFLOPS |
/// | F64  | `Native`          | 52 bits  | ~556 GFLOPS          |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Precision {
    /// 32-bit float (single precision) — default, broadly supported.
    /// coralReef: `Fp64Strategy::F32Only`.
    F32,
    /// 64-bit float (double precision) — scientific computing, gold standard.
    /// coralReef: `Fp64Strategy::Native`.
    F64,
    /// Double-float f32-pair (~48-bit mantissa, ~14 decimal digits) —
    /// unleashes FP32 cores for f64-class work. 12–18× throughput vs native
    /// f64 on consumer GPUs. The "fp48" sweet spot.
    /// coralReef: `Fp64Strategy::DoubleFloat`.
    Df64,
}

impl Precision {
    /// WGSL scalar type name.
    #[must_use]
    pub fn scalar(&self) -> &'static str {
        match self {
            Precision::F32 => "f32",
            Precision::F64 => "f64",
            Precision::Df64 => "vec2<f32>",
        }
    }

    /// WGSL vec2 type name (or scalar for f64 which lacks native vec support).
    #[must_use]
    pub fn vec2(&self) -> &'static str {
        match self {
            Precision::F32 => "vec2<f32>",
            Precision::F64 => "f64",
            Precision::Df64 => "vec2<f32>",
        }
    }

    /// WGSL vec4 type name (or scalar for f64/df64).
    #[must_use]
    pub fn vec4(&self) -> &'static str {
        match self {
            Precision::F32 => "vec4<f32>",
            Precision::F64 => "f64",
            Precision::Df64 => "vec2<f32>",
        }
    }

    /// Whether this precision supports vectorized operations (vec4).
    #[must_use]
    pub fn has_vec4(&self) -> bool {
        matches!(self, Precision::F32)
    }

    /// Bytes per element.
    #[must_use]
    pub fn bytes_per_element(&self) -> usize {
        match self {
            Precision::F32 => 4,
            Precision::F64 => 8,
            Precision::Df64 => 8,
        }
    }

    /// Required wgpu feature for this precision.
    #[must_use]
    pub fn required_feature(&self) -> Option<wgpu::Features> {
        match self {
            Precision::F32 => None,
            Precision::F64 => Some(wgpu::Features::SHADER_F64),
            Precision::Df64 => None,
        }
    }

    /// Whether this is an f64-class precision (native f64 or df64 emulation).
    #[must_use]
    pub fn is_f64_class(&self) -> bool {
        matches!(self, Precision::F64 | Precision::Df64)
    }

    /// Generate the operation preamble for this precision.
    ///
    /// The preamble defines abstract operations (`op_add`, `op_mul`, etc.)
    /// whose implementation varies per precision. Shaders written against
    /// these ops are truly universal — math is the same, precision is silicon.
    ///
    /// For f32/f64: trivial inline wrappers around native operators.
    /// For DF64: routes to `df64_add/df64_mul/etc` from the DF64 core library.
    ///
    /// All preambles provide identity `op_pack`/`op_unpack` for uniform
    /// array access patterns. DF64 uses these for `vec2<f32>` ↔ `Df64`
    /// conversion; other precisions are identity (compiler eliminates them).
    #[must_use]
    pub fn op_preamble(&self) -> &'static str {
        match self {
            Precision::F32 => OP_PREAMBLE_F32,
            Precision::F64 => OP_PREAMBLE_F64,
            Precision::Df64 => OP_PREAMBLE_DF64,
        }
    }
}

/// f32 operation preamble — trivial wrappers, compiler inlines everything.
const OP_PREAMBLE_F32: &str = r"
// Universal operation preamble — f32 precision
alias Scalar = f32;
fn op_add(a: f32, b: f32) -> f32 { return a + b; }
fn op_sub(a: f32, b: f32) -> f32 { return a - b; }
fn op_mul(a: f32, b: f32) -> f32 { return a * b; }
fn op_div(a: f32, b: f32) -> f32 { return a / b; }
fn op_neg(a: f32) -> f32 { return -a; }
fn op_abs(a: f32) -> f32 { return abs(a); }
fn op_max(a: f32, b: f32) -> f32 { return max(a, b); }
fn op_min(a: f32, b: f32) -> f32 { return min(a, b); }
fn op_gt(a: f32, b: f32) -> bool { return a > b; }
fn op_lt(a: f32, b: f32) -> bool { return a < b; }
fn op_ge(a: f32, b: f32) -> bool { return a >= b; }
fn op_le(a: f32, b: f32) -> bool { return a <= b; }
fn op_from_f32(v: f32) -> f32 { return v; }
fn op_zero() -> f32 { return 0.0; }
fn op_one() -> f32 { return 1.0; }
fn op_pack(v: f32) -> f32 { return v; }
fn op_unpack(v: f32) -> f32 { return v; }
";

/// f64 operation preamble — same structure, f64 types.
const OP_PREAMBLE_F64: &str = r"
// Universal operation preamble — f64 precision
alias Scalar = f64;
fn op_add(a: f64, b: f64) -> f64 { return a + b; }
fn op_sub(a: f64, b: f64) -> f64 { return a - b; }
fn op_mul(a: f64, b: f64) -> f64 { return a * b; }
fn op_div(a: f64, b: f64) -> f64 { return a / b; }
fn op_neg(a: f64) -> f64 { return -a; }
fn op_abs(a: f64) -> f64 { return abs(a); }
fn op_max(a: f64, b: f64) -> f64 { return max(a, b); }
fn op_min(a: f64, b: f64) -> f64 { return min(a, b); }
fn op_gt(a: f64, b: f64) -> bool { return a > b; }
fn op_lt(a: f64, b: f64) -> bool { return a < b; }
fn op_ge(a: f64, b: f64) -> bool { return a >= b; }
fn op_le(a: f64, b: f64) -> bool { return a <= b; }
fn op_from_f32(v: f32) -> f64 { return f64(v); }
fn op_zero() -> f64 { return f64(0.0); }
fn op_one() -> f64 { return f64(1.0); }
fn op_pack(v: f64) -> f64 { return v; }
fn op_unpack(v: f64) -> f64 { return v; }
";

/// DF64 operation preamble — routes to `df64_core` library functions.
/// Requires `df64_core.wgsl` + `df64_transcendentals.wgsl` prepended.
const OP_PREAMBLE_DF64: &str = r"
// Universal operation preamble — DF64 precision (f32-pair, ~48-bit mantissa)
alias Scalar = Df64;
alias StorageType = vec2<f32>;
fn op_add(a: Df64, b: Df64) -> Df64 { return df64_add(a, b); }
fn op_sub(a: Df64, b: Df64) -> Df64 { return df64_sub(a, b); }
fn op_mul(a: Df64, b: Df64) -> Df64 { return df64_mul(a, b); }
fn op_div(a: Df64, b: Df64) -> Df64 { return df64_div(a, b); }
fn op_neg(a: Df64) -> Df64 { return df64_neg(a); }
fn op_abs(a: Df64) -> Df64 { return df64_abs(a); }
fn op_max(a: Df64, b: Df64) -> Df64 { if df64_gt(a, b) { return a; } return b; }
fn op_min(a: Df64, b: Df64) -> Df64 { if df64_lt(a, b) { return a; } return b; }
fn op_gt(a: Df64, b: Df64) -> bool { return df64_gt(a, b); }
fn op_lt(a: Df64, b: Df64) -> bool { return df64_lt(a, b); }
fn op_ge(a: Df64, b: Df64) -> bool { return !df64_lt(a, b); }
fn op_le(a: Df64, b: Df64) -> bool { return !df64_gt(a, b); }
fn op_from_f32(v: f32) -> Df64 { return df64_from_f32(v); }
fn op_zero() -> Df64 { return df64_zero(); }
fn op_one() -> Df64 { return df64_from_f32(1.0); }
fn op_pack(v: Df64) -> vec2<f32> { return vec2<f32>(v.hi, v.lo); }
fn op_unpack(v: vec2<f32>) -> Df64 { return Df64(v.x, v.y); }
";

/// Inject DF64 pack/unpack helpers for array load/store patterns.
///
/// Converts:
/// - `let x: Df64 = arr[i]` → `let x: Df64 = Df64(arr[i].x, arr[i].y)`
/// - Adds pack helper: `fn df64_pack(v: Df64) -> vec2<f32> { return vec2<f32>(v.hi, v.lo); }`
///
/// This is injected into the shader source after the DF64 core library.
pub const DF64_PACK_UNPACK: &str = r"
fn df64_pack(v: Df64) -> vec2<f32> { return vec2<f32>(v.hi, v.lo); }
fn df64_unpack(v: vec2<f32>) -> Df64 { return Df64(v.x, v.y); }
";

/// Shader preparation utilities for f64-canonical WGSL.
///
/// Provides driver-aware patching, polyfill injection, and ILP optimization.
/// Math is written once in f64; these utilities prepare it for hardware dispatch.
///
/// Transitional: driver patching and polyfill injection exist because the
/// sovereign dispatch path (coralReef → coralDriver) is not yet integrated.
/// When coralReef handles compilation end-to-end, these reduce to thin IPC calls.
pub struct ShaderTemplate;

impl ShaderTemplate {
    /// Full `math_f64` polyfill preamble for shaders.
    #[must_use]
    pub fn math_f64_preamble() -> String {
        polyfill::math_f64_preamble()
    }

    /// Prepend `math_f64` preamble to a shader body.
    #[must_use]
    pub fn with_math_f64(shader_body: &str) -> String {
        format!(
            "{}\n\n// User shader:\n{}",
            Self::math_f64_preamble(),
            shader_body
        )
    }

    /// Generate f64 shader with driver-aware exp/log patching (synchronous).
    ///
    /// Uses `needs_f64_exp_log_workaround()` (name-based heuristic). For definitive
    /// detection, async callers should use `device.probe_f64_exp_capable().await` and
    /// pass `!capable` as the workaround flag — probe overrides heuristic when run.
    #[must_use]
    pub fn for_device(shader_body: &str, device: &crate::device::WgpuDevice) -> String {
        Self::for_driver_auto(shader_body, device.needs_f64_exp_log_workaround())
    }

    /// Alias for `for_device`; patches shader for the device.
    #[must_use]
    pub fn for_device_auto(shader_body: &str, device: &crate::device::WgpuDevice) -> String {
        Self::for_driver_auto(shader_body, device.needs_f64_exp_log_workaround())
    }

    /// Patch a WGSL shader's `WARP_SIZE` constant and `@workgroup_size` annotation.
    ///
    /// Replaces `const WARP_SIZE: u32 = 32u;` with the given `wave_size` and
    /// adjusts `@workgroup_size(32, 1, 1)` accordingly. Used to specialise the
    /// single-dispatch Jacobi eigensolve for AMD RDNA2/3 (`wave_size=64`) vs
    /// NVIDIA warp (`wave_size=32`) at shader-compilation time.
    #[must_use]
    pub fn patch_warp_size(shader_body: &str, wave_size: u32) -> String {
        shader_body
            .replace(
                "const WARP_SIZE: u32 = 32u;",
                &format!("const WARP_SIZE: u32 = {wave_size}u;"),
            )
            .replace(
                "@workgroup_size(32, 1, 1)",
                &format!("@workgroup_size({wave_size}, 1, 1)"),
            )
    }

    /// Replace legacy `fossil_f64` calls with native WGSL.
    #[must_use]
    pub fn substitute_fossil_f64(shader_body: &str) -> String {
        polyfill::substitute_fossil_f64(shader_body)
    }

    /// Patch shader for driver (exp/log workaround, f64 polyfills, ILP optimize).
    #[must_use]
    pub fn for_driver_auto(shader_body: &str, needs_exp_log_workaround: bool) -> String {
        // Strip `enable f64;` — naga handles f64 via capability flags, not directives.
        let stripped = shader_body
            .lines()
            .filter(|l| l.trim() != "enable f64;")
            .collect::<Vec<_>>()
            .join("\n");
        // Upgrade any legacy fossil calls to native WGSL builtins first.
        let substituted = polyfill::substitute_fossil_f64(&stripped);
        let patched = polyfill::apply_transcendental_workaround_with_sin_cos(
            &substituted,
            needs_exp_log_workaround,
            false,
        );
        let injected = polyfill::inject_f64_polyfills(&patched, None);
        // ILP-reorders @ilp_region blocks and unrolls @unroll_hint loops.
        // ConservativeModel is used as the latency model (safe fallback when no driver profile).
        crate::shaders::optimizer::WgslOptimizer::default().optimize(&injected)
    }

    /// Variant of `for_driver_auto` that uses the accurate `LatencyModel` from
    /// a `GpuDriverProfile` for precise ILP scheduling.
    ///
    /// Prefer this when a `GpuDriverProfile` is available at shader-compile time.
    #[must_use]
    pub fn for_driver_profile(
        shader_body: &str,
        needs_exp_log_workaround: bool,
        profile: &crate::device::capabilities::GpuDriverProfile,
    ) -> String {
        // Strip `enable f64;` — naga handles f64 via capability flags, not directives.
        let stripped = shader_body
            .lines()
            .filter(|l| l.trim() != "enable f64;")
            .collect::<Vec<_>>()
            .join("\n");
        let use_sin_cos_taylor =
            profile.needs_sin_f64_workaround() || profile.needs_cos_f64_workaround();
        let substituted = polyfill::substitute_fossil_f64(&stripped);
        let patched = polyfill::apply_transcendental_workaround_with_sin_cos(
            &substituted,
            needs_exp_log_workaround,
            use_sin_cos_taylor,
        );
        let extra_preamble = if use_sin_cos_taylor
            && (patched.contains("sin_f64_safe(") || patched.contains("cos_f64_safe("))
        {
            Some(polyfill::SIN_COS_F64_SAFE_PREAMBLE)
        } else {
            None
        };
        let injected = polyfill::inject_f64_polyfills(&patched, extra_preamble);
        crate::shaders::optimizer::WgslOptimizer::new(profile.latency_model()).optimize(&injected)
    }

    /// Inject only the `math_f64` functions used by the shader.
    #[must_use]
    pub fn with_math_f64_auto(shader_body: &str) -> String {
        use math_f64::F64_FUNCTION_ORDER;
        let mut used_functions: Vec<&str> = F64_FUNCTION_ORDER
            .iter()
            .filter(|func_name| {
                let call_pattern = format!("{func_name}(");
                let call_pattern_space = format!("{func_name} (");
                shader_body.contains(&call_pattern) || shader_body.contains(&call_pattern_space)
            })
            .copied()
            .collect();
        if shader_body.contains("round_f64") && !used_functions.contains(&"round_f64") {
            used_functions.push("round_f64");
        }
        if used_functions.is_empty() {
            return shader_body.to_string();
        }
        format!(
            "{}\n\n// User shader:\n{}",
            polyfill::math_f64_subset(&used_functions),
            shader_body
        )
    }

    /// Generate `math_f64` preamble for a subset of functions.
    #[must_use]
    pub fn math_f64_subset(functions: &[&str]) -> String {
        polyfill::math_f64_subset(functions)
    }

    /// Returns true if shader defines the given function.
    #[must_use]
    pub fn shader_defines_function(shader_body: &str, func_name: &str) -> bool {
        polyfill::shader_defines_function(shader_body, func_name)
    }

    /// Returns true if shader defines the given module-level variable.
    #[must_use]
    pub fn shader_defines_module_var(shader_body: &str, var_name: &str) -> bool {
        polyfill::shader_defines_module_var(shader_body, var_name)
    }

    /// Inject f64 polyfills into shader (no driver-specific patching).
    #[must_use]
    pub fn with_math_f64_safe(shader_body: &str) -> String {
        polyfill::inject_f64_polyfills(shader_body, None)
    }

    /// Alias for `with_math_f64_safe`; injects f64 polyfills.
    #[must_use]
    pub fn with_math_f64_auto_safe(shader_body: &str) -> String {
        polyfill::inject_f64_polyfills(shader_body, None)
    }
}

#[cfg(test)]
#[path = "precision_tests.rs"]
mod tests;
