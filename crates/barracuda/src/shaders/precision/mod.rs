// SPDX-License-Identifier: AGPL-3.0-or-later
//! Generic Precision Shader System
//!
//! Provides compile-time and runtime shader generation for any precision type.
//! ONE template → shaders for f16, f32, f64, and CPU implementations.

pub mod compiler;
pub mod cpu;
mod math_f64;
pub mod polyfill;
mod templates;

// Re-export public downcast/compiler API for external callers
pub use compiler::{
    downcast_f64_to_df64, downcast_f64_to_f16, downcast_f64_to_f32,
    downcast_f64_to_f32_with_transcendentals,
};

use templates::{
    TEMPLATE_DOT_PRODUCT, TEMPLATE_ELEMENTWISE_ABS, TEMPLATE_ELEMENTWISE_ADD,
    TEMPLATE_ELEMENTWISE_CLAMP, TEMPLATE_ELEMENTWISE_FMA, TEMPLATE_ELEMENTWISE_MUL,
    TEMPLATE_ELEMENTWISE_NEG, TEMPLATE_ELEMENTWISE_SUB, TEMPLATE_MAE_LOSS, TEMPLATE_MSE_LOSS,
    TEMPLATE_REDUCE_MEAN, TEMPLATE_REDUCE_SUM, TEMPLATE_SAXPY,
};

/// Supported precision types.
///
/// Math is universal — precision is a silicon detail. The same algorithm runs
/// at every precision; the compilation pipeline (`compile_shader_universal`)
/// handles type specialization, polyfill injection, and driver patching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Precision {
    /// 16-bit float (half precision) — inference, 2× memory bandwidth
    F16,
    /// 32-bit float (single precision) — default, widely supported
    F32,
    /// 64-bit float (double precision) — scientific computing
    F64,
    /// Double-float f32-pair (~48-bit mantissa, ~14 decimal digits) —
    /// unleashes FP32 cores for f64-class work. 9.9× throughput vs native
    /// f64 on consumer GPUs.
    Df64,
}

impl Precision {
    /// WGSL scalar type name.
    pub fn scalar(&self) -> &'static str {
        match self {
            Precision::F16 => "f16",
            Precision::F32 => "f32",
            Precision::F64 => "f64",
            Precision::Df64 => "vec2<f32>",
        }
    }

    /// WGSL vec2 type name (or scalar for f64 which lacks native vec support).
    pub fn vec2(&self) -> &'static str {
        match self {
            Precision::F16 => "vec2<f16>",
            Precision::F32 => "vec2<f32>",
            Precision::F64 => "f64",
            Precision::Df64 => "vec2<f32>",
        }
    }

    /// WGSL vec4 type name (or scalar for f64/df64).
    pub fn vec4(&self) -> &'static str {
        match self {
            Precision::F16 => "vec4<f16>",
            Precision::F32 => "vec4<f32>",
            Precision::F64 => "f64",
            Precision::Df64 => "vec2<f32>",
        }
    }

    /// Whether this precision supports vectorized operations (vec4).
    pub fn has_vec4(&self) -> bool {
        matches!(self, Precision::F16 | Precision::F32)
    }

    /// Bytes per element.
    pub fn bytes_per_element(&self) -> usize {
        match self {
            Precision::F16 => 2,
            Precision::F32 => 4,
            Precision::F64 => 8,
            Precision::Df64 => 8,
        }
    }

    /// Required wgpu feature for this precision.
    pub fn required_feature(&self) -> Option<wgpu::Features> {
        match self {
            Precision::F16 => Some(wgpu::Features::SHADER_F16),
            Precision::F32 => None,
            Precision::F64 => Some(wgpu::Features::SHADER_F64),
            Precision::Df64 => None,
        }
    }

    /// Whether this is an f64-class precision (native f64 or df64 emulation).
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
    /// For DF64: routes to df64_add/df64_mul/etc from the DF64 core library.
    ///
    /// All preambles provide identity `op_pack`/`op_unpack` for uniform
    /// array access patterns. DF64 uses these for `vec2<f32>` ↔ `Df64`
    /// conversion; other precisions are identity (compiler eliminates them).
    pub fn op_preamble(&self) -> &'static str {
        match self {
            Precision::F32 => OP_PREAMBLE_F32,
            Precision::F64 => OP_PREAMBLE_F64,
            Precision::Df64 => OP_PREAMBLE_DF64,
            Precision::F16 => OP_PREAMBLE_F16,
        }
    }
}

/// f32 operation preamble — trivial wrappers, compiler inlines everything.
const OP_PREAMBLE_F32: &str = r#"
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
"#;

/// f64 operation preamble — same structure, f64 types.
const OP_PREAMBLE_F64: &str = r#"
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
"#;

/// DF64 operation preamble — routes to df64_core library functions.
/// Requires df64_core.wgsl + df64_transcendentals.wgsl prepended.
const OP_PREAMBLE_DF64: &str = r#"
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
"#;

/// f16 operation preamble — trivial wrappers.
const OP_PREAMBLE_F16: &str = r#"
// Universal operation preamble — f16 precision
alias Scalar = f16;
fn op_add(a: f16, b: f16) -> f16 { return a + b; }
fn op_sub(a: f16, b: f16) -> f16 { return a - b; }
fn op_mul(a: f16, b: f16) -> f16 { return a * b; }
fn op_div(a: f16, b: f16) -> f16 { return a / b; }
fn op_neg(a: f16) -> f16 { return -a; }
fn op_abs(a: f16) -> f16 { return abs(a); }
fn op_max(a: f16, b: f16) -> f16 { return max(a, b); }
fn op_min(a: f16, b: f16) -> f16 { return min(a, b); }
fn op_gt(a: f16, b: f16) -> bool { return a > b; }
fn op_lt(a: f16, b: f16) -> bool { return a < b; }
fn op_ge(a: f16, b: f16) -> bool { return a >= b; }
fn op_le(a: f16, b: f16) -> bool { return a <= b; }
fn op_from_f32(v: f32) -> f16 { return f16(v); }
fn op_zero() -> f16 { return f16(0.0); }
fn op_one() -> f16 { return f16(1.0); }
fn op_pack(v: f16) -> f16 { return v; }
fn op_unpack(v: f16) -> f16 { return v; }
"#;

/// Inject DF64 pack/unpack helpers for array load/store patterns.
///
/// Converts:
/// - `let x: Df64 = arr[i]` → `let x: Df64 = Df64(arr[i].x, arr[i].y)`
/// - Adds pack helper: `fn df64_pack(v: Df64) -> vec2<f32> { return vec2<f32>(v.hi, v.lo); }`
///
/// This is injected into the shader source after the DF64 core library.
pub const DF64_PACK_UNPACK: &str = r#"
fn df64_pack(v: Df64) -> vec2<f32> { return vec2<f32>(v.hi, v.lo); }
fn df64_unpack(v: vec2<f32>) -> Df64 { return Df64(v.x, v.y); }
"#;

/// Shader template with precision placeholders.
pub struct ShaderTemplate {
    template: &'static str,
}

impl ShaderTemplate {
    /// Create a new shader template.
    pub const fn new(template: &'static str) -> Self {
        Self { template }
    }

    /// Render the template for the given precision.
    pub fn render(&self, precision: Precision) -> String {
        compiler::expand_template(self.template, precision)
    }

    /// Generate elementwise-add shader for the given precision.
    pub fn elementwise_add(precision: Precision) -> String {
        Self::new(TEMPLATE_ELEMENTWISE_ADD).render(precision)
    }

    /// Generate elementwise-mul shader for the given precision.
    pub fn elementwise_mul(precision: Precision) -> String {
        Self::new(TEMPLATE_ELEMENTWISE_MUL).render(precision)
    }

    /// Generate FMA (fused multiply-add) shader for the given precision.
    pub fn elementwise_fma(precision: Precision) -> String {
        Self::new(TEMPLATE_ELEMENTWISE_FMA).render(precision)
    }

    /// Generate dot-product shader for the given precision.
    pub fn dot_product(precision: Precision) -> String {
        Self::new(TEMPLATE_DOT_PRODUCT).render(precision)
    }

    /// Generate elementwise-sub shader for the given precision.
    pub fn elementwise_sub(precision: Precision) -> String {
        Self::new(TEMPLATE_ELEMENTWISE_SUB).render(precision)
    }

    /// Generate elementwise-abs shader for the given precision.
    pub fn elementwise_abs(precision: Precision) -> String {
        Self::new(TEMPLATE_ELEMENTWISE_ABS).render(precision)
    }

    /// Generate elementwise-neg shader for the given precision.
    pub fn elementwise_neg(precision: Precision) -> String {
        Self::new(TEMPLATE_ELEMENTWISE_NEG).render(precision)
    }

    /// Generate elementwise-clamp shader for the given precision.
    pub fn elementwise_clamp(precision: Precision) -> String {
        Self::new(TEMPLATE_ELEMENTWISE_CLAMP).render(precision)
    }

    /// Generate reduce-sum shader for the given precision.
    pub fn reduce_sum(precision: Precision) -> String {
        Self::new(TEMPLATE_REDUCE_SUM).render(precision)
    }

    /// Generate reduce-mean shader for the given precision.
    pub fn reduce_mean(precision: Precision) -> String {
        Self::new(TEMPLATE_REDUCE_MEAN).render(precision)
    }

    /// Generate MSE loss shader for the given precision.
    pub fn mse_loss(precision: Precision) -> String {
        Self::new(TEMPLATE_MSE_LOSS).render(precision)
    }

    /// Generate MAE loss shader for the given precision.
    pub fn mae_loss(precision: Precision) -> String {
        Self::new(TEMPLATE_MAE_LOSS).render(precision)
    }

    /// Generate SAXPY (y = αx + y) shader for the given precision.
    pub fn saxpy(precision: Precision) -> String {
        Self::new(TEMPLATE_SAXPY).render(precision)
    }

    /// Full math_f64 polyfill preamble for shaders.
    pub fn math_f64_preamble() -> String {
        polyfill::math_f64_preamble()
    }

    /// Prepend math_f64 preamble to a shader body.
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
    pub fn for_device(shader_body: &str, device: &crate::device::WgpuDevice) -> String {
        Self::for_driver_auto(shader_body, device.needs_f64_exp_log_workaround())
    }

    /// Alias for `for_device`; patches shader for the device.
    pub fn for_device_auto(shader_body: &str, device: &crate::device::WgpuDevice) -> String {
        Self::for_driver_auto(shader_body, device.needs_f64_exp_log_workaround())
    }

    /// Patch a WGSL shader's `WARP_SIZE` constant and `@workgroup_size` annotation.
    ///
    /// Replaces `const WARP_SIZE: u32 = 32u;` with the given `wave_size` and
    /// adjusts `@workgroup_size(32, 1, 1)` accordingly. Used to specialise the
    /// single-dispatch Jacobi eigensolve for AMD RDNA2/3 (wave_size=64) vs
    /// NVIDIA warp (wave_size=32) at shader-compilation time.
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

    /// Replace legacy fossil_f64 calls with native WGSL.
    pub fn substitute_fossil_f64(shader_body: &str) -> String {
        polyfill::substitute_fossil_f64(shader_body)
    }

    /// Patch shader for driver (exp/log workaround, f64 polyfills, ILP optimize).
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

    /// Inject only the math_f64 functions used by the shader.
    pub fn with_math_f64_auto(shader_body: &str) -> String {
        use math_f64::F64_FUNCTION_ORDER;
        let mut used_functions = Vec::new();
        for func_name in F64_FUNCTION_ORDER {
            let call_pattern = format!("{func_name}(");
            let call_pattern_space = format!("{func_name} (");
            if shader_body.contains(&call_pattern) || shader_body.contains(&call_pattern_space) {
                used_functions.push(*func_name);
            }
        }
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

    /// Generate math_f64 preamble for a subset of functions.
    pub fn math_f64_subset(functions: &[&str]) -> String {
        polyfill::math_f64_subset(functions)
    }

    /// Returns true if shader defines the given function.
    pub fn shader_defines_function(shader_body: &str, func_name: &str) -> bool {
        polyfill::shader_defines_function(shader_body, func_name)
    }

    /// Returns true if shader defines the given module-level variable.
    pub fn shader_defines_module_var(shader_body: &str, var_name: &str) -> bool {
        polyfill::shader_defines_module_var(shader_body, var_name)
    }

    /// Inject f64 polyfills into shader (no driver-specific patching).
    pub fn with_math_f64_safe(shader_body: &str) -> String {
        polyfill::inject_f64_polyfills(shader_body, None)
    }

    /// Alias for `with_math_f64_safe`; injects f64 polyfills.
    pub fn with_math_f64_auto_safe(shader_body: &str) -> String {
        polyfill::inject_f64_polyfills(shader_body, None)
    }
}

#[cfg(test)]
#[path = "precision_tests.rs"]
mod tests;
