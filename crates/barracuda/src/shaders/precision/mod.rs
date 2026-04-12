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
mod preambles;

// Re-export public downcast/compiler API for external callers
pub use compiler::{
    downcast_f64_to_df64, downcast_f64_to_f32, downcast_f64_to_f32_with_transcendentals,
};

/// Hardware precision tiers for shader generation.
///
/// Math is written in f64-canonical WGSL — pure math, conceptually infinite
/// precision. The compilation pipeline then targets one of these hardware
/// tiers. Each tier maps to a coralReef compilation strategy.
///
/// | Tier | coralReef strategy | Mantissa | Notes |
/// |------|-------------------|----------|-------|
/// | Binary | `binary` | 1 bit | XNOR+popcount, u32 packed |
/// | Int2 | `int2` | 2 bits | Ternary {-1,0,+1}, u32 packed |
/// | Q4 | `q4_block` | 4 bits | Block-quantized Q4_0 |
/// | Q8 | `q8_block` | 8 bits | Block-quantized Q8_0 |
/// | Fp8E5M2 | `fp8_e5m2` | 2-bit mant | Gradient comm, u32 packed |
/// | Fp8E4M3 | `fp8_e4m3` | 3-bit mant | Inference, u32 packed |
/// | Bf16 | `bf16_emulated` | 7 bits | bfloat16, u32 bit-manip |
/// | F16 | `f16_fast` | 10 bits | IEEE half, native or emulated |
/// | F32 | `f32_only` | 24 bits | Universal baseline |
/// | Df64 | `double_float` | ~48 bits | f32-pair Dekker |
/// | F64 | `native` | 52 bits | Native f64 |
/// | Qf128 | `quad_float` | ~96 bits | Bailey quad-double on f32 |
/// | Df128 | `double_double_f64` | ~104 bits | Dekker double-double on f64 |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Precision {
    /// 1-bit binary: XNOR+popcount dot products, 32 values per u32.
    Binary,
    /// 2-bit ternary {-1, 0, +1}: 16 values per u32.
    Int2,
    /// 4-bit block quantized (`Q4_0`): 8 nibbles per u32 + f16 scale.
    Q4,
    /// 8-bit block quantized (`Q8_0`): 4 bytes per u32 + f16 scale.
    Q8,
    /// 8-bit float E5M2: wider range, 2-bit mantissa. 4 per u32.
    Fp8E5M2,
    /// 8-bit float E4M3: higher precision, 3-bit mantissa. 4 per u32.
    Fp8E4M3,
    /// 16-bit bfloat (Google Brain): f32 exponent range, 7-bit mantissa.
    /// Emulated via u32 bit manipulation. No wgpu feature required.
    Bf16,
    /// 16-bit float (half precision) — ML inference, screening, tensor core path.
    /// Requires `SHADER_F16` feature. Falls back to f32 pack/unpack emulation
    /// on hardware without native f16 support.
    F16,
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
    /// Quad-double on f32 (Bailey): ~96-bit mantissa from 4× f32 components.
    /// No f64 hardware required — universally available.
    /// coralReef: `Fp64Strategy::QuadFloat`.
    Qf128,
    /// Double-double on f64 (Dekker): ~104-bit mantissa from 2× f64 components.
    /// Requires `SHADER_F64`. Preferred over Qf128 on compute GPUs.
    /// coralReef: `Fp64Strategy::DoubleDoubleF64`.
    Df128,
}

impl Precision {
    /// WGSL scalar type name.
    #[must_use]
    pub fn scalar(&self) -> &'static str {
        match self {
            Self::Binary | Self::Int2 | Self::Q4 | Self::Q8 => "u32",
            Self::Fp8E5M2 | Self::Fp8E4M3 => "u32",
            Self::Bf16 => "u32",
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::Df64 => "vec2<f32>",
            Self::Qf128 => "vec4<f32>",
            Self::Df128 => "vec2<f64>",
        }
    }

    /// WGSL vec2 type name (or scalar for types that lack native vec support).
    #[must_use]
    pub fn vec2(&self) -> &'static str {
        match self {
            Self::F16 => "vec2<f16>",
            Self::F32 => "vec2<f32>",
            Self::F64 => "f64",
            Self::Df64 => "vec2<f32>",
            Self::Qf128 => "vec4<f32>",
            Self::Df128 => "vec2<f64>",
            _ => "u32",
        }
    }

    /// WGSL vec4 type name (or scalar for types without vec support).
    #[must_use]
    pub fn vec4(&self) -> &'static str {
        match self {
            Self::F16 => "vec4<f16>",
            Self::F32 => "vec4<f32>",
            Self::F64 => "f64",
            Self::Df64 => "vec2<f32>",
            Self::Qf128 => "vec4<f32>",
            Self::Df128 => "vec2<f64>",
            _ => "u32",
        }
    }

    /// Whether this precision supports vectorized operations (vec4).
    #[must_use]
    pub fn has_vec4(&self) -> bool {
        matches!(self, Self::F16 | Self::F32)
    }

    /// Bytes per element.
    #[must_use]
    pub fn bytes_per_element(&self) -> usize {
        match self {
            Self::Binary | Self::Int2 | Self::Q4 | Self::Q8 => 1,
            Self::Fp8E5M2 | Self::Fp8E4M3 => 1,
            Self::Bf16 | Self::F16 => 2,
            Self::F32 => 4,
            Self::F64 | Self::Df64 => 8,
            Self::Qf128 | Self::Df128 => 16,
        }
    }

    /// Required wgpu feature for this precision.
    #[must_use]
    pub fn required_feature(&self) -> Option<wgpu::Features> {
        match self {
            Self::F16 => Some(wgpu::Features::SHADER_F16),
            Self::F64 | Self::Df128 => Some(wgpu::Features::SHADER_F64),
            _ => None,
        }
    }

    /// Whether this is an f64-class precision (native f64 or df64 emulation).
    #[must_use]
    pub fn is_f64_class(&self) -> bool {
        matches!(self, Self::F64 | Self::Df64 | Self::Df128)
    }

    /// Whether this is a reduced-precision tier (below f32).
    #[must_use]
    pub fn is_reduced(&self) -> bool {
        matches!(
            self,
            Self::Binary
                | Self::Int2
                | Self::Q4
                | Self::Q8
                | Self::Fp8E5M2
                | Self::Fp8E4M3
                | Self::Bf16
                | Self::F16
        )
    }

    /// Whether this is an extended-precision tier (above f64).
    #[must_use]
    pub fn is_extended(&self) -> bool {
        matches!(self, Self::Qf128 | Self::Df128)
    }

    /// Whether this is a quantized integer format.
    #[must_use]
    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Binary | Self::Int2 | Self::Q4 | Self::Q8)
    }

    /// Generate the operation preamble for this precision.
    ///
    /// The preamble defines abstract operations (`op_add`, `op_mul`, etc.)
    /// whose implementation varies per precision. Shaders written against
    /// these ops are truly universal — math is the same, precision is silicon.
    ///
    /// For f16/f32/f64: trivial inline wrappers around native operators.
    /// For DF64: routes to `df64_add/df64_mul/etc` from the DF64 core library.
    /// For DF128: routes to `df128_add/df128_mul/etc` (Dekker on f64).
    /// For QF128: routes to `qf128_add/qf128_mul/etc` (Bailey quad-double on f32).
    /// For BF16/FP8: pack/unpack helpers, compute in f32, requantize.
    /// For quantized (Binary/Int2/Q4/Q8): dequantize→f32 compute→quantize
    ///   pattern, not the `op_preamble` abstraction — returns empty preamble.
    ///
    /// All preambles provide identity `op_pack`/`op_unpack` for uniform
    /// array access patterns. DF64 uses these for `vec2<f32>` ↔ `Df64`
    /// conversion; other precisions are identity (compiler eliminates them).
    #[must_use]
    pub fn op_preamble(&self) -> &'static str {
        match self {
            Self::Binary | Self::Int2 | Self::Q4 | Self::Q8 => OP_PREAMBLE_QUANTIZED,
            Self::Fp8E5M2 => OP_PREAMBLE_FP8_E5M2,
            Self::Fp8E4M3 => OP_PREAMBLE_FP8_E4M3,
            Self::Bf16 => OP_PREAMBLE_BF16,
            Self::F16 => OP_PREAMBLE_F16,
            Self::F32 => OP_PREAMBLE_F32,
            Self::F64 => OP_PREAMBLE_F64,
            Self::Df64 => OP_PREAMBLE_DF64,
            Self::Qf128 => OP_PREAMBLE_QF128,
            Self::Df128 => OP_PREAMBLE_DF128,
        }
    }
}

pub use preambles::DF64_PACK_UNPACK;
use preambles::*;

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

    /// Variant of `for_driver_auto` that uses device capabilities for
    /// precise ILP scheduling and workaround detection.
    ///
    /// Prefer this when `DeviceCapabilities` is available at shader-compile time.
    #[must_use]
    pub fn for_device_capabilities(
        shader_body: &str,
        needs_exp_log_workaround: bool,
        caps: &crate::device::capabilities::DeviceCapabilities,
    ) -> String {
        let stripped = shader_body
            .lines()
            .filter(|l| l.trim() != "enable f64;")
            .collect::<Vec<_>>()
            .join("\n");
        let use_sin_cos_taylor = caps.needs_sin_f64_workaround() || caps.needs_cos_f64_workaround();
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
        crate::shaders::optimizer::WgslOptimizer::new(caps.latency_model()).optimize(&injected)
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
