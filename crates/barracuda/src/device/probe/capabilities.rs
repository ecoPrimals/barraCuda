// SPDX-License-Identifier: AGPL-3.0-or-later
//! f64 built-in capability result type
//!
//! Which f64 WGSL built-in functions are natively supported by a device.
//! `true` → safe to use native WGSL call; `false` → use software implementation.

/// Which f64 WGSL built-in functions are natively supported by this device.
///
/// `true`  → safe to use native WGSL call (e.g. `exp(f64(x))`)
/// `false` → use software implementation from `math_f64.wgsl`
///
/// Probed individually per function so one broken function does not shadow
/// the rest. On NVK/NAK (Feb 2026) `exp` and `log` crash the shader compiler;
/// `sqrt` and `abs`-family work everywhere since they map to non-transcendental
/// hardware instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct F64BuiltinCapabilities {
    /// Can the device compile basic f64 WGSL at all? NAK and NVVM fail this
    /// despite advertising `SHADER_F64`. When `false`, ALL other fields are
    /// meaningless and should be treated as `false`.
    pub basic_f64: bool,
    /// `exp(f64)` — transcendental, crashes on NVK ≤ Mesa 25.2
    pub exp: bool,
    /// `log(f64)` — transcendental, crashes on NVK ≤ Mesa 25.2
    pub log: bool,
    /// `exp2(f64)` — transcendental
    pub exp2: bool,
    /// `log2(f64)` — transcendental
    pub log2: bool,
    /// `sin(f64)` — transcendental (MUFU on NVIDIA, may be FP32 promoted)
    pub sin: bool,
    /// `cos(f64)` — transcendental (MUFU on NVIDIA, may be FP32 promoted)
    pub cos: bool,
    /// `sqrt(f64)` — DSQRT instruction, generally available
    pub sqrt: bool,
    /// `fma(f64, f64, f64)` → DFMA, generally available on FP64-capable hardware
    pub fma: bool,
    /// `abs(f64)`, `min(f64, f64)`, `max(f64, f64)` — bit-level ops, always work
    pub abs_min_max: bool,
    /// Combined log+exp+sqrt+sin+cos in a single shader — catches NVVM crash
    /// when the JIT compiler cannot handle multiple f64 transcendentals together.
    pub composite_transcendental: bool,
    /// Chained exp(log(x)) pattern that exercises the same op mix
    /// as Bessel K₀ / Beta shaders (lgamma-like chains).
    pub exp_log_chain: bool,
    /// `var<workgroup>` f64 reduction — writes f64 to shared memory, barriers,
    /// reads back. Fails on NVK/NAK and Ada Lovelace proprietary where
    /// shared-memory f64 accumulators return zeros.
    pub shared_mem_f64: bool,
    /// DF64 (f32-pair) arithmetic compiles and dispatches correctly.
    /// When `false`, DF64 shaders should not be used on this device.
    pub df64_arith: bool,
    /// DF64 transcendentals (`exp_df64`, `log_df64`, `pow_df64`) are safe.
    /// When `false`, DF64 shaders must omit transcendental preamble.
    /// On NVIDIA proprietary, NVVM cannot handle DF64 transcendentals
    /// and a failed compilation permanently poisons the wgpu device.
    pub df64_transcendentals_safe: bool,
    /// `fma(a, b, -p)` error-free product extraction compiles and dispatches
    /// correctly in f32. This is the core of `two_prod` in `df64_core.wgsl`.
    /// When `false`, Dekker splitting must be used instead of FMA.
    pub df64_fma_two_prod: bool,
    /// DF64 workgroup tree reduction (`shared_hi`/`shared_lo` arrays with
    /// `workgroupBarrier()`) produces correct results. When `false`,
    /// `ReduceScalarPipeline` must route through a storage-only or
    /// scalar fallback path.
    pub df64_workgroup_reduce: bool,
}

impl F64BuiltinCapabilities {
    /// Conservative fallback: no native builtins — software lib for everything.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            basic_f64: false,
            exp: false,
            log: false,
            exp2: false,
            log2: false,
            sin: false,
            cos: false,
            sqrt: false,
            fma: false,
            abs_min_max: false,
            composite_transcendental: false,
            exp_log_chain: false,
            shared_mem_f64: false,
            df64_arith: false,
            df64_transcendentals_safe: false,
            df64_fma_two_prod: false,
            df64_workgroup_reduce: false,
        }
    }

    /// Full native support (known-good proprietary drivers on FP64 hardware).
    #[must_use]
    pub const fn full() -> Self {
        Self {
            basic_f64: true,
            exp: true,
            log: true,
            exp2: true,
            log2: true,
            sin: true,
            cos: true,
            sqrt: true,
            fma: true,
            abs_min_max: true,
            composite_transcendental: true,
            exp_log_chain: true,
            shared_mem_f64: true,
            df64_arith: true,
            df64_transcendentals_safe: true,
            df64_fma_two_prod: true,
            df64_workgroup_reduce: true,
        }
    }

    /// Whether the device can compile basic f64 WGSL at all.
    /// When false, all f64 shaders must use DF64 (f32-pair) instead.
    #[must_use]
    pub fn can_compile_f64(&self) -> bool {
        self.basic_f64
    }

    /// Whether all f64 transcendental functions work correctly on this device.
    ///
    /// Tests: sqrt, abs/min/max, sin, cos, exp, log, exp2, log2, fma.
    /// When `false`, shaders using transcendentals need polyfill or DF64.
    #[must_use]
    pub fn has_f64_transcendentals(&self) -> bool {
        self.basic_f64
            && self.sqrt
            && self.abs_min_max
            && self.sin
            && self.cos
            && self.exp
            && self.log
            && self.fma
            && self.composite_transcendental
            && self.exp_log_chain
    }

    /// Whether exp/log workarounds are needed (drives `ShaderTemplate` patching).
    #[must_use]
    pub fn needs_exp_log_workaround(&self) -> bool {
        !self.basic_f64 || !self.exp || !self.log
    }

    /// Whether sqrt(f64) needs software substitution.
    #[must_use]
    pub fn needs_sqrt_f64_workaround(&self) -> bool {
        !self.basic_f64 || !self.sqrt
    }

    /// Whether sin(f64) needs software substitution.
    #[must_use]
    pub fn needs_sin_f64_workaround(&self) -> bool {
        !self.basic_f64 || !self.sin
    }

    /// Whether cos(f64) needs software substitution.
    #[must_use]
    pub fn needs_cos_f64_workaround(&self) -> bool {
        !self.basic_f64 || !self.cos
    }

    /// Whether `var<workgroup>` f64 shared-memory reductions need a workaround.
    ///
    /// When `true`, reduction shaders must use scalar f64 accumulation or
    /// DF64 workgroup accumulators instead of native `var<workgroup> array<f64>`.
    #[must_use]
    pub fn needs_shared_mem_f64_workaround(&self) -> bool {
        !self.basic_f64 || !self.shared_mem_f64
    }

    /// Whether DF64 shaders need transcendental stripping on this device.
    ///
    /// On NVIDIA proprietary, NVVM permanently poisons the wgpu device
    /// when a DF64 transcendental shader fails to compile. Callers must
    /// omit `df64_transcendentals.wgsl` from the DF64 preamble.
    #[must_use]
    pub fn needs_df64_transcendental_stripping(&self) -> bool {
        !self.df64_transcendentals_safe
    }

    /// Whether DF64 is available at all (arithmetic path).
    #[must_use]
    pub fn can_use_df64(&self) -> bool {
        self.df64_arith
    }

    /// Whether `ReduceScalarPipeline` needs a workaround for the DF64
    /// workgroup reduction pattern on this device.
    ///
    /// When `true`, the standard DF64 tree reduction in workgroup memory
    /// (`shared_hi`/`shared_lo` arrays) returns incorrect results. The pipeline
    /// must route through a storage-only, scalar, or CPU fallback path.
    #[must_use]
    pub fn needs_df64_reduce_workaround(&self) -> bool {
        !self.df64_workgroup_reduce
    }

    /// Whether the `fma(a, b, -p)` `two_prod` pattern works correctly on f32.
    /// When `false`, the Dekker splitting approach should be used instead.
    #[must_use]
    pub fn needs_df64_fma_workaround(&self) -> bool {
        !self.df64_fma_two_prod
    }

    /// Total count of natively-supported functions (excluding `basic_f64` gate).
    #[must_use]
    pub fn native_count(&self) -> u8 {
        if !self.basic_f64 {
            return 0;
        }
        [
            self.exp,
            self.log,
            self.exp2,
            self.log2,
            self.sin,
            self.cos,
            self.sqrt,
            self.fma,
            self.abs_min_max,
            self.composite_transcendental,
            self.exp_log_chain,
            self.shared_mem_f64,
            self.df64_arith,
            self.df64_transcendentals_safe,
            self.df64_fma_two_prod,
            self.df64_workgroup_reduce,
        ]
        .iter()
        .filter(|&&b| b)
        .count() as u8
    }
}

impl std::fmt::Display for F64BuiltinCapabilities {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let sym = |b: bool| if b { "✓" } else { "✗" };
        writeln!(f, "  f64 builtin capabilities:")?;
        writeln!(f, "    basic_f64={}", sym(self.basic_f64))?;
        writeln!(
            f,
            "    exp={} log={} exp2={} log2={}",
            sym(self.exp),
            sym(self.log),
            sym(self.exp2),
            sym(self.log2)
        )?;
        writeln!(
            f,
            "    sin={} cos={} sqrt={} fma={}",
            sym(self.sin),
            sym(self.cos),
            sym(self.sqrt),
            sym(self.fma)
        )?;
        writeln!(f, "    abs/min/max={}", sym(self.abs_min_max))?;
        writeln!(
            f,
            "    composite_transcendental={} exp_log_chain={}",
            sym(self.composite_transcendental),
            sym(self.exp_log_chain)
        )?;
        writeln!(f, "    shared_mem_f64={}", sym(self.shared_mem_f64))?;
        writeln!(
            f,
            "    df64_arith={} df64_transcendentals={}",
            sym(self.df64_arith),
            sym(self.df64_transcendentals_safe)
        )?;
        write!(
            f,
            "    df64_fma_two_prod={} df64_workgroup_reduce={}",
            sym(self.df64_fma_two_prod),
            sym(self.df64_workgroup_reduce)
        )
    }
}
