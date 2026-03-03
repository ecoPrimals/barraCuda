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
}

impl F64BuiltinCapabilities {
    /// Conservative fallback: no native builtins — software lib for everything.
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
        }
    }

    /// Full native support (known-good proprietary drivers on FP64 hardware).
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
        }
    }

    /// Whether the device can compile basic f64 WGSL at all.
    /// When false, all f64 shaders must use DF64 (f32-pair) instead.
    pub fn can_compile_f64(&self) -> bool {
        self.basic_f64
    }

    /// Whether exp/log workarounds are needed (drives ShaderTemplate patching).
    pub fn needs_exp_log_workaround(&self) -> bool {
        !self.basic_f64 || !self.exp || !self.log
    }

    /// Whether sin(f64) needs software substitution.
    pub fn needs_sin_f64_workaround(&self) -> bool {
        !self.basic_f64 || !self.sin
    }

    /// Whether cos(f64) needs software substitution.
    pub fn needs_cos_f64_workaround(&self) -> bool {
        !self.basic_f64 || !self.cos
    }

    /// Total count of natively-supported functions (excluding basic_f64 gate).
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
        write!(f, "    abs/min/max={}", sym(self.abs_min_max))
    }
}
