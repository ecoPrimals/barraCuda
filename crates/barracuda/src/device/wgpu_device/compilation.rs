// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shader compilation — WGSL, SPIR-V, f64, DF64, universal precision pipelines
//!
//! ## f64 in naga 28 / wgpu 28
//!
//! naga 28 does NOT support `enable f64;` as a WGSL enable-extension. The
//! only recognized enable-extension is `f16`. f64 type support is gated by
//! the `SHADER_F64` device feature at device creation time, which sets
//! `Capabilities::FLOAT64` in naga's validator. Source directives like
//! `enable f64;` must be **stripped** before passing to `create_shader_module`
//! — naga's parser will reject them as "unknown enable-extension".
//!
//! ## Compilation tiers (f64 shaders)
//!
//! `compile_shader_f64` attempts compilation in order of preference:
//!
//! 1. **Sovereign SPIR-V** (future): naga parse → FMA fusion → dead-expr
//!    elimination → SPIR-V emit → `create_shader_module_spirv`. Bypasses
//!    naga's WGSL backend entirely. Blocked by `#![forbid(unsafe_code)]`.
//!
//! 2. **Sovereign WGSL**: same naga IR optimisation, but re-emitted as WGSL
//!    text and compiled through the safe `create_shader_module` API. Works on
//!    all backends. naga parse is lenient: accepts f64 types without any
//!    enable directive when `Capabilities::FLOAT64` is set.
//!
//! 3. **Raw WGSL**: template-processed source compiled directly. Fallback when
//!    sovereign compilation fails (parse/validate/emit error).
//!
//! In parallel, `spawn_coral_compile_for_adapter` fires off a background
//! coralReef IPC compile that populates the native binary cache for future
//! sovereign-dispatch use (coral-driver direct GPU submission without wgpu).

use super::WgpuDevice;
use crate::shaders::precision::compiler::source_is_f64;

impl WgpuDevice {
    /// Compile WGSL shader (raw path — no f64 detection or routing).
    /// Use [`compile_shader`](Self::compile_shader) for the public API that handles f64.
    #[must_use]
    pub fn compile_shader_raw(&self, source: &str, label: Option<&str>) -> wgpu::ShaderModule {
        self.encoding_guard();
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label,
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        self.encoding_complete();
        module
    }

    /// Compile WGSL shader with automatic f64→f32 downcast for broad compatibility.
    ///
    /// f64-canonical architecture: shaders are authored in f64 as the source of
    /// truth. This method always downcasts f64 types to f32 via
    /// `downcast_f64_to_f32_with_transcendentals`, ensuring the shader runs on
    /// any GPU regardless of f64 support.
    ///
    /// For native f64 execution (when the Rust side uploads f64 data), use
    /// [`compile_shader_f64`](Self::compile_shader_f64) instead.
    #[must_use]
    pub fn compile_shader(&self, source: &str, label: Option<&str>) -> wgpu::ShaderModule {
        if source_is_f64(source) {
            let downcast =
                crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(source);
            return self.compile_shader_raw(&downcast, label);
        }
        self.compile_shader_raw(source, label)
    }

    /// Tier 1: Sovereign SPIR-V → wgpu passthrough.
    ///
    /// Parses WGSL → naga IR → FMA fusion + dead-expr elimination → SPIR-V
    /// emission → `create_shader_module_passthrough`. Bypasses naga's WGSL
    /// re-emission entirely, eliminating round-trip fidelity risks.
    ///
    /// Requires the `spirv-passthrough` feature flag, which pulls in the
    /// `barracuda-spirv` bridge crate (the only `unsafe` in the pipeline).
    /// Without the feature, this is a no-op that returns `None`.
    #[cfg(feature = "spirv-passthrough")]
    fn try_sovereign_spirv_compile(
        &self,
        source: &str,
        label: Option<&str>,
        tag: &str,
    ) -> Option<wgpu::ShaderModule> {
        use crate::shaders::sovereign::SovereignCompiler;
        let caps = crate::device::capabilities::DeviceCapabilities::from_device(self);
        let sovereign = SovereignCompiler::new(caps);
        match sovereign.compile(source) {
            Ok((crate::shaders::sovereign::SovereignOutput::Spirv(validated), stats)) => {
                if stats.fma_fusions > 0 || stats.dead_exprs_eliminated > 0 {
                    tracing::debug!(
                        "sovereign-spirv {tag}: {} FMA fusions, {} dead exprs eliminated",
                        stats.fma_fusions,
                        stats.dead_exprs_eliminated,
                    );
                }
                match barracuda_spirv::compile_spirv_passthrough(
                    self.device(),
                    validated.words(),
                    label,
                ) {
                    Ok(module) => Some(module),
                    Err(e) => {
                        tracing::debug!("spirv passthrough failed: {e}");
                        None
                    }
                }
            }
            Err(e) => {
                tracing::debug!("sovereign-spirv {tag} fallback to WGSL path: {e}");
                None
            }
        }
    }

    /// Tier 1 stub: returns `None` when `spirv-passthrough` feature is not enabled.
    #[cfg(not(feature = "spirv-passthrough"))]
    fn try_sovereign_spirv_compile(
        &self,
        _source: &str,
        _label: Option<&str>,
        _tag: &str,
    ) -> Option<wgpu::ShaderModule> {
        None
    }

    /// Tier 2: Sovereign WGSL — naga IR optimisation → re-emit WGSL.
    ///
    /// Applies FMA fusion and dead-expression elimination at the naga IR level,
    /// then re-emits valid WGSL through naga's WGSL writer. naga 28's parser
    /// is lenient — it accepts f64 types without `enable f64;` when
    /// `Capabilities::FLOAT64` is set (which it always is in the sovereign
    /// compiler's validation pass).
    ///
    /// Returns `None` if the sovereign pipeline fails (caller should fall
    /// back to the un-optimised WGSL).
    fn try_sovereign_wgsl_compile(
        &self,
        source: &str,
        label: Option<&str>,
        tag: &str,
    ) -> Option<wgpu::ShaderModule> {
        use crate::shaders::sovereign::SovereignCompiler;
        let caps = crate::device::capabilities::DeviceCapabilities::from_device(self);
        let sovereign = SovereignCompiler::new(caps);
        match sovereign.compile_to_wgsl(source) {
            Ok((optimized_wgsl, stats)) => {
                if stats.fma_fusions > 0 || stats.dead_exprs_eliminated > 0 {
                    tracing::debug!(
                        "sovereign-wgsl {tag}: {} FMA fusions, {} dead exprs eliminated",
                        stats.fma_fusions,
                        stats.dead_exprs_eliminated,
                    );
                }
                Some(self.compile_shader_raw(&optimized_wgsl, label))
            }
            Err(e) => {
                tracing::debug!("sovereign-wgsl {tag} fallback to raw WGSL: {e}");
                None
            }
        }
    }

    /// Compile an f64 WGSL shader through the tiered sovereign pipeline.
    ///
    /// `enable f64;` and `enable subgroups;` are stripped — naga 28 rejects
    /// `enable f64;`, and `enable subgroups;` causes naga to emit broken SPIR-V
    /// where all subgroup operations return zero. Both extensions are gated by
    /// device features (`SHADER_F64`, `SUBGROUP`) instead of WGSL directives.
    #[must_use]
    pub fn compile_shader_f64(&self, source: &str, label: Option<&str>) -> wgpu::ShaderModule {
        let source = &source
            .replace("enable f64;", "")
            .replace("enable subgroups;", "");

        let caps = crate::device::capabilities::DeviceCapabilities::from_device(self);

        let optimized = crate::shaders::precision::ShaderTemplate::for_device_capabilities(
            source,
            self.needs_f64_exp_log_workaround(),
            &caps,
        );

        // Sovereign shader compiler: adapter-aware native binary compilation via IPC.
        crate::device::coral_compiler::spawn_coral_compile_for_adapter(
            &optimized,
            self.adapter_info(),
            true,
        );

        // Tier 1: Sovereign SPIR-V passthrough (future).
        if let Some(module) = self.try_sovereign_spirv_compile(&optimized, label, "f64") {
            return module;
        }

        // Tier 2: Sovereign WGSL (naga IR → optimise → re-emit WGSL).
        // Fixed: f64 FMA fusion skip + entry point name restoration.
        if let Some(module) = self.try_sovereign_wgsl_compile(&optimized, label, "f64") {
            return module;
        }

        // Tier 3: Raw WGSL (template-processed, enable f64; already stripped).
        self.compile_shader_raw(&optimized, label)
    }

    /// Compile a DF64 (double-float, f32-pair) WGSL shader.
    ///
    /// Prepends `df64_core.wgsl` + `df64_transcendentals.wgsl` to the source,
    /// providing the full DF64 arithmetic library: `Df64`, `df64_add`, `df64_mul`,
    /// `df64_div`, `sqrt_df64`, `exp_df64`, `log_df64`, `sin_df64`, `cos_df64`,
    /// `pow_df64`, `tanh_df64`.
    ///
    /// DF64 shaders run entirely on FP32 cores (no f64 hardware needed), achieving
    /// ~48-bit mantissa (~14 decimal digits) at up to 9.9× the throughput of native
    /// f64 on consumer GPUs (Ampere/Ada fp64:fp32 = 1:64).
    ///
    /// When naga SPIR-V codegen poisons DF64 transcendentals, this method sends
    /// the full (un-stripped) source to the sovereign shader compiler. If the
    /// compiler is unavailable, the transcendentals are stripped and the shader
    /// runs in arithmetic-only mode.
    #[must_use]
    pub fn compile_shader_df64(&self, source: &str, label: Option<&str>) -> wgpu::ShaderModule {
        const DF64_CORE: &str = include_str!("../../shaders/math/df64_core.wgsl");
        const DF64_TRANSCENDENTALS: &str =
            include_str!("../../shaders/math/df64_transcendentals.wgsl");

        let caps = crate::device::capabilities::DeviceCapabilities::from_device(self);
        let naga_poisoned = caps.has_df64_spir_v_poisoning();

        let full_combined = format!("{DF64_CORE}\n{DF64_TRANSCENDENTALS}\n{source}");
        let full_combined = full_combined
            .lines()
            .filter(|l| l.trim() != "enable f64;")
            .collect::<Vec<_>>()
            .join("\n");

        if naga_poisoned {
            tracing::warn!(
                device = %caps.device_name,
                "DF64 SPIR-V poisoning (naga codegen) — requesting sovereign shader \
                 compilation to bypass naga. Falling back to arithmetic-only if unavailable."
            );

            // Send the FULL DF64 source (with transcendentals) to the sovereign
            // shader compiler. The sovereign path bypasses naga and compiles to
            // native ISA, so the poisoning is irrelevant.
            crate::device::coral_compiler::spawn_coral_compile_for_adapter(
                &full_combined,
                self.adapter_info(),
                false,
            );
        }

        let combined = if naga_poisoned {
            format!("{DF64_CORE}\n{source}")
                .lines()
                .filter(|l| l.trim() != "enable f64;")
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            full_combined
        };

        let optimized = if combined.contains("@ilp_region") || combined.contains("@unroll_hint") {
            use crate::shaders::optimizer::WgslOptimizer;
            let optimizer = WgslOptimizer::new(caps.latency_model());
            optimizer.optimize(&combined)
        } else {
            combined
        };

        if !naga_poisoned {
            crate::device::coral_compiler::spawn_coral_compile_for_adapter(
                &optimized,
                self.adapter_info(),
                false,
            );
        }

        // Tier 1: Sovereign SPIR-V (future).
        if let Some(module) = self.try_sovereign_spirv_compile(&optimized, label, "df64") {
            return module;
        }

        // Tier 2: Sovereign WGSL (naga IR → optimise → re-emit WGSL).
        if let Some(module) = self.try_sovereign_wgsl_compile(&optimized, label, "df64") {
            return module;
        }

        // Tier 3: Raw WGSL (may be arithmetic-only if naga poisoned).
        self.compile_shader_raw(&optimized, label)
    }
}
