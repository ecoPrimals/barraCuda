// SPDX-License-Identifier: AGPL-3.0-only
//! Shader compilation — WGSL, SPIR-V, f64, DF64, universal precision pipelines

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

    /// Run the sovereign compiler's naga-level optimisations on WGSL and
    /// compile the result through the safe `create_shader_module` path.
    ///
    /// Returns `None` if the sovereign pipeline fails (caller should fall
    /// back to the un-optimised WGSL).
    fn try_sovereign_compile(
        &self,
        source: &str,
        label: Option<&str>,
        tag: &str,
    ) -> Option<wgpu::ShaderModule> {
        use crate::shaders::sovereign::SovereignCompiler;
        let profile = crate::device::driver_profile::GpuDriverProfile::from_device(self);
        let sovereign = SovereignCompiler::new(profile);
        match sovereign.compile_to_wgsl(source) {
            Ok((optimized_wgsl, stats)) => {
                if stats.fma_fusions > 0 || stats.dead_exprs_eliminated > 0 {
                    tracing::debug!(
                        "sovereign {tag}: {} FMA fusions, {} dead exprs eliminated",
                        stats.fma_fusions,
                        stats.dead_exprs_eliminated,
                    );
                }
                Some(self.compile_shader_raw(&optimized_wgsl, label))
            }
            Err(e) => {
                tracing::debug!("sovereign {tag} fallback to raw WGSL: {e}");
                None
            }
        }
    }

    /// Compile an f64 WGSL shader with automatic driver-aware patching and ILP optimization.
    ///
    /// Pipeline:
    /// 1. `ShaderTemplate::for_driver_auto` — patches exp/log for drivers that lack native f64
    /// 2. `WgslOptimizer::optimize` — reorders `@ilp_region` blocks + unrolls `@unroll_hint` loops
    ///    (Phase 3 `SOVEREIGN_COMPUTE_EVOLUTION`; only active when annotations are present)
    /// 3. `SovereignCompiler::compile_to_wgsl` — Phase 4: naga IR optimisation (FMA fusion,
    ///    dead expr elimination) → re-emit optimised WGSL (safe, no SPIR-V passthrough).
    ///
    /// The optimizer is keyed to the actual GPU arch detected at device-creation time,
    /// so the ILP fill width matches the hardware (8 cy on SM70, 4 cy on RDNA2, etc.).
    #[must_use]
    pub fn compile_shader_f64(&self, source: &str, label: Option<&str>) -> wgpu::ShaderModule {
        let source = &source.replace("enable f64;", "");

        let profile = crate::device::driver_profile::GpuDriverProfile::from_device(self);

        let optimized = crate::shaders::precision::ShaderTemplate::for_driver_profile(
            source,
            self.needs_f64_exp_log_workaround(),
            &profile,
        );

        // coralReef: async native binary compilation for NVIDIA GPUs.
        if let Some(coral_arch) = crate::device::coral_compiler::arch_to_coral(&profile.arch) {
            crate::device::coral_compiler::spawn_coral_compile(&optimized, coral_arch, true);
        }

        // Sovereign compiler — naga IR → FMA fusion → optimised WGSL (safe path).
        // Runs on all backends (Vulkan, Metal, DX12, WebGPU).
        if let Some(module) = self.try_sovereign_compile(&optimized, label, "f64") {
            return module;
        }

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
    /// f64 on consumer GPUs (Ampere/Ada fp64:fp32 ≈ 1:64).
    ///
    /// Pipeline mirrors [`compile_shader_f64`] minus the f64 driver patching:
    /// 1. Prepend DF64 preamble (core + transcendentals)
    /// 2. ILP optimizer (when `@ilp_region`/`@unroll_hint` annotations present)
    /// 3. Sovereign compiler optimised WGSL path (safe, all backends)
    #[must_use]
    pub fn compile_shader_df64(&self, source: &str, label: Option<&str>) -> wgpu::ShaderModule {
        const DF64_CORE: &str = include_str!("../../shaders/math/df64_core.wgsl");
        const DF64_TRANSCENDENTALS: &str =
            include_str!("../../shaders/math/df64_transcendentals.wgsl");

        let profile = crate::device::driver_profile::GpuDriverProfile::from_device(self);
        let include_transcendentals = if profile.has_nvvm_df64_poisoning_risk() {
            tracing::warn!(
                driver = ?profile.driver,
                arch = ?profile.arch,
                "NVVM DF64 poisoning risk — omitting DF64 transcendentals (exp, log, pow) \
                 to prevent unrecoverable device invalidation. \
                 Arithmetic-only DF64 shaders remain safe."
            );
            false
        } else {
            true
        };

        let combined = if include_transcendentals {
            format!("{DF64_CORE}\n{DF64_TRANSCENDENTALS}\n{source}")
        } else {
            format!("{DF64_CORE}\n{source}")
        };

        let combined = combined
            .lines()
            .filter(|l| l.trim() != "enable f64;")
            .collect::<Vec<_>>()
            .join("\n");

        let optimized = if combined.contains("@ilp_region") || combined.contains("@unroll_hint") {
            use crate::shaders::optimizer::WgslOptimizer;
            let optimizer = WgslOptimizer::new(profile.latency_model());
            optimizer.optimize(&combined)
        } else {
            combined
        };

        // coralReef: async native binary compilation for NVIDIA GPUs.
        if let Some(coral_arch) = crate::device::coral_compiler::arch_to_coral(&profile.arch) {
            crate::device::coral_compiler::spawn_coral_compile(&optimized, coral_arch, false);
        }

        // Sovereign compiler — naga IR → FMA fusion → optimised WGSL (safe path).
        if let Some(module) = self.try_sovereign_compile(&optimized, label, "df64") {
            return module;
        }

        self.compile_shader_raw(&optimized, label)
    }
}
