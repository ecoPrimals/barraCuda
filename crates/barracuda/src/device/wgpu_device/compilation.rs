// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shader compilation — WGSL, SPIR-V, f64, DF64, universal precision pipelines

use super::WgpuDevice;

impl WgpuDevice {
    /// Compile WGSL shader
    pub fn compile_shader(&self, source: &str, label: Option<&str>) -> wgpu::ShaderModule {
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

    /// Compile a pre-built SPIR-V binary into a shader module.
    ///
    /// Requires `SPIRV_SHADER_PASSTHROUGH` — check with `has_spirv_passthrough()`.
    ///
    /// # Safety
    ///
    /// The SPIR-V binary is passed to the driver as-is. The caller must
    /// ensure the binary was produced by a trusted source (our own naga
    /// backend) and has been validated by `naga::valid::Validator`.
    pub fn compile_shader_spirv(
        &self,
        spirv_words: &[u32],
        label: Option<&str>,
    ) -> wgpu::ShaderModule {
        // SAFETY: wgpu::Device::create_shader_module_spirv is unsafe (wgpu API constraint).
        // - Why unsafe: Passes SPIR-V binary to the backend as-is; wgpu does not validate
        //   it. Malformed or malicious SPIR-V could cause driver crash, GPU hang, or UB.
        // - Invariants we maintain: SPIR-V is produced by SovereignCompiler (naga IR →
        //   spv::Writer) from naga::valid::Validator-approved module. No external or
        //   untrusted input. Callers must ensure spirv_words came from our compiler.
        // - What could go wrong: Untrusted/malformed SPIR-V → driver crash, GPU UB,
        //   or security issues (e.g. out-of-bounds access in shader).
        // - Minimum unsafe surface: wgpu 22.x has no safe SPIR-V API; create_shader_module
        //   (WGSL) validates via naga but we need SPIR-V passthrough for NVK/Vulkan
        //   optimization (Sovereign compiler bypasses NAK). This is the only path.
        self.encoding_guard();
        #[allow(unsafe_code)]
        let module = unsafe {
            self.device
                .create_shader_module_spirv(&wgpu::ShaderModuleDescriptorSpirV {
                    label,
                    source: std::borrow::Cow::Borrowed(spirv_words),
                })
        };
        self.encoding_complete();
        module
    }

    /// Compile an f64 WGSL shader with automatic driver-aware patching and ILP optimization.
    ///
    /// Pipeline:
    /// 1. `ShaderTemplate::for_driver_auto` — patches exp/log for drivers that lack native f64
    /// 2. `WgslOptimizer::optimize` — reorders `@ilp_region` blocks + unrolls `@unroll_hint` loops
    ///    (Phase 3 SOVEREIGN_COMPUTE_EVOLUTION; only active when annotations are present)
    /// 3. `SovereignCompiler::compile` — Phase 4: naga IR optimization (FMA fusion, dead expr
    ///    elimination) + SPIR-V emission via `SPIRV_SHADER_PASSTHROUGH` (when available).
    ///
    /// The optimizer is keyed to the actual GPU arch detected at device-creation time,
    /// so the ILP fill width matches the hardware (8 cy on SM70, 4 cy on RDNA2, etc.).
    pub fn compile_shader_f64(&self, source: &str, label: Option<&str>) -> wgpu::ShaderModule {
        let source = &source.replace("enable f64;", "");

        let profile = crate::device::driver_profile::GpuDriverProfile::from_device(self);

        // Single-pass: driver patching + ILP optimization with the real latency model.
        // `for_driver_profile` replaces the previous `for_driver_auto` + second optimizer
        // pass that was double-optimizing with mismatched latency models.
        let optimized = crate::shaders::precision::ShaderTemplate::for_driver_profile(
            source,
            self.needs_f64_exp_log_workaround(),
            &profile,
        );

        // Sovereign compiler — naga IR → FMA fusion → SPIR-V.
        // Bypasses NAK entirely when SPIRV_SHADER_PASSTHROUGH is available (NVK/Vulkan).
        if self.has_spirv_passthrough() {
            use crate::shaders::sovereign::{SovereignCompiler, SovereignOutput};
            let sovereign = SovereignCompiler::new(profile);
            match sovereign.compile(&optimized) {
                Ok((SovereignOutput::Spirv(words), stats)) => {
                    if stats.fma_fusions > 0 || stats.dead_exprs_eliminated > 0 {
                        tracing::debug!(
                            "sovereign: {} FMA fusions, {} dead exprs eliminated",
                            stats.fma_fusions,
                            stats.dead_exprs_eliminated,
                        );
                    }
                    return self.compile_shader_spirv(&words, label);
                }
                Err(e) => {
                    tracing::debug!("sovereign compiler fallback to WGSL: {e}");
                }
            }
        }

        self.compile_shader(&optimized, label)
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
    /// 3. Sovereign compiler SPIR-V path (when available)
    pub fn compile_shader_df64(&self, source: &str, label: Option<&str>) -> wgpu::ShaderModule {
        const DF64_CORE: &str = include_str!("../../shaders/math/df64_core.wgsl");
        const DF64_TRANSCENDENTALS: &str =
            include_str!("../../shaders/math/df64_transcendentals.wgsl");

        let combined = format!("{DF64_CORE}\n{DF64_TRANSCENDENTALS}\n{source}");

        // Strip `enable f64;` — naga handles f64 via capability flags, not directives.
        let combined = combined
            .lines()
            .filter(|l| l.trim() != "enable f64;")
            .collect::<Vec<_>>()
            .join("\n");

        let profile = crate::device::driver_profile::GpuDriverProfile::from_device(self);
        let optimized = if combined.contains("@ilp_region") || combined.contains("@unroll_hint") {
            use crate::shaders::optimizer::WgslOptimizer;
            let optimizer = WgslOptimizer::new(profile.latency_model());
            optimizer.optimize(&combined)
        } else {
            combined
        };

        if self.has_spirv_passthrough() {
            use crate::shaders::sovereign::{SovereignCompiler, SovereignOutput};
            let sovereign = SovereignCompiler::new(profile);
            match sovereign.compile(&optimized) {
                Ok((SovereignOutput::Spirv(words), stats)) => {
                    if stats.fma_fusions > 0 || stats.dead_exprs_eliminated > 0 {
                        tracing::debug!(
                            "sovereign df64: {} FMA fusions, {} dead exprs eliminated",
                            stats.fma_fusions,
                            stats.dead_exprs_eliminated,
                        );
                    }
                    return self.compile_shader_spirv(&words, label);
                }
                Err(e) => {
                    tracing::debug!("sovereign df64 fallback: {e}");
                }
            }
        }

        self.compile_shader(&optimized, label)
    }

    /// Compile a shader written as universal math, specialized to the requested precision.
    ///
    /// **Math is universal, precision is silicon.** The same algorithm written once
    /// in f64 (the conceptually true math) is compiled for any target precision:
    ///
    /// - `Precision::F32` — downcast f64 types to f32, compile via standard path
    /// - `Precision::F64` — full `compile_shader_f64()` pipeline (polyfills + sovereign compiler)
    /// - `Precision::Df64` — downcast f64 to DF64 types + transcendentals, compile via
    ///   `compile_shader_df64()` which auto-injects the DF64 core library
    /// - `Precision::F16` — downcast f64 types to f16, compile via standard path
    ///
    /// Pass the f64-canonical source (the "true math") for ALL precisions.
    /// The pipeline handles the rest.
    ///
    /// **DF64 coverage**: Full coverage via two complementary layers:
    ///
    /// 1. **Text-based downcast** — handles types, constructors, transcendentals,
    ///    storage conversions (fast, always available)
    /// 2. **Naga-guided rewrite** — parses with naga for type analysis, rewrites
    ///    f64 infix operators (`+`, `-`, `*`, `/`) to df64 function calls.
    ///    Falls back to text-only downcast if naga rewrite fails.
    ///
    /// Shaders using `op_add`/`op_mul`/etc. work at all precisions without
    /// either layer — the operation preamble provides implementations directly.
    pub fn compile_shader_universal(
        &self,
        source: &str,
        precision: crate::shaders::precision::Precision,
        label: Option<&str>,
    ) -> wgpu::ShaderModule {
        use crate::shaders::precision::{downcast_f64_to_df64, downcast_f64_to_f32, Precision};
        match precision {
            Precision::F32 => {
                let f32_source = downcast_f64_to_f32(source);
                self.compile_shader(&f32_source, label)
            }
            Precision::F64 => self.compile_shader_f64(source, label),
            Precision::Df64 => {
                // Two-layer DF64 compilation:
                //
                // Layer 1 (naga-guided): Parse f64 WGSL with naga, identify f64
                //   infix operators by type, replace with bridge functions that
                //   route computation through DF64 while keeping f64 types.
                //
                // Layer 2 (text-based): downcast_f64_to_df64 handles types,
                //   constructors, transcendentals, and storage conversions.
                //
                // Naga is tried first. If it fails (e.g., source uses polyfill
                // functions naga can't validate), fall back to text-only downcast.
                let df64_source =
                    crate::shaders::sovereign::df64_rewrite::rewrite_f64_infix_full(source)
                        .unwrap_or_else(|_| downcast_f64_to_df64(source));
                self.compile_shader_df64(&df64_source, label)
            }
            Precision::F16 => {
                let f16_source = crate::shaders::precision::downcast_f64_to_f16(source);
                self.compile_shader(&f16_source, label)
            }
        }
    }

    /// Compile a universal shader that uses `op_add`/`op_mul`/etc. operations.
    ///
    /// This is the ultimate "math is universal" entry point. The shader uses
    /// abstract operation functions (`op_add`, `op_mul`, `op_pack`, `op_unpack`,
    /// etc.) and `Scalar` as the type alias. The pipeline:
    ///
    /// 1. Injects the precision-specific operation preamble (trivial wrappers
    ///    for f32/f64, DF64 library calls for Df64)
    /// 2. Routes through the appropriate compilation pipeline
    ///
    /// Shaders written this way work at ALL precisions without naga IR rewriting.
    pub fn compile_op_shader(
        &self,
        source: &str,
        precision: crate::shaders::precision::Precision,
        label: Option<&str>,
    ) -> wgpu::ShaderModule {
        use crate::shaders::precision::Precision;
        let preamble = precision.op_preamble();
        let combined = format!("{preamble}\n{source}");
        match precision {
            Precision::F64 => self.compile_shader_f64(&combined, label),
            Precision::Df64 => self.compile_shader_df64(&combined, label),
            _ => self.compile_shader(&combined, label),
        }
    }

    /// Compile a `{{SCALAR}}`-templated shader at the given precision.
    ///
    /// Renders the template via [`ShaderTemplate::render`], then routes through
    /// the appropriate compilation pipeline for the target precision.
    pub fn compile_template(
        &self,
        template: &crate::shaders::precision::ShaderTemplate,
        precision: crate::shaders::precision::Precision,
        label: Option<&str>,
    ) -> wgpu::ShaderModule {
        use crate::shaders::precision::Precision;
        let rendered = template.render(precision);
        match precision {
            Precision::F64 => self.compile_shader_f64(&rendered, label),
            Precision::Df64 => self.compile_shader_df64(&rendered, label),
            _ => self.compile_shader(&rendered, label),
        }
    }
}
