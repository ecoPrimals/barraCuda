//! TensorSession — `SessionPipelines` cache
//!
//! All compute pipelines compiled **once** at session construction.
//! Subsequent `run()` calls pay only bind-group creation and dispatch encoding,
//! not SPIR-V translation.  Resolves D-S20-001 and the residual deep debt where
//! `encode_binary_op` / `encode_ternary_op` / `encode_scale_op` used to recreate
//! both `BindGroupLayout` and `ComputePipeline` on every `run()` invocation.

pub(super) static MATMUL_NAIVE_F32: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
        "../shaders/math/matmul_f64.wgsl"
    ))
});
pub(super) static MATMUL_TILED_F32: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
        "../shaders/math/matmul_tiled_f64.wgsl"
    ))
});
pub(super) static MATMUL_CPU_F32: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
        "../shaders/math/matmul_cpu_tiled_f64.wgsl"
    ))
});
pub(super) static MATMUL_GPU_F32: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
        "../shaders/math/matmul_gpu_evolved_f64.wgsl"
    ))
});

pub(crate) static HEAD_SPLIT_F32: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
        "../shaders/tensor/head_split_f64.wgsl"
    ))
});
pub(crate) static HEAD_CONCAT_F32: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
        "../shaders/tensor/head_concat_f64.wgsl"
    ))
});

/// All compute pipelines for a `TensorSession`.
///
/// `add_pl`, `mul_pl`, `fma_pl`, `scale_pl` are now full `ComputePipeline`
/// objects (with `layout: None` so wgpu auto-derives the BGL from the shader
/// reflection).  This eliminates the per-`run()` BGL + pipeline creation that
/// the previous `encode_binary_op/ternary_op/scale_op` helpers incurred.
pub(super) struct SessionPipelines {
    // ── Elementwise — pre-compiled pipelines (upgraded from ShaderModule) ────
    pub add_pl: wgpu::ComputePipeline,
    pub mul_pl: wgpu::ComputePipeline,
    pub fma_pl: wgpu::ComputePipeline,
    pub scale_pl: wgpu::ComputePipeline,
    // ── ML activations / norms ───────────────────────────────────────────────
    pub relu_pl: wgpu::ComputePipeline,
    pub gelu_pl: wgpu::ComputePipeline,
    pub sfmx_pl: wgpu::ComputePipeline,
    pub lnrm_pl: wgpu::ComputePipeline,
    // ── Matmul tiers ─────────────────────────────────────────────────────────
    pub mm_naive_pl: wgpu::ComputePipeline,
    pub mm_t16_pl: wgpu::ComputePipeline,
    pub mm_cpu_pl: wgpu::ComputePipeline,
    pub mm_gpu_pl: wgpu::ComputePipeline,
    // ── Attention (3-pass SDPA + head reshape) ────────────────────────────────
    pub sdpa_scores_pl: wgpu::ComputePipeline,
    pub attn_softmax_pl: wgpu::ComputePipeline,
    pub attn_apply_pl: wgpu::ComputePipeline,
    pub head_split_pl: wgpu::ComputePipeline,
    pub head_concat_pl: wgpu::ComputePipeline,
}

impl SessionPipelines {
    pub(super) fn build(device: &wgpu::Device, workgroup_size: u32) -> Self {
        // `auto_pipeline` uses `layout: None` — wgpu reflects the BGL from
        // the shader, eliminating all manual `BindGroupLayoutDescriptor` boilerplate.
        let auto_pipeline = |src: &str, label: &str| -> wgpu::ComputePipeline {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: "main",
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            })
        };

        // Inline elementwise shaders with workgroup_size embedded at compile time.
        let add_src = format!(
            "@group(0) @binding(0) var<storage, read>       a: array<f32>;\n\
             @group(0) @binding(1) var<storage, read>       b: array<f32>;\n\
             @group(0) @binding(2) var<storage, read_write> o: array<f32>;\n\
             @compute @workgroup_size({workgroup_size})\n\
             fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
                 let i = gid.x;\n\
                 if i >= arrayLength(&o) {{ return; }}\n\
                 o[i] = a[i] + b[i];\n\
             }}"
        );
        let mul_src = format!(
            "@group(0) @binding(0) var<storage, read>       a: array<f32>;\n\
             @group(0) @binding(1) var<storage, read>       b: array<f32>;\n\
             @group(0) @binding(2) var<storage, read_write> o: array<f32>;\n\
             @compute @workgroup_size({workgroup_size})\n\
             fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
                 let i = gid.x;\n\
                 if i >= arrayLength(&o) {{ return; }}\n\
                 o[i] = a[i] * b[i];\n\
             }}"
        );
        let fma_src = format!(
            "@group(0) @binding(0) var<storage, read>       a: array<f32>;\n\
             @group(0) @binding(1) var<storage, read>       b: array<f32>;\n\
             @group(0) @binding(2) var<storage, read>       c: array<f32>;\n\
             @group(0) @binding(3) var<storage, read_write> o: array<f32>;\n\
             @compute @workgroup_size({workgroup_size})\n\
             fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
                 let i = gid.x;\n\
                 if i >= arrayLength(&o) {{ return; }}\n\
                 o[i] = fma(a[i], b[i], c[i]);\n\
             }}"
        );
        let scale_src = format!(
            "struct Params {{ scalar: f32 }}\n\
             @group(0) @binding(0) var<storage, read>       a: array<f32>;\n\
             @group(0) @binding(1) var<uniform>             p: Params;\n\
             @group(0) @binding(2) var<storage, read_write> o: array<f32>;\n\
             @compute @workgroup_size({workgroup_size})\n\
             fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
                 let i = gid.x;\n\
                 if i >= arrayLength(&o) {{ return; }}\n\
                 o[i] = a[i] * p.scalar;\n\
             }}"
        );

        Self {
            add_pl: auto_pipeline(&add_src, "Session Add"),
            mul_pl: auto_pipeline(&mul_src, "Session Mul"),
            fma_pl: auto_pipeline(&fma_src, "Session FMA"),
            scale_pl: auto_pipeline(&scale_src, "Session Scale"),
            relu_pl: auto_pipeline(&crate::ops::relu::SHADER_F32, "Session ReLU"),
            gelu_pl: auto_pipeline(&crate::ops::gelu_wgsl::SHADER_F32, "Session GELU"),
            sfmx_pl: auto_pipeline(
                &crate::ops::softmax::SHADER_SOFTMAX_SIMPLE_F32,
                "Session Softmax",
            ),
            lnrm_pl: auto_pipeline(
                crate::ops::layer_norm_wgsl::LayerNorm::wgsl_shader(),
                "Session LayerNorm",
            ),
            mm_naive_pl: auto_pipeline(
                &crate::session::pipelines::MATMUL_NAIVE_F32,
                "Session MatMul Naive",
            ),
            mm_t16_pl: auto_pipeline(
                &crate::session::pipelines::MATMUL_TILED_F32,
                "Session MatMul Tiled16",
            ),
            mm_cpu_pl: auto_pipeline(
                &crate::session::pipelines::MATMUL_CPU_F32,
                "Session MatMul CpuTiled32",
            ),
            mm_gpu_pl: auto_pipeline(
                &crate::session::pipelines::MATMUL_GPU_F32,
                "Session MatMul GpuEvolved32",
            ),
            sdpa_scores_pl: auto_pipeline(
                &crate::ops::attention::SDPA_SCORES_F32,
                "Session SDPA Scores",
            ),
            attn_softmax_pl: auto_pipeline(
                &crate::ops::attention::ATTENTION_SOFTMAX_F32,
                "Session Attn Softmax",
            ),
            attn_apply_pl: auto_pipeline(
                &crate::ops::attention::ATTENTION_APPLY_F32,
                "Session Attn Apply",
            ),
            head_split_pl: auto_pipeline(
                &crate::session::pipelines::HEAD_SPLIT_F32,
                "Session Head Split",
            ),
            head_concat_pl: auto_pipeline(
                &crate::session::pipelines::HEAD_CONCAT_F32,
                "Session Head Concat",
            ),
        }
    }
}
