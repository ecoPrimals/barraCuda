// SPDX-License-Identifier: AGPL-3.0-or-later
//! Layer Normalization — GPU-resident, pipeline-cached, batchable
//!
//! Deep Debt Principles:
//! - Zero hardcoding: Capability-based workgroup dispatch
//! - Batchable: routes through `TensorContext::record_operation()`
//! - Zero-copy output: buffer pool, no GPU→CPU→GPU round-trip
//! - Pipeline cached: `GLOBAL_CACHE` eliminates recompilation overhead

use crate::device::pipeline_cache::{BindGroupLayoutSignature, GLOBAL_CACHE};
use crate::device::tensor_context::get_device_context;
use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;
use bytemuck::{Pod, Zeroable};

/// GPU shader for fused layer normalization (single-pass mean+var, normalize+affine).
#[must_use]
pub fn wgsl_layernorm_fused() -> &'static str {
    static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
        include_str!(
            "../shaders/norm/layernorm_fused_f64.wgsl"
        ).to_string()
    });
    std::sync::LazyLock::force(&SHADER).as_str()
}

/// GPU shader for fused layer normalization v2 (improved numerical stability).
#[must_use]
pub fn wgsl_layernorm_fused_v2() -> &'static str {
    static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
        include_str!(
            "../shaders/norm/layernorm_fused_v2_f64.wgsl"
        ).to_string()
    });
    std::sync::LazyLock::force(&SHADER).as_str()
}

/// GPU shader for optimized layer normalization (vectorized loads).
pub const WGSL_LAYERNORM_OPTIMIZED: &str = include_str!("../shaders/norm/layernorm_optimized.wgsl");

/// f64 canonical — f32 derived via downcast when needed.
const WGSL_LAYERNORM_MEANVAR_F64: &str = include_str!("../shaders/norm/layernorm_meanvar_f64.wgsl");
const WGSL_LAYERNORM_STATS_F64: &str = include_str!("../shaders/norm/layernorm_stats_f64.wgsl");

/// `LayerNorm` mean/variance pass.
pub static WGSL_LAYERNORM_MEANVAR: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    WGSL_LAYERNORM_MEANVAR_F64.to_string()
});

/// `LayerNorm` normalize pass.
#[must_use]
pub fn wgsl_layernorm_normalize() -> &'static str {
    static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
        include_str!(
            "../shaders/norm/layernorm_normalize_f64.wgsl"
        ).to_string()
    });
    std::sync::LazyLock::force(&SHADER).as_str()
}

/// `LayerNorm` stats pass.
pub static WGSL_LAYERNORM_STATS: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    WGSL_LAYERNORM_STATS_F64.to_string()
});

/// `LayerNorm` optimized variant.
#[must_use]
pub fn wgsl_layernorm_opt() -> &'static str {
    static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
        include_str!(
            "../shaders/norm/layernorm_opt_f64.wgsl"
        ).to_string()
    });
    std::sync::LazyLock::force(&SHADER).as_str()
}

/// Base layer norm shader.
#[must_use]
pub fn wgsl_layernorm_base() -> &'static str {
    static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
        include_str!(
            "../shaders/norm/layernorm_f64.wgsl"
        ).to_string()
    });
    std::sync::LazyLock::force(&SHADER).as_str()
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    size: u32,
    feature_size: u32,
    epsilon: f32,
    _pad: u32,
}

/// Layer normalization: normalize each row along the last (feature) dimension.
pub struct LayerNorm {
    input: Tensor,
    epsilon: f32,
}

impl LayerNorm {
    /// Create layer norm with given epsilon for numerical stability.
    #[must_use]
    pub fn new(input: Tensor, epsilon: f32) -> Self {
        Self { input, epsilon }
    }

    pub(crate) fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            include_str!(
                "../shaders/norm/layer_norm_f64.wgsl"
            ).to_string()
        });
        std::sync::LazyLock::force(&SHADER).as_str()
    }

    /// Execute layer normalization.
    /// Dispatch: one workgroup per batch row (shape[:-1]) so each workgroup
    /// normalizes one full feature vector.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let size: usize = shape.iter().product();
        let feature_size = shape[shape.len() - 1];
        let num_rows = (size / feature_size) as u32;

        let ctx = get_device_context(device);
        let caps = DeviceCapabilities::from_device(device);
        let wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
        let workgroups = num_rows.div_ceil(wg_size);
        let adapter_info = device.adapter_info();

        let output_buffer = ctx.acquire_pooled_output(size);

        let params_buf = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("LayerNorm Params"),
                contents: bytemuck::bytes_of(&Params {
                    size: size as u32,
                    feature_size: feature_size as u32,
                    epsilon: self.epsilon,
                    _pad: 0,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let layout_sig = BindGroupLayoutSignature::reduction();
        let bgl = GLOBAL_CACHE.get_or_create_layout(
            device.device(),
            adapter_info,
            layout_sig,
            Some("LayerNorm BGL"),
        );

        let bind_group =
            std::sync::Arc::new(device.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("LayerNorm BG"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.input.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            }));

        let pipeline = GLOBAL_CACHE.get_or_create_pipeline(
            device.device(),
            adapter_info,
            Self::wgsl_shader(),
            layout_sig,
            "main",
            Some("LayerNorm Pipeline"),
        );

        ctx.record_operation(move |encoder| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LayerNorm Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&*bind_group), &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
            drop(params_buf);
        })?;

        Ok(Tensor::from_pooled_buffer(
            output_buffer,
            shape.to_vec(),
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Apply layer normalization (normalize along last dimension).
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn layer_norm_wgsl(self, epsilon: f32) -> Result<Self> {
        LayerNorm::new(self, epsilon).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn get_test_device() -> Option<std::sync::Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_gpu_available().await
    }

    #[tokio::test]
    async fn test_layer_norm_1d() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4], device.clone());
        let output = input.layer_norm_wgsl(1e-5).unwrap();
        assert_eq!(output.shape(), &[1, 4]);
        let result = output.to_vec().unwrap();
        let mean: f32 = result.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "mean={mean}");
    }

    #[tokio::test]
    async fn test_layer_norm_batch() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let input = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            device.clone(),
        );
        let output = input.layer_norm_wgsl(1e-5).unwrap();
        assert_eq!(output.shape(), &[2, 3]);
        let r = output.to_vec().unwrap();
        let mean1 = (r[0] + r[1] + r[2]) / 3.0;
        let mean2 = (r[3] + r[4] + r[5]) / 3.0;
        assert!(mean1.abs() < 1e-5, "batch0 mean={mean1}");
        assert!(mean2.abs() < 1e-5, "batch1 mean={mean2}");
    }
}
