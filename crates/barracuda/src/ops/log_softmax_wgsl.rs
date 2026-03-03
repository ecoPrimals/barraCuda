//! Log-Softmax — GPU-resident, pipeline-cached, batchable
//!
//! `log_softmax(x_i) = x_i − log(Σ exp(x_j))` (numerically stable)
//!
//! Deep Debt Principles:
//! - Zero hardcoding: Capability-based workgroup dispatch
//! - Batchable: routes through TensorContext::record_operation()
//! - Zero-copy output: buffer pool
//! - Pipeline cached: GLOBAL_CACHE eliminates recompilation overhead

use crate::device::pipeline_cache::{BindGroupLayoutSignature, GLOBAL_CACHE};
use crate::device::tensor_context::get_device_context;
use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// f64 is the canonical source — math is universal, precision is silicon.
const SHADER_F64: &str = include_str!("../shaders/activation/log_softmax_f64.wgsl");
static SHADER_F32: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(SHADER_F64)
});

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    batch_size: u32,
    feature_size: u32,
}

/// Log-Softmax along the last (feature) dimension.
pub struct LogSoftmax {
    input: Tensor,
}

impl LogSoftmax {
    pub fn new(input: Tensor) -> Self {
        Self { input }
    }

    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let size: usize = shape.iter().product();
        let feature_size = shape[shape.len() - 1];
        let batch_size = (size / feature_size) as u32;

        let ctx = get_device_context(device);
        let caps = DeviceCapabilities::from_device(device);
        let wg_size = caps.optimal_workgroup_size(WorkloadType::Reduction);
        let workgroups = batch_size.div_ceil(wg_size);
        let adapter_info = device.adapter_info();

        // reduction() = (1 read-only, 1 read-write, 1 uniform)
        let layout_sig = BindGroupLayoutSignature::reduction();
        let output_buffer = ctx.acquire_pooled_output(size);

        let params_buf = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("LogSoftmax Params"),
                contents: bytemuck::bytes_of(&Params {
                    batch_size,
                    feature_size: feature_size as u32,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bgl = GLOBAL_CACHE.get_or_create_layout(
            device.device(),
            adapter_info,
            layout_sig,
            Some("LogSoftmax BGL"),
        );

        let bind_group =
            std::sync::Arc::new(device.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("LogSoftmax BG"),
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
            Some("LogSoftmax Pipeline"),
        );

        ctx.record_operation(move |encoder| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LogSoftmax Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
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
    /// Apply log-softmax along last dimension (GPU-resident, batchable).
    pub fn log_softmax_wgsl(self) -> Result<Self> {
        LogSoftmax::new(self).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn get_test_device() -> Option<std::sync::Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_gpu_available().await
    }

    #[tokio::test]
    async fn test_log_softmax_negative() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3], device.clone());
        let output = input.log_softmax_wgsl().unwrap();
        assert_eq!(output.shape(), &[1, 3]);
        let result = output.to_vec().unwrap();
        // log-softmax values must all be ≤ 0 (since softmax values are in (0,1])
        assert!(
            result.iter().all(|&v| v <= 0.0),
            "All log-softmax values must be ≤ 0: {result:?}"
        );
    }

    #[tokio::test]
    async fn test_log_softmax_batch() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let input = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            device.clone(),
        );
        let output = input.log_softmax_wgsl().unwrap();
        assert_eq!(output.shape(), &[2, 3]);
        assert_eq!(output.to_vec().unwrap().len(), 6);
    }
}
