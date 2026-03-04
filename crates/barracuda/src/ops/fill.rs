// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fill operation - Fill tensor with constant value
//! Pure WGSL implementation

use crate::device::{DeviceCapabilities, WgpuDevice, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;
use std::sync::Arc;

/// f64 is the canonical source — math is universal, precision is silicon.
const SHADER_F64: &str = include_str!("../shaders/misc/fill_f64.wgsl");

/// f32 variant derived from f64 via precision downcast.
static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| crate::shaders::precision::downcast_f64_to_f32(SHADER_F64));

/// Meshgrid shader (expand coords to grid).
pub fn wgsl_meshgrid() -> &'static str {
    static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
        crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
            "../shaders/misc/meshgrid_f64.wgsl"
        ))
    });
    std::sync::LazyLock::force(&SHADER).as_str()
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct FillParams {
    value: f32,
    _padding: [f32; 7],
}

pub struct Fill {
    shape: Vec<usize>,
    value: f32,
    device: Arc<WgpuDevice>,
}

impl Fill {
    pub fn new(shape: Vec<usize>, value: f32, device: Arc<WgpuDevice>) -> Self {
        Self {
            shape,
            value,
            device,
        }
    }

    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

    pub fn execute(self) -> Result<Tensor> {
        let size: usize = self.shape.iter().product();

        // Create output buffer
        let output_buffer = self.device.create_buffer_f32(size)?;

        // Create params
        let params = FillParams {
            value: self.value,
            _padding: [0.0; 7],
        };
        let params_buffer =
            self.device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Fill Params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        // Create shader module
        let shader = self
            .device
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Fill Shader"),
                source: wgpu::ShaderSource::Wgsl(Self::wgsl_shader().into()),
            });

        // Create compute pipeline
        let pipeline =
            self.device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Fill Pipeline"),
                    layout: None,
                    module: &shader,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        // Create bind group
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Fill Bind Group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: output_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

        // Execute
        let mut encoder = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("Fill Encoder"),
            });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Fill Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);
            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(&self.device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let workgroups = (size as u32).div_ceil(optimal_wg_size);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.device.submit_and_poll(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            self.shape,
            self.device.clone(),
        ))
    }
}

impl Tensor {
    /// Create tensor filled with constant value
    /// # Arguments
    /// * `shape` - Shape of the tensor
    /// * `value` - Value to fill with
    /// * `device` - Device to create tensor on
    pub fn fill(shape: Vec<usize>, value: f32, device: Arc<WgpuDevice>) -> Result<Self> {
        Fill::new(shape, value, device).execute()
    }

    /// Fill this tensor with a constant value (in-place operation concept)
    pub fn fill_with(self, value: f32) -> Result<Self> {
        Fill::new(self.shape().to_vec(), value, self.device().clone()).execute()
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    async fn get_test_device() -> Option<std::sync::Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_gpu_available().await
    }

    #[tokio::test]
    async fn test_fill_basic() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // Fill [3, 4] tensor with 7.5
        let result = Tensor::fill(vec![3, 4], 7.5, device).unwrap();
        let output = result.to_vec().unwrap();

        // All 12 elements should be 7.5
        assert_eq!(output.len(), 12);
        for val in output.iter() {
            assert_eq!(*val, 7.5);
        }
    }

    #[tokio::test]
    async fn test_fill_edge_cases() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // Single element
        let result = Tensor::fill(vec![1], 99.0, device.clone()).unwrap();
        let output = result.to_vec().unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0], 99.0);

        // Fill with zero
        let result = Tensor::fill(vec![5, 5], 0.0, device.clone()).unwrap();
        let output = result.to_vec().unwrap();
        assert_eq!(output.len(), 25);
        assert!(output.iter().all(|&x| x == 0.0));

        // Negative value
        let result = Tensor::fill(vec![3], -5.5, device).unwrap();
        let output = result.to_vec().unwrap();
        assert!(output.iter().all(|&x| x == -5.5));
    }

    #[tokio::test]
    async fn test_fill_boundary() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // Very small value
        let result = Tensor::fill(vec![4], 1e-10, device.clone()).unwrap();
        let output = result.to_vec().unwrap();
        assert!(output.iter().all(|&x| (x - 1e-10).abs() < 1e-15));

        // Very large value
        let result = Tensor::fill(vec![4], 1e10, device.clone()).unwrap();
        let output = result.to_vec().unwrap();
        assert!(output.iter().all(|&x| (x - 1e10).abs() < 1e5));

        // Different shapes
        let result = Tensor::fill(vec![2, 3, 4], 2.78, device).unwrap();
        let output = result.to_vec().unwrap();
        assert_eq!(output.len(), 24);
        assert!(output.iter().all(|&x| (x - 2.78).abs() < 1e-6));
    }

    #[tokio::test]
    async fn test_fill_large_batch() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // Large tensor
        let result = Tensor::fill(vec![100, 100], 42.0, device).unwrap();
        let output = result.to_vec().unwrap();
        assert_eq!(output.len(), 10_000);
        assert!(output.iter().all(|&x| x == 42.0));
    }

    #[tokio::test]
    async fn test_fill_precision() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // Fractional value with exact FP32 representation
        let result = Tensor::fill(vec![10], 2.5, device.clone()).unwrap();
        let output = result.to_vec().unwrap();
        assert!(output.iter().all(|&x| x == 2.5));

        // Value requiring precision
        let result = Tensor::fill(vec![5], 1.234567, device).unwrap();
        let output = result.to_vec().unwrap();
        for val in output.iter() {
            assert!((val - 1.234567).abs() < 1e-6);
        }
    }
}
