//! GLU - Gated Linear Unit - Pure WGSL
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (GPU-optimized)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//!
//! ## Algorithm
//!
//! Gated Linear Unit:
//! ```text
//! Input x split into two halves: [a, b]
//! glu(x) = a ⊙ sigmoid(b)
//!
//! Output size is half of input size
//! Used in language models and transformers
//! ```

/// f64 is the canonical source — math is universal, precision is silicon.
const SHADER_F64: &str = include_str!("../shaders/activation/glu_f64.wgsl");
pub(crate) static SHADER_F32: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(SHADER_F64)
});

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;

pub struct GLU {
    input: Tensor,
}

impl GLU {
    pub fn new(input: Tensor) -> Self {
        Self { input }
    }

    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let size = self.input.len();

        // Input must be even-sized for splitting
        if !size.is_multiple_of(2) {
            return Err(crate::error::BarracudaError::InvalidShape {
                expected: vec![size / 2 * 2],
                actual: vec![size],
            });
        }

        let half_size = size / 2;
        let output_buffer = device.create_buffer_f32(half_size)?;

        let params_data = [
            size as u32,
            0u32, // split_dim: last dim (standard GLU behavior)
        ];
        let params_buffer = device.create_uniform_buffer("Params", &params_data);

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("GLU BGL"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GLU BG"),
            layout: &bind_group_layout,
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
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let shader = device.compile_shader(Self::wgsl_shader(), Some("GLU"));
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("GLU PL"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("GLU Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GLU Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GLU Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let workgroups = (half_size as u32).div_ceil(optimal_wg_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        // Output shape is half the input size
        let mut output_shape = self.input.shape().to_vec();
        if let Some(last) = output_shape.last_mut() {
            *last = half_size;
        }

        Ok(Tensor::from_buffer(
            output_buffer,
            output_shape,
            device.clone(),
        ))
    }
}

impl Tensor {
    pub fn glu_wgsl(self) -> Result<Self> {
        GLU::new(self).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_glu() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // 6 elements: first 3 for a, last 3 for b
        let input_data = vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0];
        let input = Tensor::from_vec_on(input_data, vec![6], device)
            .await
            .unwrap();

        let result = input.glu_wgsl().unwrap();
        let output = result.to_vec().unwrap();

        // Output should be 3 elements (half of input)
        assert_eq!(output.len(), 3);
        // glu([a, b]) = a * sigmoid(b)
        // sigmoid(0) = 0.5, so output ≈ [0.5, 1.0, 1.5]
        assert!((output[0] - 0.5).abs() < 0.1);
    }
}
