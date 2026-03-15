// SPDX-License-Identifier: AGPL-3.0-only
//! Bincount - Count occurrences of values - Pure WGSL
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (GPU-optimized with atomics)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//!
//! ## Algorithm
//!
//! Counts occurrences of each non-negative integer value:
//! ```text
//! Input:  [0, 1, 1, 2, 2, 2]
//! Output: [1, 2, 3]  (counts for values 0, 1, 2)
//! ```
//! Uses atomic operations for thread-safe GPU counting.

use crate::device::DeviceCapabilities;
use crate::error::Result;
use crate::tensor::Tensor;

/// Count occurrences of non-negative integer values.
pub struct Bincount {
    input: Tensor,
    num_bins: Option<usize>,
}

impl Bincount {
    /// Create a bincount operation. `num_bins` defaults to 256 if None.
    #[must_use]
    pub fn new(input: Tensor, num_bins: Option<usize>) -> Self {
        Self { input, num_bins }
    }

    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            include_str!("../shaders/misc/bincount_f64.wgsl").to_string()
        });
        &SHADER
    }

    /// Execute bincount on GPU. Returns counts as f32.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let input_size = self.input.len();

        // Determine number of bins
        let num_bins = self.num_bins.unwrap_or({
            // Default: max value + 1 (would need to compute max first)
            // For now, use a reasonable default
            256
        });

        // Create output buffer initialized to zeros
        let output_buffer = device.create_buffer_u32_zeros(num_bins)?;

        // Create params buffer
        let params_data = [input_size as u32, num_bins as u32];
        let params_buffer = device.create_uniform_buffer("Params", &params_data);

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Bincount BGL"),
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
            label: Some("Bincount BG"),
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

        let shader = device.compile_shader(Self::wgsl_shader(), Some("Bincount"));
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Bincount PL"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Bincount Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Bincount Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Bincount Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);

            // Dispatch using standard 1D shader workgroup size (256)
            let caps = DeviceCapabilities::from_device(device);
            let workgroups = caps.dispatch_1d(input_size as u32);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_commands(Some(encoder.finish()));

        // Read u32 buffer and convert to f32 for Tensor compatibility
        let u32_data = crate::utils::read_buffer_u32(device, &output_buffer, num_bins)?;
        let f32_data: Vec<f32> = u32_data.iter().map(|&x| x as f32).collect();

        Ok(Tensor::new(f32_data, vec![num_bins], device.clone()))
    }
}

impl Tensor {
    /// Count value occurrences into bins. Input values as f32 (cast from u32).
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn bincount_wgsl(self, num_bins: Option<usize>) -> Result<Self> {
        Bincount::new(self, num_bins).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_bincount_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input_data = [0u32, 1, 1, 2, 2, 2];
        // Convert u32 to f32 for Tensor
        let input_f32: Vec<f32> = input_data.iter().map(|&x| x as f32).collect();
        let input = Tensor::from_vec_on(input_f32, vec![6], device.clone())
            .await
            .unwrap();

        let result = input.bincount_wgsl(Some(3)).unwrap();
        let output_f32 = result.to_vec().unwrap();
        let output: Vec<u32> = output_f32.iter().map(|&x| x as u32).collect();

        // Value 0 appears 1 time
        // Value 1 appears 2 times
        // Value 2 appears 3 times
        assert_eq!(output[0], 1);
        assert_eq!(output[1], 2);
        assert_eq!(output[2], 3);
    }

    #[tokio::test]
    async fn test_bincount_sparse() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input_data = [0u32, 0, 5, 5, 5];
        // Convert u32 to f32 for Tensor
        let input_f32: Vec<f32> = input_data.iter().map(|&x| x as f32).collect();
        let input = Tensor::from_vec_on(input_f32, vec![5], device.clone())
            .await
            .unwrap();

        let result = input.bincount_wgsl(Some(10)).unwrap();
        let output_f32 = result.to_vec().unwrap();
        let output: Vec<u32> = output_f32.iter().map(|&x| x as u32).collect();

        assert_eq!(output[0], 2); // 0 appears twice
        assert_eq!(output[5], 3); // 5 appears three times
        assert_eq!(output[1], 0); // 1 never appears
    }
}
