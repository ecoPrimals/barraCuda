// SPDX-License-Identifier: AGPL-3.0-or-later
//! FHE Coefficient Extraction Operation
//!
//! **Purpose**: Extract specific polynomial coefficients from ciphertext
//!
//! **Algorithm**: Masked coefficient selection
//! - Apply zero-mask except at target index
//! - Extract single coefficient value
//! - Useful for slot-based decryption (CKKS)
//!
//! **Deep Debt Compliance**:
//! - ✅ Pure Rust + WGSL (no unsafe)
//! - ✅ GPU-accelerated (parallel masking)
//! - ✅ Numerically exact (no approximation)
//! - ✅ Production-ready (bounds checking)
//!
//! ## Use Cases
//!
//! 1. **Selective Decryption**: Decrypt single slot without full decryption
//! 2. **Debugging**: Extract specific coefficients for validation
//! 3. **CKKS Slots**: Access individual encrypted values in SIMD packing
//! 4. **Noise Analysis**: Examine specific coefficient noise levels
//!
//! ## Mathematical Background
//!
//! For a polynomial ciphertext:
//! ```text
//! ct(X) = c_0 + c_1*X + c_2*X² + ... + c_{n-1}*X^{n-1}
//! ```
//!
//! Extraction creates masked polynomial:
//! ```text
//! extract(ct, i) = c_i*X^i (all other coefficients = 0)
//! ```
//!
//! After decryption, only the i-th plaintext slot is visible.
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use barracuda::ops::fhe_extract::FheExtract;
//! use barracuda::prelude::Tensor;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Ciphertext polynomial (degree 4096)
//! let ct = Tensor::from_u64_poly(&ciphertext, 4096).await?;
//!
//! // Extract coefficient at index 42
//! let extract_op = FheExtract::new(ct, 4096, 42)?;
//! let ct_masked = extract_op.execute()?;
//!
//! // ct_masked has only coefficient 42 non-zero
//! // Decrypt to get plaintext[42]
//! # Ok(())
//! # }
//! ```

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// FHE Coefficient Extraction operation
///
/// Masks all coefficients except the target index.
pub struct FheExtract {
    input: Tensor,
    degree: u32,
    index: u32,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl FheExtract {
    /// Create a new coefficient extraction operation
    ///
    /// **Parameters**:
    /// - `input`: Ciphertext polynomial (2*degree u32 values, u64 emulated)
    /// - `degree`: Polynomial degree (power of 2)
    /// - `index`: Coefficient index to extract (0 <= index < degree)
    ///
    /// **Returns**: FheExtract operation ready to execute
    ///
    /// **Errors**:
    /// - Invalid degree (not power of 2)
    /// - Index out of bounds (>= degree)
    /// - Input tensor size mismatch
    pub fn new(input: Tensor, degree: u32, index: u32) -> Result<Self> {
        // ✅ VALIDATION: Degree must be power of 2
        if !degree.is_power_of_two() || degree < 4 {
            return Err(BarracudaError::InvalidInput {
                message: format!("Degree must be power of 2 >= 4, got {degree}"),
            });
        }

        // ✅ VALIDATION: Index must be in bounds
        if index >= degree {
            return Err(BarracudaError::InvalidInput {
                message: format!("Index {index} out of bounds for degree {degree}"),
            });
        }

        // ✅ VALIDATION: Input tensor must be 2*degree (u64 as 2xu32)
        let expected_size = (degree * 2) as usize;
        if input.shape()[0] != expected_size {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "Input must have {} elements (degree={}, u64 emulated), got {}",
                    expected_size,
                    degree,
                    input.shape()[0]
                ),
            });
        }

        let device = input.device();

        // Load WGSL shader
        let shader_source = include_str!("fhe_extract.wgsl");
        let shader_module = device
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("FHE Extract Shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("FHE Extract Bind Group Layout"),
                    entries: &[
                        // Input buffer
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
                        // Output buffer
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
                        // Parameters buffer
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

        // Create pipeline
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("FHE Extract Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("FHE Extract Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("extract_coefficient"),
                cache: None,
                compilation_options: Default::default(),
            });

        Ok(Self {
            input,
            degree,
            index,
            pipeline,
            bind_group_layout,
        })
    }

    /// Execute coefficient extraction on GPU
    ///
    /// **Returns**: Tensor with all coefficients zero except at target index
    ///
    /// **Performance**: O(n) GPU parallel execution
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();

        // Create output buffer
        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FHE Extract Output"),
            size: self.input.buffer().size(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create parameters buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct ExtractParams {
            degree: u32,
            target_index: u32,
            _padding: [u32; 2], // Align to 16 bytes
        }

        let params = ExtractParams {
            degree: self.degree,
            target_index: self.index,
            _padding: [0; 2],
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("FHE Extract Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FHE Extract Bind Group"),
            layout: &self.bind_group_layout,
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

        // ✅ GPU EXECUTION: Parallel coefficient masking
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("FHE Extract Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FHE Extract Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);

            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::FHE);
            let workgroups = self.degree.div_ceil(optimal_wg_size);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(std::iter::once(encoder.finish()));

        // Return result tensor
        Ok(Tensor::from_buffer(
            output_buffer,
            self.input.shape().to_vec(),
            device.clone(),
        ))
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_extract_validation() {
        // Test invalid degree
        let result = FheExtract::new(Tensor::zeros(vec![8]).await.unwrap(), 3, 0);
        assert!(result.is_err());

        // Test index out of bounds
        let result = FheExtract::new(Tensor::zeros(vec![8]).await.unwrap(), 4, 4);
        assert!(result.is_err());

        // Test index >= degree
        let result = FheExtract::new(Tensor::zeros(vec![8]).await.unwrap(), 4, 5);
        assert!(result.is_err());
    }

    // NOTE: Full integration tests require GPU + encryption setup
    // See examples/fhe_extract_validation.rs for round-trip tests
}
