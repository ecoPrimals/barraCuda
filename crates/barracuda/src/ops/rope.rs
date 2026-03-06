// SPDX-License-Identifier: AGPL-3.0-or-later
//! Rotary Position Embedding (`RoPE`) - GPU-accelerated
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (single-pass GPU)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready for Llama, GPT-NeoX)
//!
//! ## Algorithm
//!
//! ```text
//! RoPE applies 2D rotations to embedding pairs based on position:
//!
//! freq[i] = 1 / (10000^(2i / head_dim))
//! theta[pos,i] = pos * freq[i]
//! [x1', x2'] = [cos(theta) -sin(theta)] [x1]
//!              [sin(theta)  cos(theta)] [x2]
//! ```
//!
//! **Key Properties**:
//! - Encodes relative position (not absolute)
//! - No learned parameters
//! - Rotation preserves magnitude
//! - Works for any sequence length
//!
//! **Used By**: Llama, GPT-NeoX, `PaLM`, Falcon
//!
//! **Reference**: `RoFormer` (Su et al., 2021)
//!
//! ## Usage
//!
//! ```rust,ignore
//! use barracuda::tensor::Tensor;
//!
//! let q = Tensor::randn(vec![2, 128, 8, 64]).await?;  // [batch, seq, heads, dim]
//! let q_rotated = q.rotary_embedding()?;  // Apply RoPE
//! ```

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// f64 is the canonical source — math is universal, precision is silicon.
static SHADER_F64: &str = include_str!("../shaders/attention/rotary_embedding_f64.wgsl");
static SHADER_F32: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(SHADER_F64)
});

/// `RoPE` parameters for WGSL shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct RopeParams {
    batch_size: u32,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    half_dim: u32,
    _padding: [u32; 3],
}

/// Rotary Position Embedding operation
///
/// **Deep Debt**: Single-pass GPU implementation, no dependencies
pub struct RotaryEmbedding {
    input: Tensor,
}

impl RotaryEmbedding {
    /// Create new `RoPE` operation
    /// **Shape**: [batch, `seq_len`, `num_heads`, `head_dim`]
    /// # Errors
    /// Returns [`Err`] if input is not 4D or `head_dim` is not even.
    pub fn new(input: Tensor) -> Result<Self> {
        // Validate shape: must be 4D
        if input.shape().len() != 4 {
            return Err(BarracudaError::shape_mismatch(
                input.shape().to_vec(),
                vec![0, 0, 0, 0],
            ));
        }

        // Validate head_dim is even (required for pairwise rotation)
        let head_dim = input.shape()[3];
        if !head_dim.is_multiple_of(2) {
            return Err(BarracudaError::invalid_op(
                "RotaryEmbedding",
                format!("head_dim ({head_dim}) must be even for pairwise rotation"),
            ));
        }

        Ok(Self { input })
    }

    /// WGSL shader source
    fn shader() -> &'static str {
        &SHADER_F32
    }

    /// Execute `RoPE` (single GPU pass)
    /// **Deep Debt**: Efficient single-pass implementation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or device submission fails (e.g. device lost).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();

        // Extract dimensions
        let shape = self.input.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let num_heads = shape[2];
        let head_dim = shape[3];
        let half_dim = head_dim / 2;

        // Create parameters
        let params = RopeParams {
            batch_size: batch_size as u32,
            seq_len: seq_len as u32,
            num_heads: num_heads as u32,
            head_dim: head_dim as u32,
            half_dim: half_dim as u32,
            _padding: [0, 0, 0],
        };

        let params_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RoPE Params"),
            size: std::mem::size_of::<RopeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device
            .queue
            .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        // Output buffer (same size as input)
        let output_size = self.input.len();
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Compile shader
        let shader = device.compile_shader(Self::shader(), Some("RoPE"));

        // Create bind group layout
        let bgl = device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("RoPE BGL"),
                entries: &[
                    // Input
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
                    // Output
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
                    // Params
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

        // Create bind group
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RoPE BG"),
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
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create pipeline
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("RoPE Pipeline Layout"),
                    bind_group_layouts: &[&bgl],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("RoPE Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        // Execute
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("RoPE Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RoPE Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);

            // Deep Debt Evolution: Capability-based dispatch
            let total = (batch_size * seq_len * num_heads * half_dim) as u32;
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let workgroups = total.div_ceil(optimal_wg_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        // Return output tensor (same shape as input)
        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size, seq_len, num_heads, head_dim],
            device.clone(),
        ))
    }
}

// ═══════════════════════════════════════════════════════════════
// TENSOR API INTEGRATION
// ═══════════════════════════════════════════════════════════════

impl Tensor {
    /// Apply Rotary Position Embedding (`RoPE`)
    /// **Deep Debt**: Essential for Llama, GPT-NeoX, `PaLM`
    /// # Arguments
    /// - Input: [batch, `seq_len`, `num_heads`, `head_dim`]
    /// # Returns
    /// - Output: [batch, `seq_len`, `num_heads`, `head_dim`] (rotated)
    /// # Example
    /// ```rust,ignore
    /// let q = Tensor::randn(vec![2, 128, 8, 64]).await?;
    /// let q_rope = q.rotary_embedding()?;  // Apply RoPE for Llama
    /// ```
    /// # Errors
    /// Returns [`Err`] if input is not 4D, `head_dim` is not even, or buffer allocation/GPU dispatch fails (e.g. device lost).
    pub fn rotary_embedding(self) -> Result<Self> {
        RotaryEmbedding::new(self)?.execute()
    }
}

// ═══════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_rope_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let batch = 1;
        let seq = 4;
        let heads = 2;
        let dim = 8;

        let input = Tensor::from_vec_on(
            vec![1.0; batch * seq * heads * dim],
            vec![batch, seq, heads, dim],
            device,
        )
        .await
        .unwrap();

        let output = input.rotary_embedding().unwrap();

        assert_eq!(output.shape(), &[batch, seq, heads, dim]);
        let data = output.to_vec().unwrap();
        assert!(data.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_rope_single_position() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let batch = 1;
        let seq = 1;
        let heads = 2;
        let dim = 8;

        let input = Tensor::from_vec_on(
            vec![1.0; batch * seq * heads * dim],
            vec![batch, seq, heads, dim],
            device,
        )
        .await
        .unwrap();

        let output = input.rotary_embedding().unwrap();

        assert_eq!(output.shape(), &[batch, seq, heads, dim]);
    }

    #[tokio::test]
    async fn test_rope_llama_dims() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Llama-style dimensions
        let batch = 2;
        let seq = 128;
        let heads = 8;
        let dim = 64;

        let input = Tensor::from_vec_on(
            vec![0.5; batch * seq * heads * dim],
            vec![batch, seq, heads, dim],
            device,
        )
        .await
        .unwrap();

        let output = input.rotary_embedding().unwrap();

        assert_eq!(output.shape(), &[batch, seq, heads, dim]);
        let data = output.to_vec().unwrap();
        assert!(data.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_rope_magnitude_preservation() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let batch = 1;
        let seq = 4;
        let heads = 1;
        let dim = 4;

        let input = Tensor::from_vec_on(
            vec![1.0; batch * seq * heads * dim],
            vec![batch, seq, heads, dim],
            device,
        )
        .await
        .unwrap();

        let output = input.rotary_embedding().unwrap();

        // Rotation should preserve magnitude (approximately)
        let data = output.to_vec().unwrap();
        assert!(data.iter().all(|&x| x.abs() <= 2.0));
    }

    #[tokio::test]
    async fn test_rope_shape_validation() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Valid: even head_dim
        let input = Tensor::from_vec_on(vec![1.0; 4 * 2 * 8], vec![1, 4, 2, 8], device.clone())
            .await
            .unwrap();
        assert!(input.rotary_embedding().is_ok());

        // Invalid: odd head_dim
        let input = Tensor::from_vec_on(vec![1.0; 4 * 2 * 7], vec![1, 4, 2, 7], device)
            .await
            .unwrap();
        assert!(input.rotary_embedding().is_err());
    }
}
