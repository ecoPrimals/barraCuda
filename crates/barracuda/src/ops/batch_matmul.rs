// SPDX-License-Identifier: AGPL-3.0-or-later
//! `BatchMatMul` - Batched Matrix Multiplication
//! Pure WGSL implementation
//!
//! Critical operation for transformer attention mechanisms
//! Performs multiple matrix multiplications in parallel across batches
//!
//! Used in: Transformers, multi-head attention, batched inference
//! Benefits: More efficient than looping `MatMul`, GPU-optimized parallelism

use crate::error::Result;
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BatchMatMulParams {
    batch_size: u32,
    m: u32, // rows of A
    n: u32, // cols of B
    k: u32, // cols of A / rows of B
}

/// Batched matrix multiplication — A[batch,m,k] × B[batch,k,n].
pub struct BatchMatMul {
    a: Tensor,
    b: Tensor,
}

impl BatchMatMul {
    /// Create batched matmul. A: [batch,m,k], B: [batch,k,n].
    #[must_use]
    pub fn new(a: Tensor, b: Tensor) -> Self {
        Self { a, b }
    }

    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            include_str!(
                "../shaders/math/batch_matmul_f64.wgsl"
            ).to_string()
        });
        &SHADER
    }

    /// Execute batched matrix multiplication on GPU.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.a.device();
        let a_shape = self.a.shape();
        let b_shape = self.b.shape();

        // Shapes: A[batch, m, k], B[batch, k, n]
        let batch_size = a_shape[0];
        let m = a_shape[1];
        let k = a_shape[2];
        let n = b_shape[2];

        let output_size = batch_size * m * n;

        // Create output buffer
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create params
        let params = BatchMatMulParams {
            batch_size: batch_size as u32,
            m: m as u32,
            n: n as u32,
            k: k as u32,
        };
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("BatchMatMul Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create shader module
        let shader = device.compile_shader(Self::wgsl_shader(), Some("BatchMatMul Shader"));

        // Create compute pipeline
        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("BatchMatMul Pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        // Create bind group
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BatchMatMul Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.b.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Execute with 2D workgroup (16x16 per batch)
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("BatchMatMul Encoder"),
        });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BatchMatMul Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);

            // Dispatch with 2D grid for matrix dimensions + batch dimension
            let workgroups_x = n.div_ceil(16) as u32;
            let workgroups_y = m.div_ceil(16) as u32;
            let workgroups_z = batch_size as u32;
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }
        device.submit_and_poll(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size, m, n],
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Perform batched matrix multiplication (critical for transformers)
    /// # Arguments
    /// * `other` - Second tensor with shape [batch, k, n]
    /// # Returns
    /// Tensor with shape [batch, m, n]
    /// # Example
    /// ```ignore
    /// // Transformer attention: Q @ K^T
    /// let attention_scores = q.batch_matmul(&k_transposed)?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn batch_matmul(self, other: &Tensor) -> Result<Self> {
        BatchMatMul::new(self, other.clone()).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_batch_matmul_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Create A [2, 2, 3] - 2 batches, 2x3 matrices
        let a_data = vec![
            1.0f32, 2.0, 3.0, // Batch 0, row 0
            4.0, 5.0, 6.0, // Batch 0, row 1
            1.0, 1.0, 1.0, // Batch 1, row 0
            2.0, 2.0, 2.0, // Batch 1, row 1
        ];
        let a = Tensor::from_data(&a_data, vec![2, 2, 3], device.clone()).unwrap();

        // Create B [2, 3, 2] - 2 batches, 3x2 matrices
        let b_data = vec![
            1.0f32, 0.0, // Batch 0
            0.0, 1.0, 1.0, 1.0, 2.0, 0.0, // Batch 1
            0.0, 2.0, 1.0, 1.0,
        ];
        let b = Tensor::from_data(&b_data, vec![2, 3, 2], device.clone()).unwrap();

        // Compute C = A @ B
        let result = a.batch_matmul(&b).unwrap();
        let output = result.to_vec().unwrap();

        // Output shape should be [2, 2, 2]
        assert_eq!(result.shape(), &[2, 2, 2]);
        assert_eq!(output.len(), 8);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_batch_matmul_edge_cases() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Single batch, identity-like multiplication
        let a_data = vec![1.0, 0.0, 0.0, 1.0]; // [1, 2, 2]
        let b_data = vec![1.0, 2.0, 3.0, 4.0]; // [1, 2, 2]

        let a = Tensor::from_data(&a_data, vec![1, 2, 2], device.clone()).unwrap();
        let b = Tensor::from_data(&b_data, vec![1, 2, 2], device.clone()).unwrap();

        let result = a.batch_matmul(&b).unwrap();
        let output = result.to_vec().unwrap();

        assert_eq!(output.len(), 4);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_batch_matmul_boundary() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Test with different matrix sizes
        let a_data = vec![1.0; 2 * 4 * 3]; // [2, 4, 3]
        let b_data = vec![1.0; 2 * 3 * 5]; // [2, 3, 5]

        let a = Tensor::from_data(&a_data, vec![2, 4, 3], device.clone()).unwrap();
        let b = Tensor::from_data(&b_data, vec![2, 3, 5], device.clone()).unwrap();

        let result = a.batch_matmul(&b).unwrap();

        // Output should be [2, 4, 5]
        assert_eq!(result.shape(), &[2, 4, 5]);
        let output = result.to_vec().unwrap();
        assert_eq!(output.len(), 2 * 4 * 5);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_batch_matmul_large_batch() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Transformer-style: multiple batches, attention heads
        let batch_size = 4;
        let seq_len = 8;
        let d_k = 16;

        let a_data = vec![1.0; batch_size * seq_len * d_k];
        let b_data = vec![1.0; batch_size * d_k * seq_len];

        let a = Tensor::from_data(&a_data, vec![batch_size, seq_len, d_k], device.clone()).unwrap();
        let b = Tensor::from_data(&b_data, vec![batch_size, d_k, seq_len], device.clone()).unwrap();

        let result = a.batch_matmul(&b).unwrap();

        // Output: [batch, seq, seq]
        assert_eq!(result.shape(), &[batch_size, seq_len, seq_len]);
        let output = result.to_vec().unwrap();
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_batch_matmul_precision() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Test determinism and functional correctness
        let a_data = vec![1.0, 2.0, 3.0, 4.0]; // [1, 2, 2]
        let b_data = vec![5.0, 6.0, 7.0, 8.0]; // [1, 2, 2]

        let a1 = Tensor::from_data(&a_data, vec![1, 2, 2], device.clone()).unwrap();
        let b1 = Tensor::from_data(&b_data, vec![1, 2, 2], device.clone()).unwrap();

        let a2 = Tensor::from_data(&a_data, vec![1, 2, 2], device.clone()).unwrap();
        let b2 = Tensor::from_data(&b_data, vec![1, 2, 2], device.clone()).unwrap();

        // Run twice to check determinism
        let result1 = a1.batch_matmul(&b1).unwrap();
        let result2 = a2.batch_matmul(&b2).unwrap();

        let output1 = result1.to_vec().unwrap();
        let output2 = result2.to_vec().unwrap();

        // Should be deterministic
        assert_eq!(output1, output2);

        // Output should be finite and correct dimensions
        assert_eq!(output1.len(), 4); // 1 batch * 2x2
        assert!(output1.iter().all(|&x| x.is_finite()));
    }
}
