// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cosine Similarity
//!
//! **Pure WGSL**: Single implementation via WebGPU shader
//! Computes cosine similarity between pairs of vectors

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CosineSimilarityParams {
    num_vectors_a: u32,
    num_vectors_b: u32,
    vector_dim: u32,
    _padding: u32,
}

/// Cosine similarity between pairs of vectors: cos(θ) = (A·B)/(||A|| ||B||).
pub struct CosineSimilarity {
    vectors_a: Tensor,
    vectors_b: Tensor,
}

impl CosineSimilarity {
    /// Create `CosineSimilarity` operation
    /// # Errors
    /// This function does not return errors; the [`Result`] type is for API consistency.
    pub fn new(vectors_a: Tensor, vectors_b: Tensor) -> Result<Self> {
        Ok(Self {
            vectors_a,
            vectors_b,
        })
    }

    /// WGSL shader source (embedded at compile time)
    fn wgsl_shader() -> &'static str {
        include_str!("../shaders/math/cosine_similarity.wgsl")
    }

    /// Execute `CosineSimilarity` on tensor
    /// # Errors
    /// Returns [`Err`] if vectors are not 2D, vector dimensions do not match,
    /// or buffer allocation/GPU dispatch fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.vectors_a.device();
        let a_shape = self.vectors_a.shape();
        let b_shape = self.vectors_b.shape();

        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(BarracudaError::invalid_op(
                "CosineSimilarity",
                format!(
                    "vectors must be 2D [num_vectors, dim], got shapes {a_shape:?} and {b_shape:?}"
                ),
            ));
        }

        let num_vectors_a = a_shape[0];
        let num_vectors_b = b_shape[0];
        let vector_dim = a_shape[1];

        if b_shape[1] != vector_dim {
            return Err(BarracudaError::invalid_op(
                "CosineSimilarity",
                format!(
                    "vector dimensions must match: {} != {}",
                    vector_dim, b_shape[1]
                ),
            ));
        }

        // Create output buffer: [num_vectors_a, num_vectors_b]
        let output_size = num_vectors_a * num_vectors_b;
        let output_buffer = device.create_buffer_f32(output_size)?;

        let params = CosineSimilarityParams {
            num_vectors_a: num_vectors_a as u32,
            num_vectors_b: num_vectors_b as u32,
            vector_dim: vector_dim as u32,
            _padding: 0,
        };

        let params_buffer = device.create_uniform_buffer("CosineSimilarity Params", &params);

        let workgroups_x = (num_vectors_b as u32).div_ceil(16);
        let workgroups_y = (num_vectors_a as u32).div_ceil(16);

        ComputeDispatch::new(device, "CosineSimilarity")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, self.vectors_a.buffer())
            .storage_read(1, self.vectors_b.buffer())
            .storage_rw(2, &output_buffer)
            .uniform(3, &params_buffer)
            .dispatch(workgroups_x, workgroups_y, 1)
            .submit()?;

        // Create output tensor
        Ok(Tensor::from_buffer(
            output_buffer,
            vec![num_vectors_a, num_vectors_b],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_cosine_similarity_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let num_vectors_a = 3;
        let num_vectors_b = 4;
        let vector_dim = 5;

        let vectors_a = Tensor::from_vec_on(
            vec![1.0; num_vectors_a * vector_dim],
            vec![num_vectors_a, vector_dim],
            device.clone(),
        )
        .await
        .unwrap();

        let vectors_b = Tensor::from_vec_on(
            vec![1.0; num_vectors_b * vector_dim],
            vec![num_vectors_b, vector_dim],
            device.clone(),
        )
        .await
        .unwrap();

        let result = CosineSimilarity::new(vectors_a, vectors_b)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(result.shape(), &[num_vectors_a, num_vectors_b]);
    }
}
