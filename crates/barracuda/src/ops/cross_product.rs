// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross product operation - Vector cross product
//!
//! Computes cross product of 3D vectors
//! a × b = (`a_y`*`b_z` - `a_z`*`b_y`, `a_z`*`b_x` - `a_x`*`b_z`, `a_x`*`b_y` - `a_y`*`b_x`)

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct CrossProductParams {
    num_vectors: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
    _pad5: u32,
    _pad6: u32,
}

/// Cross product operation
pub struct CrossProduct {
    input_a: Tensor,
    input_b: Tensor,
}

impl CrossProduct {
    /// Create cross product operation
    /// # Errors
    /// Returns [`Err`] if inputs are not [N, 3] or batch sizes differ.
    pub fn new(input_a: Tensor, input_b: Tensor) -> Result<Self> {
        let shape_a = input_a.shape();
        let shape_b = input_b.shape();

        if shape_a.len() != 2 || shape_a[1] != 3 {
            return Err(BarracudaError::invalid_op(
                "cross_product",
                format!("input_a must be [N, 3], got shape {shape_a:?}"),
            ));
        }

        if shape_b.len() != 2 || shape_b[1] != 3 {
            return Err(BarracudaError::invalid_op(
                "cross_product",
                format!("input_b must be [N, 3], got shape {shape_b:?}"),
            ));
        }

        if shape_a[0] != shape_b[0] {
            return Err(BarracudaError::invalid_op(
                "cross_product",
                format!(
                    "input_a and input_b must have same batch size, got {} and {}",
                    shape_a[0], shape_b[0]
                ),
            ));
        }

        Ok(Self { input_a, input_b })
    }

    /// WGSL shader source (embedded at compile time)
    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            include_str!(
                "../shaders/misc/cross_product_f64.wgsl"
            ).to_string()
        });
        &SHADER
    }

    /// Execute cross product on tensors
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input_a.device();
        let shape_a = self.input_a.shape();
        let num_vectors = shape_a[0];

        // Create output buffer [N, 3]
        let output_size = num_vectors * 3;
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create params
        let params = CrossProductParams {
            num_vectors: num_vectors as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
            _pad4: 0,
            _pad5: 0,
            _pad6: 0,
        };

        let params_buffer = device.create_uniform_buffer("CrossProduct Params", &params);

        ComputeDispatch::new(device, "CrossProduct")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, self.input_a.buffer())
            .storage_read(1, self.input_b.buffer())
            .storage_rw(2, &output_buffer)
            .uniform(3, &params_buffer)
            .dispatch_1d(num_vectors as u32)
            .submit()?;

        // Create output tensor
        Ok(Tensor::from_buffer(
            output_buffer,
            vec![num_vectors, 3],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_cross_product_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Test vectors: [1, 0, 0] × [0, 1, 0] = [0, 0, 1]
        let input_a = Tensor::from_vec_on(vec![1.0, 0.0, 0.0], vec![1, 3], device.clone())
            .await
            .unwrap();

        let input_b = Tensor::from_vec_on(vec![0.0, 1.0, 0.0], vec![1, 3], device)
            .await
            .unwrap();

        let output = CrossProduct::new(input_a, input_b)
            .unwrap()
            .execute()
            .unwrap();
        let result = output.to_vec().unwrap();

        assert_eq!(result.len(), 3);
        // Result should be approximately [0, 0, 1]
        assert!((result[0] - 0.0).abs() < 1e-5);
        assert!((result[1] - 0.0).abs() < 1e-5);
        assert!((result[2] - 1.0).abs() < 1e-5);
    }
}
