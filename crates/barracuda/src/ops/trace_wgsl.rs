// SPDX-License-Identifier: AGPL-3.0-or-later
//! Trace - Sum of diagonal elements - Pure WGSL
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (GPU-optimized)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//!
//! ## Algorithm
//!
//! Computes the trace of a square matrix:
//! ```text
//! trace(A) = sum of diagonal elements
//! For [[a, b], [c, d]]: trace = a + d
//! ```

use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;

/// Sum of diagonal elements of a square matrix.
pub struct Trace {
    input: Tensor,
}

impl Trace {
    /// Create a trace operation for the given square matrix.
    #[must_use]
    pub fn new(input: Tensor) -> Self {
        Self { input }
    }

    /// Execute trace (sum of diagonal elements) on a square matrix.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();

        // Expect 2D square matrix
        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(crate::error::BarracudaError::InvalidShape {
                expected: vec![0, 0],
                actual: shape.to_vec(),
            });
        }

        let n = shape[0];

        // Deep Debt Evolution: Capability-based dispatch
        let caps = DeviceCapabilities::from_device(device);
        let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Reduction);
        let workgroups = (n as u32).div_ceil(optimal_wg_size);

        // Output buffer: single element for final result, or partial results if multi-workgroup
        let output_size = if workgroups > 1 {
            workgroups as usize
        } else {
            1
        };
        let output_buffer = device.create_buffer_f32(output_size)?;

        let params_buffer = device.create_uniform_buffer("Params", &[n as u32, 0u32, 0u32, 0u32]);

        ComputeDispatch::new(device, "Trace")
            .shader(include_str!("../shaders/linalg/trace_f64.wgsl"), "main")
            .storage_read(0, self.input.buffer())
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch(workgroups, 1, 1)
            .submit()?;

        // If multiple workgroups, reduce partial results in a second pass using reduce shader
        let final_buffer = if workgroups > 1 {
            // Second pass: reduce partial results using reduce shader
            let reduce_shader_source = crate::ops::reduce::Reduce::wgsl_shader();

            #[repr(C)]
            #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
            struct ReduceParams {
                size: u32,
                operation: u32, // 0 = Sum
                _pad0: u32,
                _pad1: u32,
            }

            let reduce_params = ReduceParams {
                size: workgroups,
                operation: 0u32, // Sum operation
                _pad0: 0,
                _pad1: 0,
            };

            let final_output_buffer = device.create_buffer_f32(1)?;
            let reduce_params_buffer =
                device.create_uniform_buffer("Trace Reduce Params", &reduce_params);

            let caps_2 = DeviceCapabilities::from_device(device);
            let optimal_wg_size_2 = caps_2.optimal_workgroup_size(WorkloadType::Reduction);
            let workgroups_2 = workgroups.div_ceil(optimal_wg_size_2);

            ComputeDispatch::new(device, "Trace Reduce")
                .shader(reduce_shader_source, "main")
                .storage_read(0, &output_buffer)
                .storage_rw(1, &final_output_buffer)
                .uniform(2, &reduce_params_buffer)
                .dispatch(workgroups_2.max(1), 1, 1)
                .submit()?;

            final_output_buffer
        } else {
            output_buffer
        };

        // Return scalar tensor with trace result
        Ok(Tensor::from_buffer(final_buffer, vec![1], device.clone()))
    }
}

impl Tensor {
    /// Compute the trace (sum of diagonal elements) of this square matrix.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn trace_wgsl(self) -> Result<Self> {
        Trace::new(self).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_trace_2x2() {
        let device = crate::device::test_pool::get_test_device().await;
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::from_vec_on(input_data, vec![2, 2], device)
            .await
            .unwrap();

        let result = input.trace_wgsl().unwrap();
        let trace_result = result.to_vec().unwrap();

        // Result should be scalar tensor [trace_value]
        assert_eq!(trace_result.len(), 1);
        // Trace = 1.0 + 4.0 = 5.0
        assert!((trace_result[0] - 5.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_trace_3x3() {
        let device = crate::device::test_pool::get_test_device().await;
        // Matrix: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        // Diagonal: [1, 5, 9], trace = 15
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let input = Tensor::from_vec_on(input_data, vec![3, 3], device)
            .await
            .unwrap();

        let result = input.trace_wgsl().unwrap();
        let trace_result = result.to_vec().unwrap();

        assert_eq!(trace_result.len(), 1);
        assert!((trace_result[0] - 15.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_trace_large_matrix() {
        let device = crate::device::test_pool::get_test_device().await;
        let n = 512; // Larger than workgroup size to test multi-workgroup reduction
        let mut input_data = vec![0.0; n * n];

        // Fill diagonal with sequential values: 1, 2, 3, ..., n
        for i in 0..n {
            input_data[i * n + i] = (i + 1) as f32;
        }

        let input = Tensor::from_vec_on(input_data, vec![n, n], device)
            .await
            .unwrap();

        let result = input.trace_wgsl().unwrap();
        let trace_result = result.to_vec().unwrap();

        assert_eq!(trace_result.len(), 1);
        // Sum of 1..n = n*(n+1)/2
        let expected = (n * (n + 1) / 2) as f32;
        assert!((trace_result[0] - expected).abs() < 1e-4);
    }
}
