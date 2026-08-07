// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU compute operations for Expand
//!
//! This module contains broadcasting shape computation, stride calculation,
//! and GPU pipeline execution for the expand operation.

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};

/// f64 is the canonical source — f32 derived via `downcast_f64_to_f32` when needed.
const SHADER_F64: &str = include_str!("../../shaders/math/expand_f64.wgsl");
use crate::tensor::Tensor;

/// Compute broadcasted input shape following `NumPy` broadcasting rules
///
/// Broadcasting rules:
/// - Dimensions are compared right-to-left
/// - Each dimension must either:
///   - Be equal
///   - One of them is 1
///   - One of them doesn't exist (implicitly 1)
/// - For expand operation:
///   - If `target_rank` > `input_rank`: pad dimensions at the back (right) with 1s
///   - If `target_rank` == `input_rank`: try padding at front first, then validate
pub fn compute_broadcast_shape(
    input_shape: &[usize],
    target_shape: &[usize],
) -> Result<Vec<usize>> {
    let input_rank = input_shape.len();
    let target_rank = target_shape.len();

    let mut broadcasted_input_shape = vec![1; target_rank];

    match target_rank.cmp(&input_rank) {
        std::cmp::Ordering::Greater => {
            for (i, &dim) in input_shape.iter().enumerate() {
                broadcasted_input_shape[i] = dim;
            }
        }
        std::cmp::Ordering::Equal => {
            // Same rank: check if dimensions are compatible
            // For [3] → [9]: this will be handled specially in execute_expand
            // For validation here, we allow it if target is a multiple of input (1D case)
            let mut compatible = true;
            for i in (0..target_rank).rev() {
                let input_dim = input_shape[i];
                let target_dim = target_shape[i];
                if input_dim != target_dim && input_dim != 1 && target_dim != 1 {
                    // Special case: if this is the last (only) dimension and target is a multiple of input
                    // This will be handled specially in execute_expand by reshaping
                    if i == target_rank - 1
                        && target_rank == 1
                        && target_dim.is_multiple_of(input_dim)
                    {
                        // Allow it - will be handled in execute_expand
                        compatible = true;
                        break;
                    }
                    compatible = false;
                    break;
                }
            }

            if compatible {
                broadcasted_input_shape = input_shape.to_vec();
            } else {
                // Try back-padding
                for (i, &dim) in input_shape.iter().enumerate() {
                    broadcasted_input_shape[i] = dim;
                }
            }
        }
        std::cmp::Ordering::Less => {
            let offset = target_rank.saturating_sub(input_rank);
            for (i, &dim) in input_shape.iter().enumerate() {
                if offset + i < target_rank {
                    broadcasted_input_shape[offset + i] = dim;
                }
            }
        }
    }

    // Validate broadcasting compatibility (right-to-left)
    for i in (0..target_rank).rev() {
        let input_dim = broadcasted_input_shape[i];
        let target_dim = target_shape[i];

        if input_dim != target_dim && input_dim != 1 && target_dim != 1 {
            return Err(BarracudaError::InvalidShape {
                expected: target_shape.to_vec(),
                actual: input_shape.to_vec(),
            });
        }
    }

    Ok(broadcasted_input_shape)
}

/// Compute input strides for broadcasting
///
/// For broadcasting: if shape[i] == 1, stride[i] = 0
fn compute_input_strides(broadcasted_input_shape: &[usize], num_dims: usize) -> Vec<usize> {
    let mut input_strides = vec![0; num_dims];

    // Compute strides backwards
    // Start with last dimension
    if broadcasted_input_shape[num_dims - 1] != 1 {
        input_strides[num_dims - 1] = 1;
    }

    // For each dimension, compute stride as product of subsequent dimensions
    for i in (0..num_dims - 1).rev() {
        if broadcasted_input_shape[i] == 1 {
            // Broadcast dimension: stride is 0
            input_strides[i] = 0;
        } else {
            // Compute stride as product of all dimensions after this one
            let mut stride = 1u32;
            for &dim in &broadcasted_input_shape[(i + 1)..num_dims] {
                stride *= dim as u32;
            }
            input_strides[i] = stride as usize;
        }
    }

    input_strides
}

/// Compute output strides
fn compute_output_strides(target_shape: &[usize], num_dims: usize) -> Vec<usize> {
    let mut output_strides = vec![1; num_dims];
    for i in (0..num_dims - 1).rev() {
        output_strides[i] = output_strides[i + 1] * target_shape[i + 1];
    }
    output_strides
}

/// Execute expand operation on GPU
pub fn execute_expand(input: Tensor, target_shape: Vec<usize>) -> Result<Tensor> {
    let device = input.device();
    let input_shape = input.shape();
    let output_size: usize = target_shape.iter().product();
    // Keep the original target shape for the output tensor
    let original_target_shape = target_shape.clone();

    // Special case: handle [3] → [9] type expansions where ranks are equal
    // but target size is a multiple of input size
    // We treat this as adding a leading dimension: [3] → [1, 3] → [3, 3] → [9]
    let (effective_target_shape, effective_broadcasted_input_shape) = if input_shape.len()
        == target_shape.len()
        && input_shape.len() == 1
        && target_shape[0].is_multiple_of(input_shape[0])
        && target_shape[0] != input_shape[0]
    {
        // [3] → [9]: treat as [1, 3] → [3, 3]
        let leading_dim = target_shape[0] / input_shape[0];
        let effective_target = vec![leading_dim, input_shape[0]];
        let effective_input = vec![1, input_shape[0]];
        (effective_target, effective_input)
    } else {
        // Normal case: use target_shape as-is
        let broadcasted = compute_broadcast_shape(input_shape, &target_shape)?;
        (target_shape, broadcasted)
    };

    let broadcasted_input_shape = effective_broadcasted_input_shape;
    let target_shape = effective_target_shape;

    // Compute strides
    let num_dims = target_shape.len();
    let input_strides = compute_input_strides(&broadcasted_input_shape, num_dims);
    let output_strides = compute_output_strides(&target_shape, num_dims);

    let output_buffer = device.create_buffer_f32(output_size)?;

    device.encoding_guard();

    let input_shape_buffer = device
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Expand Input Shape"),
            contents: bytemuck::cast_slice(
                &broadcasted_input_shape
                    .iter()
                    .map(|&x| x as u32)
                    .collect::<Vec<_>>(),
            ),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

    let output_shape_buffer = device
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Expand Output Shape"),
            contents: bytemuck::cast_slice(
                &target_shape.iter().map(|&x| x as u32).collect::<Vec<_>>(),
            ),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

    let input_strides_buffer =
        device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Expand Input Strides"),
                contents: bytemuck::cast_slice(
                    &input_strides.iter().map(|&x| x as u32).collect::<Vec<_>>(),
                ),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

    let output_strides_buffer =
        device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Expand Output Strides"),
                contents: bytemuck::cast_slice(
                    &output_strides.iter().map(|&x| x as u32).collect::<Vec<_>>(),
                ),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

    // Create params buffer
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params {
        output_size: u32,
        num_dims: u32,
        _pad1: u32,
        _pad2: u32,
    }

    let params = Params {
        output_size: output_size as u32,
        num_dims: num_dims as u32,
        _pad1: 0,
        _pad2: 0,
    };
    let params_buffer = device
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Expand Params"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

    ComputeDispatch::new(device, "Expand")
        .shader(SHADER_F64, "main")
        .uniform(0, &params_buffer)
        .storage_read(1, input.buffer())
        .storage_read(2, &input_shape_buffer)
        .storage_read(3, &output_shape_buffer)
        .storage_read(4, &input_strides_buffer)
        .storage_read(5, &output_strides_buffer)
        .storage_rw(6, &output_buffer)
        .dispatch_1d(output_size as u32)
        .submit()?;

    device.encoding_complete();

    Ok(Tensor::from_buffer(
        output_buffer,
        original_target_shape,
        device.clone(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_shape_same_shape() {
        let result = compute_broadcast_shape(&[3, 4], &[3, 4]).unwrap();
        assert_eq!(result, vec![3, 4]);
    }

    #[test]
    fn test_broadcast_shape_expand_ones() {
        let result = compute_broadcast_shape(&[1, 4], &[3, 4]).unwrap();
        assert_eq!(result, vec![1, 4]);
    }

    #[test]
    fn test_broadcast_shape_rank_expansion() {
        let result = compute_broadcast_shape(&[3], &[3, 5]).unwrap();
        assert_eq!(result, vec![3, 1]);
    }

    #[test]
    fn test_broadcast_shape_1d_exact_match() {
        let result = compute_broadcast_shape(&[9], &[9]).unwrap();
        assert_eq!(result, vec![9]);
    }

    #[test]
    fn test_broadcast_shape_scalar_to_vector() {
        let result = compute_broadcast_shape(&[1], &[5]).unwrap();
        assert_eq!(result, vec![1]);
    }

    #[test]
    fn test_broadcast_shape_multi_expand() {
        let result = compute_broadcast_shape(&[1, 1, 4], &[2, 3, 4]).unwrap();
        assert_eq!(result, vec![1, 1, 4]);
    }

    #[test]
    fn test_broadcast_shape_3d_to_4d() {
        let result = compute_broadcast_shape(&[3, 1, 5], &[3, 1, 5, 7]).unwrap();
        assert_eq!(result, vec![3, 1, 5, 1]);
    }

    #[test]
    fn test_broadcast_shape_incompatible_fails() {
        let result = compute_broadcast_shape(&[3, 5], &[3, 4]);
        assert!(result.is_err());
        if let Err(BarracudaError::InvalidShape { expected, actual }) = result {
            assert_eq!(expected, vec![3, 4]);
            assert_eq!(actual, vec![3, 5]);
        } else {
            panic!("Expected InvalidShape error");
        }
    }

    #[test]
    fn test_broadcast_shape_incompatible_non_divisible() {
        let result = compute_broadcast_shape(&[4], &[7]);
        assert!(result.is_err());
    }

    #[test]
    fn test_broadcast_shape_empty_target() {
        let result = compute_broadcast_shape(&[3], &[]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_input_strides_simple_2d() {
        let strides = compute_input_strides(&[3, 4], 2);
        assert_eq!(strides, vec![4, 1]);
    }

    #[test]
    fn test_input_strides_broadcast_first_dim() {
        let strides = compute_input_strides(&[1, 4], 2);
        assert_eq!(strides, vec![0, 1]);
    }

    #[test]
    fn test_input_strides_broadcast_last_dim() {
        let strides = compute_input_strides(&[3, 1], 2);
        assert_eq!(strides, vec![1, 0]);
    }

    #[test]
    fn test_input_strides_all_broadcast() {
        let strides = compute_input_strides(&[1, 1, 1], 3);
        assert_eq!(strides, vec![0, 0, 0]);
    }

    #[test]
    fn test_input_strides_3d() {
        let strides = compute_input_strides(&[2, 3, 4], 3);
        assert_eq!(strides, vec![12, 4, 1]);
    }

    #[test]
    fn test_output_strides_simple_2d() {
        let strides = compute_output_strides(&[3, 4], 2);
        assert_eq!(strides, vec![4, 1]);
    }

    #[test]
    fn test_output_strides_3d() {
        let strides = compute_output_strides(&[2, 3, 4], 3);
        assert_eq!(strides, vec![12, 4, 1]);
    }

    #[test]
    fn test_output_strides_4d() {
        let strides = compute_output_strides(&[2, 3, 4, 5], 4);
        assert_eq!(strides, vec![60, 20, 5, 1]);
    }

    #[test]
    fn test_output_strides_1d() {
        let strides = compute_output_strides(&[10], 1);
        assert_eq!(strides, vec![1]);
    }
}
