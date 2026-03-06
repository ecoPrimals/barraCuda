// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fused MLP forward pass — `BatchedEncoder` for single-submit across all layers.

use crate::device::BatchedEncoder;
use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::error::{BarracudaError, Result};
use crate::shaders::precision::downcast_f64_to_f32;
use crate::tensor::Tensor;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const LINEAR_F64: &str = include_str!("../shaders/misc/linear_f64.wgsl");
const RELU_F64: &str = include_str!("../shaders/activation/relu_f64.wgsl");

static LINEAR_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| downcast_f64_to_f32(LINEAR_F64));
static RELU_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| downcast_f64_to_f32(RELU_F64));

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct LinearParams {
    batch_size: u32,
    in_features: u32,
    out_features: u32,
    has_bias: u32,
}

/// Activation for MLP layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// Rectified Linear Unit: max(0, x)
    ReLU,
}

/// Run a fused MLP forward pass using `BatchedEncoder` (single submit).
///
/// # Errors
///
/// Returns [`Err`] if weights/biases lengths mismatch, shape validation fails,
/// or buffer allocation, GPU dispatch, or readback fails (e.g. device lost or
/// out of memory).
pub async fn fused_mlp(
    input: &Tensor,
    weights: &[Tensor],
    biases: &[Tensor],
    activation: Activation,
) -> Result<Tensor> {
    if weights.len() != biases.len() {
        return Err(BarracudaError::InvalidInput {
            message: format!(
                "fused_mlp: weights len {} != biases len {}",
                weights.len(),
                biases.len()
            ),
        });
    }
    if weights.is_empty() {
        return Err(BarracudaError::InvalidInput {
            message: "fused_mlp: at least one layer required".into(),
        });
    }
    let device = input.device();
    let shape = input.shape();
    if shape.len() != 2 {
        return Err(BarracudaError::InvalidInput {
            message: format!("fused_mlp: input must be 2D, got {shape:?}"),
        });
    }
    let batch = shape[0];
    let mut in_features = shape[1];

    let mut layer_dims: Vec<(usize, usize)> = Vec::with_capacity(weights.len());
    let mut layer_buffers: Vec<Arc<wgpu::Buffer>> = Vec::with_capacity(weights.len());

    for (i, (w, b)) in weights.iter().zip(biases.iter()).enumerate() {
        let w_shape = w.shape();
        let (out_features, in_feat) = if w_shape.len() != 2 {
            return Err(BarracudaError::InvalidInput {
                message: format!("fused_mlp: weight[{i}] must be 2D"),
            });
        } else if w_shape[1] == in_features {
            (w_shape[0], w_shape[1])
        } else if w_shape[0] == in_features {
            (w_shape[1], w_shape[0])
        } else {
            return Err(BarracudaError::InvalidInput {
                message: format!("fused_mlp: weight[{i}] shape {w_shape:?} incompatible"),
            });
        };
        if b.shape() != [out_features] {
            return Err(BarracudaError::InvalidInput {
                message: format!("fused_mlp: bias[{i}] shape mismatch"),
            });
        }
        layer_dims.push((in_feat, out_features));
        let buf = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("fused_mlp_layer_{i}")),
            size: ((batch * out_features) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        layer_buffers.push(Arc::new(buf));
        in_features = out_features;
    }

    let params_buffers: Vec<wgpu::Buffer> = layer_dims
        .iter()
        .map(|(in_f, out_f)| {
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("fused_mlp_params"),
                    contents: bytemuck::bytes_of(&LinearParams {
                        batch_size: batch as u32,
                        in_features: *in_f as u32,
                        out_features: *out_f as u32,
                        has_bias: 1,
                    }),
                    usage: wgpu::BufferUsages::UNIFORM,
                })
        })
        .collect();

    let mut batch_enc = BatchedEncoder::new(device.as_ref());
    let mut prev_input = input.buffer();

    for (i, ((_, out_features), (w, b))) in layer_dims
        .iter()
        .zip(weights.iter().zip(biases.iter()))
        .enumerate()
    {
        let out_buf = &layer_buffers[i];
        let _ = batch_enc
            .dispatch("fused_mlp_linear", &LINEAR_F32, "main")
            .uniform(0, &params_buffers[i])
            .storage_read(1, prev_input)
            .storage_read(2, w.buffer())
            .storage_read(3, b.buffer())
            .storage_rw(4, out_buf)
            .workgroups(
                (*out_features as u32).div_ceil(16).max(1),
                (batch as u32).div_ceil(16).max(1),
                1,
            );

        if matches!(activation, Activation::ReLU) {
            let _ = batch_enc
                .dispatch("fused_mlp_relu", &RELU_F32, "main")
                .storage_read(0, out_buf)
                .storage_rw(1, out_buf)
                .workgroups(
                    ((batch * out_features) as u32)
                        .div_ceil(WORKGROUP_SIZE_1D)
                        .max(1),
                    1,
                    1,
                );
        }
        prev_input = out_buf;
    }

    batch_enc.submit();
    let last = layer_buffers
        .last()
        .ok_or_else(|| BarracudaError::InvalidInput {
            message: "fused_mlp: no layer buffers".into(),
        })?
        .clone();
    Ok(Tensor::from_arc_buffer(
        last,
        vec![batch, in_features],
        device.clone(),
    ))
}
