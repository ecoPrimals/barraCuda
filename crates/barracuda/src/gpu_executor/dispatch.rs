// SPDX-License-Identifier: AGPL-3.0-only
//! `MathOp` dispatch logic for GPU execution.
//!
//! Contains the actual operation dispatch (match on `MathOp`) and the
//! `build_tensor` helper for zero-copy / fallback tensor construction.

use super::GpuExecutor;
use super::storage::GpuTensorStorage;
use crate::cpu_executor::CpuExecutor;
use crate::device::WgpuDevice;
use crate::error::Result;
use crate::unified_hardware::{ComputeExecutor, TensorStorage};
use crate::unified_math::{DType, MathOp};
use std::sync::Arc;

/// Build a `Tensor` from storage (zero-copy fast path or CPU fallback).
///
/// Fast path (D-S19-001 resolved): if the storage is already a
/// `GpuTensorStorage` on this device, reuse its `Arc<wgpu::Buffer>`
/// directly via `Tensor::from_arc_buffer` — zero GPU↔CPU transfers.
///
/// Slow path (fallback): CPU round-trip for cross-device or non-GPU storage.
pub async fn build_tensor(
    storage: &Arc<dyn TensorStorage>,
    device: &Arc<WgpuDevice>,
) -> Result<crate::tensor::Tensor> {
    let desc = storage.descriptor();
    let shape = desc.shape.clone();

    // Zero-copy path: storage already has a wgpu::Buffer
    if let Some(buffer) = storage.as_wgpu_buffer() {
        return Ok(crate::tensor::Tensor::from_arc_buffer(
            buffer,
            shape,
            device.clone(),
        ));
    }

    // Fallback: read from CPU and upload (cross-device or CPU storage)
    let data_bytes = storage.read_to_cpu().await?;
    let numel = desc.numel;
    let elem = desc.dtype.size_bytes();
    if data_bytes.len() < numel * elem {
        return Err(crate::error::BarracudaError::InvalidInput {
            message: format!(
                "execute: expected {} bytes for {numel} × {dtype:?} elements, got {}",
                numel * elem,
                data_bytes.len(),
                dtype = desc.dtype
            ),
        });
    }
    let floats: Vec<f32> = match desc.dtype {
        DType::F32 => data_bytes
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        DType::F64 => data_bytes
            .chunks_exact(8)
            .map(|c| f64::from_ne_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect(),
        DType::I32 => data_bytes
            .chunks_exact(4)
            .map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]]) as f32)
            .collect(),
        DType::I64 => data_bytes
            .chunks_exact(8)
            .map(|c| i64::from_ne_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect(),
        DType::U32 => data_bytes
            .chunks_exact(4)
            .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]) as f32)
            .collect(),
        DType::U64 => data_bytes
            .chunks_exact(8)
            .map(|c| u64::from_ne_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect(),
        DType::Bool => data_bytes
            .iter()
            .map(|&b| if b != 0 { 1.0f32 } else { 0.0 })
            .collect(),
    };
    crate::tensor::Tensor::from_data(&floats, desc.shape.clone(), device.clone())
}

/// Execute the given `MathOp` on the GPU, returning the result as `TensorStorage`.
pub(super) async fn execute_dispatch(
    op: &MathOp,
    inputs: Vec<Arc<dyn TensorStorage>>,
    executor: &GpuExecutor,
) -> Result<Arc<dyn TensorStorage>> {
    let device = executor.wgpu_device_arc();

    let output_tensor: crate::tensor::Tensor = match op {
        // ── Unary ops ───────────────────────────────────────────────────
        MathOp::Negate => build_tensor(&inputs[0], device)
            .await?
            .mul_scalar(-1.0f32)?,
        MathOp::Abs => build_tensor(&inputs[0], device).await?.abs_wgsl()?,
        MathOp::Sqrt => build_tensor(&inputs[0], device).await?.sqrt_wgsl()?,
        MathOp::Exp => build_tensor(&inputs[0], device).await?.exp_wgsl()?,
        MathOp::Log => build_tensor(&inputs[0], device).await?.log_wgsl()?,
        MathOp::Sin => build_tensor(&inputs[0], device).await?.sin_wgsl()?,
        MathOp::Cos => build_tensor(&inputs[0], device).await?.cos_wgsl()?,
        MathOp::Tan => build_tensor(&inputs[0], device).await?.tan_wgsl()?,
        MathOp::Reciprocal => build_tensor(&inputs[0], device).await?.reciprocal_wgsl()?,
        MathOp::Square => {
            let t = build_tensor(&inputs[0], device).await?;
            t.mul(&t)?
        }

        // ── Binary ops ──────────────────────────────────────────────────
        MathOp::Add => {
            let a = build_tensor(&inputs[0], device).await?;
            let b = build_tensor(&inputs[1], device).await?;
            a.add(&b)?
        }
        MathOp::Sub => {
            let a = build_tensor(&inputs[0], device).await?;
            let b = build_tensor(&inputs[1], device).await?;
            a.sub(&b)?
        }
        MathOp::Mul => {
            let a = build_tensor(&inputs[0], device).await?;
            let b = build_tensor(&inputs[1], device).await?;
            a.mul(&b)?
        }
        MathOp::Div => {
            let a = build_tensor(&inputs[0], device).await?;
            let b = build_tensor(&inputs[1], device).await?;
            a.div(&b)?
        }

        // ── Matrix multiply ─────────────────────────────────────────────
        MathOp::MatMul { .. } | MathOp::BatchMatMul { .. } => {
            let a = build_tensor(&inputs[0], device).await?;
            let b = build_tensor(&inputs[1], device).await?;
            a.matmul(&b)?
        }

        // ── Activation ops ──────────────────────────────────────────────
        MathOp::Softmax { .. } => build_tensor(&inputs[0], device).await?.softmax()?,
        MathOp::ReLU => build_tensor(&inputs[0], device).await?.relu()?,
        MathOp::Sigmoid => build_tensor(&inputs[0], device).await?.sigmoid()?,
        MathOp::Tanh => build_tensor(&inputs[0], device).await?.tanh()?,
        MathOp::GELU => build_tensor(&inputs[0], device).await?.gelu_wgsl()?,

        // ── Reductions ──────────────────────────────────────────────────
        MathOp::ReduceSum { .. } => build_tensor(&inputs[0], device).await?.sum()?,
        MathOp::ReduceMean { .. } => build_tensor(&inputs[0], device).await?.mean()?,
        MathOp::ReduceMax { .. } => build_tensor(&inputs[0], device).await?.max()?,
        MathOp::ReduceMin { .. } => build_tensor(&inputs[0], device).await?.min()?,
        MathOp::ReduceProd { .. } => build_tensor(&inputs[0], device).await?.prod()?,

        // ── Pow (scalar exponent via GPU, extracts first element of b) ─
        MathOp::Pow => {
            let a = build_tensor(&inputs[0], device).await?;
            let b_data = inputs[1].read_to_cpu().await?;
            let exp = if b_data.len() >= 4 {
                f32::from_ne_bytes([b_data[0], b_data[1], b_data[2], b_data[3]])
            } else {
                2.0f32
            };
            a.pow_wgsl(exp)?
        }

        // ── Binary Max / Min (elementwise, CPU fallback pending GPU kernel) ─
        MathOp::Max | MathOp::Min => {
            let a_data = inputs[0].read_to_cpu().await?;
            let b_data = inputs[1].read_to_cpu().await?;
            let a_f32: Vec<f32> = a_data
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let b_f32: Vec<f32> = b_data
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let result: Vec<f32> = a_f32
                .iter()
                .zip(b_f32.iter())
                .map(|(&a, &b)| {
                    if matches!(op, MathOp::Max) {
                        a.max(b)
                    } else {
                        a.min(b)
                    }
                })
                .collect();
            crate::tensor::Tensor::from_data(
                &result,
                inputs[0].descriptor().shape.clone(),
                executor.wgpu_device_arc().clone(),
            )?
        }

        // ── Shape ops ───────────────────────────────────────────────────
        MathOp::Reshape { new_shape } => {
            let t = build_tensor(&inputs[0], device).await?;
            t.reshape(new_shape.iter().map(|&x| x as usize).collect())?
        }
        MathOp::Transpose { .. } => build_tensor(&inputs[0], device).await?.transpose()?,
        MathOp::Squeeze { .. } => build_tensor(&inputs[0], device).await?.squeeze()?,
        MathOp::Unsqueeze { dims } => {
            let axis = dims.first().copied().unwrap_or(0);
            build_tensor(&inputs[0], device).await?.unsqueeze(axis)?
        }
        MathOp::Broadcast { target_shape } => build_tensor(&inputs[0], device)
            .await?
            .broadcast(target_shape.clone())?,
        MathOp::Concat { .. } => {
            if inputs.len() < 2 {
                return Err(crate::error::BarracudaError::InvalidInput {
                    message: "Concat requires at least 2 inputs".to_string(),
                });
            }
            let a = build_tensor(&inputs[0], device).await?;
            let b = build_tensor(&inputs[1], device).await?;
            a.concat(&b)?
        }
        MathOp::Split { sizes, .. } => {
            let t = build_tensor(&inputs[0], device).await?;
            let split_point = sizes.first().copied().unwrap_or(t.len() / 2);
            let (first, _second) = t.split(split_point)?;
            first
        }

        // ── Convolution ops (full NCHW GPU via Conv2dGpu) ──────────────────
        // Uses the full NCHW shader with stride, padding, dilation, groups.
        // Handles 2D inputs by promoting to [1,1,H,W], and falls back to
        // the simple Conv2D op for trivial 2D cases without NCHW overhead.
        MathOp::Conv2D {
            stride: (stride_h, stride_w),
            padding: (pad_h, pad_w),
            dilation: (dil_h, dil_w),
            groups,
        } => {
            if inputs.len() < 2 {
                return Err(crate::error::BarracudaError::InvalidInput {
                    message: "Conv2D requires 2 inputs (input, kernel)".to_string(),
                });
            }
            let in_desc = inputs[0].descriptor();
            let kernel_desc = inputs[1].descriptor();

            let can_nchw = in_desc.shape.len() == 4 && kernel_desc.shape.len() == 4;
            let can_promote_2d = in_desc.shape.len() == 2
                && kernel_desc.shape.len() >= 2
                && *stride_h == 1
                && *stride_w == 1
                && *pad_h == 0
                && *pad_w == 0
                && *dil_h == 1
                && *dil_w == 1
                && *groups == 1;

            if can_nchw {
                let input_t = build_tensor(&inputs[0], device).await?;
                let kernel_t = build_tensor(&inputs[1], device).await?;
                let bias_t = if inputs.len() > 2 {
                    Some(build_tensor(&inputs[2], device).await?)
                } else {
                    None
                };
                crate::ops::nn::Conv2dGpu {
                    input: input_t,
                    kernel: kernel_t,
                    bias: bias_t,
                    stride: (*stride_h, *stride_w),
                    padding: (*pad_h, *pad_w),
                    dilation: (*dil_h, *dil_w),
                    groups: *groups,
                }
                .execute()?
            } else if can_promote_2d {
                let input_t = build_tensor(&inputs[0], device).await?;
                let kernel_t = build_tensor(&inputs[1], device).await?;
                let (k_h, k_w) = if kernel_desc.shape.len() == 2 {
                    (kernel_desc.shape[0], kernel_desc.shape[1])
                } else {
                    (kernel_desc.shape[2], kernel_desc.shape[3])
                };
                let kernel_2d = kernel_t.reshape(vec![k_h, k_w])?;
                input_t.conv2d(&kernel_2d)?
            } else {
                let cpu = CpuExecutor::new();
                let mut cpu_inputs = Vec::with_capacity(inputs.len());
                for inp in &inputs {
                    let on_cpu = cpu.transfer(inp.clone()).await?;
                    cpu_inputs.push(on_cpu);
                }
                let cpu_result = cpu.execute(op, cpu_inputs).await?;
                return executor.transfer(cpu_result).await;
            }
        }

        MathOp::MaxPool2D {
            kernel_size: (k_h, k_w),
            stride: (stride_h, stride_w),
            padding: (pad_h, pad_w),
        } => {
            let in_desc = inputs[0].descriptor();
            let use_gpu = *k_h == *k_w && *stride_h == *stride_w && in_desc.shape.len() >= 2;

            let gpu_params = if use_gpu && in_desc.shape.len() == 4 {
                let (n, c, h, w) = (
                    in_desc.shape[0],
                    in_desc.shape[1],
                    in_desc.shape[2],
                    in_desc.shape[3],
                );
                if n == 1 && c == 1 {
                    Some((vec![h, w], *k_h, *stride_h, *pad_h, *pad_w))
                } else {
                    None
                }
            } else if use_gpu && in_desc.shape.len() == 2 {
                Some((in_desc.shape.clone(), *k_h, *stride_h, *pad_h, *pad_w))
            } else {
                None
            };

            if let Some((in_shape, pool_size, stride, ph, pw)) = gpu_params {
                let input_t = build_tensor(&inputs[0], device).await?;
                let input_2d_t = input_t.reshape(in_shape)?;
                let out = input_2d_t.maxpool2d_padded(pool_size, stride, ph, pw)?;
                if in_desc.shape.len() == 4 {
                    out.reshape(vec![1, 1, out.shape()[0], out.shape()[1]])?
                } else {
                    out
                }
            } else {
                let cpu = CpuExecutor::new();
                let mut cpu_inputs = Vec::with_capacity(inputs.len());
                for inp in &inputs {
                    let on_cpu = cpu.transfer(inp.clone()).await?;
                    cpu_inputs.push(on_cpu);
                }
                let cpu_result = cpu.execute(op, cpu_inputs).await?;
                return executor.transfer(cpu_result).await;
            }
        }

        MathOp::AvgPool2D {
            kernel_size: (k_h, k_w),
            stride: (stride_h, stride_w),
            padding: (pad_h, pad_w),
        } => {
            let in_desc = inputs[0].descriptor();
            let use_gpu = *k_h == *k_w && *stride_h == *stride_w && in_desc.shape.len() >= 2;

            let gpu_params = if use_gpu && in_desc.shape.len() == 4 {
                let (n, c, h, w) = (
                    in_desc.shape[0],
                    in_desc.shape[1],
                    in_desc.shape[2],
                    in_desc.shape[3],
                );
                if n == 1 && c == 1 {
                    Some((vec![h, w], *k_h, *stride_h, *pad_h, *pad_w))
                } else {
                    None
                }
            } else if use_gpu && in_desc.shape.len() == 2 {
                Some((in_desc.shape.clone(), *k_h, *stride_h, *pad_h, *pad_w))
            } else {
                None
            };

            if let Some((in_shape, pool_size, stride, ph, pw)) = gpu_params {
                let input_t = build_tensor(&inputs[0], device).await?;
                let input_2d_t = input_t.reshape(in_shape)?;
                input_2d_t.avgpool2d_padded(pool_size, stride, ph, pw)?
            } else {
                let cpu = CpuExecutor::new();
                let mut cpu_inputs = Vec::with_capacity(inputs.len());
                for inp in &inputs {
                    let on_cpu = cpu.transfer(inp.clone()).await?;
                    cpu_inputs.push(on_cpu);
                }
                let cpu_result = cpu.execute(op, cpu_inputs).await?;
                return executor.transfer(cpu_result).await;
            }
        }
    };

    // ── Wrap output Tensor as GpuTensorStorage — zero-copy when possible ───
    // `GpuTensorStorage::from_tensor` shares the Tensor's Arc<wgpu::Buffer>
    // (owned path) or issues a GPU-side copy_buffer_to_buffer (pooled path).
    // In either case, no GPU→CPU→GPU round-trip occurs. D-S16-001 resolved.
    let out_dtype = inputs[0].descriptor().dtype;
    let out_storage = GpuTensorStorage::from_tensor(&output_tensor, out_dtype);
    Ok(Arc::new(out_storage))
}
