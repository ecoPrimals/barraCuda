// SPDX-License-Identifier: AGPL-3.0-or-later
//! CPU op dispatch - `MathOp` execution logic

use super::executor::CpuExecutor;
use crate::error::Result;
use crate::unified_hardware::TensorStorage;
use crate::unified_math::{MathOp, TensorDescriptor};
use std::sync::Arc;

pub(super) fn dispatch(
    executor: &CpuExecutor,
    op: &MathOp,
    inputs: Vec<Arc<dyn TensorStorage>>,
) -> Result<Arc<dyn TensorStorage>> {
    if inputs.is_empty() {
        return Err(crate::error::BarracudaError::InvalidInput {
            message: "No inputs provided".to_string(),
        });
    }

    use MathOp::{
        Abs, Add, AvgPool2D, BatchMatMul, Broadcast, Concat, Conv2D, Cos, Div, Exp, GELU, Log,
        MatMul, Max, MaxPool2D, Min, Mul, Negate, Pow, ReLU, Reciprocal, ReduceMax, ReduceMean,
        ReduceMin, ReduceProd, ReduceSum, Reshape, Sigmoid, Sin, Softmax, Split, Sqrt, Square,
        Squeeze, Sub, Tan, Tanh, Transpose, Unsqueeze,
    };
    match op {
        ReLU | Sigmoid | Tanh | GELU | Negate | Abs | Square | Sqrt | Reciprocal | Exp | Log
        | Sin | Cos | Tan => {
            let data = CpuExecutor::read_f32(inputs[0].as_ref())?;
            let result = executor.execute_unary_cpu(op, &data)?;
            Ok(CpuExecutor::pack_f32(
                result,
                inputs[0].descriptor().clone(),
            ))
        }
        Add | Sub | Mul | Div | Pow | Max | Min => {
            if inputs.len() < 2 {
                return Err(crate::error::BarracudaError::InvalidInput {
                    message: format!("{op:?} requires 2 inputs, got {}", inputs.len()),
                });
            }
            let a = CpuExecutor::read_f32(inputs[0].as_ref())?;
            let b = CpuExecutor::read_f32(inputs[1].as_ref())?;
            let result = executor.execute_binary_cpu(op, &a, &b)?;
            Ok(CpuExecutor::pack_f32(
                result,
                inputs[0].descriptor().clone(),
            ))
        }
        ReduceSum { .. }
        | ReduceMean { .. }
        | ReduceMax { .. }
        | ReduceMin { .. }
        | ReduceProd { .. } => {
            let data = CpuExecutor::read_f32(inputs[0].as_ref())?;
            let scalar = executor.execute_reduce_cpu(op, &data)?;
            let desc = TensorDescriptor::new(vec![1], inputs[0].descriptor().dtype);
            Ok(CpuExecutor::pack_f32(vec![scalar], desc))
        }
        MatMul {
            transpose_a,
            transpose_b,
        } => {
            if inputs.len() < 2 {
                return Err(crate::error::BarracudaError::InvalidInput {
                    message: "MatMul requires 2 inputs".to_string(),
                });
            }
            let a_desc = inputs[0].descriptor();
            let b_desc = inputs[1].descriptor();
            let a_data = CpuExecutor::read_f32(inputs[0].as_ref())?;
            let b_data = CpuExecutor::read_f32(inputs[1].as_ref())?;

            let (m, k_a) = if a_desc.shape.len() >= 2 {
                let r = a_desc.shape.len();
                if *transpose_a {
                    (a_desc.shape[r - 1], a_desc.shape[r - 2])
                } else {
                    (a_desc.shape[r - 2], a_desc.shape[r - 1])
                }
            } else {
                return Err(crate::error::BarracudaError::InvalidInput {
                    message: "MatMul requires 2D+ tensors".to_string(),
                });
            };

            let (k_b, n) = if b_desc.shape.len() >= 2 {
                let r = b_desc.shape.len();
                if *transpose_b {
                    (b_desc.shape[r - 1], b_desc.shape[r - 2])
                } else {
                    (b_desc.shape[r - 2], b_desc.shape[r - 1])
                }
            } else {
                return Err(crate::error::BarracudaError::InvalidInput {
                    message: "MatMul requires 2D+ tensors".to_string(),
                });
            };

            if k_a != k_b {
                return Err(crate::error::BarracudaError::InvalidInput {
                    message: format!("MatMul inner dimension mismatch: {k_a} vs {k_b}"),
                });
            }

            let result = executor.execute_matmul_cpu(&a_data, &b_data, m, k_a, n)?;
            Ok(CpuExecutor::pack_f32(
                result,
                TensorDescriptor::new(vec![m, n], a_desc.dtype),
            ))
        }
        Softmax { .. } => {
            let data = CpuExecutor::read_f32(inputs[0].as_ref())?;
            let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_vals: Vec<f32> = data.iter().map(|&x| (x - max_val).exp()).collect();
            let sum: f32 = exp_vals.iter().sum();
            let result: Vec<f32> = exp_vals.iter().map(|&x| x / sum).collect();
            Ok(CpuExecutor::pack_f32(
                result,
                inputs[0].descriptor().clone(),
            ))
        }
        BatchMatMul {
            transpose_a,
            transpose_b,
        } => {
            if inputs.len() < 2 {
                return Err(crate::error::BarracudaError::InvalidInput {
                    message: "BatchMatMul requires 2 inputs".to_string(),
                });
            }
            let a_desc = inputs[0].descriptor();
            let b_desc = inputs[1].descriptor();
            let a_data = CpuExecutor::read_f32(inputs[0].as_ref())?;
            let b_data = CpuExecutor::read_f32(inputs[1].as_ref())?;

            let (m, k_a) = if a_desc.shape.len() >= 2 {
                let r = a_desc.shape.len();
                if *transpose_a {
                    (a_desc.shape[r - 1], a_desc.shape[r - 2])
                } else {
                    (a_desc.shape[r - 2], a_desc.shape[r - 1])
                }
            } else {
                return Err(crate::error::BarracudaError::InvalidInput {
                    message: "BatchMatMul requires 2D+ tensors".to_string(),
                });
            };

            let (k_b, n) = if b_desc.shape.len() >= 2 {
                let r = b_desc.shape.len();
                if *transpose_b {
                    (b_desc.shape[r - 1], b_desc.shape[r - 2])
                } else {
                    (b_desc.shape[r - 2], b_desc.shape[r - 1])
                }
            } else {
                return Err(crate::error::BarracudaError::InvalidInput {
                    message: "BatchMatMul requires 2D+ tensors".to_string(),
                });
            };

            if k_a != k_b {
                return Err(crate::error::BarracudaError::InvalidInput {
                    message: format!("BatchMatMul inner dimension mismatch: {k_a} vs {k_b}"),
                });
            }

            let result = executor.execute_matmul_cpu(&a_data, &b_data, m, k_a, n)?;
            Ok(CpuExecutor::pack_f32(
                result,
                TensorDescriptor::new(vec![m, n], a_desc.dtype),
            ))
        }
        Reshape { new_shape } => {
            let data = CpuExecutor::read_f32(inputs[0].as_ref())?;
            let shape: Vec<usize> = new_shape.iter().map(|&x| x as usize).collect();
            let desc = TensorDescriptor::new(shape, inputs[0].descriptor().dtype);
            Ok(CpuExecutor::pack_f32(data, desc))
        }
        Squeeze { .. } => {
            let data = CpuExecutor::read_f32(inputs[0].as_ref())?;
            let shape: Vec<usize> = inputs[0]
                .descriptor()
                .shape
                .iter()
                .copied()
                .filter(|&d| d != 1)
                .collect();
            let shape = if shape.is_empty() { vec![1] } else { shape };
            let desc = TensorDescriptor::new(shape, inputs[0].descriptor().dtype);
            Ok(CpuExecutor::pack_f32(data, desc))
        }
        Unsqueeze { dims } => {
            let data = CpuExecutor::read_f32(inputs[0].as_ref())?;
            let mut shape = inputs[0].descriptor().shape.clone();
            for &d in dims.iter().rev() {
                let pos = d.min(shape.len());
                shape.insert(pos, 1);
            }
            let desc = TensorDescriptor::new(shape, inputs[0].descriptor().dtype);
            Ok(CpuExecutor::pack_f32(data, desc))
        }
        Transpose { .. } => {
            let desc = inputs[0].descriptor();
            let data = CpuExecutor::read_f32(inputs[0].as_ref())?;
            if desc.shape.len() == 2 {
                let (rows, cols) = (desc.shape[0], desc.shape[1]);
                let mut transposed = vec![0.0f32; data.len()];
                for r in 0..rows {
                    for c in 0..cols {
                        transposed[c * rows + r] = data[r * cols + c];
                    }
                }
                let new_desc = TensorDescriptor::new(vec![cols, rows], desc.dtype);
                Ok(CpuExecutor::pack_f32(transposed, new_desc))
            } else {
                Ok(CpuExecutor::pack_f32(data, desc.clone()))
            }
        }
        Concat { .. } => {
            if inputs.len() < 2 {
                return Err(crate::error::BarracudaError::InvalidInput {
                    message: "Concat requires at least 2 inputs".to_string(),
                });
            }
            let a = CpuExecutor::read_f32(inputs[0].as_ref())?;
            let b = CpuExecutor::read_f32(inputs[1].as_ref())?;
            let mut result = a;
            result.extend_from_slice(&b);
            let desc = TensorDescriptor::new(vec![result.len()], inputs[0].descriptor().dtype);
            Ok(CpuExecutor::pack_f32(result, desc))
        }
        Split { sizes, .. } => {
            let data = CpuExecutor::read_f32(inputs[0].as_ref())?;
            let split_at = sizes.first().copied().unwrap_or(data.len() / 2);
            let first = data[..split_at.min(data.len())].to_vec();
            let desc = TensorDescriptor::new(vec![first.len()], inputs[0].descriptor().dtype);
            Ok(CpuExecutor::pack_f32(first, desc))
        }
        Broadcast { target_shape } => {
            let data = CpuExecutor::read_f32(inputs[0].as_ref())?;
            let target_size: usize = target_shape.iter().product();
            let mut result = Vec::with_capacity(target_size);
            if data.is_empty() {
                result.resize(target_size, 0.0);
            } else {
                for i in 0..target_size {
                    result.push(data[i % data.len()]);
                }
            }
            let desc = TensorDescriptor::new(target_shape.clone(), inputs[0].descriptor().dtype);
            Ok(CpuExecutor::pack_f32(result, desc))
        }
        Conv2D {
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
            if *groups != 1 {
                return Err(crate::error::BarracudaError::InvalidInput {
                    message: "Conv2D groups > 1 not yet supported".to_string(),
                });
            }

            let in_desc = inputs[0].descriptor();
            let kernel_desc = inputs[1].descriptor();

            if in_desc.shape.len() != 4 {
                return Err(crate::error::BarracudaError::InvalidInput {
                    message: format!(
                        "Conv2D input must be 4D [N, C_in, H, W], got {:?}",
                        in_desc.shape
                    ),
                });
            }
            if kernel_desc.shape.len() != 4 {
                return Err(crate::error::BarracudaError::InvalidInput {
                    message: format!(
                        "Conv2D kernel must be 4D [C_out, C_in, kH, kW], got {:?}",
                        kernel_desc.shape
                    ),
                });
            }

            let n = in_desc.shape[0];
            let c_in = in_desc.shape[1];
            let h = in_desc.shape[2];
            let w = in_desc.shape[3];

            let c_out = kernel_desc.shape[0];
            let k_c_in = kernel_desc.shape[1];
            let k_h = kernel_desc.shape[2];
            let k_w = kernel_desc.shape[3];

            if c_in != k_c_in {
                return Err(crate::error::BarracudaError::InvalidInput {
                    message: format!("Conv2D input channels {c_in} != kernel in-channels {k_c_in}"),
                });
            }

            let input_data = CpuExecutor::read_f32(inputs[0].as_ref())?;
            let kernel_data = CpuExecutor::read_f32(inputs[1].as_ref())?;

            let result = crate::cpu_conv_pool::conv2d(
                &input_data,
                &kernel_data,
                crate::cpu_conv_pool::TensorShape { n, c: c_in, h, w },
                crate::cpu_conv_pool::Conv2dConfig::new(c_out, k_h, k_w)
                    .stride(*stride_h, *stride_w)
                    .padding(*pad_h, *pad_w)
                    .dilation(*dil_h, *dil_w),
            )?;

            let eff_k_h = (k_h - 1) * *dil_h + 1;
            let eff_k_w = (k_w - 1) * *dil_w + 1;
            let h_out = (h + 2 * *pad_h - eff_k_h) / *stride_h + 1;
            let w_out = (w + 2 * *pad_w - eff_k_w) / *stride_w + 1;

            let out_desc = TensorDescriptor::new(vec![n, c_out, h_out, w_out], in_desc.dtype);
            Ok(CpuExecutor::pack_f32(result, out_desc))
        }
        MaxPool2D {
            kernel_size: (k_h, k_w),
            stride: (stride_h, stride_w),
            padding: (pad_h, pad_w),
        } => {
            let in_desc = inputs[0].descriptor();
            if in_desc.shape.len() != 4 {
                return Err(crate::error::BarracudaError::InvalidInput {
                    message: format!(
                        "MaxPool2D input must be 4D [N, C, H, W], got {:?}",
                        in_desc.shape
                    ),
                });
            }

            let n = in_desc.shape[0];
            let c = in_desc.shape[1];
            let h = in_desc.shape[2];
            let w = in_desc.shape[3];

            let input_data = CpuExecutor::read_f32(inputs[0].as_ref())?;
            let result = crate::cpu_conv_pool::max_pool2d(
                &input_data,
                crate::cpu_conv_pool::TensorShape { n, c, h, w },
                crate::cpu_conv_pool::Pool2dConfig::new(*k_h, *k_w)
                    .stride(*stride_h, *stride_w)
                    .padding(*pad_h, *pad_w),
            )?;

            let h_out = (h + 2 * *pad_h - *k_h) / *stride_h + 1;
            let w_out = (w + 2 * *pad_w - *k_w) / *stride_w + 1;

            let out_desc = TensorDescriptor::new(vec![n, c, h_out, w_out], in_desc.dtype);
            Ok(CpuExecutor::pack_f32(result, out_desc))
        }
        AvgPool2D {
            kernel_size: (k_h, k_w),
            stride: (stride_h, stride_w),
            padding: (pad_h, pad_w),
        } => {
            let in_desc = inputs[0].descriptor();
            if in_desc.shape.len() != 4 {
                return Err(crate::error::BarracudaError::InvalidInput {
                    message: format!(
                        "AvgPool2D input must be 4D [N, C, H, W], got {:?}",
                        in_desc.shape
                    ),
                });
            }

            let n = in_desc.shape[0];
            let c = in_desc.shape[1];
            let h = in_desc.shape[2];
            let w = in_desc.shape[3];

            let input_data = CpuExecutor::read_f32(inputs[0].as_ref())?;
            let result = crate::cpu_conv_pool::avg_pool2d(
                &input_data,
                crate::cpu_conv_pool::TensorShape { n, c, h, w },
                crate::cpu_conv_pool::Pool2dConfig::new(*k_h, *k_w)
                    .stride(*stride_h, *stride_w)
                    .padding(*pad_h, *pad_w),
            )?;

            let h_out = (h + 2 * *pad_h - *k_h) / *stride_h + 1;
            let w_out = (w + 2 * *pad_w - *k_w) / *stride_w + 1;

            let out_desc = TensorDescriptor::new(vec![n, c, h_out, w_out], in_desc.dtype);
            Ok(CpuExecutor::pack_f32(result, out_desc))
        }
    }
}
