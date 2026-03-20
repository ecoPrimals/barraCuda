// SPDX-License-Identifier: AGPL-3.0-or-later
//! CPU executor tests

use super::executor::CpuExecutor;
use super::storage::CpuTensorStorage;
use crate::unified_hardware::{ComputeExecutor, HardwareType, TensorStorage};
use crate::unified_math::{DType, MathOp, TensorDescriptor};
use std::sync::Arc;

fn make_storage(data: &[f32], shape: Vec<usize>) -> Arc<dyn TensorStorage> {
    let desc = TensorDescriptor::new(shape, DType::F32);
    Arc::new(CpuTensorStorage {
        descriptor: desc,
        data: bytes::BytesMut::from(bytemuck::cast_slice::<f32, u8>(data)),
    })
}

async fn read_f32_output(storage: &dyn TensorStorage) -> Vec<f32> {
    let bytes = storage.read_to_cpu().await.unwrap();
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn test_cpu_executor_creation() {
    let cpu = CpuExecutor::new();
    assert_eq!(cpu.name(), "CPU (Native Rust + SIMD)");
    assert_eq!(cpu.hardware_type(), HardwareType::CPU);
    assert!(cpu.num_threads > 0);
}

#[test]
fn test_simd_detection() {
    let width = CpuExecutor::detect_simd_width();
    assert!(width >= 1);
    tracing::debug!("SIMD width: {}", width);
}

#[test]
fn test_cpu_capabilities() {
    let cpu = CpuExecutor::new();
    let caps = cpu.capabilities();
    assert!(caps.operations.matmul);
    assert!(caps.precision.fp32);
    assert!(caps.parallelism.max_parallel_units > 0);
}

#[test]
fn test_cpu_can_execute_all() {
    let cpu = CpuExecutor::new();
    let desc = TensorDescriptor::new(vec![10, 10], crate::unified_math::DType::F32);
    assert!(cpu.can_execute(&MathOp::ReLU, std::slice::from_ref(&desc)));
    assert!(cpu.can_execute(&MathOp::Add, &[desc.clone(), desc]));
}

#[test]
fn test_scoring_small_vs_large() {
    let cpu = CpuExecutor::new();
    let small = TensorDescriptor::new(vec![10, 10], crate::unified_math::DType::F32);
    let score_small = cpu.score_operation(&MathOp::ReLU, &[small]);
    let large = TensorDescriptor::new(vec![4096, 4096], crate::unified_math::DType::F32);
    let score_large = cpu.score_operation(&MathOp::ReLU, &[large]);
    assert!(score_small > score_large);
    tracing::debug!("Small: {:.2}, Large: {:.2}", score_small, score_large);
}

#[test]
fn test_unary_relu() {
    let cpu = CpuExecutor::new();
    let input = vec![-1.0, 0.0, 1.0, 2.0, -2.0];
    let output = cpu.execute_unary_cpu(&MathOp::ReLU, &input).unwrap();
    assert_eq!(output, vec![0.0, 0.0, 1.0, 2.0, 0.0]);
}

#[test]
fn test_binary_add() {
    let cpu = CpuExecutor::new();
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let output = cpu.execute_binary_cpu(&MathOp::Add, &a, &b).unwrap();
    assert_eq!(output, vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_reduce_sum() {
    let cpu = CpuExecutor::new();
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let op = MathOp::ReduceSum {
        dim: None,
        keepdim: false,
    };
    let output = cpu.execute_reduce_cpu(&op, &input).unwrap();
    assert_eq!(output, 10.0);
}

#[test]
fn test_matmul_small() {
    let cpu = CpuExecutor::new();
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let c = cpu.execute_matmul_cpu(&a, &b, 2, 2, 2).unwrap();
    assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_conv2d_simple() {
    let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let kernel = vec![1.0, 0.0, 0.0, 1.0];

    let out = crate::cpu_conv_pool::conv2d(
        &input,
        &kernel,
        crate::cpu_conv_pool::TensorShape {
            n: 1,
            c: 1,
            h: 4,
            w: 4,
        },
        crate::cpu_conv_pool::Conv2dConfig::new(1, 2, 2),
    )
    .unwrap();

    assert_eq!(out.len(), 9);
    assert!((out[0] - 7.0).abs() < 1e-5);
}

#[test]
fn test_maxpool2d_simple() {
    let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let out = crate::cpu_conv_pool::max_pool2d(
        &input,
        crate::cpu_conv_pool::TensorShape {
            n: 1,
            c: 1,
            h: 4,
            w: 4,
        },
        crate::cpu_conv_pool::Pool2dConfig::new(2, 2),
    )
    .unwrap();
    assert_eq!(out.len(), 4);
    assert!((out[0] - 6.0).abs() < 1e-5);
    assert!((out[1] - 8.0).abs() < 1e-5);
    assert!((out[2] - 14.0).abs() < 1e-5);
    assert!((out[3] - 16.0).abs() < 1e-5);
}

#[test]
fn test_avgpool2d_simple() {
    let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let out = crate::cpu_conv_pool::avg_pool2d(
        &input,
        crate::cpu_conv_pool::TensorShape {
            n: 1,
            c: 1,
            h: 4,
            w: 4,
        },
        crate::cpu_conv_pool::Pool2dConfig::new(2, 2),
    )
    .unwrap();
    assert_eq!(out.len(), 4);
    assert!((out[0] - 3.5).abs() < 1e-5);
    assert!((out[1] - 5.5).abs() < 1e-5);
}

// ═══════════════════════════════════════════════════════════════════════════
// Dispatch tests — exercise CpuExecutor::execute() (ComputeExecutor trait)
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_unary_relu() {
    let cpu = CpuExecutor::new();
    let input = make_storage(&[-1.0, 0.0, 1.0, 2.0, -2.0], vec![5]);
    let out = cpu.execute(&MathOp::ReLU, vec![input]).await.unwrap();
    let data = read_f32_output(out.as_ref()).await;
    assert_eq!(data, vec![0.0, 0.0, 1.0, 2.0, 0.0]);
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_unary_sigmoid() {
    let cpu = CpuExecutor::new();
    let input = make_storage(&[0.0], vec![1]);
    let out = cpu.execute(&MathOp::Sigmoid, vec![input]).await.unwrap();
    let data = read_f32_output(out.as_ref()).await;
    assert!((data[0] - 0.5).abs() < 1e-5);
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_unary_tanh() {
    let cpu = CpuExecutor::new();
    let input = make_storage(&[0.0], vec![1]);
    let out = cpu.execute(&MathOp::Tanh, vec![input]).await.unwrap();
    let data = read_f32_output(out.as_ref()).await;
    assert!((data[0] - 0.0).abs() < 1e-5);
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_unary_gelu() {
    let cpu = CpuExecutor::new();
    let input = make_storage(&[0.0, 1.0], vec![2]);
    let out = cpu.execute(&MathOp::GELU, vec![input]).await.unwrap();
    let data = read_f32_output(out.as_ref()).await;
    assert!((data[0] - 0.0).abs() < 1e-5);
    assert!(data[1] > 0.0 && data[1] < 1.0);
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_unary_negate() {
    let cpu = CpuExecutor::new();
    let input = make_storage(&[1.0, -2.0, 0.0], vec![3]);
    let out = cpu.execute(&MathOp::Negate, vec![input]).await.unwrap();
    let data = read_f32_output(out.as_ref()).await;
    assert_eq!(data, vec![-1.0, 2.0, 0.0]);
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_unary_abs() {
    let cpu = CpuExecutor::new();
    let input = make_storage(&[-1.0, 2.0, -3.0], vec![3]);
    let out = cpu.execute(&MathOp::Abs, vec![input]).await.unwrap();
    let data = read_f32_output(out.as_ref()).await;
    assert_eq!(data, vec![1.0, 2.0, 3.0]);
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_unary_square() {
    let cpu = CpuExecutor::new();
    let input = make_storage(&[2.0, -3.0], vec![2]);
    let out = cpu.execute(&MathOp::Square, vec![input]).await.unwrap();
    let data = read_f32_output(out.as_ref()).await;
    assert_eq!(data, vec![4.0, 9.0]);
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_unary_sqrt() {
    let cpu = CpuExecutor::new();
    let input = make_storage(&[4.0, 9.0], vec![2]);
    let out = cpu.execute(&MathOp::Sqrt, vec![input]).await.unwrap();
    let data = read_f32_output(out.as_ref()).await;
    assert!((data[0] - 2.0).abs() < 1e-5);
    assert!((data[1] - 3.0).abs() < 1e-5);
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_unary_exp() {
    let cpu = CpuExecutor::new();
    let input = make_storage(&[0.0, 1.0], vec![2]);
    let out = cpu.execute(&MathOp::Exp, vec![input]).await.unwrap();
    let data = read_f32_output(out.as_ref()).await;
    assert!((data[0] - 1.0).abs() < 1e-5);
    assert!((data[1] - std::f32::consts::E).abs() < 1e-5);
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_unary_log() {
    let cpu = CpuExecutor::new();
    let input = make_storage(&[1.0, std::f32::consts::E], vec![2]);
    let out = cpu.execute(&MathOp::Log, vec![input]).await.unwrap();
    let data = read_f32_output(out.as_ref()).await;
    assert!((data[0] - 0.0).abs() < 1e-5);
    assert!((data[1] - 1.0).abs() < 1e-5);
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_unary_sin() {
    let cpu = CpuExecutor::new();
    let input = make_storage(&[0.0, std::f32::consts::FRAC_PI_2], vec![2]);
    let out = cpu.execute(&MathOp::Sin, vec![input]).await.unwrap();
    let data = read_f32_output(out.as_ref()).await;
    assert!((data[0] - 0.0).abs() < 1e-5);
    assert!((data[1] - 1.0).abs() < 1e-5);
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_unary_cos() {
    let cpu = CpuExecutor::new();
    let input = make_storage(&[0.0, std::f32::consts::FRAC_PI_2], vec![2]);
    let out = cpu.execute(&MathOp::Cos, vec![input]).await.unwrap();
    let data = read_f32_output(out.as_ref()).await;
    assert!((data[0] - 1.0).abs() < 1e-5);
    assert!((data[1] - 0.0).abs() < 1e-5);
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_unary_tan() {
    let cpu = CpuExecutor::new();
    let input = make_storage(&[0.0], vec![1]);
    let out = cpu.execute(&MathOp::Tan, vec![input]).await.unwrap();
    let data = read_f32_output(out.as_ref()).await;
    assert!((data[0] - 0.0).abs() < 1e-5);
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_binary_add() {
    let cpu = CpuExecutor::new();
    let a = make_storage(&[1.0, 2.0, 3.0], vec![3]);
    let b = make_storage(&[4.0, 5.0, 6.0], vec![3]);
    let out = cpu.execute(&MathOp::Add, vec![a, b]).await.unwrap();
    let data = read_f32_output(out.as_ref()).await;
    assert_eq!(data, vec![5.0, 7.0, 9.0]);
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_binary_sub() {
    let cpu = CpuExecutor::new();
    let a = make_storage(&[5.0, 7.0], vec![2]);
    let b = make_storage(&[2.0, 3.0], vec![2]);
    let out = cpu.execute(&MathOp::Sub, vec![a, b]).await.unwrap();
    let data = read_f32_output(out.as_ref()).await;
    assert_eq!(data, vec![3.0, 4.0]);
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_binary_mul() {
    let cpu = CpuExecutor::new();
    let a = make_storage(&[2.0, 3.0], vec![2]);
    let b = make_storage(&[4.0, 5.0], vec![2]);
    let out = cpu.execute(&MathOp::Mul, vec![a, b]).await.unwrap();
    let data = read_f32_output(out.as_ref()).await;
    assert_eq!(data, vec![8.0, 15.0]);
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_binary_div() {
    let cpu = CpuExecutor::new();
    let a = make_storage(&[6.0, 10.0], vec![2]);
    let b = make_storage(&[2.0, 5.0], vec![2]);
    let out = cpu.execute(&MathOp::Div, vec![a, b]).await.unwrap();
    let data = read_f32_output(out.as_ref()).await;
    assert_eq!(data, vec![3.0, 2.0]);
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_binary_pow() {
    let cpu = CpuExecutor::new();
    let a = make_storage(&[2.0, 3.0], vec![2]);
    let b = make_storage(&[3.0, 2.0], vec![2]);
    let out = cpu.execute(&MathOp::Pow, vec![a, b]).await.unwrap();
    let data = read_f32_output(out.as_ref()).await;
    assert!((data[0] - 8.0).abs() < 1e-5);
    assert!((data[1] - 9.0).abs() < 1e-5);
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_binary_max() {
    let cpu = CpuExecutor::new();
    let a = make_storage(&[1.0, 5.0, 3.0], vec![3]);
    let b = make_storage(&[4.0, 2.0, 6.0], vec![3]);
    let out = cpu.execute(&MathOp::Max, vec![a, b]).await.unwrap();
    let data = read_f32_output(out.as_ref()).await;
    assert_eq!(data, vec![4.0, 5.0, 6.0]);
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_binary_min() {
    let cpu = CpuExecutor::new();
    let a = make_storage(&[1.0, 5.0, 3.0], vec![3]);
    let b = make_storage(&[4.0, 2.0, 6.0], vec![3]);
    let out = cpu.execute(&MathOp::Min, vec![a, b]).await.unwrap();
    let data = read_f32_output(out.as_ref()).await;
    assert_eq!(data, vec![1.0, 2.0, 3.0]);
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_reduce_sum() {
    let cpu = CpuExecutor::new();
    let input = make_storage(&[1.0, 2.0, 3.0, 4.0], vec![4]);
    let op = MathOp::ReduceSum {
        dim: None,
        keepdim: false,
    };
    let out = cpu.execute(&op, vec![input]).await.unwrap();
    let data = read_f32_output(out.as_ref()).await;
    assert_eq!(data.len(), 1);
    assert!((data[0] - 10.0).abs() < 1e-5);
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_reduce_mean() {
    let cpu = CpuExecutor::new();
    let input = make_storage(&[2.0, 4.0, 6.0, 8.0], vec![4]);
    let op = MathOp::ReduceMean {
        dim: None,
        keepdim: false,
    };
    let out = cpu.execute(&op, vec![input]).await.unwrap();
    let data = read_f32_output(out.as_ref()).await;
    assert_eq!(data.len(), 1);
    assert!((data[0] - 5.0).abs() < 1e-5);
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_reduce_max() {
    let cpu = CpuExecutor::new();
    let input = make_storage(&[1.0, 5.0, 3.0, 2.0], vec![4]);
    let op = MathOp::ReduceMax {
        dim: None,
        keepdim: false,
    };
    let out = cpu.execute(&op, vec![input]).await.unwrap();
    let data = read_f32_output(out.as_ref()).await;
    assert_eq!(data.len(), 1);
    assert!((data[0] - 5.0).abs() < 1e-5);
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_reduce_min() {
    let cpu = CpuExecutor::new();
    let input = make_storage(&[1.0, 5.0, 3.0, 2.0], vec![4]);
    let op = MathOp::ReduceMin {
        dim: None,
        keepdim: false,
    };
    let out = cpu.execute(&op, vec![input]).await.unwrap();
    let data = read_f32_output(out.as_ref()).await;
    assert_eq!(data.len(), 1);
    assert!((data[0] - 1.0).abs() < 1e-5);
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_reduce_prod() {
    let cpu = CpuExecutor::new();
    let input = make_storage(&[2.0, 3.0, 4.0], vec![3]);
    let op = MathOp::ReduceProd {
        dim: None,
        keepdim: false,
    };
    let out = cpu.execute(&op, vec![input]).await.unwrap();
    let data = read_f32_output(out.as_ref()).await;
    assert_eq!(data.len(), 1);
    assert!((data[0] - 24.0).abs() < 1e-5);
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_matmul_normal() {
    let cpu = CpuExecutor::new();
    let a = make_storage(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = make_storage(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    let op = MathOp::MatMul {
        transpose_a: false,
        transpose_b: false,
    };
    let out = cpu.execute(&op, vec![a, b]).await.unwrap();
    let data = read_f32_output(out.as_ref()).await;
    assert_eq!(data, vec![19.0, 22.0, 43.0, 50.0]);
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_matmul_transposed() {
    let cpu = CpuExecutor::new();
    // A^T @ B: A is 2x2 [[1,3],[2,4]], A^T is [[1,2],[3,4]]
    // B is 2x2 [[5,6],[7,8]]
    // A^T @ B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
    let a = make_storage(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = make_storage(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    let op = MathOp::MatMul {
        transpose_a: true,
        transpose_b: false,
    };
    let out = cpu.execute(&op, vec![a, b]).await.unwrap();
    let data = read_f32_output(out.as_ref()).await;
    assert_eq!(data, vec![19.0, 22.0, 43.0, 50.0]);
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_softmax() {
    let cpu = CpuExecutor::new();
    let input = make_storage(&[1.0, 2.0, 3.0], vec![3]);
    let op = MathOp::Softmax { dim: 0 };
    let out = cpu.execute(&op, vec![input]).await.unwrap();
    let data = read_f32_output(out.as_ref()).await;
    assert_eq!(data.len(), 3);
    let sum: f32 = data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_reshape() {
    let cpu = CpuExecutor::new();
    let input = make_storage(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let op = MathOp::Reshape {
        new_shape: vec![3, 2],
    };
    let out = cpu.execute(&op, vec![input]).await.unwrap();
    assert_eq!(out.descriptor().shape, vec![3, 2]);
    let data = read_f32_output(out.as_ref()).await;
    assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_squeeze() {
    let cpu = CpuExecutor::new();
    let input = make_storage(&[1.0, 2.0, 3.0], vec![1, 3, 1]);
    let op = MathOp::Squeeze { dims: None };
    let out = cpu.execute(&op, vec![input]).await.unwrap();
    assert_eq!(out.descriptor().shape, vec![3]);
    let data = read_f32_output(out.as_ref()).await;
    assert_eq!(data, vec![1.0, 2.0, 3.0]);
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_unsqueeze() {
    let cpu = CpuExecutor::new();
    let input = make_storage(&[1.0, 2.0], vec![2]);
    let op = MathOp::Unsqueeze { dims: vec![0] };
    let out = cpu.execute(&op, vec![input]).await.unwrap();
    assert_eq!(out.descriptor().shape, vec![1, 2]);
    let data = read_f32_output(out.as_ref()).await;
    assert_eq!(data, vec![1.0, 2.0]);
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_transpose_2d() {
    let cpu = CpuExecutor::new();
    let input = make_storage(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let op = MathOp::Transpose { perm: vec![1, 0] };
    let out = cpu.execute(&op, vec![input]).await.unwrap();
    assert_eq!(out.descriptor().shape, vec![2, 2]);
    let data = read_f32_output(out.as_ref()).await;
    assert_eq!(data, vec![1.0, 3.0, 2.0, 4.0]);
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_concat() {
    let cpu = CpuExecutor::new();
    let a = make_storage(&[1.0, 2.0], vec![2]);
    let b = make_storage(&[3.0, 4.0], vec![2]);
    let op = MathOp::Concat { dim: 0 };
    let out = cpu.execute(&op, vec![a, b]).await.unwrap();
    let data = read_f32_output(out.as_ref()).await;
    assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_split() {
    let cpu = CpuExecutor::new();
    let input = make_storage(&[1.0, 2.0, 3.0, 4.0], vec![4]);
    let op = MathOp::Split {
        dim: 0,
        sizes: vec![2, 2],
    };
    let out = cpu.execute(&op, vec![input]).await.unwrap();
    let data = read_f32_output(out.as_ref()).await;
    assert_eq!(data, vec![1.0, 2.0]);
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_broadcast() {
    let cpu = CpuExecutor::new();
    let input = make_storage(&[1.0, 2.0], vec![2]);
    let op = MathOp::Broadcast {
        target_shape: vec![2, 3],
    };
    let out = cpu.execute(&op, vec![input]).await.unwrap();
    assert_eq!(out.descriptor().shape, vec![2, 3]);
    let data = read_f32_output(out.as_ref()).await;
    assert_eq!(data, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_error_empty_inputs() {
    let cpu = CpuExecutor::new();
    let result = cpu.execute(&MathOp::ReLU, vec![]).await;
    assert!(result.is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_error_binary_wrong_input_count() {
    let cpu = CpuExecutor::new();
    let a = make_storage(&[1.0, 2.0], vec![2]);
    let result = cpu.execute(&MathOp::Add, vec![a]).await;
    assert!(result.is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_error_matmul_dimension_mismatch() {
    let cpu = CpuExecutor::new();
    let a = make_storage(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = make_storage(&[1.0, 2.0, 3.0, 4.0], vec![4, 1]);
    let op = MathOp::MatMul {
        transpose_a: false,
        transpose_b: false,
    };
    let result = cpu.execute(&op, vec![a, b]).await;
    assert!(result.is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_unary_reciprocal() {
    let cpu = CpuExecutor::new();
    let input = make_storage(&[2.0, 4.0, 0.5], vec![3]);
    let out = cpu.execute(&MathOp::Reciprocal, vec![input]).await.unwrap();
    let data = read_f32_output(out.as_ref()).await;
    assert!((data[0] - 0.5).abs() < 1e-5);
    assert!((data[1] - 0.25).abs() < 1e-5);
    assert!((data[2] - 2.0).abs() < 1e-5);
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_batch_matmul() {
    let cpu = CpuExecutor::new();
    let a = make_storage(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = make_storage(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    let op = MathOp::BatchMatMul {
        transpose_a: false,
        transpose_b: false,
    };
    let out = cpu.execute(&op, vec![a, b]).await.unwrap();
    let data = read_f32_output(out.as_ref()).await;
    assert_eq!(data, vec![19.0, 22.0, 43.0, 50.0]);
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_conv2d() {
    let cpu = CpuExecutor::new();
    // Input: [N=1, C_in=1, H=4, W=4], Kernel: [C_out=1, C_in=1, kH=2, kW=2]
    let input_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let kernel_data = vec![1.0, 0.0, 0.0, 1.0];
    let input = make_storage(&input_data, vec![1, 1, 4, 4]);
    let kernel = make_storage(&kernel_data, vec![1, 1, 2, 2]);
    let op = MathOp::Conv2D {
        stride: (1, 1),
        padding: (0, 0),
        dilation: (1, 1),
        groups: 1,
    };
    let out = cpu.execute(&op, vec![input, kernel]).await.unwrap();
    let desc = out.descriptor();
    assert_eq!(desc.shape, vec![1, 1, 3, 3]);
    let data = read_f32_output(out.as_ref()).await;
    assert_eq!(data.len(), 9);
    assert!((data[0] - 7.0).abs() < 1e-5);
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_maxpool2d() {
    let cpu = CpuExecutor::new();
    // Input: [N=1, C=1, H=4, W=4]
    let input_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let input = make_storage(&input_data, vec![1, 1, 4, 4]);
    let op = MathOp::MaxPool2D {
        kernel_size: (2, 2),
        stride: (2, 2),
        padding: (0, 0),
    };
    let out = cpu.execute(&op, vec![input]).await.unwrap();
    let desc = out.descriptor();
    assert_eq!(desc.shape, vec![1, 1, 2, 2]);
    let data = read_f32_output(out.as_ref()).await;
    assert!((data[0] - 6.0).abs() < 1e-5);
    assert!((data[1] - 8.0).abs() < 1e-5);
    assert!((data[2] - 14.0).abs() < 1e-5);
    assert!((data[3] - 16.0).abs() < 1e-5);
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_avgpool2d() {
    let cpu = CpuExecutor::new();
    let input_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let input = make_storage(&input_data, vec![1, 1, 4, 4]);
    let op = MathOp::AvgPool2D {
        kernel_size: (2, 2),
        stride: (2, 2),
        padding: (0, 0),
    };
    let out = cpu.execute(&op, vec![input]).await.unwrap();
    let desc = out.descriptor();
    assert_eq!(desc.shape, vec![1, 1, 2, 2]);
    let data = read_f32_output(out.as_ref()).await;
    assert!((data[0] - 3.5).abs() < 1e-5);
    assert!((data[1] - 5.5).abs() < 1e-5);
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_conv2d_missing_input() {
    let cpu = CpuExecutor::new();
    let input = make_storage(&[1.0; 16], vec![1, 1, 4, 4]);
    let op = MathOp::Conv2D {
        stride: (1, 1),
        padding: (0, 0),
        dilation: (1, 1),
        groups: 1,
    };
    let result = cpu.execute(&op, vec![input]).await;
    assert!(result.is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_maxpool2d_wrong_dims() {
    let cpu = CpuExecutor::new();
    let input = make_storage(&[1.0, 2.0, 3.0], vec![3]);
    let op = MathOp::MaxPool2D {
        kernel_size: (2, 2),
        stride: (2, 2),
        padding: (0, 0),
    };
    let result = cpu.execute(&op, vec![input]).await;
    assert!(result.is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_matmul_1d_error() {
    let cpu = CpuExecutor::new();
    let a = make_storage(&[1.0, 2.0], vec![2]);
    let b = make_storage(&[3.0, 4.0], vec![2]);
    let op = MathOp::MatMul {
        transpose_a: false,
        transpose_b: false,
    };
    let result = cpu.execute(&op, vec![a, b]).await;
    assert!(result.is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_batch_matmul_missing_input() {
    let cpu = CpuExecutor::new();
    let a = make_storage(&[1.0; 4], vec![2, 2]);
    let op = MathOp::BatchMatMul {
        transpose_a: false,
        transpose_b: false,
    };
    let result = cpu.execute(&op, vec![a]).await;
    assert!(result.is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_concat_missing_input() {
    let cpu = CpuExecutor::new();
    let a = make_storage(&[1.0, 2.0], vec![2]);
    let op = MathOp::Concat { dim: 0 };
    let result = cpu.execute(&op, vec![a]).await;
    assert!(result.is_err());
}
