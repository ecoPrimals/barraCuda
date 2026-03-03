//! CPU executor tests

use super::executor::CpuExecutor;
use crate::unified_hardware::{ComputeExecutor, HardwareType};
use crate::unified_math::{MathOp, TensorDescriptor};

#[test]
fn test_cpu_executor_creation() {
    let cpu = CpuExecutor::new();
    assert_eq!(cpu.name(), "CPU (Native Rust + SIMD)");
    assert_eq!(cpu.hardware_type(), HardwareType::CPU);
    assert!(cpu._num_threads > 0);
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

    let out = crate::cpu_conv_pool::conv2d(&input, &kernel, 1, 1, 4, 4, 1, 2, 2, 1, 1, 0, 0, 1, 1)
        .unwrap();

    assert_eq!(out.len(), 9);
    assert!((out[0] - 7.0).abs() < 1e-5);
}

#[test]
fn test_maxpool2d_simple() {
    let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let out = crate::cpu_conv_pool::max_pool2d(&input, 1, 1, 4, 4, 2, 2, 2, 2, 0, 0).unwrap();
    assert_eq!(out.len(), 4);
    assert!((out[0] - 6.0).abs() < 1e-5);
    assert!((out[1] - 8.0).abs() < 1e-5);
    assert!((out[2] - 14.0).abs() < 1e-5);
    assert!((out[3] - 16.0).abs() < 1e-5);
}

#[test]
fn test_avgpool2d_simple() {
    let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let out = crate::cpu_conv_pool::avg_pool2d(&input, 1, 1, 4, 4, 2, 2, 2, 2, 0, 0).unwrap();
    assert_eq!(out.len(), 4);
    assert!((out[0] - 3.5).abs() < 1e-5);
    assert!((out[1] - 5.5).abs() < 1e-5);
}
