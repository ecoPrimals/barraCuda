// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

#[test]
fn test_elementwise_add_f64() {
    let a = vec![1.0_f64, 2.0, 3.0];
    let b = vec![4.0_f64, 5.0, 6.0];
    let mut out = vec![0.0_f64; 3];
    cpu::elementwise_add(&a, &b, &mut out);
    assert_eq!(out, vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_elementwise_add_f32() {
    let a = vec![1.0_f32, -2.0, 0.0];
    let b = vec![4.0_f32, 3.0, 1.0];
    let mut out = vec![0.0_f32; 3];
    cpu::elementwise_add(&a, &b, &mut out);
    assert_eq!(out, vec![5.0, 1.0, 1.0]);
}

#[test]
fn test_elementwise_add_single_element() {
    let a = vec![std::f64::consts::PI];
    let b = vec![2.86_f64];
    let mut out = vec![0.0_f64; 1];
    cpu::elementwise_add(&a, &b, &mut out);
    let expected = std::f64::consts::PI + 2.86;
    assert!((out[0] - expected).abs() < 1e-15);
}

#[test]
fn test_elementwise_mul_f64() {
    let a = vec![2.0_f64, 3.0, 4.0];
    let b = vec![5.0_f64, 2.0, 0.5];
    let mut out = vec![0.0_f64; 3];
    cpu::elementwise_mul(&a, &b, &mut out);
    assert_eq!(out, vec![10.0, 6.0, 2.0]);
}

#[test]
fn test_elementwise_mul_f32() {
    let a = vec![1.0_f32, -2.0, 0.0];
    let b = vec![4.0_f32, 3.0, 1.0];
    let mut out = vec![0.0_f32; 3];
    cpu::elementwise_mul(&a, &b, &mut out);
    assert_eq!(out, vec![4.0, -6.0, 0.0]);
}

#[test]
fn test_elementwise_fma_f64() {
    let a = vec![2.0_f64, 3.0, 4.0];
    let b = vec![5.0_f64, 2.0, 0.5];
    let c = vec![1.0_f64, 0.0, -1.0];
    let mut out = vec![0.0_f64; 3];
    cpu::elementwise_fma(&a, &b, &c, &mut out);
    assert_eq!(out, vec![11.0, 6.0, 1.0]); // 2*5+1, 3*2+0, 4*0.5-1
}

#[test]
fn test_elementwise_fma_f32() {
    let a = vec![1.0_f32, 2.0];
    let b = vec![3.0_f32, 4.0];
    let c = vec![1.0_f32, -1.0];
    let mut out = vec![0.0_f32; 2];
    cpu::elementwise_fma(&a, &b, &c, &mut out);
    assert_eq!(out, vec![4.0, 7.0]);
}

#[test]
fn test_dot_product_f64() {
    let a = vec![1.0_f64, 2.0, 3.0];
    let b = vec![4.0_f64, 5.0, 6.0];
    let result = cpu::dot_product(&a, &b);
    assert!((result - 32.0).abs() < 1e-10); // 4 + 10 + 18
}

#[test]
fn test_dot_product_f32() {
    let a = vec![1.0_f32, 0.0, -1.0];
    let b = vec![1.0_f32, 1.0, 1.0];
    let result = cpu::dot_product(&a, &b);
    assert!((result - 0.0).abs() < 1e-6);
}

#[test]
fn test_dot_product_single_element() {
    let a = vec![3.0_f64];
    let b = vec![7.0_f64];
    let result = cpu::dot_product(&a, &b);
    assert!((result - 21.0).abs() < 1e-10);
}

#[test]
fn test_dot_product_orthogonal() {
    let a = vec![1.0_f64, 0.0, 0.0];
    let b = vec![0.0_f64, 1.0, 0.0];
    let result = cpu::dot_product(&a, &b);
    assert!((result - 0.0).abs() < 1e-15);
}

#[test]
fn test_kahan_sum() {
    let input = vec![1e-10_f64, 1e-10, 1e-10, 1e10];
    let result = cpu::kahan_sum(&input);
    let naive = cpu::reduce_sum(&input);
    assert!(result.is_finite());
    assert!(naive.is_finite());
    assert!((result - 3.0f64.mul_add(1e-10, 1e10)).abs() < 1e-6);
}

#[test]
fn test_kahan_sum_empty() {
    let input: Vec<f64> = vec![];
    let result = cpu::kahan_sum(&input);
    assert_eq!(result, 0.0);
}

#[test]
fn test_kahan_sum_single() {
    let input = vec![42.0_f64];
    let result = cpu::kahan_sum(&input);
    assert_eq!(result, 42.0);
}

#[test]
fn test_reduce_sum() {
    let input = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let result = cpu::reduce_sum(&input);
    assert!((result - 15.0).abs() < 1e-10);
}

#[test]
fn test_reduce_sum_empty() {
    let input: Vec<f64> = vec![];
    let result = cpu::reduce_sum(&input);
    assert_eq!(result, 0.0);
}

#[test]
fn test_reduce_sum_f32() {
    let input = vec![0.1_f32, 0.2, 0.3];
    let result = cpu::reduce_sum(&input);
    assert!((result - 0.6).abs() < 1e-5);
}

#[test]
#[should_panic(expected = "assertion")]
fn test_elementwise_add_panics_length_mismatch() {
    let a = vec![1.0_f64, 2.0];
    let b = vec![1.0_f64, 2.0, 3.0];
    let mut out = vec![0.0_f64; 2];
    cpu::elementwise_add(&a, &b, &mut out);
}

#[test]
#[should_panic(expected = "assertion")]
fn test_elementwise_mul_panics_output_mismatch() {
    let a = vec![1.0_f64, 2.0];
    let b = vec![1.0_f64, 2.0];
    let mut out = vec![0.0_f64; 3];
    cpu::elementwise_mul(&a, &b, &mut out);
}

#[test]
#[should_panic(expected = "assertion")]
fn test_dot_product_panics_length_mismatch() {
    let a = vec![1.0_f64, 2.0];
    let b = vec![1.0_f64];
    let _ = cpu::dot_product(&a, &b);
}
