// SPDX-License-Identifier: AGPL-3.0-only
//! CPU implementations that match GPU algorithms exactly

use num_traits::Float;

/// Element-wise addition: C = A + B
/// # Panics
/// Panics if `a.len() != b.len()` or `a.len() != output.len()`.
#[inline]
pub fn elementwise_add<T: Float>(a: &[T], b: &[T], output: &mut [T]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), output.len());
    for ((o, &ai), &bi) in output.iter_mut().zip(a).zip(b) {
        *o = ai + bi;
    }
}

/// Element-wise multiplication: C = A * B
/// # Panics
/// Panics if `a.len() != b.len()` or `a.len() != output.len()`.
#[inline]
pub fn elementwise_mul<T: Float>(a: &[T], b: &[T], output: &mut [T]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), output.len());
    for ((o, &ai), &bi) in output.iter_mut().zip(a).zip(b) {
        *o = ai * bi;
    }
}

/// Fused multiply-add: D = A * B + C
/// # Panics
/// Panics if `a.len() != b.len()`, `a.len() != c.len()`, or `a.len() != output.len()`.
#[inline]
pub fn elementwise_fma<T: Float>(a: &[T], b: &[T], c: &[T], output: &mut [T]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), c.len());
    assert_eq!(a.len(), output.len());
    for (((o, &ai), &bi), &ci) in output.iter_mut().zip(a).zip(b).zip(c) {
        *o = ai.mul_add(bi, ci);
    }
}

/// Dot product: sum(A * B)
/// # Panics
/// Panics if `a.len() != b.len()`.
#[inline]
pub fn dot_product<T: Float>(a: &[T], b: &[T]) -> T {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(T::zero(), |acc, (&ai, &bi)| acc + ai * bi)
}

/// Kahan summation for high-precision reduction
#[inline]
pub fn kahan_sum<T: Float>(input: &[T]) -> T {
    let mut sum = T::zero();
    let mut c = T::zero(); // Compensation
    for &x in input {
        let y = x - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

/// Naive sum
#[inline]
pub fn reduce_sum<T: Float>(input: &[T]) -> T {
    input.iter().fold(T::zero(), |acc, &x| acc + x)
}
