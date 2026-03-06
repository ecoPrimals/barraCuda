// SPDX-License-Identifier: AGPL-3.0-or-later
//! CPU implementations that match GPU algorithms exactly

use num_traits::Float;

/// Element-wise addition: C = A + B
/// # Panics
/// Panics if `a.len() != b.len()` or `a.len() != output.len()`.
#[inline]
pub fn elementwise_add<T: Float>(a: &[T], b: &[T], output: &mut [T]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), output.len());
    for i in 0..output.len() {
        output[i] = a[i] + b[i]; // Same as GPU
    }
}

/// Element-wise multiplication: C = A * B
/// # Panics
/// Panics if `a.len() != b.len()` or `a.len() != output.len()`.
#[inline]
pub fn elementwise_mul<T: Float>(a: &[T], b: &[T], output: &mut [T]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), output.len());
    for i in 0..output.len() {
        output[i] = a[i] * b[i]; // Same as GPU
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
    for i in 0..output.len() {
        output[i] = a[i].mul_add(b[i], c[i]); // Same as GPU fma()
    }
}

/// Dot product: sum(A * B)
/// # Panics
/// Panics if `a.len() != b.len()`.
#[inline]
pub fn dot_product<T: Float>(a: &[T], b: &[T]) -> T {
    assert_eq!(a.len(), b.len());
    let mut sum = T::zero();
    for i in 0..a.len() {
        sum = sum + a[i] * b[i]; // Same as GPU
    }
    sum
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
