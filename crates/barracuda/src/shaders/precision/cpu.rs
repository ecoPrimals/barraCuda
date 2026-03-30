// SPDX-License-Identifier: AGPL-3.0-or-later
//! CPU implementations that match GPU algorithms exactly.
//!
//! Uses a minimal local trait instead of `num_traits::Float` to avoid
//! pulling in an external dependency for three methods.

/// Minimal float trait: the subset of float operations needed by CPU
/// reference implementations. Implemented for `f32` and `f64`.
pub trait CpuFloat:
    Copy
    + Default
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
{
    /// Fused multiply-add: `self * a + b`.
    #[must_use]
    fn mul_add(self, a: Self, b: Self) -> Self;
}

impl CpuFloat for f32 {
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        Self::mul_add(self, a, b)
    }
}

impl CpuFloat for f64 {
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        Self::mul_add(self, a, b)
    }
}

/// Element-wise addition: C = A + B
/// # Panics
/// Panics if `a.len() != b.len()` or `a.len() != output.len()`.
#[inline]
pub fn elementwise_add<T: CpuFloat>(a: &[T], b: &[T], output: &mut [T]) {
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
pub fn elementwise_mul<T: CpuFloat>(a: &[T], b: &[T], output: &mut [T]) {
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
pub fn elementwise_fma<T: CpuFloat>(a: &[T], b: &[T], c: &[T], output: &mut [T]) {
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
pub fn dot_product<T: CpuFloat>(a: &[T], b: &[T]) -> T {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(T::default(), |acc, (&ai, &bi)| acc + ai * bi)
}

/// Kahan summation for high-precision reduction.
#[inline]
pub fn kahan_sum<T: CpuFloat>(input: &[T]) -> T {
    let mut sum = T::default();
    let mut c = T::default();
    for &x in input {
        let y = x - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

/// Naive sum.
#[inline]
pub fn reduce_sum<T: CpuFloat>(input: &[T]) -> T {
    input.iter().fold(T::default(), |acc, &x| acc + x)
}
