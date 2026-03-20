// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scalar arithmetic and random tensor generation.
//!
//! Extracted from `tensor/mod.rs` to keep files under the 1 000-line limit.

use super::Tensor;
use crate::error::Result;

impl Tensor {
    // ── Scalar arithmetic ────────────────────────────────────────────────────

    /// `C = A * scalar` — broadcast scalar multiplication.
    /// ```rust,ignore
    /// let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).await?;
    /// let y = x.mul_scalar(2.0)?;  // [2.0, 4.0, 6.0]
    /// ```
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, buffer write fails, or the
    /// element-wise multiply operation fails (e.g., device lost, out of memory).
    pub fn mul_scalar(&self, scalar: f32) -> Result<Tensor> {
        let scalar_tensor = self.broadcast_scalar(scalar)?;
        self.mul(&scalar_tensor)
    }

    /// `C = A + scalar` — broadcast scalar addition.
    /// ```rust,ignore
    /// let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).await?;
    /// let y = x.add_scalar(10.0)?;  // [11.0, 12.0, 13.0]
    /// ```
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, buffer write fails, or the
    /// element-wise add operation fails (e.g., device lost, out of memory).
    pub fn add_scalar(&self, scalar: f32) -> Result<Tensor> {
        let scalar_tensor = self.broadcast_scalar(scalar)?;
        self.add(&scalar_tensor)
    }

    /// `C = A / scalar` — implemented as `A * (1/scalar)`.
    /// ```rust,ignore
    /// let x = Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![3]).await?;
    /// let y = x.div_scalar(2.0)?;  // [5.0, 10.0, 15.0]
    /// ```
    /// # Errors
    /// Returns [`Err`] if `mul_scalar(1.0 / scalar)` fails (buffer allocation,
    /// buffer write, or multiply operation failure).
    pub fn div_scalar(&self, scalar: f32) -> Result<Tensor> {
        self.mul_scalar(1.0 / scalar)
    }

    fn broadcast_scalar(&self, scalar: f32) -> Result<Tensor> {
        let data = vec![scalar; self.len()];
        Tensor::from_vec_on_sync(data, self.shape.clone(), self.device.clone())
    }

    // ── Random generation ────────────────────────────────────────────────────

    /// Random tensor sampled from N(0, 1) via Box-Muller transform.
    /// ```rust,ignore
    /// let x = Tensor::randn(vec![100, 100]).await?;
    /// ```
    /// # Errors
    /// Returns [`Err`] if device discovery fails, buffer allocation fails, or
    /// buffer write fails.
    pub async fn randn(shape: Vec<usize>) -> Result<Self> {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::from_os_rng();
        Self::randn_with_rng(shape, &mut rng).await
    }

    /// Random normal tensor with a caller-supplied RNG (reproducible).
    /// ```rust,ignore
    /// use rand::SeedableRng;
    /// let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    /// let x = Tensor::randn_with_rng(vec![10, 10], &mut rng).await?;
    /// ```
    /// # Errors
    /// Returns [`Err`] if device discovery fails, buffer allocation fails, or
    /// buffer write fails.
    pub async fn randn_with_rng<R: rand::Rng>(shape: Vec<usize>, rng: &mut R) -> Result<Self> {
        let size: usize = shape.iter().product();

        let mut data = Vec::with_capacity(size);
        for _ in 0..(size / 2) {
            let u1: f32 = rng.random::<f32>().max(1e-10);
            let u2: f32 = rng.random();

            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;

            data.push(r * theta.cos());
            data.push(r * theta.sin());
        }

        if size % 2 == 1 {
            let u1: f32 = rng.random::<f32>().max(1e-10);
            let u2: f32 = rng.random();
            data.push((-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos());
        }

        data.truncate(size);
        Self::from_vec(data, shape).await
    }

    /// Random tensor sampled from U(0, 1).
    /// ```rust,ignore
    /// let x = Tensor::rand(vec![100, 100]).await?;
    /// ```
    /// # Errors
    /// Returns [`Err`] if device discovery fails, buffer allocation fails, or
    /// buffer write fails.
    pub async fn rand(shape: Vec<usize>) -> Result<Self> {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::from_os_rng();
        Self::rand_with_rng(shape, &mut rng).await
    }

    /// Random uniform tensor with a caller-supplied RNG.
    /// # Errors
    /// Returns [`Err`] if device discovery fails, buffer allocation fails, or
    /// buffer write fails.
    pub async fn rand_with_rng<R: rand::Rng>(shape: Vec<usize>, rng: &mut R) -> Result<Self> {
        let size: usize = shape.iter().product();
        let data: Vec<f32> = (0..size).map(|_| rng.random()).collect();
        Self::from_vec(data, shape).await
    }

    /// Random tensor sampled from U(min, max).
    /// ```rust,ignore
    /// let x = Tensor::rand_range(vec![100], -1.0, 1.0).await?;
    /// ```
    /// # Errors
    /// Returns [`Err`] if `rand`, `mul_scalar`, or `add_scalar` fails (device
    /// discovery, buffer allocation, buffer write, or element-wise ops).
    pub async fn rand_range(shape: Vec<usize>, min: f32, max: f32) -> Result<Self> {
        let uniform = Self::rand(shape).await?;
        let range = max - min;
        uniform.mul_scalar(range)?.add_scalar(min)
    }
}
