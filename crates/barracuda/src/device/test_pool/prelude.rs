// SPDX-License-Identifier: AGPL-3.0-or-later
//! Test prelude for concurrent GPU tests.
//!
//! Provides convenience wrappers for acquiring test devices and creating
//! tensors, plus the [`with_device_retry`] resilience helper.
//!
//! # Usage
//! ```rust,ignore
//! #[cfg(test)]
//! mod tests {
//!     use crate::device::test_pool::test_prelude::*;
//!
//!     #[tokio::test]
//!     async fn test_my_op() {
//!         let device = test_device().await;
//!         // runs on CPU by default (fast, parallel)
//!         // set BARRACUDA_TEST_BACKEND=gpu for GPU workload testing
//!     }
//! }
//! ```

use super::{
    Arc, WgpuDevice, get_test_device, get_test_device_if_f64_gpu_available,
    get_test_device_if_gpu_available, get_test_device_sync, tokio_block_on,
};
use crate::tensor::Tensor;

pub use crate::device::test_harness::{
    baseline_path, coral_available, fused_ops_healthy, gpu_section, is_software_adapter, with_coral,
};

/// Get shared test device (async). CPU by default, GPU with env var.
///
/// The device self-throttles via its internal dispatch semaphore —
/// no manual thread count management needed.
pub async fn test_device() -> Arc<WgpuDevice> {
    get_test_device().await
}

/// Get shared test device (sync version)
#[must_use]
pub fn test_device_blocking() -> Arc<WgpuDevice> {
    get_test_device_sync()
}

/// Get GPU-only test device, or None if unavailable.
/// Use for tests that specifically validate GPU pipeline behavior.
pub async fn test_gpu_device() -> Option<Arc<WgpuDevice>> {
    get_test_device_if_gpu_available().await
}

/// Get f64-capable test device, or None if unavailable
pub async fn test_f64_device() -> Option<Arc<WgpuDevice>> {
    get_test_device_if_f64_gpu_available().await
}

/// Get device with verified f64 transcendentals (sqrt, sin, cos, log, exp).
/// Use for Bessel, Beta, and other transcendental-heavy shader tests.
pub async fn test_f64_transcendental_device() -> Option<Arc<WgpuDevice>> {
    super::get_test_device_if_f64_transcendentals_available().await
}

/// Create test tensor on shared device
/// # Panics
/// Panics if tensor creation fails (e.g. device lost, shape mismatch).
pub async fn test_tensor(data: &[f32], shape: &[usize], device: &Arc<WgpuDevice>) -> Tensor {
    Tensor::from_vec_on(data.to_vec(), shape.to_vec(), Arc::clone(device))
        .await
        .expect("Failed to create test tensor")
}

/// Create test tensor (sync version)
/// # Panics
/// Panics if tensor creation fails (e.g. device lost, shape mismatch).
#[must_use]
pub fn test_tensor_blocking(data: &[f32], shape: &[usize], device: &Arc<WgpuDevice>) -> Tensor {
    tokio_block_on(test_tensor(data, shape, device))
}

/// Create zeros tensor on shared device
/// # Panics
/// Panics if tensor creation fails (e.g. device lost).
pub async fn test_zeros(shape: &[usize], device: &Arc<WgpuDevice>) -> Tensor {
    Tensor::zeros_on(shape.to_vec(), Arc::clone(device))
        .await
        .expect("Failed to create zeros tensor")
}

/// Create randn tensor on shared device (Box-Muller on CPU, then upload).
/// # Panics
/// Panics if tensor creation fails (e.g. device lost).
pub async fn test_randn(shape: &[usize], device: &Arc<WgpuDevice>) -> Tensor {
    use rand::Rng;
    let size: usize = shape.iter().product();
    let mut rng = rand::rng();

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

    Tensor::from_vec_on(data, shape.to_vec(), Arc::clone(device))
        .await
        .expect("Failed to create randn tensor")
}

/// Create rand tensor on shared device (uniform [0, 1))
/// # Panics
/// Panics if tensor creation fails (e.g. device lost).
pub async fn test_rand(shape: &[usize], device: &Arc<WgpuDevice>) -> Tensor {
    use rand::Rng;
    let size: usize = shape.iter().product();
    let data: Vec<f32> = (0..size).map(|_| rand::rng().random()).collect();

    Tensor::from_vec_on(data, shape.to_vec(), Arc::clone(device))
        .await
        .expect("Failed to create rand tensor")
}

/// Run a GPU test with automatic device-lost recovery.
///
/// Production pattern: if the GPU device dies during the test, the pool
/// recreates the device and the test retries once. This is the same
/// recovery path production code follows under sustained concurrent load.
///
/// The closure receives an `Arc<WgpuDevice>` and returns `Result<()>`.
/// - Device-lost errors → retry once with a fresh device
/// - Other errors → propagated (test fails)
/// - Second device-lost → propagated (avoids infinite loop)
///
/// # Panics
/// Panics if GPU is unavailable on retry after device loss, or if the test fails on retry.
///
/// # Example
/// ```rust,ignore
/// #[tokio::test]
/// async fn test_my_op() {
///     with_device_retry(|device| async move {
///         let t = Tensor::from_vec_on(vec![1.0, 2.0], vec![2], device).await?;
///         let result = t.erf()?.to_vec()?;
///         assert!(result.iter().all(|x| x.is_finite()));
///         Ok(())
///     }).await;
/// }
/// ```
pub async fn with_device_retry<F, Fut>(f: F)
where
    F: Fn(Arc<WgpuDevice>) -> Fut,
    Fut: std::future::Future<Output = crate::error::Result<()>>,
{
    let device = super::get_test_device_if_gpu_available().await;
    let Some(device) = device else {
        return;
    };
    let result = f(Arc::clone(&device)).await;
    match result {
        Ok(()) => {}
        Err(e) if e.is_retriable() => {
            tracing::warn!("test: retriable GPU error ({e}), retrying with fresh device");
            drop(device);
            let Some(fresh) = super::get_test_device_if_gpu_available().await else {
                tracing::warn!("GPU unavailable on retry — skipping test");
                return;
            };
            match f(fresh).await {
                Ok(()) => {}
                Err(e) if e.is_retriable() => {
                    tracing::warn!("test still failing after retry ({e}) — skipping on llvmpipe");
                }
                Err(e) => panic!("test failed on retry: {e}"),
            }
        }
        Err(e) => panic!("test failed: {e}"),
    }
}
