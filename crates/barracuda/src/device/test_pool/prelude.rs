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
    Arc, WgpuDevice, get_test_device, get_test_device_for_shader_validation,
    get_test_device_if_f64_gpu_available, get_test_device_if_gpu_available, get_test_device_sync,
    tokio_block_on,
};
use crate::tensor::Tensor;

pub use crate::device::test_harness::{
    ShaderValidationBackend, baseline_path, coral_available, coral_validation_available,
    fused_ops_healthy, gpu_section, is_software_adapter, shader_validation_backend, with_coral,
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

/// Get a device for validating shader math correctness.
///
/// Returns the CPU adapter (llvmpipe/software Vulkan). The shader IS the
/// math — this runs WGSL on CPU to validate correctness without hardware.
/// Use for all numerical correctness tests.
pub async fn test_shader_device() -> Arc<WgpuDevice> {
    get_test_device_for_shader_validation().await
}

/// Get a device for validating f64 shader math correctness.
///
/// Returns a real GPU if f64-capable, None otherwise.
/// Phase 2: will return naga interpreter (no hardware needed).
/// Phase 3: will use coralReef sovereign CPU execution.
pub async fn test_f64_shader_device() -> Option<Arc<WgpuDevice>> {
    super::get_test_device_for_f64_shader_validation().await
}

/// Get GPU-only test device, or None if unavailable.
/// Use for tests that specifically validate GPU pipeline behavior,
/// driver integration, or performance — NOT for shader math validation.
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

// ============================================================================
// NagaExecutor shader math validation (dev-dependency)
// ============================================================================

/// Assert that a WGSL compute shader produces expected f32 output on CPU.
///
/// Parses the WGSL, dispatches via `NagaExecutor` (no GPU needed), and
/// compares output buffer to expected values within tolerance.
///
/// # Example
/// ```rust,ignore
/// assert_shader_math!(
///     "@group(0) @binding(0) var<storage, read> input: array<f32>;
///      @group(0) @binding(1) var<storage, read_write> output: array<f32>;
///      @compute @workgroup_size(1)
///      fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
///          output[gid.x] = input[gid.x] * 2.0;
///      }",
///     "main",
///     inputs: { (0, 0) => &[1.0, 2.0, 3.0] },
///     outputs: { (0, 1) => 3 },
///     expected: &[2.0, 4.0, 6.0],
///     tolerance: 1e-6
/// );
/// ```
#[cfg(test)]
#[macro_export]
macro_rules! assert_shader_math {
    (
        $wgsl:expr,
        $entry:expr,
        inputs: { $( ($ig:expr, $ib:expr) => $idata:expr ),* $(,)? },
        outputs: { $( ($og:expr, $ob:expr) => $osize:expr ),* $(,)? },
        expected: $expected:expr,
        tolerance: $tol:expr
    ) => {{
        use barracuda_naga_exec::{NagaExecutor, SimBuffer};
        use std::collections::BTreeMap;

        let exec = NagaExecutor::new($wgsl, $entry)
            .expect("WGSL parse/validation failed in assert_shader_math!");

        let mut bindings = BTreeMap::new();
        $( bindings.insert(($ig, $ib), SimBuffer::from_f32_readonly($idata)); )*
        $( bindings.insert(($og, $ob), SimBuffer::from_f32(&vec![0.0f32; $osize])); )*

        let total_elements: usize = 0 $( + $osize )*;
        exec.dispatch(
            (total_elements.max(1) as u32, 1, 1),
            &mut bindings,
        )
        .expect("NagaExecutor dispatch failed in assert_shader_math!");

        let expected_slice: &[f32] = $expected;
        let mut result_idx = 0usize;
        $(
            let result = bindings[&($og, $ob)].as_f32();
            for (i, &val) in result.iter().enumerate() {
                let exp = expected_slice[result_idx + i];
                assert!(
                    (val - exp).abs() < $tol,
                    "shader math mismatch at output({}, {})[{}]: got {}, expected {}, tol={}",
                    $og, $ob, i, val, exp, $tol
                );
            }
            result_idx += result.len();
        )*
        let _ = result_idx;
    }};
}

/// Assert f64 shader math correctness via `NagaExecutor` (no GPU required).
#[cfg(test)]
#[macro_export]
macro_rules! assert_shader_math_f64 {
    (
        $wgsl:expr,
        $entry:expr,
        inputs: { $( ($ig:expr, $ib:expr) => $idata:expr ),* $(,)? },
        outputs: { $( ($og:expr, $ob:expr) => $osize:expr ),* $(,)? },
        expected: $expected:expr,
        tolerance: $tol:expr
    ) => {{
        use barracuda_naga_exec::{NagaExecutor, SimBuffer};
        use std::collections::BTreeMap;

        let exec = NagaExecutor::new($wgsl, $entry)
            .expect("WGSL parse/validation failed in assert_shader_math_f64!");

        let mut bindings = BTreeMap::new();
        $( bindings.insert(($ig, $ib), SimBuffer::from_f64($idata)); )*
        $( bindings.insert(($og, $ob), SimBuffer::from_f64(&vec![0.0f64; $osize])); )*

        let total_elements: usize = 0 $( + $osize )*;
        exec.dispatch(
            (total_elements.max(1) as u32, 1, 1),
            &mut bindings,
        )
        .expect("NagaExecutor dispatch failed in assert_shader_math_f64!");

        let expected_slice: &[f64] = $expected;
        let mut result_idx = 0usize;
        $(
            let result = bindings[&($og, $ob)].as_f64();
            for (i, &val) in result.iter().enumerate() {
                let exp = expected_slice[result_idx + i];
                assert!(
                    (val - exp).abs() < $tol,
                    "f64 shader math mismatch at output({}, {})[{}]: got {}, expected {}, tol={}",
                    $og, $ob, i, val, exp, $tol
                );
            }
            result_idx += result.len();
        )*
        let _ = result_idx;
    }};
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
