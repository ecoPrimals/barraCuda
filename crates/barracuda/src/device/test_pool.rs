// SPDX-License-Identifier: AGPL-3.0-or-later
//! Test device pool — dual-backend with automatic recovery
//!
//! **Architecture**: Math is the contract, hardware is the target.
//! - **CPU pool** (default): validates shader math via llvmpipe/software.
//!   Fast, fully parallel, zero GPU contention. No device loss possible.
//! - **GPU pool** (opt-in): validates the hardware pipeline, streaming,
//!   and driver integration. Used for perf tests and hardware CI.
//!
//! Set `BARRACUDA_TEST_BACKEND=gpu` to run all tests on GPU (workload testing).
//! Set `BARRACUDA_GPU_ADAPTER=<name|index>` to pin the GPU adapter.
//!
//! The test suite IS a workload test of barraCuda — test failures reveal
//! runtime flaws that would appear in production under sustained load.

use crate::device::WgpuDevice;
use std::sync::{Arc, OnceLock, RwLock};

/// Block on a future, compatible with both sync and tokio contexts.
///
/// - Multi-threaded runtime: uses `block_in_place` (zero overhead).
/// - Current-thread runtime: spawns an OS thread with a dedicated runtime
///   (avoids both the `block_in_place` panic and nested `block_on` issues).
/// - No runtime: uses a lazily-created static runtime directly.
pub fn tokio_block_on<F>(f: F) -> F::Output
where
    F: std::future::Future + Send,
    F::Output: Send,
{
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();

    fn get_or_create_rt() -> &'static tokio::runtime::Runtime {
        RT.get_or_init(|| tokio::runtime::Runtime::new().expect("Failed to create tokio runtime"))
    }

    match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            let can_block_in_place = std::panic::catch_unwind(|| {
                tokio::task::block_in_place(|| {});
            })
            .is_ok();
            if can_block_in_place {
                tokio::task::block_in_place(|| handle.block_on(f))
            } else {
                std::thread::scope(|s| {
                    s.spawn(|| get_or_create_rt().block_on(f))
                        .join()
                        .expect("tokio_block_on: spawned thread panicked")
                })
            }
        }
        Err(_) => get_or_create_rt().block_on(f),
    }
}

// ============================================================================
// Pool storage
// ============================================================================

static CPU_POOL: std::sync::LazyLock<RwLock<Option<Arc<WgpuDevice>>>> =
    std::sync::LazyLock::new(|| RwLock::new(None));

static GPU_POOL: std::sync::LazyLock<RwLock<Option<Arc<WgpuDevice>>>> =
    std::sync::LazyLock::new(|| RwLock::new(None));

/// Serializes device creation so only one thread pays the cost.
/// Other threads wait, then get the cached device.
static CPU_CREATE_GUARD: std::sync::LazyLock<tokio::sync::Mutex<()>> =
    std::sync::LazyLock::new(|| tokio::sync::Mutex::new(()));
static GPU_CREATE_GUARD: std::sync::LazyLock<tokio::sync::Mutex<()>> =
    std::sync::LazyLock::new(|| tokio::sync::Mutex::new(()));

/// Cached adapter capabilities (survive device recreation).
static GPU_IS_REAL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static GPU_HAS_F64: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static CPU_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Whether tests default to GPU backend.
fn prefer_gpu() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("BARRACUDA_TEST_BACKEND")
            .map(|v| v.eq_ignore_ascii_case("gpu"))
            .unwrap_or(false)
    })
}

// ============================================================================
// GPU adapter pinning
// ============================================================================

static GPU_ADAPTER_SELECTOR: std::sync::OnceLock<String> = std::sync::OnceLock::new();

fn resolve_gpu_adapter_selector() -> String {
    if let Ok(v) = std::env::var("BARRACUDA_GPU_ADAPTER") {
        if !v.is_empty() {
            return v;
        }
    }
    if let Ok(v) = std::env::var("HOTSPRING_GPU_ADAPTER") {
        if !v.is_empty() {
            return v.split(',').next().unwrap_or("auto").to_string();
        }
    }
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    for adapter in instance.enumerate_adapters(wgpu::Backends::all()) {
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::DiscreteGpu
            && adapter.features().contains(wgpu::Features::SHADER_F64)
        {
            return info.name.clone();
        }
    }
    "auto".to_string()
}

// ============================================================================
// Device creation
// ============================================================================

async fn create_gpu_device() -> Arc<WgpuDevice> {
    let selector = GPU_ADAPTER_SELECTOR.get_or_init(resolve_gpu_adapter_selector);
    let device = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        WgpuDevice::with_adapter_selector(selector),
    )
    .await
    .expect("GPU device creation timed out after 30s -- check driver")
    .expect("Failed to create GPU test device");
    tracing::info!(
        "test_pool[gpu]: '{}' ({:?})",
        device.adapter_info().name,
        device.adapter_info().device_type,
    );
    Arc::new(device)
}

async fn create_cpu_device() -> Option<Arc<WgpuDevice>> {
    match WgpuDevice::new_cpu_relaxed().await {
        Ok(device) => {
            tracing::info!(
                "test_pool[cpu]: '{}' ({:?})",
                device.adapter_info().name,
                device.adapter_info().device_type,
            );
            Some(Arc::new(device))
        }
        Err(e) => {
            tracing::warn!("CPU backend unavailable: {e}");
            None
        }
    }
}

fn is_device_healthy(device: &WgpuDevice) -> bool {
    !device.is_lost()
}

/// Fast-path check: returns cached healthy device or None.
fn try_get_cached(pool: &RwLock<Option<Arc<WgpuDevice>>>) -> Option<Arc<WgpuDevice>> {
    let guard = pool.read().unwrap_or_else(|e| e.into_inner());
    if let Some(ref dev) = *guard {
        if is_device_healthy(dev) {
            return Some(Arc::clone(dev));
        }
    }
    None
}

/// Insert a new device into the pool (double-checked locking).
fn insert_into_pool(
    pool: &RwLock<Option<Arc<WgpuDevice>>>,
    device: Arc<WgpuDevice>,
) -> Arc<WgpuDevice> {
    let mut guard = pool.write().unwrap_or_else(|e| e.into_inner());
    if let Some(ref dev) = *guard {
        if is_device_healthy(dev) {
            return Arc::clone(dev);
        }
        tracing::warn!("test_pool: device lost — recreating");
        crate::device::tensor_context::clear_global_contexts();
        crate::device::pipeline_cache::clear_global_cache();
    }
    *guard = Some(Arc::clone(&device));
    device
}

// ============================================================================
// Public API
// ============================================================================

/// Get the default test device.
///
/// Returns CPU device for math validation (fast, parallel, no GPU contention).
/// Set `BARRACUDA_TEST_BACKEND=gpu` to use GPU instead (workload testing).
/// Falls back to GPU if CPU backend is unavailable.
pub async fn get_test_device() -> Arc<WgpuDevice> {
    if prefer_gpu() {
        return get_test_gpu_device()
            .await
            .expect("BARRACUDA_TEST_BACKEND=gpu but no GPU available");
    }

    // Try CPU first (check cache synchronously)
    if let Some(dev) = try_get_cached(&CPU_POOL) {
        return dev;
    }

    // Create CPU device asynchronously
    if let Some(dev) = get_test_cpu_device_async().await {
        return dev;
    }

    // Fall back to GPU if no CPU backend
    get_test_gpu_device()
        .await
        .expect("No test device available (neither CPU nor GPU)")
}

/// Get the CPU test device (llvmpipe/software) — async version.
async fn get_test_cpu_device_async() -> Option<Arc<WgpuDevice>> {
    if let Some(dev) = try_get_cached(&CPU_POOL) {
        return Some(dev);
    }
    if CPU_AVAILABLE.get() == Some(&false) {
        return None;
    }
    let _guard = CPU_CREATE_GUARD.lock().await;
    if let Some(dev) = try_get_cached(&CPU_POOL) {
        return Some(dev);
    }
    match create_cpu_device().await {
        Some(dev) => {
            CPU_AVAILABLE.get_or_init(|| true);
            Some(insert_into_pool(&CPU_POOL, dev))
        }
        None => {
            CPU_AVAILABLE.get_or_init(|| false);
            None
        }
    }
}

/// Get the CPU test device (llvmpipe/software) — sync version.
/// Returns None if no CPU backend is available.
pub fn get_test_cpu_device() -> Option<Arc<WgpuDevice>> {
    if let Some(dev) = try_get_cached(&CPU_POOL) {
        return Some(dev);
    }
    if CPU_AVAILABLE.get() == Some(&false) {
        return None;
    }
    tokio_block_on(get_test_cpu_device_async())
}

/// Get the GPU test device. Returns None if no real GPU is available.
pub async fn get_test_gpu_device() -> Option<Arc<WgpuDevice>> {
    if let Some(dev) = try_get_cached(&GPU_POOL) {
        return Some(dev);
    }
    let _guard = GPU_CREATE_GUARD.lock().await;
    if let Some(dev) = try_get_cached(&GPU_POOL) {
        return Some(dev);
    }
    let dev = create_gpu_device().await;
    GPU_IS_REAL.get_or_init(|| dev.adapter_info().device_type != wgpu::DeviceType::Cpu);
    GPU_HAS_F64.get_or_init(|| dev.has_f64_shaders());
    Some(insert_into_pool(&GPU_POOL, dev))
}

/// Get a GPU device only if it's real hardware (not software fallback).
pub async fn get_test_device_if_gpu_available() -> Option<Arc<WgpuDevice>> {
    let device = get_test_gpu_device().await?;
    if *GPU_IS_REAL.get_or_init(|| device.adapter_info().device_type != wgpu::DeviceType::Cpu) {
        Some(device)
    } else {
        None
    }
}

/// Get a device only if it supports f64 shader operations.
pub async fn get_test_device_if_f64_gpu_available() -> Option<Arc<WgpuDevice>> {
    let device = get_test_gpu_device().await?;
    if *GPU_HAS_F64.get_or_init(|| device.has_f64_shaders()) {
        Some(device)
    } else {
        None
    }
}

// ============================================================================
// Sync helpers
// ============================================================================

/// Run a closure with the shared test device.
pub fn run_with_sync_device<F, R>(f: F) -> R
where
    F: FnOnce(Arc<WgpuDevice>) -> R,
{
    f(get_test_device_sync())
}

/// Sync wrapper for `get_test_device`.
pub fn get_test_device_sync() -> Arc<WgpuDevice> {
    tokio_block_on(get_test_device())
}

/// Sync wrapper for `get_test_device_if_gpu_available`.
pub fn get_test_device_if_gpu_available_sync() -> Option<Arc<WgpuDevice>> {
    tokio_block_on(get_test_device_if_gpu_available())
}

/// Sync wrapper for `get_test_device_if_f64_gpu_available`.
pub fn get_test_device_if_f64_gpu_available_sync() -> Option<Arc<WgpuDevice>> {
    tokio_block_on(get_test_device_if_f64_gpu_available())
}

// ============================================================================
// Test prelude - import this in test modules for easy device access
// ============================================================================

/// Test prelude for concurrent GPU tests
///
/// # Usage
/// ```rust,ignore
/// #[cfg(test)]
/// mod tests {
///     use crate::device::test_pool::test_prelude::*;
///     
///     #[tokio::test]
///     async fn test_my_op() {
///         let device = test_device().await;
///         // runs on CPU by default (fast, parallel)
///         // set BARRACUDA_TEST_BACKEND=gpu for GPU workload testing
///     }
/// }
/// ```
pub mod test_prelude {
    use super::*;
    use crate::tensor::Tensor;

    /// Get shared test device (async). CPU by default, GPU with env var.
    ///
    /// The device self-throttles via its internal dispatch semaphore —
    /// no manual thread count management needed.
    pub async fn test_device() -> Arc<WgpuDevice> {
        get_test_device().await
    }

    /// Get shared test device (sync version)
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

    /// Create test tensor on shared device
    pub async fn test_tensor(data: &[f32], shape: &[usize], device: &Arc<WgpuDevice>) -> Tensor {
        Tensor::from_vec_on(data.to_vec(), shape.to_vec(), Arc::clone(device))
            .await
            .expect("Failed to create test tensor")
    }

    /// Create test tensor (sync version)
    pub fn test_tensor_blocking(data: &[f32], shape: &[usize], device: &Arc<WgpuDevice>) -> Tensor {
        tokio_block_on(test_tensor(data, shape, device))
    }

    /// Create zeros tensor on shared device
    pub async fn test_zeros(shape: &[usize], device: &Arc<WgpuDevice>) -> Tensor {
        Tensor::zeros_on(shape.to_vec(), Arc::clone(device))
            .await
            .expect("Failed to create zeros tensor")
    }

    /// Create randn tensor on shared device (Box-Muller on CPU, then upload).
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
        match f(Arc::clone(&device)).await {
            Ok(()) => {}
            Err(e) if e.is_device_lost() => {
                tracing::warn!("test: device lost during execution, retrying with fresh device");
                drop(device);
                let fresh = super::get_test_device_if_gpu_available()
                    .await
                    .expect("GPU unavailable on retry after device loss");
                f(fresh)
                    .await
                    .expect("test failed on retry after device recovery");
            }
            Err(e) => panic!("test failed: {e}"),
        }
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_device_pool_reuse() {
        let dev1 = get_test_device().await;
        let ptr1 = Arc::as_ptr(&dev1);

        let dev2 = get_test_device().await;
        let ptr2 = Arc::as_ptr(&dev2);

        assert_eq!(ptr1, ptr2, "Device pool should reuse same device");
    }

    #[tokio::test]
    async fn test_device_pool_concurrent() {
        let handles: Vec<_> = (0..10).map(|_| tokio::spawn(get_test_device())).collect();

        let mut devices = Vec::with_capacity(handles.len());
        for h in handles {
            devices.push(h.await.unwrap());
        }

        let first_ptr = Arc::as_ptr(&devices[0]);
        for dev in &devices[1..] {
            assert_eq!(
                Arc::as_ptr(dev),
                first_ptr,
                "Concurrent accesses should get same device"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_cpu_device_available() {
        // On systems with llvmpipe, CPU device should work
        if let Some(dev) = get_test_cpu_device() {
            assert!(dev.is_cpu(), "CPU device should report as CPU");
        }
    }
}
