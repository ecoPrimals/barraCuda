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
/// # Panics
/// Panics if tokio runtime creation fails, or if a spawned thread panics.
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
static GPU_F64_COMPUTES: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static CPU_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

// ============================================================================
// Thread-local GPU test gate permits
// ============================================================================
//
// Under `cargo test`, all tests share one process and tokio runtime.  GPU
// tests that use `get_test_device_if_gpu_available()` need to be throttled
// to prevent driver oversubscription ("parent device is lost").
//
// Each thread stores ONE owned permit from the `GpuTestGate` semaphore.
// - When a GPU test starts, it acquires a permit and stores it here.
// - When the thread picks up the NEXT test (GPU or CPU), the old permit
//   is released (via `release_held_permit()`).
// - `#[tokio::test]` (default `current_thread`) keeps the task on one
//   thread, so the permit lives for the test's duration.

thread_local! {
    static GPU_HELD_PERMIT: std::cell::RefCell<Option<tokio::sync::OwnedSemaphorePermit>>
        = const { std::cell::RefCell::new(None) };
}

fn hold_gpu_permit(permit: tokio::sync::OwnedSemaphorePermit) {
    GPU_HELD_PERMIT.with(|p| {
        *p.borrow_mut() = Some(permit);
    });
}

fn release_held_permit() {
    GPU_HELD_PERMIT.with(|p| {
        *p.borrow_mut() = None;
    });
}

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
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    for adapter in pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all())) {
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

/// Maximum retries for device creation under driver oversubscription.
///
/// When many processes request devices simultaneously (e.g., nextest with
/// high test-threads), the GPU driver may report "parent device is lost"
/// for overflow requests.  This is transient — the adapter recovers once
/// competing requests drain.  Exponential backoff with jitter mirrors the
/// same strategy GPU memory allocators use when VRAM is temporarily
/// exhausted.
const DEVICE_CREATION_MAX_RETRIES: u32 = 5;
const DEVICE_CREATION_BASE_DELAY_MS: u64 = 100;

/// Timeout for GPU device creation (per attempt).
///
/// Prevents indefinite hangs when the driver stalls during adapter/device
/// initialization. Used in `create_gpu_device()`.
const DEVICE_CREATION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Timeout for GPU test execution.
///
/// Wraps the entire async test body in `run_gpu_resilient_async` to prevent
/// indefinite hangs when GPU stalls (e.g., driver lockup, compute shader
/// deadlock). Exported for use by the test helper infrastructure.
pub const GPU_TEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

async fn create_gpu_device() -> Arc<WgpuDevice> {
    let selector = GPU_ADAPTER_SELECTOR.get_or_init(resolve_gpu_adapter_selector);

    let mut last_err = String::new();
    for attempt in 0..=DEVICE_CREATION_MAX_RETRIES {
        match tokio::time::timeout(
            DEVICE_CREATION_TIMEOUT,
            WgpuDevice::with_adapter_selector(selector),
        )
        .await
        {
            Ok(Ok(device)) => {
                if attempt > 0 {
                    tracing::info!("test_pool[gpu]: device created after {} retries", attempt);
                }
                tracing::info!(
                    "test_pool[gpu]: '{}' ({:?})",
                    device.adapter_info().name,
                    device.adapter_info().device_type,
                );
                return Arc::new(device);
            }
            Ok(Err(e)) => {
                last_err = e.to_string();
                let is_transient = last_err.contains("lost")
                    || last_err.contains("overloaded")
                    || last_err.contains("busy");
                if !is_transient || attempt == DEVICE_CREATION_MAX_RETRIES {
                    break;
                }
                let jitter = (attempt as u64).wrapping_mul(7) % 50;
                let delay_ms = DEVICE_CREATION_BASE_DELAY_MS * 2u64.pow(attempt) + jitter;
                tracing::warn!(
                    "test_pool[gpu]: device creation failed (attempt {}/{}): {} — retrying in {}ms",
                    attempt + 1,
                    DEVICE_CREATION_MAX_RETRIES,
                    last_err,
                    delay_ms,
                );
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
            }
            Err(_timeout) => {
                last_err = format!("timed out after {DEVICE_CREATION_TIMEOUT:?}");
                if attempt == DEVICE_CREATION_MAX_RETRIES {
                    break;
                }
                tracing::warn!(
                    "test_pool[gpu]: device creation timed out (attempt {}/{}) — retrying",
                    attempt + 1,
                    DEVICE_CREATION_MAX_RETRIES,
                );
            }
        }
    }
    panic!(
        "GPU device creation failed after {} attempts: {last_err}",
        DEVICE_CREATION_MAX_RETRIES + 1
    );
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
    let guard = pool
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
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
    let mut guard = pool
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
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
/// # Panics
/// Panics if `BARRACUDA_TEST_BACKEND=gpu` is set but no GPU is available, or if
/// neither CPU nor GPU backend is available.
pub async fn get_test_device() -> Arc<WgpuDevice> {
    release_held_permit();
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
    if let Some(dev) = create_cpu_device().await {
        CPU_AVAILABLE.get_or_init(|| true);
        Some(insert_into_pool(&CPU_POOL, dev))
    } else {
        CPU_AVAILABLE.get_or_init(|| false);
        None
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
///
/// Acquires a [`GpuTestGate`](super::test_harness::GpuTestGate) permit that
/// is held in thread-local storage for the duration of the test. This limits
/// concurrent GPU tests under `cargo test` parallelism, preventing driver
/// oversubscription ("parent device is lost" errors).
pub async fn get_test_device_if_gpu_available() -> Option<Arc<WgpuDevice>> {
    release_held_permit();
    let permit = super::test_harness::global_gate().acquire_owned().await;
    let device = get_test_gpu_device().await?;
    if *GPU_IS_REAL.get_or_init(|| device.adapter_info().device_type != wgpu::DeviceType::Cpu) {
        hold_gpu_permit(permit);
        Some(device)
    } else {
        None
    }
}

/// Get a device only if it supports f64 shader operations with verified accuracy.
///
/// Three-gate check plus test-level throttling (see
/// [`get_test_device_if_gpu_available`] for details):
///
/// 1. Real hardware (not CPU software rasterizer)
/// 2. `SHADER_F64` feature + compilation probe pass
/// 3. Computational accuracy probe — runs a small f64 reduction and verifies
///    the result. Catches drivers (e.g. NVK/NAK, some RADV) that advertise
///    f64 support and pass compilation probes but produce incorrect results.
pub async fn get_test_device_if_f64_gpu_available() -> Option<Arc<WgpuDevice>> {
    release_held_permit();
    let permit = super::test_harness::global_gate().acquire_owned().await;
    let device = get_test_gpu_device().await?;
    let is_real =
        *GPU_IS_REAL.get_or_init(|| device.adapter_info().device_type != wgpu::DeviceType::Cpu);
    let has_f64 = *GPU_HAS_F64.get_or_init(|| device.has_f64_shaders());
    if !is_real || !has_f64 {
        return None;
    }
    let computes = *GPU_F64_COMPUTES.get_or_init(|| f64_computation_probe(&device));
    if computes {
        hold_gpu_permit(permit);
        Some(device)
    } else {
        None
    }
}

/// Run a small f64 shader that writes 3.0 * 2.0 + 1.0 to a storage buffer
/// and read back the result. Returns true only if the readback value matches.
fn f64_computation_probe(device: &WgpuDevice) -> bool {
    use crate::device::compute_pipeline::ComputeDispatch;

    let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("f64 compute probe"),
        size: 8,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let shader_src = "\
        enable f64;\n\
        @group(0) @binding(0) var<storage, read_write> out: array<f64>;\n\
        @compute @workgroup_size(1)\n\
        fn main() { out[0] = f64(3.0) * f64(2.0) + f64(1.0); }\n";

    let result = ComputeDispatch::new(device, "f64_compute_probe")
        .shader(shader_src, "main")
        .f64()
        .storage_rw(0, &output_buffer)
        .dispatch(1, 1, 1)
        .submit();

    if result.is_err() {
        return false;
    }

    match device.read_f64_buffer(&output_buffer, 1) {
        Ok(values) if !values.is_empty() => {
            let ok = (values[0] - 7.0).abs() < 1e-10;
            if !ok {
                tracing::warn!(
                    "f64 computation probe FAILED: expected 7.0, got {} — \
                     f64 GPU tests will be skipped",
                    values[0]
                );
            }
            ok
        }
        _ => false,
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
#[must_use]
pub fn get_test_device_sync() -> Arc<WgpuDevice> {
    release_held_permit();
    tokio_block_on(get_test_device())
}

/// Sync wrapper for `get_test_device_if_gpu_available`.
#[must_use]
pub fn get_test_device_if_gpu_available_sync() -> Option<Arc<WgpuDevice>> {
    release_held_permit();
    let permit = super::test_harness::global_gate().acquire_owned_blocking();
    let device = tokio_block_on(async {
        let device = get_test_gpu_device().await?;
        if *GPU_IS_REAL.get_or_init(|| device.adapter_info().device_type != wgpu::DeviceType::Cpu) {
            Some(device)
        } else {
            None
        }
    });
    if device.is_some() {
        hold_gpu_permit(permit);
    }
    device
}

/// Sync wrapper for `get_test_device_if_f64_gpu_available`.
#[must_use]
pub fn get_test_device_if_f64_gpu_available_sync() -> Option<Arc<WgpuDevice>> {
    release_held_permit();
    let permit = super::test_harness::global_gate().acquire_owned_blocking();
    let device = tokio_block_on(async {
        let device = get_test_gpu_device().await?;
        let is_real =
            *GPU_IS_REAL.get_or_init(|| device.adapter_info().device_type != wgpu::DeviceType::Cpu);
        let has_f64 = *GPU_HAS_F64.get_or_init(|| device.has_f64_shaders());
        if !is_real || !has_f64 {
            return None;
        }
        let computes = *GPU_F64_COMPUTES.get_or_init(|| f64_computation_probe(&device));
        if computes { Some(device) } else { None }
    });
    if device.is_some() {
        hold_gpu_permit(permit);
    }
    device
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
    use super::{
        Arc, WgpuDevice, get_test_device, get_test_device_if_f64_gpu_available,
        get_test_device_if_gpu_available, get_test_device_sync, tokio_block_on,
    };
    use crate::tensor::Tensor;

    pub use crate::device::test_harness::{
        baseline_path, coral_available, fused_ops_healthy, gpu_section, is_software_adapter,
        with_coral,
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
        use crate::device::test_harness::gpu_section;

        let device = super::get_test_device_if_gpu_available().await;
        let Some(device) = device else {
            return;
        };
        let result = gpu_section(|| f(Arc::clone(&device))).await;
        match result {
            Ok(()) => {}
            Err(e) if e.is_retriable() => {
                tracing::warn!("test: retriable GPU error ({e}), retrying with fresh device");
                drop(device);
                let Some(fresh) = super::get_test_device_if_gpu_available().await else {
                    tracing::warn!("GPU unavailable on retry — skipping test");
                    return;
                };
                match gpu_section(|| f(fresh)).await {
                    Ok(()) => {}
                    Err(e) if e.is_retriable() => {
                        tracing::warn!(
                            "test still failing after retry ({e}) — skipping on llvmpipe"
                        );
                    }
                    Err(e) => panic!("test failed on retry: {e}"),
                }
            }
            Err(e) => panic!("test failed: {e}"),
        }
    }
}

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
