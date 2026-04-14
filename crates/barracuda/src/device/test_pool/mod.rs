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
use std::sync::{Arc, RwLock};

/// Block on a future, compatible with both sync and tokio contexts.
///
/// Delegates to [`crate::runtime::tokio_block_on`]. Re-exported here for
/// backward compatibility with test code that imports from `test_pool`.
pub fn tokio_block_on<F>(f: F) -> F::Output
where
    F: std::future::Future + Send,
    F::Output: Send,
{
    crate::runtime::tokio_block_on(f)
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
static GPU_F64_TRANSCENDENTALS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static CPU_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static GPU_CREATION_FAILED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

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
            return info.name;
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
const DEVICE_CREATION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Total timeout for GPU device creation (adapter resolution + device creation).
/// Returns None instead of hanging when no GPU is present.
pub const TEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Timeout for GPU test execution.
///
/// Wraps the entire async test body in `run_gpu_resilient_async` to prevent
/// indefinite hangs when GPU stalls (e.g., driver lockup, compute shader
/// deadlock). Exported for use by the test helper infrastructure.
pub const GPU_TEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

async fn create_gpu_device() -> Option<Arc<WgpuDevice>> {
    let create_future = async {
        let selector = tokio::task::spawn_blocking(resolve_gpu_adapter_selector)
            .await
            .unwrap_or_else(|_| "auto".to_string());

        for attempt in 0..=DEVICE_CREATION_MAX_RETRIES {
            match tokio::time::timeout(
                DEVICE_CREATION_TIMEOUT,
                WgpuDevice::with_adapter_selector(&selector),
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
                    return Some(Arc::new(device));
                }
                Ok(Err(e)) => {
                    let err_str = e.to_string();
                    let is_transient = err_str.contains("lost")
                        || err_str.contains("overloaded")
                        || err_str.contains("busy");
                    if !is_transient || attempt == DEVICE_CREATION_MAX_RETRIES {
                        break;
                    }
                    let jitter = (attempt as u64).wrapping_mul(7) % 50;
                    let delay_ms = DEVICE_CREATION_BASE_DELAY_MS * 2u64.pow(attempt) + jitter;
                    tracing::warn!(
                        "test_pool[gpu]: device creation failed (attempt {}/{}): {} — retrying in {}ms",
                        attempt + 1,
                        DEVICE_CREATION_MAX_RETRIES,
                        err_str,
                        delay_ms,
                    );
                    tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                }
                Err(_timeout) => {
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
        None
    };

    match tokio::time::timeout(TEST_TIMEOUT, create_future).await {
        Ok(Some(device)) => Some(device),
        Ok(None) => None,
        Err(_) => None,
    }
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
    if *GPU_CREATION_FAILED.get().unwrap_or(&false) {
        return None;
    }
    if let Some(dev) = try_get_cached(&GPU_POOL) {
        return Some(dev);
    }
    let _guard = GPU_CREATE_GUARD.lock().await;
    if *GPU_CREATION_FAILED.get().unwrap_or(&false) {
        return None;
    }
    if let Some(dev) = try_get_cached(&GPU_POOL) {
        return Some(dev);
    }
    let Some(dev) = create_gpu_device().await else {
        GPU_CREATION_FAILED.get_or_init(|| true);
        return None;
    };
    GPU_IS_REAL.get_or_init(|| dev.adapter_info().device_type != wgpu::DeviceType::Cpu);
    GPU_HAS_F64.get_or_init(|| dev.has_f64_shaders());
    Some(insert_into_pool(&GPU_POOL, dev))
}

/// Get a device for validating shader math correctness on CPU.
///
/// Returns the CPU adapter (llvmpipe/software Vulkan) for validating that
/// WGSL shaders compute correct results. The shader IS the math — the CPU
/// adapter runs it without hardware.
///
/// Semantically identical to [`get_test_device`] — this alias makes test
/// intent explicit: "I am validating shader math, not testing hardware."
pub async fn get_test_device_for_shader_validation() -> Arc<WgpuDevice> {
    get_test_device().await
}

/// Get a device for validating f64 shader math correctness.
///
/// Phase 1: returns real GPU if f64-capable (existing behavior).
/// Phase 2: will return naga interpreter device (no hardware needed).
/// Phase 3: will use coralReef sovereign CPU execution.
pub async fn get_test_device_for_f64_shader_validation() -> Option<Arc<WgpuDevice>> {
    get_test_device_if_f64_gpu_available().await
}

/// Sync wrapper for [`get_test_device_for_shader_validation`].
#[must_use]
pub fn get_test_device_for_shader_validation_sync() -> Arc<WgpuDevice> {
    get_test_device_sync()
}

/// Get a GPU device only if it's real hardware (not software fallback).
///
/// Use for tests that validate hardware pipeline behavior, driver integration,
/// or performance — NOT for validating shader math correctness. For math
/// validation, use [`get_test_device_for_shader_validation`] instead.
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

/// Run the full probe suite asynchronously and log the per-operation
/// capability matrix. Returns whether all f64 transcendentals are working.
async fn probe_and_log_f64_transcendentals(device: &WgpuDevice) -> bool {
    let caps = crate::device::probe::probe_f64_builtins(device).await;
    let info = device.adapter_info();

    tracing::info!(
        adapter = %info.name,
        vendor = format_args!("{:#06x}", info.vendor),
        driver = %info.driver,
        driver_info = %info.driver_info,
        basic_f64 = caps.basic_f64,
        sqrt = caps.sqrt,
        abs_min_max = caps.abs_min_max,
        sin = caps.sin,
        cos = caps.cos,
        exp = caps.exp,
        log = caps.log,
        exp2 = caps.exp2,
        log2 = caps.log2,
        fma = caps.fma,
        shared_mem_f64 = caps.shared_mem_f64,
        df64_arith = caps.df64_arith,
        composite_transcendental = caps.composite_transcendental,
        exp_log_chain = caps.exp_log_chain,
        transcendentals = caps.has_f64_transcendentals(),
        native_count = caps.native_count(),
        "f64 capability probe complete"
    );

    if !caps.has_f64_transcendentals() {
        let mut missing = Vec::new();
        if !caps.sqrt {
            missing.push("sqrt");
        }
        if !caps.abs_min_max {
            missing.push("abs/min/max");
        }
        if !caps.sin {
            missing.push("sin");
        }
        if !caps.cos {
            missing.push("cos");
        }
        if !caps.exp {
            missing.push("exp");
        }
        if !caps.log {
            missing.push("log");
        }
        if !caps.fma {
            missing.push("fma");
        }
        if !caps.composite_transcendental {
            missing.push("composite_transcendental");
        }
        if !caps.exp_log_chain {
            missing.push("exp_log_chain");
        }
        tracing::warn!(
            adapter = %info.name,
            driver = %info.driver,
            missing = %missing.join(", "),
            "f64 transcendentals BROKEN — shaders using these ops need polyfill"
        );
    }

    caps.has_f64_transcendentals()
}

/// Get a device only if f64 transcendentals (sqrt, sin, cos, log, exp)
/// all work correctly with full f64 precision.
///
/// Runs the full probe suite (cached per adapter). Use this for tests
/// that exercise transcendental-heavy shaders (Bessel, Beta, etc.).
pub async fn get_test_device_if_f64_transcendentals_available() -> Option<Arc<WgpuDevice>> {
    release_held_permit();
    let permit = super::test_harness::global_gate().acquire_owned().await;
    let device = get_test_gpu_device().await?;
    let is_real =
        *GPU_IS_REAL.get_or_init(|| device.adapter_info().device_type != wgpu::DeviceType::Cpu);
    let has_f64 = *GPU_HAS_F64.get_or_init(|| device.has_f64_shaders());
    if !is_real || !has_f64 {
        return None;
    }

    let transcendentals = if let Some(&cached) = GPU_F64_TRANSCENDENTALS.get() {
        cached
    } else {
        let result = probe_and_log_f64_transcendentals(&device).await;
        let _ = GPU_F64_TRANSCENDENTALS.set(result);
        result
    };

    if transcendentals {
        hold_gpu_permit(permit);
        Some(device)
    } else {
        None
    }
}

/// Sync wrapper for `get_test_device_if_f64_transcendentals_available`.
#[must_use]
pub fn get_test_device_if_f64_transcendentals_available_sync() -> Option<Arc<WgpuDevice>> {
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
        let transcendentals = if let Some(&cached) = GPU_F64_TRANSCENDENTALS.get() {
            cached
        } else {
            let result = probe_and_log_f64_transcendentals(&device).await;
            let _ = GPU_F64_TRANSCENDENTALS.set(result);
            result
        };
        if transcendentals { Some(device) } else { None }
    });
    if device.is_some() {
        hold_gpu_permit(permit);
    }
    device
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

#[path = "prelude.rs"]
pub mod test_prelude;

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
