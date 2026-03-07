// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU test coordination harness — admission gate for concurrent test execution.
//!
//! Provides [`GpuTestGate`] to limit how many test threads execute GPU work
//! simultaneously, preventing driver contention under full parallel load.
//! Also provides [`CoralProbe`] for cached coralReef availability checks
//! and [`gpu_section`] as the primary entry point for gated GPU work.
//!
//! # Architecture
//!
//! The device-level [`DispatchSemaphore`](super::wgpu_device::dispatch) limits
//! concurrent *operations* on a single device. This module limits concurrent
//! *tests* — a coarser gate that prevents the driver oversubscription that
//! causes device-lost errors under full `cargo test` parallelism.
//!
//! Budget is derived from device type (matching dispatch semaphore tiers)
//! or the `BARRACUDA_TEST_GPU_BUDGET` environment variable.

use std::sync::OnceLock;

const GPU_TEST_BUDGET_CPU: u32 = 2;
const GPU_TEST_BUDGET_IGPU: u32 = 4;
const GPU_TEST_BUDGET_DGPU: u32 = 8;
const GPU_TEST_BUDGET_ENV: &str = "BARRACUDA_TEST_GPU_BUDGET";

static GPU_GATE: OnceLock<GpuTestGate> = OnceLock::new();
static CORAL_AVAILABLE: OnceLock<bool> = OnceLock::new();

/// Process-wide admission gate for GPU test execution.
///
/// Wraps a [`tokio::sync::Semaphore`] with a budget derived from device type.
/// Acquire a permit via [`gpu_section`] before executing GPU work in tests.
pub struct GpuTestGate {
    semaphore: tokio::sync::Semaphore,
    budget: u32,
}

impl GpuTestGate {
    fn new(budget: u32) -> Self {
        Self {
            semaphore: tokio::sync::Semaphore::new(budget as usize),
            budget,
        }
    }

    /// Acquire a test-level GPU permit. Waits until capacity is available.
    ///
    /// # Panics
    /// Panics if the semaphore has been closed (should never happen — the
    /// gate lives for the process lifetime).
    pub async fn acquire(&self) -> tokio::sync::SemaphorePermit<'_> {
        self.semaphore
            .acquire()
            .await
            .expect("GPU test gate semaphore closed unexpectedly")
    }

    /// The configured budget for this gate.
    #[must_use]
    pub fn budget(&self) -> u32 {
        self.budget
    }
}

impl std::fmt::Debug for GpuTestGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuTestGate")
            .field("budget", &self.budget)
            .field("available", &self.semaphore.available_permits())
            .finish()
    }
}

/// Resolve the GPU test budget from env or device type heuristic.
fn resolve_budget() -> u32 {
    if let Ok(val) = std::env::var(GPU_TEST_BUDGET_ENV) {
        if let Ok(n) = val.parse::<u32>() {
            if n > 0 {
                return n;
            }
        }
    }

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
    if let Some(adapter) = adapters.first() {
        let info = adapter.get_info();
        match info.device_type {
            wgpu::DeviceType::DiscreteGpu => GPU_TEST_BUDGET_DGPU,
            wgpu::DeviceType::IntegratedGpu => GPU_TEST_BUDGET_IGPU,
            wgpu::DeviceType::Cpu => GPU_TEST_BUDGET_CPU,
            _ => GPU_TEST_BUDGET_IGPU,
        }
    } else {
        GPU_TEST_BUDGET_CPU
    }
}

/// Get or initialize the global GPU test gate.
pub fn global_gate() -> &'static GpuTestGate {
    GPU_GATE.get_or_init(|| {
        let budget = resolve_budget();
        tracing::debug!(budget, "GPU test gate initialized");
        GpuTestGate::new(budget)
    })
}

/// Execute a closure under the GPU test admission gate.
///
/// Acquires a test-level GPU permit before running `f`, releasing it
/// when the future completes. This is the primary entry point for
/// coordinated GPU test execution.
///
/// ```rust,ignore
/// gpu_section(|| async {
///     let device = get_test_device().await;
///     let result = my_gpu_op(&device).await?;
///     Ok(())
/// }).await;
/// ```
pub async fn gpu_section<F, Fut, T>(f: F) -> T
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = T>,
{
    let _permit = global_gate().acquire().await;
    f().await
}

/// Check whether coralReef is reachable (cached after first probe).
///
/// Uses the global [`CoralCompiler`](super::coral_compiler::CoralCompiler)
/// singleton. The result is cached process-wide so 3000+ tests don't each
/// make an IPC call.
pub async fn coral_available() -> bool {
    if let Some(&cached) = CORAL_AVAILABLE.get() {
        return cached;
    }
    let available = super::coral_compiler::probe_health().await;
    *CORAL_AVAILABLE.get_or_init(|| available)
}

/// Execute a closure only if coralReef is available.
///
/// Returns `Some(result)` if coralReef is reachable and the closure succeeds,
/// `None` if coralReef is unavailable (test should skip the coralReef path).
pub async fn with_coral<F, Fut, T>(f: F) -> Option<T>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = T>,
{
    if coral_available().await {
        Some(f().await)
    } else {
        None
    }
}

/// Validate that a WGSL shader parses and validates through naga.
///
/// Used by cross-spring validation to verify absorbed shaders compile
/// without requiring GPU hardware. Returns the shader module info on
/// success, or an error description on failure.
///
/// # Errors
///
/// Returns `Err` with a description if naga parsing or validation fails.
#[cfg(feature = "gpu")]
pub fn validate_wgsl_shader(wgsl_source: &str) -> Result<(), String> {
    let module = naga::front::wgsl::parse_str(wgsl_source).map_err(|e| format!("parse: {e}"))?;

    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .map_err(|e| format!("validate: {e}"))?;

    Ok(())
}

/// Validate a shader with DF64 preamble injection.
///
/// Prepends `df64_core.wgsl` + `df64_transcendentals.wgsl` before validation,
/// matching the runtime DF64 compilation path used by hotSpring lattice QCD
/// and neuralSpring coralForge attention shaders.
///
/// # Errors
///
/// Returns `Err` if preamble concatenation produces invalid WGSL.
#[cfg(feature = "gpu")]
pub fn validate_df64_shader(shader_body: &str) -> Result<(), String> {
    let df64_core = include_str!("../shaders/math/df64_core.wgsl");
    let df64_trans = include_str!("../shaders/math/df64_transcendentals.wgsl");
    let full_source = format!("{df64_core}\n{df64_trans}\n{shader_body}");
    validate_wgsl_shader(&full_source)
}

/// Run a cross-spring shader validation suite.
///
/// Validates a batch of shader sources and returns a summary of results.
/// Intended for integration tests that verify all absorbed shaders from
/// a specific spring domain still compile.
#[cfg(feature = "gpu")]
#[must_use]
pub fn validate_shader_batch<'a>(
    shaders: &'a [(&'a str, &'a str)],
) -> Vec<(&'a str, Result<(), String>)> {
    shaders
        .iter()
        .map(|(name, source)| (*name, validate_wgsl_shader(source)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_resolves_positive() {
        let budget = resolve_budget();
        assert!(budget > 0, "budget should be positive: {budget}");
        assert!(budget <= 128, "budget should be bounded: {budget}");
    }

    #[test]
    fn global_gate_is_consistent() {
        let gate1 = global_gate();
        let gate2 = global_gate();
        assert_eq!(gate1.budget(), gate2.budget());
    }

    #[tokio::test]
    async fn gpu_section_executes_closure() {
        let result = gpu_section(|| async { 42 }).await;
        assert_eq!(result, 42);
    }

    #[tokio::test]
    async fn gpu_section_concurrent_bounded() {
        let gate = global_gate();
        let budget = gate.budget() as usize;

        let barrier = std::sync::Arc::new(tokio::sync::Barrier::new(budget));
        let mut handles = Vec::with_capacity(budget);

        for _ in 0..budget {
            let b = barrier.clone();
            handles.push(tokio::spawn(gpu_section(move || async move {
                b.wait().await;
                true
            })));
        }

        for h in handles {
            assert!(h.await.unwrap());
        }
    }

    #[tokio::test]
    async fn coral_probe_is_cached() {
        let a = coral_available().await;
        let b = coral_available().await;
        assert_eq!(a, b, "coral probe should return cached result");
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn validate_wgsl_shader_accepts_valid() {
        let wgsl = "@compute @workgroup_size(64) fn main() {}";
        assert!(validate_wgsl_shader(wgsl).is_ok());
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn validate_wgsl_shader_rejects_invalid() {
        let bad = "fn main() { let x = ; }";
        assert!(validate_wgsl_shader(bad).is_err());
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn validate_shader_batch_mixed() {
        let shaders = vec![
            ("valid", "@compute @workgroup_size(64) fn main() {}"),
            ("invalid", "fn main() { let x = ; }"),
        ];
        let results = validate_shader_batch(&shaders);
        assert!(results[0].1.is_ok());
        assert!(results[1].1.is_err());
    }
}
