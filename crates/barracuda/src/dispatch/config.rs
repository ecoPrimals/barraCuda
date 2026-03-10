// SPDX-License-Identifier: AGPL-3.0-only
//! Dispatch configuration and routing logic.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

/// Global dispatch configuration (lazy-initialized)
static GLOBAL_CONFIG: OnceLock<DispatchConfig> = OnceLock::new();

/// Dispatch configuration for CPU/GPU routing
#[derive(Debug, Clone)]
pub struct DispatchConfig {
    /// Per-operation CPU thresholds (input size below which CPU is used)
    thresholds: HashMap<Arc<str>, usize>,
    /// Whether GPU is available (cached at init)
    gpu_available: bool,
    /// Force CPU for all operations (useful for testing, f64 precision)
    force_cpu: bool,
    /// Force GPU for all operations (useful for benchmarking)
    force_gpu: bool,
}

impl Default for DispatchConfig {
    fn default() -> Self {
        Self {
            thresholds: default_thresholds(),
            gpu_available: check_gpu_available(),
            force_cpu: false,
            force_gpu: false,
        }
    }
}

impl DispatchConfig {
    /// Create a new dispatch config with default thresholds
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create config with custom thresholds
    #[must_use]
    pub fn with_thresholds(thresholds: HashMap<Arc<str>, usize>) -> Self {
        Self {
            thresholds,
            ..Self::default()
        }
    }

    /// Force all operations to CPU
    #[must_use]
    pub fn force_cpu(mut self) -> Self {
        self.force_cpu = true;
        self.force_gpu = false;
        self
    }

    /// Force all operations to GPU (if available)
    #[must_use]
    pub fn force_gpu(mut self) -> Self {
        self.force_gpu = true;
        self.force_cpu = false;
        self
    }

    /// Set threshold for a specific operation
    pub fn set_threshold(&mut self, operation: impl Into<Arc<str>>, threshold: usize) {
        self.thresholds.insert(operation.into(), threshold);
    }

    /// Get threshold for an operation (returns default if not set)
    #[must_use]
    pub fn threshold(&self, operation: &str) -> usize {
        self.thresholds
            .get(operation)
            .copied()
            .unwrap_or(DEFAULT_THRESHOLD)
    }

    /// Check if GPU should be used for given input size and operation
    ///
    /// # Arguments
    ///
    /// * `input_size` - Size of input (elements, not bytes)
    /// * `operation` - Name of operation (e.g., "matmul", "erf", "cdist")
    ///
    /// # Returns
    ///
    /// `true` if GPU should be used, `false` for CPU
    #[must_use]
    pub fn should_use_gpu(&self, input_size: usize, operation: &str) -> bool {
        // Force flags take precedence
        if self.force_cpu {
            return false;
        }
        if self.force_gpu && self.gpu_available {
            return true;
        }

        // GPU must be available
        if !self.gpu_available {
            return false;
        }

        // Size-based dispatch
        let threshold = self.threshold(operation);
        input_size >= threshold
    }

    /// Check if GPU is available
    #[must_use]
    pub fn has_gpu(&self) -> bool {
        self.gpu_available
    }
}

/// Default threshold when operation not specified.
pub const DEFAULT_THRESHOLD: usize = 1024;

/// HMM forward-pass dispatch threshold (states × observations).
const HMM_FORWARD_THRESHOLD: usize = 5000;

/// Default per-operation thresholds (empirically determined)
fn default_thresholds() -> HashMap<Arc<str>, usize> {
    let mut m = HashMap::new();

    // === Special Functions ===
    m.insert(Arc::from("erf"), 512); // Error function
    m.insert(Arc::from("erfc"), 512); // Complementary error function
    m.insert(Arc::from("gamma"), 256); // Gamma function
    m.insert(Arc::from("lgamma"), 256); // Log-gamma
    m.insert(Arc::from("digamma"), 256); // Digamma (psi)
    m.insert(Arc::from("bessel_j0"), 512); // Bessel J0
    m.insert(Arc::from("bessel_j1"), 512); // Bessel J1
    m.insert(Arc::from("bessel_i0"), 512); // Modified Bessel I0
    m.insert(Arc::from("bessel_k0"), 512); // Modified Bessel K0

    // === Linear Algebra ===
    m.insert(Arc::from("matmul"), 64); // Matrix multiply: 64×64 = 4096 elements
    m.insert(Arc::from("frobenius_norm"), 4096);
    m.insert(Arc::from("transpose"), 4096);
    m.insert(Arc::from("cholesky"), 128); // Cholesky decomposition
    m.insert(Arc::from("eigh"), 128); // Symmetric eigenvalue
    m.insert(Arc::from("lu"), 128); // LU decomposition
    m.insert(Arc::from("qr"), 128); // QR decomposition
    m.insert(Arc::from("svd"), 128); // Singular value decomposition
    m.insert(Arc::from("solve"), 128); // Linear solve
    m.insert(Arc::from("tridiagonal"), 256); // Tridiagonal solve (Thomas alg is fast)

    // === Distance/Similarity ===
    m.insert(Arc::from("cdist"), 200); // Pairwise distances (O(N²))
    m.insert(Arc::from("pdist"), 200); // Pairwise distances (condensed)
    m.insert(Arc::from("cosine_similarity"), 256);

    // === Transforms ===
    m.insert(Arc::from("fft"), 1024); // FFT
    m.insert(Arc::from("ifft"), 1024); // Inverse FFT
    m.insert(Arc::from("dct"), 1024); // Discrete cosine transform

    // === Reductions ===
    m.insert(Arc::from("sum"), 4096); // Reduction needs large N for GPU win
    m.insert(Arc::from("mean"), 4096);
    m.insert(Arc::from("variance"), 4096);
    m.insert(Arc::from("l2_distance"), 4096);
    m.insert(Arc::from("max"), 4096);
    m.insert(Arc::from("min"), 4096);
    m.insert(Arc::from("argmax"), 4096);
    m.insert(Arc::from("argmin"), 4096);

    // === Element-wise ===
    m.insert(Arc::from("softmax"), 2048);
    m.insert(Arc::from("gelu"), 2048);
    m.insert(Arc::from("relu"), 2048);
    m.insert(Arc::from("sigmoid"), 2048);
    m.insert(Arc::from("tanh"), 2048);
    m.insert(Arc::from("exp"), 2048);
    m.insert(Arc::from("log"), 2048);
    m.insert(Arc::from("sqrt"), 2048);
    m.insert(Arc::from("sin"), 2048);
    m.insert(Arc::from("cos"), 2048);

    // === Surrogate/Optimization ===
    m.insert(Arc::from("rbf_kernel"), 200); // RBF kernel evaluation
    m.insert(Arc::from("surrogate_predict"), 100); // Single-point prediction is CPU-only
    m.insert(Arc::from("surrogate_train"), 200); // Training benefits from GPU

    // === Bio (HMM) ===
    m.insert(Arc::from("hmm"), HMM_FORWARD_THRESHOLD);

    m
}

/// Check if a hardware GPU is available at runtime.
///
/// Uses `WgpuDevice::new()` for a consistent probe rather than duplicating
/// low-level wgpu setup code.  Returns `false` on software/CPU adapters.
fn check_gpu_available() -> bool {
    crate::device::test_pool::tokio_block_on(crate::device::WgpuDevice::new()).is_ok()
}

/// Get global dispatch config (lazy-initialized)
pub fn global_config() -> &'static DispatchConfig {
    GLOBAL_CONFIG.get_or_init(DispatchConfig::default)
}

/// Dispatch trait for types that can auto-dispatch
pub trait Dispatch {
    /// Get the dispatch configuration
    fn dispatch_config(&self) -> &DispatchConfig;

    /// Should this workload use GPU?
    fn should_use_gpu(&self, operation: &str) -> bool;

    /// Get input size for dispatch decision
    fn dispatch_size(&self) -> usize;
}

/// Dispatch decision result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchTarget {
    /// Use CPU (f64 precision available)
    Cpu,
    /// Use GPU (f32 precision, high throughput)
    Gpu,
}

impl DispatchTarget {
    /// Check if this is CPU target
    #[must_use]
    pub fn is_cpu(self) -> bool {
        matches!(self, DispatchTarget::Cpu)
    }

    /// Check if this is GPU target
    #[must_use]
    pub fn is_gpu(self) -> bool {
        matches!(self, DispatchTarget::Gpu)
    }
}

/// Decide dispatch target for given operation and size
#[must_use]
pub fn dispatch_for(operation: &str, input_size: usize) -> DispatchTarget {
    if global_config().should_use_gpu(input_size, operation) {
        DispatchTarget::Gpu
    } else {
        DispatchTarget::Cpu
    }
}

// -----------------------------------------------------------------------------
// Domain-specific dispatch heuristics (M-009, from neuralSpring metalForge)
// -----------------------------------------------------------------------------

/// Recommend GPU vs CPU for pairwise distance computation.
#[must_use]
pub const fn pairwise_substrate(n_items: usize, item_dim: usize) -> DispatchTarget {
    let n_pairs = n_items.saturating_mul(n_items.saturating_sub(1)) / 2;
    let estimated_work = n_pairs * item_dim;
    if estimated_work > 500_000 {
        DispatchTarget::Gpu
    } else {
        DispatchTarget::Cpu
    }
}

/// Recommend GPU vs CPU for batch fitness evaluation.
#[must_use]
pub const fn batch_fitness_substrate(pop_size: usize, genome_len: usize) -> DispatchTarget {
    let total_work = pop_size * genome_len;
    if total_work > 50_000 {
        DispatchTarget::Gpu
    } else {
        DispatchTarget::Cpu
    }
}

/// Recommend GPU vs CPU for parallel ODE integration.
#[must_use]
pub const fn ode_substrate(n_systems: usize, n_steps: usize) -> DispatchTarget {
    let total_work = n_systems * n_steps;
    if total_work > 10_000 {
        DispatchTarget::Gpu
    } else {
        DispatchTarget::Cpu
    }
}

/// Recommend GPU vs CPU for HMM forward pass.
#[must_use]
pub const fn hmm_substrate(n_states: usize, n_observations: usize) -> DispatchTarget {
    let total_work = n_states * n_observations;
    if total_work > HMM_FORWARD_THRESHOLD {
        DispatchTarget::Gpu
    } else {
        DispatchTarget::Cpu
    }
}

/// Recommend GPU vs CPU for spatial payoff (game theory grid).
#[must_use]
pub const fn spatial_substrate(grid_cells: usize) -> DispatchTarget {
    if grid_cells > 4_000 {
        DispatchTarget::Gpu
    } else {
        DispatchTarget::Cpu
    }
}

/// Decide dispatch target using custom config
#[must_use]
pub fn dispatch_with_config(
    config: &DispatchConfig,
    operation: &str,
    input_size: usize,
) -> DispatchTarget {
    if config.should_use_gpu(input_size, operation) {
        DispatchTarget::Gpu
    } else {
        DispatchTarget::Cpu
    }
}

/// Dispatch with transfer cost awareness.
///
/// Like [`dispatch_for`] but additionally penalizes GPU dispatch when the
/// `PCIe` data transfer would dominate the compute savings. Uses the
/// [`BandwidthTier`] to estimate transfer overhead and compares against
/// a heuristic GPU compute advantage.
///
/// For shared-memory or `NVLink` tiers the transfer cost is negligible, so
/// this falls through to the basic threshold check.
#[must_use]
pub fn dispatch_with_transfer_cost(
    operation: &str,
    input_size: usize,
    data_bytes: usize,
    bandwidth: crate::unified_hardware::BandwidthTier,
) -> DispatchTarget {
    let config = global_config();

    if !config.should_use_gpu(input_size, operation) {
        return DispatchTarget::Cpu;
    }

    let cost = bandwidth.transfer_cost();
    let transfer_us =
        cost.estimated_us(data_bytes) + crate::unified_hardware::GPU_DISPATCH_OVERHEAD_US;

    // Heuristic: GPU advantage grows ~1 ns per element above the threshold.
    // This is conservative — real GPU throughput is much higher for parallel ops,
    // but we only need a rough breakeven estimate.
    let threshold = config.threshold(operation);
    let compute_advantage_us = input_size.saturating_sub(threshold) as f64 * 0.001;

    if compute_advantage_us > transfer_us {
        DispatchTarget::Gpu
    } else {
        DispatchTarget::Cpu
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = DispatchConfig::default();

        // Check some default thresholds
        assert_eq!(config.threshold("matmul"), 64);
        assert_eq!(config.threshold("erf"), 512);
        assert_eq!(config.threshold("cdist"), 200);

        // Unknown operation gets default
        assert_eq!(config.threshold("unknown_op"), DEFAULT_THRESHOLD);
    }

    #[test]
    fn test_force_cpu() {
        let config = DispatchConfig::default().force_cpu();

        // Should never use GPU when forced to CPU
        assert!(!config.should_use_gpu(1_000_000, "matmul"));
        assert!(!config.should_use_gpu(1_000_000, "erf"));
    }

    #[test]
    fn test_size_threshold() {
        let config = DispatchConfig::default();

        // Below threshold: CPU
        // Note: if GPU not available, always returns false
        if config.has_gpu() {
            assert!(!config.should_use_gpu(32, "matmul")); // 32 < 64
            assert!(config.should_use_gpu(128, "matmul")); // 128 >= 64
        }
    }

    #[test]
    fn test_custom_threshold() {
        let mut config = DispatchConfig::default();
        config.set_threshold("custom_op", 500);

        assert_eq!(config.threshold("custom_op"), 500);
    }

    #[test]
    fn test_dispatch_target() {
        assert!(DispatchTarget::Cpu.is_cpu());
        assert!(!DispatchTarget::Cpu.is_gpu());
        assert!(DispatchTarget::Gpu.is_gpu());
        assert!(!DispatchTarget::Gpu.is_cpu());
    }

    #[test]
    fn test_dispatch_for() {
        // Small input should go to CPU
        let target = dispatch_for("matmul", 10);
        assert!(target.is_cpu());
    }

    #[test]
    fn test_force_gpu_and_cpu_mutual_exclusion() {
        let config = DispatchConfig::default().force_gpu().force_cpu();
        assert!(!config.should_use_gpu(1_000_000, "matmul"));

        let config = DispatchConfig::default().force_cpu().force_gpu();
        if config.has_gpu() {
            assert!(config.should_use_gpu(1, "matmul"));
        }
    }

    #[test]
    fn test_with_thresholds_custom_map() {
        let mut thresholds = HashMap::new();
        thresholds.insert(Arc::from("custom_fft"), 256_usize);
        thresholds.insert(Arc::from("custom_sum"), 8192_usize);

        let config = DispatchConfig::with_thresholds(thresholds);

        assert_eq!(config.threshold("custom_fft"), 256);
        assert_eq!(config.threshold("custom_sum"), 8192);
        assert_eq!(config.threshold("unknown"), DEFAULT_THRESHOLD);
    }

    #[test]
    fn test_dispatch_with_config() {
        let config = DispatchConfig::default().force_cpu();
        let target = dispatch_with_config(&config, "matmul", 1_000_000);
        assert!(target.is_cpu());
    }

    #[test]
    fn test_dispatch_with_config_gpu_available() {
        let config = DispatchConfig::default();
        if config.has_gpu() {
            let target = dispatch_with_config(&config, "matmul", 1_000_000);
            assert!(target.is_gpu());
        }
    }

    #[test]
    fn test_dispatch_with_transfer_cost_small_input_stays_cpu() {
        let target = dispatch_with_transfer_cost(
            "matmul",
            10,
            10 * 4,
            crate::unified_hardware::BandwidthTier::PciE4x16,
        );
        assert!(target.is_cpu(), "small input should stay on CPU");
    }

    #[test]
    fn test_dispatch_with_transfer_cost_shared_memory() {
        let config = DispatchConfig::default();
        if config.has_gpu() {
            let target = dispatch_with_transfer_cost(
                "matmul",
                200,
                200 * 4,
                crate::unified_hardware::BandwidthTier::SharedMemory,
            );
            // SharedMemory has near-zero transfer cost, but the heuristic
            // compute advantage (200-64)*0.001 = 0.136 µs is still less
            // than GPU dispatch overhead (1500 µs). So this should be CPU.
            // Only truly massive workloads overcome the dispatch overhead.
            assert!(target.is_cpu());
        }
    }

    #[test]
    fn test_dispatch_with_transfer_cost_regression_basic_dispatch() {
        let config = DispatchConfig::default();
        if config.has_gpu() {
            // Verify that dispatch_for and dispatch_with_transfer_cost agree
            // for small inputs (both should return CPU)
            let basic = dispatch_for("erf", 100);
            let transfer_aware = dispatch_with_transfer_cost(
                "erf",
                100,
                100 * 4,
                crate::unified_hardware::BandwidthTier::PciE4x16,
            );
            assert_eq!(basic, transfer_aware);
        }
    }

    #[test]
    fn test_all_default_threshold_operations() {
        let config = DispatchConfig::default();

        assert_eq!(config.threshold("erf"), 512);
        assert_eq!(config.threshold("erfc"), 512);
        assert_eq!(config.threshold("gamma"), 256);
        assert_eq!(config.threshold("lgamma"), 256);
        assert_eq!(config.threshold("digamma"), 256);
        assert_eq!(config.threshold("bessel_j0"), 512);
        assert_eq!(config.threshold("bessel_j1"), 512);
        assert_eq!(config.threshold("bessel_i0"), 512);
        assert_eq!(config.threshold("bessel_k0"), 512);
        assert_eq!(config.threshold("cholesky"), 128);
        assert_eq!(config.threshold("eigh"), 128);
        assert_eq!(config.threshold("lu"), 128);
        assert_eq!(config.threshold("qr"), 128);
        assert_eq!(config.threshold("svd"), 128);
        assert_eq!(config.threshold("solve"), 128);
        assert_eq!(config.threshold("tridiagonal"), 256);
        assert_eq!(config.threshold("cdist"), 200);
        assert_eq!(config.threshold("pdist"), 200);
        assert_eq!(config.threshold("cosine_similarity"), 256);
        assert_eq!(config.threshold("fft"), 1024);
        assert_eq!(config.threshold("ifft"), 1024);
        assert_eq!(config.threshold("dct"), 1024);
        assert_eq!(config.threshold("sum"), 4096);
        assert_eq!(config.threshold("mean"), 4096);
        assert_eq!(config.threshold("max"), 4096);
        assert_eq!(config.threshold("min"), 4096);
        assert_eq!(config.threshold("argmax"), 4096);
        assert_eq!(config.threshold("argmin"), 4096);
        assert_eq!(config.threshold("relu"), 2048);
        assert_eq!(config.threshold("sigmoid"), 2048);
        assert_eq!(config.threshold("tanh"), 2048);
        assert_eq!(config.threshold("exp"), 2048);
        assert_eq!(config.threshold("log"), 2048);
        assert_eq!(config.threshold("sqrt"), 2048);
        assert_eq!(config.threshold("sin"), 2048);
        assert_eq!(config.threshold("cos"), 2048);
        assert_eq!(config.threshold("rbf_kernel"), 200);
        assert_eq!(config.threshold("surrogate_predict"), 100);
        assert_eq!(config.threshold("surrogate_train"), 200);
    }

    #[test]
    fn test_boundary_conditions() {
        let config = DispatchConfig::default();
        if config.has_gpu() {
            assert!(!config.should_use_gpu(63, "matmul"));
            assert!(config.should_use_gpu(64, "matmul"));
            assert!(config.should_use_gpu(65, "matmul"));
        }
    }

    #[test]
    fn test_zero_size_input() {
        let target = dispatch_for("matmul", 0);
        assert!(target.is_cpu());
    }

    #[test]
    fn test_new_equals_default() {
        let new_config = DispatchConfig::new();
        let default_config = DispatchConfig::default();

        assert_eq!(
            new_config.threshold("matmul"),
            default_config.threshold("matmul")
        );
        assert_eq!(new_config.has_gpu(), default_config.has_gpu());
    }

    // Domain-specific dispatch heuristics (M-009)
    #[test]
    fn test_pairwise_substrate_small_uses_cpu() {
        assert_eq!(pairwise_substrate(20, 500), DispatchTarget::Cpu);
    }

    #[test]
    fn test_pairwise_substrate_large_uses_gpu() {
        assert_eq!(pairwise_substrate(200, 1000), DispatchTarget::Gpu);
    }

    #[test]
    fn test_batch_fitness_substrate_small_uses_cpu() {
        assert_eq!(batch_fitness_substrate(100, 10), DispatchTarget::Cpu);
    }

    #[test]
    fn test_batch_fitness_substrate_large_uses_gpu() {
        assert_eq!(batch_fitness_substrate(50_000, 64), DispatchTarget::Gpu);
    }

    #[test]
    fn test_ode_substrate_small_uses_cpu() {
        assert_eq!(ode_substrate(10, 100), DispatchTarget::Cpu);
    }

    #[test]
    fn test_ode_substrate_large_uses_gpu() {
        assert_eq!(ode_substrate(1000, 2000), DispatchTarget::Gpu);
    }

    #[test]
    fn test_hmm_substrate_small_uses_cpu() {
        assert_eq!(hmm_substrate(3, 100), DispatchTarget::Cpu);
    }

    #[test]
    fn test_hmm_substrate_large_uses_gpu() {
        assert_eq!(hmm_substrate(3, 5000), DispatchTarget::Gpu);
    }

    #[test]
    fn test_spatial_substrate_small_uses_cpu() {
        assert_eq!(spatial_substrate(100), DispatchTarget::Cpu);
    }

    #[test]
    fn test_spatial_substrate_large_uses_gpu() {
        assert_eq!(spatial_substrate(10_000), DispatchTarget::Gpu);
    }
}
