// SPDX-License-Identifier: AGPL-3.0-only
//! Auto-dispatch system for CPU/GPU routing
//!
//! Provides intelligent, size-based dispatch for mathematical operations.
//! Small workloads stay on CPU (avoiding dispatch overhead), large workloads
//! use GPU acceleration.
//!
//! # Dual-Precision Architecture
//!
//! ```text
//! User calls: erf(x) or matmul(a, b)
//!       ↓
//! DispatchConfig checks:
//! - Input size vs threshold
//! - GPU availability
//! - Force flags
//!       ↓
//! Routes to:
//! - CPU f64 (small N, precision-critical)
//! - GPU f32 (large N, throughput-critical)
//! ```
//!
//! # Per-Function Thresholds
//!
//! Thresholds are empirically determined via benchmarking:
//!
//! | Operation | CPU Threshold | Reason |
//! |-----------|---------------|--------|
//! | erf | 512 | GPU dispatch overhead ~0.1ms |
//! | matmul | 64 | GPU wins at 64×64 matrices |
//! | eigh | 128 | Jacobi iteration memory-bound |
//! | cdist | 200 | Distance computation O(N²) |
//! | fft | 1024 | FFT benefits from parallelism |
//!
//! # Example
//!
//! ```
//! use barracuda::dispatch::{Dispatch, DispatchConfig};
//!
//! // Configure dispatch
//! let config = DispatchConfig::default();
//!
//! // Auto-route based on size
//! if config.should_use_gpu(1000, "matmul") {
//!     // GPU path
//! } else {
//!     // CPU path
//! }
//! ```
//!
//! # Benchmarking
//!
//! Use the benchmark module to empirically determine optimal thresholds:
//!
//! ```ignore
//! use barracuda::dispatch::benchmark::{BenchmarkSuite, BenchmarkConfig};
//!
//! let suite = BenchmarkSuite::new(BenchmarkConfig::default());
//! let results = suite.run_all()?;
//! println!("{}", results.summary());
//!
//! // Update thresholds based on results
//! let new_thresholds = results.optimal_thresholds();
//! ```

pub mod benchmark;
mod config;
pub mod domain_ops;

pub use benchmark::{
    BenchmarkConfig, BenchmarkResult, BenchmarkSuite, OperationBenchmark, ThresholdResult,
};
pub use config::{
    DEFAULT_THRESHOLD, Dispatch, DispatchConfig, DispatchTarget, batch_fitness_substrate,
    dispatch_for, dispatch_with_config, dispatch_with_transfer_cost, global_config, hmm_substrate,
    ode_substrate, pairwise_substrate, spatial_substrate,
};
pub use domain_ops::{
    frobenius_norm_dispatch, gelu_dispatch, hmm_forward_dispatch, l2_distance_dispatch,
    matmul_dispatch, mean_dispatch, softmax_dispatch, transpose_dispatch, variance_dispatch,
};

/// wgpu limits dispatch to 65535 workgroups per dimension. For large lattices
/// (32^4+ sites) the total workgroup count can exceed this limit. This helper
/// splits the count into a 2D `(x, y, 1)` dispatch that covers the full range.
///
/// Shaders must linearize via `gid.x + gid.y * num_workgroups.x * WG_SIZE`.
#[must_use]
pub fn split_workgroups(total: u32) -> (u32, u32, u32) {
    const MAX_DIM: u32 = 65535;
    if total <= MAX_DIM {
        (total, 1, 1)
    } else {
        let y = total.div_ceil(MAX_DIM);
        let x = total.div_ceil(y);
        (x, y, 1)
    }
}

#[cfg(test)]
mod split_workgroups_tests {
    use super::split_workgroups;

    #[test]
    fn small_stays_1d() {
        assert_eq!(split_workgroups(100), (100, 1, 1));
    }

    #[test]
    fn exact_limit_stays_1d() {
        assert_eq!(split_workgroups(65535), (65535, 1, 1));
    }

    #[test]
    fn just_over_limit_splits() {
        let (x, y, z) = split_workgroups(65536);
        assert!(x <= 65535);
        assert!(y > 1);
        assert_eq!(z, 1);
        assert!(x * y >= 65536, "product {x}*{y} must cover total");
    }

    #[test]
    fn large_lattice_coverage() {
        let total = 32_u32.pow(4) / 64; // 32^4 sites / wg_size=64 = 16384
        let (x, y, z) = split_workgroups(total);
        assert_eq!((x, y, z), (total, 1, 1), "16384 fits in 1D");

        let huge = 500_000_u32;
        let (x, y, z) = split_workgroups(huge);
        assert!(x <= 65535);
        assert!(y <= 65535);
        assert_eq!(z, 1);
        assert!(x * y >= huge);
    }
}
