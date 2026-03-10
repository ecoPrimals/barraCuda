// SPDX-License-Identifier: AGPL-3.0-only
//! `SparsitySampler` configuration and penalty filter types.

use crate::surrogate::RBFKernel;
use std::sync::Arc;

use crate::device::WgpuDevice;

/// Penalty filter strategy for surrogate training.
///
/// When training the RBF surrogate, large penalty values from infeasible
/// regions can corrupt the approximation. These filters remove or cap
/// penalty values before training.
///
/// # Reference
///
/// hotSpring validation: `surrogate.rs::filter_training_data()`
#[derive(Debug, Clone, Copy, Default)]
pub enum PenaltyFilter {
    /// No filtering (default)
    #[default]
    None,
    /// Remove all y values exceeding threshold
    Threshold(f64),
    /// Remove top q% outliers (0.0 to 1.0)
    Quantile(f64),
    /// Median + k×MAD (robust outlier detection)
    AdaptiveMAD(f64),
}

/// Configuration for the `SparsitySampler`.
#[derive(Clone)]
pub struct SparsitySamplerConfig {
    /// Number of initial samples via LHS (default: 10 × `n_dims`)
    pub n_initial: usize,
    /// Number of NM solvers per iteration (default: 8)
    pub n_solvers: usize,
    /// Number of NM solvers running on TRUE objective (default: 2)
    ///
    /// These solvers run Nelder-Mead directly on the true objective function,
    /// accumulating more true evaluations per iteration for exploration.
    /// Remaining solvers (`n_solvers` - `n_direct_solvers`) run on the surrogate
    /// for efficient exploitation.
    ///
    /// # Hybrid Evaluation Mode
    ///
    /// With `n_direct_solvers` > 0, the sampler balances:
    /// - **Exploration**: Direct solvers accumulate true evaluations, densely
    ///   sampling the objective landscape
    /// - **Exploitation**: Surrogate solvers efficiently find local optima
    ///
    /// # Reference
    ///
    /// hotSpring L2 validation: Python's mystic does ~100-200 true evals per round.
    /// `BarraCuda` with `n_direct_solvers=2` and `n_solvers=8` approaches this density.
    pub n_direct_solvers: usize,
    /// Max evaluations per NM solver per iteration (default: 50)
    pub max_eval_per_solver: usize,
    /// Number of surrogate refinement iterations (default: 5)
    pub n_iterations: usize,
    /// NM convergence tolerance (default: 1e-6)
    pub tol: f64,
    /// RBF kernel for surrogate (default: `ThinPlateSpline`)
    pub kernel: RBFKernel,
    /// RBF smoothing parameter (default: 1e-3, but see `auto_smoothing`)
    pub smoothing: f64,
    /// Enable LOO-CV auto-tuning of smoothing (default: true)
    ///
    /// When enabled, the sampler will run LOO-CV grid search after each
    /// iteration to find the optimal smoothing parameter. This prevents
    /// both overfitting (smoothing too low) and underfitting (smoothing too high).
    ///
    /// # Reference
    ///
    /// hotSpring validation: `surrogate.rs::loo_cv_optimal_smoothing()`
    pub auto_smoothing: bool,
    /// Penalty filter for surrogate training (default: None)
    ///
    /// Filters out penalty values before training the surrogate, preventing
    /// corruption from large infeasible-region penalties.
    pub penalty_filter: PenaltyFilter,
    /// Warm-start seeds (default: empty)
    ///
    /// Pre-computed starting points for optimization (e.g., from L1 layer
    /// for L2 optimization). When non-empty, these seeds are used as
    /// additional starting points alongside LHS samples.
    ///
    /// # Reference
    ///
    /// hotSpring validation: `nuclear_eos_l2_ref.rs` L1-seeded L2 pattern
    pub warm_start_seeds: Vec<Vec<f64>>,
    /// Random seed
    pub seed: u64,
    /// GPU device for hybrid evaluation (None = CPU only)
    pub gpu_device: Option<Arc<WgpuDevice>>,
    /// Minimum dataset size to trigger GPU acceleration (default: 100)
    pub gpu_threshold: usize,
}

impl std::fmt::Debug for SparsitySamplerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SparsitySamplerConfig")
            .field("n_initial", &self.n_initial)
            .field("n_solvers", &self.n_solvers)
            .field("n_direct_solvers", &self.n_direct_solvers)
            .field("max_eval_per_solver", &self.max_eval_per_solver)
            .field("n_iterations", &self.n_iterations)
            .field("tol", &self.tol)
            .field("kernel", &self.kernel)
            .field("smoothing", &self.smoothing)
            .field("auto_smoothing", &self.auto_smoothing)
            .field("penalty_filter", &self.penalty_filter)
            .field("warm_start_seeds", &self.warm_start_seeds.len())
            .field("seed", &self.seed)
            .field(
                "gpu_device",
                &self.gpu_device.as_ref().map(|_| "Some(WgpuDevice)"),
            )
            .field("gpu_threshold", &self.gpu_threshold)
            .finish()
    }
}

impl SparsitySamplerConfig {
    /// Create a default configuration scaled to the problem dimension.
    ///
    /// Default uses `auto_smoothing = true` with LOO-CV to select optimal
    /// smoothing parameter, preventing both overfitting and underfitting.
    ///
    /// # Note
    ///
    /// Previous default was `auto_smoothing = false` with `smoothing = 1e-12`,
    /// which caused severe overfitting (0.0 training RMSE but poor prediction).
    /// Changed to `auto_smoothing = true` after hotSpring L2 validation.
    #[must_use]
    pub fn new(n_dims: usize, seed: u64) -> Self {
        Self {
            n_initial: 10 * n_dims,
            n_solvers: 8,
            n_direct_solvers: 2, // Hybrid mode: 2 direct + 6 surrogate (hotSpring L2 fix)
            max_eval_per_solver: 50,
            n_iterations: 5,
            tol: 1e-6,
            kernel: RBFKernel::ThinPlateSpline,
            smoothing: 1e-3,      // Reasonable default if auto_smoothing fails
            auto_smoothing: true, // CHANGED: prevent overfitting (hotSpring L2 fix)
            penalty_filter: PenaltyFilter::None,
            warm_start_seeds: Vec::new(),
            seed,
            gpu_device: None,
            gpu_threshold: 100,
        }
    }

    /// Set number of initial LHS samples.
    #[must_use]
    pub fn with_initial_samples(mut self, n: usize) -> Self {
        self.n_initial = n;
        self
    }

    /// Set number of NM solvers per iteration.
    #[must_use]
    pub fn with_solvers(mut self, n: usize) -> Self {
        self.n_solvers = n;
        self
    }

    /// Set number of direct solvers (running on true objective).
    #[must_use]
    pub fn with_direct_solvers(mut self, n: usize) -> Self {
        self.n_direct_solvers = n.min(self.n_solvers);
        self
    }

    /// Set max evaluations per solver.
    #[must_use]
    pub fn with_eval_budget(mut self, n: usize) -> Self {
        self.max_eval_per_solver = n;
        self
    }

    /// Set number of refinement iterations.
    #[must_use]
    pub fn with_iterations(mut self, n: usize) -> Self {
        self.n_iterations = n;
        self
    }

    /// Set RBF kernel type.
    #[must_use]
    pub fn with_kernel(mut self, kernel: RBFKernel) -> Self {
        self.kernel = kernel;
        self
    }

    /// Enable GPU-accelerated surrogate training.
    #[must_use]
    pub fn with_gpu(mut self, device: Arc<WgpuDevice>) -> Self {
        self.gpu_device = Some(device);
        self
    }

    /// Set minimum dataset size to trigger GPU acceleration (default: 100).
    #[must_use]
    pub fn with_gpu_threshold(mut self, n: usize) -> Self {
        self.gpu_threshold = n;
        self
    }

    /// Set RBF smoothing parameter explicitly.
    #[must_use]
    pub fn with_smoothing(mut self, smoothing: f64) -> Self {
        self.smoothing = smoothing;
        self
    }

    /// Enable automatic smoothing via LOO-CV grid search.
    #[must_use]
    pub fn with_auto_smoothing(mut self, enabled: bool) -> Self {
        self.auto_smoothing = enabled;
        self
    }

    /// Set penalty filtering strategy for surrogate training.
    #[must_use]
    pub fn with_penalty_filter(mut self, filter: PenaltyFilter) -> Self {
        self.penalty_filter = filter;
        self
    }

    /// Set warm-start seeds from a previous optimization layer.
    #[must_use]
    pub fn with_warm_start(mut self, seeds: Vec<Vec<f64>>) -> Self {
        self.warm_start_seeds = seeds;
        self
    }

    /// Total evaluation budget (approximate).
    #[must_use]
    pub fn total_budget(&self) -> usize {
        self.n_initial + self.n_iterations * self.n_solvers * self.max_eval_per_solver
    }

    /// Check if GPU acceleration is configured and applicable.
    #[must_use]
    pub fn should_use_gpu(&self, dataset_size: usize) -> bool {
        self.gpu_device.is_some() && dataset_size >= self.gpu_threshold
    }
}
