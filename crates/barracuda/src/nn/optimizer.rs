//! Optimizer Implementations
//!
//! Various optimization algorithms for neural network training.
//! Deep Debt compliant: Runtime parameter configuration.

/// Optimizer types (capability-based)
#[derive(Debug, Clone)]
pub enum Optimizer {
    /// Adam optimizer
    Adam {
        lr: f32,
        betas: (f32, f32),
        eps: f32,
    },
    /// AdaGrad optimizer
    AdaGrad { lr: f32, eps: f32 },
    /// AdaDelta optimizer
    AdaDelta { rho: f32, eps: f32 },
    /// SGD with momentum
    SGD { lr: f32, momentum: f32 },
}
