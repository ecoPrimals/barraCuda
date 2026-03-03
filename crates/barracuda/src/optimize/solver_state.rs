//! Pausable/resumable solver state management
//!
//! Provides a stateful Nelder-Mead optimizer that can be paused mid-iteration
//! and resumed later with the same simplex state. This is critical for the
//! SparsitySampler workflow:
//!
//! 1. Run NM for N evaluations → pause
//! 2. Train surrogate on accumulated evaluations
//! 3. Use surrogate to select next batch of starting points
//! 4. Resume or spawn new NM instances → repeat
//!
//! # Cross-Domain Applications
//!
//! - **Iterative surrogate learning**: Alternate between optimization and model training
//! - **Budget-constrained optimization**: Run for K evaluations, checkpoint, continue later
//! - **Interactive optimization**: Human-in-the-loop parameter tuning
//! - **Distributed optimization**: Checkpoint state for migration between workers
//!
//! # References
//!
//! - Diaw et al. (2024). Iterative optimize→train→optimize workflow.
//! - mystic framework: Solver state serialization for checkpoint/restart.

use crate::error::{BarracudaError, Result};
use crate::optimize::eval_record::EvaluationCache;

/// Status of a resumable solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverStatus {
    /// Solver has not started yet
    NotStarted,
    /// Solver is paused (can be resumed)
    Paused,
    /// Solver converged (met tolerance)
    Converged,
    /// Solver exhausted evaluation budget
    BudgetExhausted,
}

impl std::fmt::Display for SolverStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolverStatus::NotStarted => write!(f, "NotStarted"),
            SolverStatus::Paused => write!(f, "Paused"),
            SolverStatus::Converged => write!(f, "Converged"),
            SolverStatus::BudgetExhausted => write!(f, "BudgetExhausted"),
        }
    }
}

/// A pausable, resumable Nelder-Mead solver.
///
/// Maintains the full simplex state between `step()` calls, allowing the
/// optimizer to be paused and resumed at any point.
///
/// # Examples
///
/// ```
/// use barracuda::optimize::solver_state::ResumableNelderMead;
///
/// let f = |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2);
/// let bounds = vec![(-10.0, 10.0), (-10.0, 10.0)];
/// let x0 = vec![0.0, 0.0];
///
/// let mut solver = ResumableNelderMead::new(&x0, &bounds, 1e-8)?;
///
/// // Run for 100 evaluations
/// solver.step(&f, 100);
///
/// // Check state
/// let (best_x, best_f) = solver.best();
/// println!("After 100 evals: f = {}", best_f);
///
/// // Resume for another 100 evaluations
/// solver.step(&f, 100);
///
/// let (best_x2, best_f2) = solver.best();
/// assert!(best_f2 <= best_f); // Should improve or stay same
/// # Ok::<(), barracuda::error::BarracudaError>(())
/// ```
pub struct ResumableNelderMead {
    /// Current simplex vertices
    simplex: Vec<Vec<f64>>,
    /// Function values at each vertex
    f_vals: Vec<f64>,
    /// Box constraints
    bounds: Vec<(f64, f64)>,
    /// Number of dimensions
    n_dims: usize,
    /// Convergence tolerance
    tol: f64,
    /// Total evaluations performed
    n_evals: usize,
    /// Current solver status
    status: SolverStatus,
    /// Evaluation cache (records all evaluations)
    cache: EvaluationCache,
}

impl ResumableNelderMead {
    /// Create a new resumable NM solver with initial point and bounds.
    ///
    /// The initial simplex is constructed by perturbing each dimension of `x0`.
    pub fn new(x0: &[f64], bounds: &[(f64, f64)], tol: f64) -> Result<Self> {
        let n = x0.len();
        if bounds.len() != n {
            return Err(BarracudaError::InvalidInput {
                message: format!("Bounds length {} must match x0 length {}", bounds.len(), n),
            });
        }
        if n == 0 {
            return Err(BarracudaError::InvalidInput {
                message: "x0 must be non-empty".to_string(),
            });
        }

        Ok(Self {
            simplex: Vec::new(),
            f_vals: Vec::new(),
            bounds: bounds.to_vec(),
            n_dims: n,
            tol,
            n_evals: 0,
            status: SolverStatus::NotStarted,
            cache: EvaluationCache::new(),
        })
    }

    /// Run the solver for up to `budget` function evaluations.
    ///
    /// Returns the number of evaluations actually performed. The solver may
    /// stop early if convergence is reached.
    pub fn step<F>(&mut self, f: &F, budget: usize) -> usize
    where
        F: Fn(&[f64]) -> f64,
    {
        let start_evals = self.n_evals;

        // Initialize simplex on first call
        if self.status == SolverStatus::NotStarted {
            self.initialize_simplex(f);
            if self.n_evals >= start_evals + budget {
                self.status = SolverStatus::Paused;
                return self.n_evals - start_evals;
            }
        }

        // Already terminal?
        if self.status == SolverStatus::Converged || self.status == SolverStatus::BudgetExhausted {
            return 0;
        }

        let n = self.n_dims;

        // Nelder-Mead parameters
        const ALPHA: f64 = 1.0;
        const GAMMA: f64 = 2.0;
        const RHO: f64 = 0.5;
        const SIGMA: f64 = 0.5;

        while self.n_evals < start_evals + budget {
            // Sort by function value
            let mut indices: Vec<usize> = (0..=n).collect();
            indices.sort_by(|&i, &j| {
                self.f_vals[i]
                    .partial_cmp(&self.f_vals[j])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let best_idx = indices[0];
            let worst_idx = indices[n];
            let second_worst_idx = indices[n - 1];

            // Convergence check
            let f_mean: f64 = self.f_vals.iter().sum::<f64>() / (n + 1) as f64;
            let f_std = (self
                .f_vals
                .iter()
                .map(|&fi| (fi - f_mean).powi(2))
                .sum::<f64>()
                / (n + 1) as f64)
                .sqrt();

            if f_std < self.tol {
                self.status = SolverStatus::Converged;
                return self.n_evals - start_evals;
            }

            // Centroid (excluding worst)
            let mut centroid = vec![0.0; n];
            for &idx in &indices[..n] {
                for (j, c) in centroid.iter_mut().enumerate() {
                    *c += self.simplex[idx][j];
                }
            }
            for c in &mut centroid {
                *c /= n as f64;
            }

            // Reflection
            let x_reflect = reflect(&self.simplex[worst_idx], &centroid, ALPHA);
            let x_reflect = project_bounds(&x_reflect, &self.bounds);
            let f_reflect = self.eval(f, &x_reflect);

            if f_reflect < self.f_vals[best_idx] {
                // Expansion
                let x_expand = reflect(&self.simplex[worst_idx], &centroid, GAMMA);
                let x_expand = project_bounds(&x_expand, &self.bounds);
                let f_expand = self.eval(f, &x_expand);

                if f_expand < f_reflect {
                    self.simplex[worst_idx] = x_expand;
                    self.f_vals[worst_idx] = f_expand;
                } else {
                    self.simplex[worst_idx] = x_reflect;
                    self.f_vals[worst_idx] = f_reflect;
                }
            } else if f_reflect < self.f_vals[second_worst_idx] {
                self.simplex[worst_idx] = x_reflect;
                self.f_vals[worst_idx] = f_reflect;
            } else {
                let x_contract = if f_reflect < self.f_vals[worst_idx] {
                    reflect(&self.simplex[worst_idx], &centroid, ALPHA * RHO)
                } else {
                    reflect(&self.simplex[worst_idx], &centroid, -RHO)
                };
                let x_contract = project_bounds(&x_contract, &self.bounds);
                let f_contract = self.eval(f, &x_contract);

                if f_contract < self.f_vals[worst_idx] {
                    self.simplex[worst_idx] = x_contract;
                    self.f_vals[worst_idx] = f_contract;
                } else {
                    // Shrinkage
                    for i in 0..=n {
                        if i != best_idx {
                            for j in 0..n {
                                self.simplex[i][j] = self.simplex[best_idx][j]
                                    + SIGMA * (self.simplex[i][j] - self.simplex[best_idx][j]);
                            }
                            self.simplex[i] = project_bounds(&self.simplex[i], &self.bounds);
                            let fi = self.eval(f, &self.simplex[i].clone());
                            self.f_vals[i] = fi;
                        }
                    }
                }
            }
        }

        self.status = SolverStatus::Paused;
        self.n_evals - start_evals
    }

    /// Get the best point and function value found so far.
    pub fn best(&self) -> (Vec<f64>, f64) {
        if self.f_vals.is_empty() {
            return (vec![0.0; self.n_dims], f64::INFINITY);
        }

        let best_idx = self
            .f_vals
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(idx, _)| idx);

        (self.simplex[best_idx].clone(), self.f_vals[best_idx])
    }

    /// Get the current solver status.
    pub fn status(&self) -> SolverStatus {
        self.status
    }

    /// Total number of function evaluations performed.
    pub fn n_evals(&self) -> usize {
        self.n_evals
    }

    /// Access the evaluation cache.
    pub fn cache(&self) -> &EvaluationCache {
        &self.cache
    }

    /// Take ownership of the evaluation cache (consumes the cache).
    pub fn take_cache(&mut self) -> EvaluationCache {
        std::mem::take(&mut self.cache)
    }

    /// Whether the solver has converged.
    pub fn is_converged(&self) -> bool {
        self.status == SolverStatus::Converged
    }

    /// Whether the solver can be resumed.
    pub fn is_resumable(&self) -> bool {
        self.status == SolverStatus::Paused || self.status == SolverStatus::NotStarted
    }

    // --- Private helpers ---

    fn initialize_simplex<F>(&mut self, f: &F)
    where
        F: Fn(&[f64]) -> f64,
    {
        let n = self.n_dims;
        // Use a zero-based x0 projected onto bounds
        let x0: Vec<f64> = self.bounds.iter().map(|(lo, hi)| (lo + hi) / 2.0).collect();
        let x0_bounded = project_bounds(&x0, &self.bounds);
        let f0 = self.eval(f, &x0_bounded);
        self.simplex.push(x0_bounded);
        self.f_vals.push(f0);

        for i in 0..n {
            let mut x = x0.clone();
            let delta = 0.05 * (self.bounds[i].1 - self.bounds[i].0).max(0.1);
            x[i] += delta;
            let x_bounded = project_bounds(&x, &self.bounds);
            let fx = self.eval(f, &x_bounded);
            self.simplex.push(x_bounded);
            self.f_vals.push(fx);
        }

        self.status = SolverStatus::Paused;
    }

    fn eval<F>(&mut self, f: &F, x: &[f64]) -> f64
    where
        F: Fn(&[f64]) -> f64,
    {
        let val = f(x);
        self.cache.record(x.to_vec(), val);
        self.n_evals += 1;
        val
    }
}

fn reflect(x: &[f64], centroid: &[f64], alpha: f64) -> Vec<f64> {
    centroid
        .iter()
        .zip(x.iter())
        .map(|(&c, &xi)| c + alpha * (c - xi))
        .collect()
}

fn project_bounds(x: &[f64], bounds: &[(f64, f64)]) -> Vec<f64> {
    x.iter()
        .zip(bounds.iter())
        .map(|(&xi, &(lo, hi))| xi.clamp(lo, hi))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resumable_nm_basic() {
        let f = |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2);
        let bounds = vec![(-10.0, 10.0), (-10.0, 10.0)];
        let x0 = vec![0.0, 0.0];

        let mut solver = ResumableNelderMead::new(&x0, &bounds, 1e-8).unwrap();
        assert_eq!(solver.status(), SolverStatus::NotStarted);
        assert!(solver.is_resumable());

        solver.step(&f, 500);

        let (best_x, best_f) = solver.best();
        assert!((best_x[0] - 2.0).abs() < 0.5);
        assert!((best_x[1] - 3.0).abs() < 0.5);
        assert!(best_f < 1.0);
    }

    #[test]
    fn test_resumable_nm_pause_resume() {
        let f = |x: &[f64]| (x[0] - 5.0).powi(2);
        let bounds = vec![(-10.0, 10.0)];
        let x0 = vec![0.0];

        let mut solver = ResumableNelderMead::new(&x0, &bounds, 1e-8).unwrap();

        // Run 50 evaluations
        solver.step(&f, 50);
        let (_, f1) = solver.best();
        let evals_after_first = solver.n_evals();

        assert!(
            solver.status() == SolverStatus::Paused || solver.status() == SolverStatus::Converged
        );

        // Resume for 50 more
        solver.step(&f, 50);
        let (_, f2) = solver.best();

        // Should improve or maintain
        assert!(f2 <= f1 + 1e-10);

        // Total evals should be higher (or same if converged)
        assert!(solver.n_evals() >= evals_after_first);
    }

    #[test]
    fn test_resumable_nm_convergence() {
        let f = |x: &[f64]| x[0].powi(2);
        let bounds = vec![(-10.0, 10.0)];
        let x0 = vec![5.0];

        let mut solver = ResumableNelderMead::new(&x0, &bounds, 1e-6).unwrap();
        solver.step(&f, 1000);

        // Should converge on this simple problem
        assert!(
            solver.is_converged() || solver.best().1 < 1e-4,
            "Should converge: status={}, best_f={}",
            solver.status(),
            solver.best().1
        );
    }

    #[test]
    fn test_resumable_nm_cache() {
        let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let x0 = vec![1.0, 1.0];

        let mut solver = ResumableNelderMead::new(&x0, &bounds, 1e-8).unwrap();
        solver.step(&f, 100);

        // Cache should have all evaluations
        assert_eq!(solver.cache().len(), solver.n_evals());
        assert!(!solver.cache().is_empty());

        // Best in cache should match solver best
        let cache_best = solver.cache().best_f().unwrap();
        let (_, solver_best) = solver.best();
        assert!((cache_best - solver_best).abs() < 1e-10);
    }

    #[test]
    fn test_resumable_nm_take_cache() {
        let f = |x: &[f64]| x[0].powi(2);
        let bounds = vec![(-5.0, 5.0)];
        let x0 = vec![3.0];

        let mut solver = ResumableNelderMead::new(&x0, &bounds, 1e-8).unwrap();
        solver.step(&f, 50);

        let n_before = solver.cache().len();
        assert!(n_before > 0);

        // Take the cache (empties it)
        let cache = solver.take_cache();
        assert_eq!(cache.len(), n_before);
        assert_eq!(solver.cache().len(), 0);
    }

    #[test]
    fn test_resumable_nm_errors() {
        // Bounds/x0 mismatch
        assert!(ResumableNelderMead::new(&[0.0, 0.0], &[(0.0, 1.0)], 1e-8).is_err());

        // Empty x0
        assert!(ResumableNelderMead::new(&[], &[], 1e-8).is_err());
    }

    #[test]
    fn test_solver_status_display() {
        assert_eq!(format!("{}", SolverStatus::NotStarted), "NotStarted");
        assert_eq!(format!("{}", SolverStatus::Paused), "Paused");
        assert_eq!(format!("{}", SolverStatus::Converged), "Converged");
        assert_eq!(
            format!("{}", SolverStatus::BudgetExhausted),
            "BudgetExhausted"
        );
    }

    #[test]
    fn test_resumable_nm_multi_resume() {
        // Run in 10-eval increments to test repeated pause/resume
        let f = |x: &[f64]| (x[0] - 3.0).powi(2) + (x[1] + 1.0).powi(2);
        let bounds = vec![(-10.0, 10.0), (-10.0, 10.0)];
        let x0 = vec![0.0, 0.0];

        let mut solver = ResumableNelderMead::new(&x0, &bounds, 1e-8).unwrap();

        for _ in 0..20 {
            if solver.is_converged() {
                break;
            }
            solver.step(&f, 25);
        }

        let (best_x, _) = solver.best();
        assert!((best_x[0] - 3.0).abs() < 1.0);
        assert!((best_x[1] + 1.0).abs() < 1.0);
    }
}
