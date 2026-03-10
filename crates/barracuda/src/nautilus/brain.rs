// SPDX-License-Identifier: AGPL-3.0-only
//! High-level `NautilusBrain` API for physics observables.

use serde::{Deserialize, Serialize};

use super::board::ReservoirInput;
use super::shell::{GenerationRecord, InstanceId, NautilusShell, ShellConfig};

const ANOMALY_WINDOW: usize = 10;

/// GPU memory pressure estimate (bytes) for common algorithms.
///
/// Absorbed from hotSpring `MdBrain`. Helps the brain predict whether a
/// workload will fit in GPU VRAM and whether to recommend smaller batch
/// sizes, cell-list fallback, or CPU offload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AlgorithmClass {
    /// O(N²) all-pairs: 2 position arrays + force array.
    AllPairs,
    /// Cell-list: positions + cell indices + cell list + forces.
    CellList,
    /// Verlet neighbour list: positions + neighbour list + forces.
    VerletList,
    /// Lattice field (SU(3)/gauge): 4 link arrays + staple + force.
    LatticeGauge,
    /// Generic: caller provides estimate directly.
    Custom,
}

/// Estimate GPU memory usage in bytes for a workload.
///
/// Returns a conservative estimate of the minimum GPU allocation
/// required for `n` particles/sites using the given algorithm class.
/// Uses f64 (8 bytes) per component, 3D positions.
#[must_use]
pub fn memory_pressure(algo: AlgorithmClass, n: usize) -> u64 {
    let n = n as u64;
    let f64_bytes: u64 = 8;
    let dims: u64 = 3;
    match algo {
        AlgorithmClass::AllPairs => {
            // 2 × position (N×3×f64) + force (N×3×f64) + params
            3 * n * dims * f64_bytes + 1024
        }
        AlgorithmClass::CellList => {
            // positions + cell indices + cell head + forces + params
            // cell count ≈ N for uniform distribution
            n * dims * f64_bytes + n * 4 + n * 4 + n * dims * f64_bytes + 4096
        }
        AlgorithmClass::VerletList => {
            // positions + neighbour list (≈100 neighbours/particle) + forces
            let max_neighbours: u64 = 100;
            n * dims * f64_bytes + n * max_neighbours * 4 + n * dims * f64_bytes + 4096
        }
        AlgorithmClass::LatticeGauge => {
            // 4 link directions × N sites × 18 f64 (SU(3) 3×3 complex)
            // + staple + force buffers
            let su3_size = 18 * f64_bytes;
            4 * n * su3_size + 2 * n * su3_size + 4096
        }
        AlgorithmClass::Custom => 0,
    }
}

/// Detect energy anomalies from sudden jumps vs running statistics.
///
/// Absorbed from hotSpring `MdBrain`. Domain-agnostic: works with any
/// scalar energy proxy (`delta_h`, `total_energy`/N, etc.).
/// Returns 1.0 if current value deviates by more than 10σ from the
/// recent window, 0.0 otherwise.
#[must_use]
pub fn force_anomaly(current: f64, recent_window: &[f64]) -> f64 {
    let take = recent_window.len().min(ANOMALY_WINDOW);
    if take < 3 {
        return 0.0;
    }
    let start = recent_window.len().saturating_sub(take);
    let slice = &recent_window[start..];
    let mean: f64 = slice.iter().sum::<f64>() / slice.len() as f64;
    let variance: f64 = slice.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / slice.len() as f64;
    let std = (variance + 1e-20).sqrt();
    let deviation = (current - mean).abs();
    if deviation > 10.0 * std { 1.0 } else { 0.0 }
}

fn obs_to_input(o: &BetaObservation) -> ReservoirInput {
    ReservoirInput::Continuous(vec![
        o.beta,
        o.plaquette,
        o.acceptance,
        o.delta_h_abs,
        o.anderson_r.unwrap_or(0.0),
    ])
}

/// Observation from an HMC trajectory used for brain training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BetaObservation {
    /// Gauge coupling β.
    pub beta: f64,
    /// Average plaquette value.
    pub plaquette: f64,
    /// Mean CG iterations per trajectory.
    pub cg_iters: f64,
    /// Metropolis acceptance rate.
    pub acceptance: f64,
    /// Mean |ΔH| (Hamiltonian drift).
    pub delta_h_abs: f64,
    /// Quenched plaquette (if precomputed).
    pub quenched_plaq: Option<f64>,
    /// Quenched plaquette variance.
    pub quenched_plaq_var: Option<f64>,
    /// Anderson acceleration residual (if available).
    pub anderson_r: Option<f64>,
    /// Minimum eigenvalue from Anderson (if available).
    pub anderson_lambda_min: Option<f64>,
}

/// Monitors effective population size drift to detect evolutionary stagnation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftMonitor {
    /// History of Nₑ×S (effective size × selection) per generation.
    pub ne_s_history: Vec<f64>,
    /// Threshold below which drift is flagged.
    pub drift_threshold: f64,
    /// Number of recent generations to check.
    pub window: usize,
}

impl Default for DriftMonitor {
    fn default() -> Self {
        Self {
            ne_s_history: Vec::new(),
            drift_threshold: 1.0,
            window: 3,
        }
    }
}

impl DriftMonitor {
    /// Record Nₑ×S for the given generation.
    pub fn record(&mut self, generation: &GenerationRecord, pop_size: usize) {
        self.ne_s_history
            .push((pop_size as f64 * generation.best_fitness) / (1.0 + generation.best_fitness));
    }
    /// Returns true if recent generations show stagnation.
    #[must_use]
    pub fn is_drifting(&self) -> bool {
        self.ne_s_history.len() >= self.window
            && self.ne_s_history[self.ne_s_history.len() - self.window..]
                .iter()
                .all(|&v| v < self.drift_threshold)
    }
}

/// Configuration for the Nautilus evolutionary optimizer brain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NautilusBrainConfig {
    /// Reservoir/shell configuration.
    pub shell_config: ShellConfig,
    /// Generations to evolve per training call.
    pub generations_per_train: usize,
    /// Minimum observations before training is allowed.
    pub min_observations: usize,
}

impl Default for NautilusBrainConfig {
    fn default() -> Self {
        Self {
            shell_config: ShellConfig::default(),
            generations_per_train: 20,
            min_observations: 5,
        }
    }
}

/// Evolutionary optimizer brain that predicts HMC observables from β.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NautilusBrain {
    /// Brain configuration.
    pub config: NautilusBrainConfig,
    /// Reservoir computing shell (evolved boards).
    pub shell: NautilusShell,
    /// Observed (β, plaquette, CG iters, acceptance) from HMC runs.
    pub observations: Vec<BetaObservation>,
    /// Recent delta-H values for anomaly detection.
    #[serde(default)]
    pub recent_delta_h: Vec<f64>,
    /// Whether training has been performed.
    pub trained: bool,
    /// Drift monitor for evolutionary stagnation.
    pub drift: DriftMonitor,
}

impl NautilusBrain {
    /// Create a new brain with the given config and instance name.
    #[must_use]
    pub fn new(config: NautilusBrainConfig, instance_name: &str) -> Self {
        let shell = NautilusShell::from_seed(
            config.shell_config.clone(),
            InstanceId(instance_name.to_string()),
            0x9a17b5,
        );
        Self {
            config: config.clone(),
            shell,
            observations: Vec::new(),
            recent_delta_h: Vec::new(),
            trained: false,
            drift: DriftMonitor::default(),
        }
    }

    /// Record an HMC observation for training.
    pub fn observe(&mut self, obs: BetaObservation) {
        self.recent_delta_h.push(obs.delta_h_abs);
        self.observations.push(obs);
    }

    /// Check if the most recent observation shows an energy anomaly.
    #[must_use]
    pub fn has_force_anomaly(&self) -> bool {
        self.observations.last().is_some_and(|obs| {
            force_anomaly(
                obs.delta_h_abs,
                &self.recent_delta_h[..self.recent_delta_h.len().saturating_sub(1)],
            ) > 0.5
        })
    }

    /// Evolve the shell and train on observations. Returns MSE if trained.
    pub fn train(&mut self) -> Option<f64> {
        if self.observations.len() < self.config.min_observations {
            return None;
        }
        let inputs: Vec<_> = self.observations.iter().map(obs_to_input).collect();
        let targets: Vec<Vec<f64>> = self
            .observations
            .iter()
            .enumerate()
            .map(|(i, o)| {
                let anomaly = if i > 0 {
                    force_anomaly(o.delta_h_abs, &self.recent_delta_h[..i])
                } else {
                    0.0
                };
                vec![o.cg_iters, o.plaquette, o.acceptance, anomaly]
            })
            .collect();
        let mut mse = 0.0;
        for _ in 0..self.config.generations_per_train {
            if let Ok(m) = self.shell.evolve_generation(&inputs, &targets) {
                mse = m;
                if let Some(rec) = self.shell.history.last() {
                    self.drift.record(rec, self.config.shell_config.pop_size);
                }
            }
        }
        self.trained = true;
        Some(mse)
    }

    /// Predict (CG iters, plaquette, acceptance) for a given β.
    #[must_use]
    pub fn predict_dynamical(
        &self,
        beta: f64,
        quenched_plaq: Option<f64>,
    ) -> Option<(f64, f64, f64)> {
        let pred = self.shell.predict(&ReservoirInput::Continuous(vec![
            beta,
            quenched_plaq.unwrap_or(0.5),
            0.5,
            0.0,
            0.0,
        ]))?;
        (pred.len() >= 3).then(|| (pred[0], pred[1], pred[2]))
    }
    /// Score candidate β values by predicted quality.
    #[must_use]
    pub fn screen_candidates(&self, betas: &[f64]) -> Vec<(f64, f64)> {
        betas
            .iter()
            .map(|&b| {
                (
                    b,
                    self.predict_dynamical(b, None)
                        .map_or(0.0, |(a, x, c)| (a * a + x * x + c * c) / 3.0),
                )
            })
            .collect()
    }

    /// Detect β values where prediction error spikes (concept boundaries).
    pub fn detect_concept_edges(&mut self) -> Vec<(f64, f64)> {
        if self.observations.len() < self.config.min_observations {
            return Vec::new();
        }
        let mut edges = Vec::new();
        for (i, obs) in self.observations.iter().enumerate() {
            let (inputs, targets): (Vec<_>, Vec<_>) = self
                .observations
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, o)| (obs_to_input(o), vec![o.cg_iters, o.plaquette, o.acceptance]))
                .unzip();
            if inputs.is_empty() {
                continue;
            }
            let mut shell = self.shell.clone();
            let _ = shell.evolve_generation(&inputs, &targets);
            if let Some(pred) = shell.predict(&obs_to_input(obs)) {
                edges.push((
                    obs.beta,
                    ((pred[0] - obs.cg_iters).powi(2)
                        + (pred[1] - obs.plaquette).powi(2)
                        + (pred[2] - obs.acceptance).powi(2))
                    .sqrt(),
                ));
            }
        }
        edges
    }
    /// Estimate GPU memory pressure for a workload.
    ///
    /// Returns `true` if the estimated allocation exceeds the device's
    /// safe limit. Use this before dispatching to decide between GPU and
    /// CPU paths, or to recommend smaller batch sizes.
    #[must_use]
    pub fn would_exceed_memory(algo: AlgorithmClass, n: usize, max_safe_bytes: u64) -> bool {
        memory_pressure(algo, n) > max_safe_bytes
    }

    /// Returns true if the shell is drifting (stagnating).
    #[must_use]
    pub fn is_drifting(&self) -> bool {
        self.drift.is_drifting()
    }
    /// Serialize brain to JSON.
    /// # Errors
    /// Returns [`Err`] if JSON serialization fails.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
    /// Deserialize brain from JSON.
    /// # Errors
    /// Returns [`Err`] if JSON parsing fails.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_obs(beta: f64, plaquette: f64, cg_iters: f64, acceptance: f64) -> BetaObservation {
        BetaObservation {
            beta,
            plaquette,
            cg_iters,
            acceptance,
            delta_h_abs: 0.01,
            quenched_plaq: None,
            quenched_plaq_var: None,
            anderson_r: None,
            anderson_lambda_min: None,
        }
    }

    #[test]
    fn test_brain_observe_and_train() {
        let config = NautilusBrainConfig {
            shell_config: ShellConfig {
                pop_size: 4,
                ..ShellConfig::default()
            },
            generations_per_train: 2,
            min_observations: 5,
        };
        let mut brain = NautilusBrain::new(config, "observe-train-test");

        for i in 0..6 {
            brain.observe(make_obs(
                (i as f64) * 0.1,
                0.5 + (i as f64) * 0.01,
                (i as f64) * 10.0,
                0.7 + (i as f64) * 0.02,
            ));
        }

        let mse = brain
            .train()
            .expect("train should return Some with enough observations");
        assert!(mse.is_finite());
        assert!(brain.trained);
    }

    #[test]
    fn test_brain_predict_after_training() {
        let config = NautilusBrainConfig {
            shell_config: ShellConfig {
                pop_size: 4,
                ..ShellConfig::default()
            },
            generations_per_train: 2,
            min_observations: 5,
        };
        let mut brain = NautilusBrain::new(config, "predict-test");

        for i in 0..6 {
            brain.observe(make_obs((i as f64) * 0.1, 0.5, (i as f64) * 5.0, 0.8));
        }
        brain.train().unwrap();

        let pred = brain.predict_dynamical(0.5, None);
        let (a, x, c) = pred.expect("predict should return Some after training");
        assert!(a.is_finite());
        assert!(x.is_finite());
        assert!(c.is_finite());
    }

    #[test]
    fn test_brain_serialization_roundtrip() {
        let config = NautilusBrainConfig {
            shell_config: ShellConfig {
                pop_size: 4,
                ..ShellConfig::default()
            },
            generations_per_train: 2,
            min_observations: 5,
        };
        let mut brain = NautilusBrain::new(config, "serialization-test");

        for i in 0..6 {
            brain.observe(make_obs((i as f64) * 0.1, 0.5, (i as f64) * 3.0, 0.75));
        }
        brain.train().unwrap();

        let json = brain.to_json().expect("to_json should succeed");
        let restored = NautilusBrain::from_json(&json).expect("from_json should succeed");

        assert_eq!(restored.observations.len(), brain.observations.len());
        assert_eq!(restored.trained, brain.trained);
        assert_eq!(
            restored.shell.population.boards.len(),
            brain.shell.population.boards.len()
        );
    }

    #[test]
    fn test_force_anomaly_no_history() {
        assert!((force_anomaly(1.0, &[]) - 0.0).abs() < f64::EPSILON);
        assert!((force_anomaly(1.0, &[1.0, 2.0]) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_force_anomaly_normal() {
        let window = vec![1.0, 1.1, 0.9, 1.05, 0.95];
        assert!((force_anomaly(1.0, &window) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_force_anomaly_spike() {
        let window = vec![1.0, 1.01, 0.99, 1.005, 0.995];
        assert!((force_anomaly(100.0, &window) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_brain_anomaly_tracking() {
        let config = NautilusBrainConfig {
            shell_config: ShellConfig::default(),
            generations_per_train: 2,
            min_observations: 5,
        };
        let mut brain = NautilusBrain::new(config, "anomaly-test");
        for i in 0..6 {
            brain.observe(make_obs(
                (i as f64) * 0.1,
                0.5 + (i as f64) * 0.01,
                (i as f64) * 10.0,
                0.8,
            ));
        }
        assert_eq!(brain.recent_delta_h.len(), 6);
    }

    #[test]
    fn test_memory_pressure_all_pairs() {
        let bytes = memory_pressure(AlgorithmClass::AllPairs, 10_000);
        assert!(bytes > 0);
        assert!(bytes < 10 * 1024 * 1024, "10K particles should be < 10 MB");
    }

    #[test]
    fn test_memory_pressure_scaling() {
        let small = memory_pressure(AlgorithmClass::AllPairs, 1_000);
        let large = memory_pressure(AlgorithmClass::AllPairs, 100_000);
        assert!(large > small * 50, "memory should scale roughly linearly");
    }

    #[test]
    fn test_memory_pressure_lattice_gauge() {
        let bytes = memory_pressure(AlgorithmClass::LatticeGauge, 32 * 32 * 32 * 8);
        assert!(bytes > 100 * 1024 * 1024, "32³×8 lattice needs > 100 MB");
    }

    #[test]
    fn test_would_exceed_memory() {
        let limit = 1024 * 1024 * 1024; // 1 GB
        assert!(!NautilusBrain::would_exceed_memory(
            AlgorithmClass::AllPairs,
            10_000,
            limit
        ));
        assert!(NautilusBrain::would_exceed_memory(
            AlgorithmClass::LatticeGauge,
            64 * 64 * 64 * 16,
            limit
        ));
    }

    #[test]
    fn test_brain_drift_detection() {
        let mut drift = DriftMonitor {
            ne_s_history: Vec::new(),
            drift_threshold: 1.0,
            window: 3,
        };

        assert!(!drift.is_drifting());

        for _ in 0..3 {
            drift.record(
                &GenerationRecord {
                    generation: 0,
                    mean_fitness: 0.01,
                    best_fitness: 0.01,
                    pop_size: 16,
                    origin: InstanceId("test".to_string()),
                    training_size: 5,
                },
                16,
            );
        }
        assert!(drift.is_drifting());
    }
}
