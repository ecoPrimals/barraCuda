// SPDX-License-Identifier: AGPL-3.0-or-later
//! High-level NautilusBrain API for physics observables.

use serde::{Deserialize, Serialize};

use super::board::ReservoirInput;
use super::shell::{GenerationRecord, InstanceId, NautilusShell, ShellConfig};

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
    /// Whether training has been performed.
    pub trained: bool,
    /// Drift monitor for evolutionary stagnation.
    pub drift: DriftMonitor,
}

impl NautilusBrain {
    /// Create a new brain with the given config and instance name.
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
            trained: false,
            drift: DriftMonitor::default(),
        }
    }

    /// Record an HMC observation for training.
    pub fn observe(&mut self, obs: BetaObservation) {
        self.observations.push(obs);
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
            .map(|o| vec![o.cg_iters, o.plaquette, o.acceptance])
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
    pub fn screen_candidates(&self, betas: &[f64]) -> Vec<(f64, f64)> {
        betas
            .iter()
            .map(|&b| {
                (
                    b,
                    self.predict_dynamical(b, None)
                        .map(|(a, x, c)| (a * a + x * x + c * c) / 3.0)
                        .unwrap_or(0.0),
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
    /// Returns true if the shell is drifting (stagnating).
    pub fn is_drifting(&self) -> bool {
        self.drift.is_drifting()
    }
    /// Serialize brain to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
    /// Deserialize brain from JSON.
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
            shell_config: ShellConfig::default(),
            generations_per_train: 5,
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
            shell_config: ShellConfig::default(),
            generations_per_train: 3,
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
            shell_config: ShellConfig::default(),
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
