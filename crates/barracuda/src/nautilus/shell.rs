// SPDX-License-Identifier: AGPL-3.0-only
//! Nautilus shell with layered evolutionary history.

use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use super::board::{BoardConfig, ReservoirInput};
use super::evolution::{EvolutionConfig, evolve};
use super::population::Population;
use super::readout::LinearReadout;

/// Machine identifier for lineage tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceId(pub String);

/// Record for one generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationRecord {
    /// Generation number
    pub generation: usize,
    /// Mean population fitness
    pub mean_fitness: f64,
    /// Best individual fitness
    pub best_fitness: f64,
    /// Population size
    pub pop_size: usize,
    /// Origin instance ID
    pub origin: InstanceId,
    /// Training set size
    pub training_size: usize,
}

/// Shell configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShellConfig {
    /// Board configuration.
    pub board_config: BoardConfig,
    /// Population size
    pub pop_size: usize,
    /// Evolution parameters
    pub evolution: EvolutionConfig,
    /// Ridge regression regularization
    pub lambda: f64,
    /// Output dimension override. When `Some(n)`, the readout layer is
    /// constructed with `n` outputs instead of the default 3. Springs with
    /// extended target vectors (e.g. 4-head anomaly signal) set this to
    /// match their target dimensionality.
    #[serde(default)]
    pub output_dim: Option<usize>,
}

impl Default for ShellConfig {
    fn default() -> Self {
        Self {
            board_config: BoardConfig::default(),
            pop_size: 16,
            evolution: EvolutionConfig::default(),
            lambda: 1e-3,
            output_dim: None,
        }
    }
}

impl ShellConfig {
    /// Set the output dimension for the readout layer.
    ///
    /// By default the readout has 3 outputs (CG iters, plaquette, acceptance).
    /// Use this when training targets have a different width (e.g. 4 outputs
    /// when including the anomaly signal).
    #[must_use]
    pub fn with_output_dim(mut self, n: usize) -> Self {
        self.output_dim = Some(n);
        self
    }
}

/// Nautilus shell: population + readout + history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NautilusShell {
    /// Shell configuration
    pub config: ShellConfig,
    /// Reservoir population
    pub population: Population,
    /// Linear readout layer
    pub readout: LinearReadout,
    /// Generation history
    pub history: Vec<GenerationRecord>,
    /// Current origin instance
    pub origin: InstanceId,
    /// Lineage chain
    pub lineage: Vec<InstanceId>,
}

impl NautilusShell {
    /// Create from seed with random population.
    #[must_use]
    pub fn from_seed(config: ShellConfig, origin: InstanceId, seed: u64) -> Self {
        let pop = Population::new(&config.board_config, config.pop_size, seed);
        let l2 = config.board_config.grid_size * config.board_config.grid_size;
        let input_dim = config.pop_size * l2;
        let n_out = config.output_dim.unwrap_or(3);
        let readout = LinearReadout::new(input_dim, n_out, config.lambda);
        Self {
            config,
            population: pop,
            readout,
            history: Vec::new(),
            origin: origin.clone(),
            lineage: vec![origin],
        }
    }

    /// Evolve one generation: train readout, evaluate fitness, breed. Returns MSE.
    /// # Errors
    /// Returns [`Err`] if readout training fails or fitness evaluation fails.
    pub fn evolve_generation(
        &mut self,
        inputs: &[ReservoirInput],
        targets: &[Vec<f64>],
    ) -> crate::error::Result<f64> {
        let l2 = self.config.board_config.grid_size * self.config.board_config.grid_size;
        let input_dim = self.config.pop_size * l2;

        let responses: Vec<Vec<f64>> = inputs
            .iter()
            .map(|inp| self.population.respond_all(inp))
            .collect();

        let n_out = targets.first().map_or(3, std::vec::Vec::len);
        self.readout = LinearReadout::new(input_dim, n_out, self.config.lambda);
        let mse = self.readout.train(&responses, targets)?;

        self.population.evaluate_fitness(inputs, targets);
        let fitness: Vec<f64> = self
            .population
            .fitness
            .iter()
            .map(|f| f.pearson_r)
            .collect();
        let mean_fitness = fitness.iter().sum::<f64>() / fitness.len() as f64;
        let best_fitness = fitness.iter().copied().fold(0.0f64, f64::max);

        let mut rng =
            StdRng::seed_from_u64(self.population.generation as u64 ^ 0x9e37_79b9_7f4a_7c15);
        let next_boards = evolve(
            &self.population.boards,
            &fitness,
            &self.config.evolution,
            self.config.pop_size,
            &mut rng,
        );
        self.population.boards = next_boards;
        self.population.generation += 1;

        self.history.push(GenerationRecord {
            generation: self.population.generation,
            mean_fitness,
            best_fitness,
            pop_size: self.config.pop_size,
            origin: self.origin.clone(),
            training_size: inputs.len(),
        });

        Ok(mse)
    }

    /// Predict through population then readout.
    #[must_use]
    pub fn predict(&self, input: &ReservoirInput) -> Option<Vec<f64>> {
        let response = self.population.respond_all(input);
        self.readout.predict(&response)
    }

    /// Continue from another shell, inheriting and adding to lineage.
    #[must_use]
    pub fn continue_from(shell: Self, new_origin: InstanceId) -> Self {
        let mut lineage = shell.lineage.clone();
        lineage.push(new_origin.clone());
        Self {
            lineage,
            origin: new_origin,
            ..shell
        }
    }

    /// Merge best boards from another shell into this population.
    pub fn merge_shell(&mut self, other: &NautilusShell) {
        let self_fit: Vec<f64> = self
            .population
            .fitness
            .iter()
            .map(|f| f.pearson_r)
            .collect();
        let other_fit: Vec<f64> = other
            .population
            .fitness
            .iter()
            .map(|f| f.pearson_r)
            .collect();
        let mut all: Vec<_> = self
            .population
            .boards
            .iter()
            .enumerate()
            .map(|(i, b)| (b.clone(), self_fit.get(i).copied().unwrap_or(0.0)))
            .collect();
        all.extend(
            other
                .population
                .boards
                .iter()
                .enumerate()
                .map(|(i, b)| (b.clone(), other_fit.get(i).copied().unwrap_or(0.0))),
        );
        all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        self.population.boards = all
            .into_iter()
            .take(self.config.pop_size)
            .map(|(b, _)| b)
            .collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal config for fast unit tests.
    /// Tests validate mechanics (generation counter, MSE finiteness, etc.),
    /// not convergence — that's the springs' responsibility.
    fn test_config() -> ShellConfig {
        ShellConfig {
            board_config: BoardConfig {
                grid_size: 2,
                range_per_column: 4,
                seed: 0,
            },
            pop_size: 4,
            evolution: EvolutionConfig::default(),
            lambda: 1e-3,
            output_dim: None,
        }
    }

    fn test_inputs() -> Vec<ReservoirInput> {
        (0..3)
            .map(|i| ReservoirInput::Continuous(vec![(i as f64) * 0.2, 0.5]))
            .collect()
    }

    fn test_targets() -> Vec<Vec<f64>> {
        vec![vec![1.0, 0.5], vec![2.0, 0.6], vec![1.5, 0.55]]
    }

    #[test]
    fn test_shell_creation() {
        let config = test_config();
        let shell =
            NautilusShell::from_seed(config.clone(), InstanceId("test-instance".to_string()), 42);
        assert_eq!(shell.population.boards.len(), config.pop_size);
        assert_eq!(shell.population.generation, 0);
        assert_eq!(shell.lineage.len(), 1);
        assert_eq!(shell.lineage[0].0, "test-instance");
        assert_eq!(shell.origin.0, "test-instance");
    }

    #[test]
    fn test_shell_evolve_generation() {
        let config = test_config();
        let mut shell =
            NautilusShell::from_seed(config, InstanceId("evolve-test".to_string()), 123);
        let inputs = test_inputs();
        let targets = test_targets();

        let mse1 = shell.evolve_generation(&inputs, &targets).unwrap();
        assert!(mse1.is_finite());
        assert_eq!(shell.population.generation, 1);
        assert_eq!(shell.history.len(), 1);

        let mse2 = shell.evolve_generation(&inputs, &targets).unwrap();
        assert!(mse2.is_finite());
        assert_eq!(shell.population.generation, 2);
    }

    #[test]
    fn test_shell_predict() {
        let config = test_config();
        let mut shell =
            NautilusShell::from_seed(config, InstanceId("predict-test".to_string()), 456);
        let inputs = test_inputs();
        let targets = test_targets();
        let _ = shell.evolve_generation(&inputs, &targets).unwrap();

        let pred = shell.predict(&ReservoirInput::Continuous(vec![0.5, 0.5]));
        let pred = pred.expect("predict should return Some after training");
        assert_eq!(pred.len(), 2);
    }

    #[test]
    fn test_shell_continue_from() {
        let config = test_config();
        let shell = NautilusShell::from_seed(config, InstanceId("original".to_string()), 789);
        let continued = NautilusShell::continue_from(shell, InstanceId("continued".to_string()));

        assert_eq!(continued.lineage.len(), 2);
        assert_eq!(continued.lineage[0].0, "original");
        assert_eq!(continued.lineage[1].0, "continued");
        assert_eq!(continued.origin.0, "continued");
    }

    #[test]
    fn test_shell_merge() {
        let config = test_config();
        let mut shell1 =
            NautilusShell::from_seed(config.clone(), InstanceId("shell1".to_string()), 111);
        let shell2 =
            NautilusShell::from_seed(config.clone(), InstanceId("shell2".to_string()), 222);
        let inputs = test_inputs();
        let targets = test_targets();

        let _ = shell1.evolve_generation(&inputs, &targets).unwrap();
        let mut shell2_mut = shell2;
        let _ = shell2_mut.evolve_generation(&inputs, &targets).unwrap();

        shell1.merge_shell(&shell2_mut);
        assert_eq!(shell1.population.boards.len(), config.pop_size);
    }
}
