// SPDX-License-Identifier: AGPL-3.0-or-later
//! Board ensembles and fitness evaluation.

use serde::{Deserialize, Serialize};

use super::board::{Board, BoardConfig, ReservoirInput};

/// Fitness record for a board.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitnessRecord {
    pub board_idx: usize,
    pub pearson_r: f64,
}

/// Population of boards with fitness tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Population {
    pub boards: Vec<Board>,
    pub generation: usize,
    pub fitness: Vec<FitnessRecord>,
}

impl Population {
    /// Create a new random population.
    pub fn new(config: &BoardConfig, size: usize, seed: u64) -> Self {
        let boards: Vec<Board> = (0..size)
            .map(|i| Board::from_seed(config.clone(), seed.wrapping_add(i as u64)))
            .collect();
        Self {
            boards,
            generation: 0,
            fitness: Vec::new(),
        }
    }

    /// Concatenated responses from all boards (pop_size × L²).
    pub fn respond_all(&self, input: &ReservoirInput) -> Vec<f64> {
        let mut out = Vec::new();
        for board in &self.boards {
            let r = board.respond(input);
            out.extend(r.activations);
        }
        out
    }

    /// Evaluate fitness: Pearson correlation between each board's response pattern and targets.
    pub fn evaluate_fitness(&mut self, inputs: &[ReservoirInput], targets: &[Vec<f64>]) {
        let l2 = self.boards[0].config.grid_size * self.boards[0].config.grid_size;
        let n_out = targets.first().map(|t| t.len()).unwrap_or(0);
        self.fitness.clear();

        for (idx, board) in self.boards.iter().enumerate() {
            let responses: Vec<Vec<f64>> = inputs
                .iter()
                .map(|inp| board.respond(inp).activations)
                .collect();
            let n_samples = responses.len().min(targets.len());

            let mut pearson_sum = 0.0;
            let mut count = 0usize;
            for o in 0..n_out {
                let t_col: Vec<f64> = (0..n_samples)
                    .map(|s| targets[s].get(o).copied().unwrap_or(0.0))
                    .collect();
                let mut best_r = 0.0f64;
                for r in 0..l2 {
                    let r_col: Vec<f64> = (0..n_samples)
                        .map(|s| responses[s].get(r).copied().unwrap_or(0.0))
                        .collect();
                    let n = r_col.len().min(t_col.len());
                    if n < 2 {
                        continue;
                    }
                    let mean_r: f64 = r_col[..n].iter().sum::<f64>() / (n as f64);
                    let mean_t: f64 = t_col[..n].iter().sum::<f64>() / (n as f64);
                    let mut ss_r = 0.0;
                    let mut ss_t = 0.0;
                    let mut sp = 0.0;
                    for i in 0..n {
                        let dr = r_col[i] - mean_r;
                        let dt = t_col[i] - mean_t;
                        ss_r += dr * dr;
                        ss_t += dt * dt;
                        sp += dr * dt;
                    }
                    let denom = (ss_r * ss_t).sqrt();
                    if denom > 1e-15 {
                        let corr = (sp / denom).abs();
                        best_r = best_r.max(corr);
                    }
                }
                pearson_sum += best_r;
                count += 1;
            }
            let pearson_r = if count > 0 {
                (pearson_sum / (count as f64)).clamp(0.0, 1.0)
            } else {
                0.0
            };
            self.fitness.push(FitnessRecord {
                board_idx: idx,
                pearson_r,
            });
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_population_creation() {
        let config = BoardConfig {
            grid_size: 5,
            range_per_column: 15,
            seed: 0,
        };
        let pop = Population::new(&config, 12, 999);
        assert_eq!(pop.boards.len(), 12);
        assert_eq!(pop.generation, 0);
        assert!(pop.fitness.is_empty());
    }

    #[test]
    fn test_respond_all_dimension() {
        let config = BoardConfig {
            grid_size: 5,
            range_per_column: 15,
            seed: 0,
        };
        let pop = Population::new(&config, 8, 111);
        let l = config.grid_size;
        let input = ReservoirInput::Continuous(vec![0.1, 0.2, 0.3, 0.4, 0.5]);
        let out = pop.respond_all(&input);
        assert_eq!(out.len(), 8 * l * l);
    }

    #[test]
    fn test_evaluate_fitness_produces_records() {
        let config = BoardConfig {
            grid_size: 5,
            range_per_column: 15,
            seed: 0,
        };
        let mut pop = Population::new(&config, 4, 222);
        let inputs: Vec<ReservoirInput> = (0..3)
            .map(|i| ReservoirInput::Continuous(vec![(i as f64) * 0.1, 0.2, 0.3, 0.4, 0.5]))
            .collect();
        let targets = vec![
            vec![1.0, 0.5, 0.8],
            vec![2.0, 0.6, 0.7],
            vec![1.5, 0.55, 0.75],
        ];
        pop.evaluate_fitness(&inputs, &targets);
        assert_eq!(pop.fitness.len(), 4);
        for (i, rec) in pop.fitness.iter().enumerate() {
            assert_eq!(rec.board_idx, i);
            assert!(rec.pearson_r >= 0.0 && rec.pearson_r <= 1.0);
        }
    }
}
