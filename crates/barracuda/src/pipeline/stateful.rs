// SPDX-License-Identifier: AGPL-3.0-or-later
//! Stateful pipeline — day-over-day state tracking for water balance (airSpring V039).
//!
//! Soil moisture from day N becomes input for day N+1.

/// A stage that executes with mutable state and returns output.
pub trait PipelineStage<S> {
    /// Run this stage; may read and mutate `state`, consume `input`, return output.
    fn execute(&self, state: &mut S, input: &[f64]) -> Vec<f64>;
}

/// Pipeline that carries state between invocations.
pub struct StatefulPipeline<S: Default + Clone> {
    /// Mutable state passed to each stage.
    pub state: S,
    /// Ordered stages executed in sequence.
    pub stages: Vec<Box<dyn PipelineStage<S>>>,
}

impl<S: Default + Clone> Default for StatefulPipeline<S> {
    fn default() -> Self {
        Self {
            state: S::default(),
            stages: Vec::new(),
        }
    }
}

impl<S: Default + Clone> StatefulPipeline<S> {
    /// Create a new empty pipeline with default state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a stage to the pipeline.
    pub fn add_stage(&mut self, stage: Box<dyn PipelineStage<S>>) {
        self.stages.push(stage);
    }

    /// Run all stages; state is updated in place, output flows between stages.
    pub fn run(&mut self, input: &[f64]) -> Vec<f64> {
        let mut data = input.to_vec();
        for stage in &self.stages {
            data = stage.execute(&mut self.state, &data);
        }
        data
    }
}

/// Water balance state for day-over-day accumulation.
#[derive(Debug, Clone, Default)]
pub struct WaterBalanceState {
    /// Soil moisture content (mm or equivalent).
    pub soil_moisture: f64,
    /// Snow water equivalent (mm).
    pub snow_water_eq: f64,
    /// Deep percolation to groundwater (mm).
    pub deep_percolation: f64,
}

impl WaterBalanceState {
    /// Create water balance state with given values.
    #[must_use]
    pub fn new(soil_moisture: f64, snow_water_eq: f64, deep_percolation: f64) -> Self {
        Self {
            soil_moisture,
            snow_water_eq,
            deep_percolation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DoubleStage;
    impl PipelineStage<f64> for DoubleStage {
        fn execute(&self, _state: &mut f64, input: &[f64]) -> Vec<f64> {
            input.iter().map(|x| x * 2.0).collect()
        }
    }

    struct AccumulateStage;
    impl PipelineStage<f64> for AccumulateStage {
        fn execute(&self, state: &mut f64, input: &[f64]) -> Vec<f64> {
            let sum: f64 = input.iter().sum();
            *state += sum;
            vec![*state]
        }
    }

    #[test]
    fn empty_pipeline_passthrough() {
        let mut pipeline: StatefulPipeline<f64> = StatefulPipeline::new();
        let result = pipeline.run(&[1.0, 2.0, 3.0]);
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn single_stage() {
        let mut pipeline: StatefulPipeline<f64> = StatefulPipeline::new();
        pipeline.add_stage(Box::new(DoubleStage));
        let result = pipeline.run(&[1.0, 2.0]);
        assert_eq!(result, vec![2.0, 4.0]);
    }

    #[test]
    fn chained_stages() {
        let mut pipeline: StatefulPipeline<f64> = StatefulPipeline::new();
        pipeline.add_stage(Box::new(DoubleStage));
        pipeline.add_stage(Box::new(DoubleStage));
        let result = pipeline.run(&[1.0]);
        assert_eq!(result, vec![4.0]);
    }

    #[test]
    fn state_persists_across_runs() {
        let mut pipeline: StatefulPipeline<f64> = StatefulPipeline::new();
        pipeline.add_stage(Box::new(AccumulateStage));

        let r1 = pipeline.run(&[1.0, 2.0, 3.0]);
        assert_eq!(r1, vec![6.0]);
        assert!((pipeline.state - 6.0).abs() < 1e-15);

        let r2 = pipeline.run(&[4.0]);
        assert_eq!(r2, vec![10.0]);
        assert!((pipeline.state - 10.0).abs() < 1e-15);
    }

    #[test]
    fn water_balance_state_defaults() {
        let state = WaterBalanceState::default();
        assert_eq!(state.soil_moisture, 0.0);
        assert_eq!(state.snow_water_eq, 0.0);
        assert_eq!(state.deep_percolation, 0.0);
    }

    #[test]
    fn water_balance_state_constructor() {
        let state = WaterBalanceState::new(100.0, 50.0, 25.0);
        assert_eq!(state.soil_moisture, 100.0);
        assert_eq!(state.snow_water_eq, 50.0);
        assert_eq!(state.deep_percolation, 25.0);
    }
}
