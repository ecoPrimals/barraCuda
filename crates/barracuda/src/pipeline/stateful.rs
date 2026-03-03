//! Stateful pipeline — day-over-day state tracking for water balance (airSpring V039).
//!
//! Soil moisture from day N becomes input for day N+1.

/// A stage that executes with mutable state and returns output.
pub trait PipelineStage<S> {
    fn execute(&self, state: &mut S, input: &[f64]) -> Vec<f64>;
}

/// Pipeline that carries state between invocations.
pub struct StatefulPipeline<S: Default + Clone> {
    pub state: S,
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
    pub fn new() -> Self {
        Self::default()
    }

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
    pub soil_moisture: f64,
    pub snow_water_eq: f64,
    pub deep_percolation: f64,
}

impl WaterBalanceState {
    pub fn new(soil_moisture: f64, snow_water_eq: f64, deep_percolation: f64) -> Self {
        Self {
            soil_moisture,
            snow_water_eq,
            deep_percolation,
        }
    }
}
