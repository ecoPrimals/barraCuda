//! # `BarraCUDA` Core
//!
//! Core device management and primal lifecycle for `BarraCUDA` —
//! **C**ross-platform **U**nified **D**ispatch **A**rithmetic.
//!
//! `BarraCUDA` is the sovereign math engine for the ecoPrimals ecosystem.
//! It provides GPU-accelerated scientific computing across any vendor's
//! hardware via WGSL shaders compiled through wgpu.
//!
//! ## Naming
//!
//! "CUDA" in `BarraCUDA` stands for **C**ross-platform **U**nified **D**ispatch
//! **A**rithmetic — not NVIDIA's "Compute Unified Device Architecture."
//! `BarraCUDA` has zero code, zero dependencies, and zero API compatibility
//! with NVIDIA CUDA.

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]

pub mod error;

use sourdough_core::{
    health::{HealthReport, HealthStatus},
    PrimalError, PrimalHealth, PrimalLifecycle, PrimalState,
};

/// `BarraCUDA` primal — sovereign GPU compute engine.
///
/// Manages GPU device discovery, shader compilation, and compute dispatch
/// across any vendor's hardware. Composes with `BearDog` (crypto) for FHE
/// and with `ToadStool` (orchestration) for workload routing.
pub struct BarraCudaPrimal {
    state: PrimalState,
}

impl BarraCudaPrimal {
    /// Create a new `BarraCUDA` primal instance.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: PrimalState::Created,
        }
    }
}

impl Default for BarraCudaPrimal {
    fn default() -> Self {
        Self::new()
    }
}

impl PrimalLifecycle for BarraCudaPrimal {
    fn state(&self) -> PrimalState {
        self.state
    }

    async fn start(&mut self) -> Result<(), PrimalError> {
        if !self.state.can_start() {
            return Err(PrimalError::lifecycle("Cannot start from current state"));
        }
        self.state = PrimalState::Running;
        Ok(())
    }

    async fn stop(&mut self) -> Result<(), PrimalError> {
        if !self.state.can_stop() {
            return Err(PrimalError::lifecycle("Cannot stop from current state"));
        }
        self.state = PrimalState::Stopped;
        Ok(())
    }
}

impl PrimalHealth for BarraCudaPrimal {
    fn health_status(&self) -> HealthStatus {
        if self.state.is_running() {
            HealthStatus::Healthy
        } else {
            HealthStatus::Unknown
        }
    }

    async fn health_check(&self) -> Result<HealthReport, PrimalError> {
        Ok(HealthReport::new("barraCuda", env!("CARGO_PKG_VERSION"))
            .with_status(self.health_status()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_lifecycle() {
        let mut primal = BarraCudaPrimal::new();

        assert_eq!(primal.state(), PrimalState::Created);

        primal.start().await.unwrap();
        assert_eq!(primal.state(), PrimalState::Running);

        primal.stop().await.unwrap();
        assert_eq!(primal.state(), PrimalState::Stopped);
    }

    #[tokio::test]
    async fn test_health() {
        let mut primal = BarraCudaPrimal::new();
        primal.start().await.unwrap();

        assert!(primal.health_status().is_healthy());

        let report = primal.health_check().await.unwrap();
        assert_eq!(report.name, "barraCuda");
    }
}
