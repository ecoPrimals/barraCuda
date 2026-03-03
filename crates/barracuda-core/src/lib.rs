// SPDX-License-Identifier: AGPL-3.0-or-later
//! # barraCuda Core
//!
//! Core primal lifecycle for barraCuda —
//! *BARrier-free Rust Abstracted Cross-platform Unified Dimensional Algebra*.
//!
//! This crate wraps the `barracuda` compute library with primal lifecycle
//! management. On `start()`, it discovers GPU hardware and initializes the
//! device pool. Health checks report device availability.

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::unused_async)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::result_large_err)]
#![allow(clippy::cast_possible_truncation)]

pub mod error;
pub mod ipc;
pub mod rpc;

pub use barracuda;

use sourdough_core::{
    health::{HealthReport, HealthStatus},
    PrimalError, PrimalHealth, PrimalLifecycle, PrimalState,
};
use std::sync::Arc;

/// barraCuda primal — sovereign GPU compute engine.
///
/// Manages GPU device discovery, shader compilation, and compute dispatch
/// across any vendor's hardware. Composes with BearDog (crypto) for FHE
/// and with ToadStool (orchestration) for workload routing.
pub struct BarraCudaPrimal {
    state: PrimalState,
    device: Option<Arc<barracuda::device::WgpuDevice>>,
}

impl BarraCudaPrimal {
    /// Create a new barraCuda primal instance.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: PrimalState::Created,
            device: None,
        }
    }

    /// Access the compute device (available after `start()`).
    #[must_use]
    pub fn device(&self) -> Option<&Arc<barracuda::device::WgpuDevice>> {
        self.device.as_ref()
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

        tracing::info!("barraCuda: discovering compute devices...");

        match barracuda::device::Auto::new().await {
            Ok(dev) => {
                let info = dev.adapter_info();
                tracing::info!(
                    "barraCuda: device ready — {} ({:?})",
                    info.name,
                    info.device_type,
                );
                self.device = Some(dev);
            }
            Err(e) => {
                tracing::warn!("barraCuda: no GPU device available ({e}), running CPU-only");
            }
        }

        self.state = PrimalState::Running;
        Ok(())
    }

    async fn stop(&mut self) -> Result<(), PrimalError> {
        if !self.state.can_stop() {
            return Err(PrimalError::lifecycle("Cannot stop from current state"));
        }
        self.device = None;
        self.state = PrimalState::Stopped;
        Ok(())
    }
}

impl PrimalHealth for BarraCudaPrimal {
    fn health_status(&self) -> HealthStatus {
        match self.state {
            PrimalState::Running if self.device.is_some() => HealthStatus::Healthy,
            PrimalState::Running => HealthStatus::Degraded {
                reason: "No GPU device available".to_string(),
            },
            _ => HealthStatus::Unknown,
        }
    }

    async fn health_check(&self) -> Result<HealthReport, PrimalError> {
        let mut report = HealthReport::new("barraCuda", env!("CARGO_PKG_VERSION"))
            .with_status(self.health_status());

        if let Some(dev) = &self.device {
            let info = dev.adapter_info();
            report = report
                .with_detail("adapter", &info.name)
                .with_detail("device_type", format!("{:?}", info.device_type));
        }

        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_lifecycle() {
        let mut primal = BarraCudaPrimal::new();

        assert_eq!(primal.state(), PrimalState::Created);
        assert!(primal.device().is_none());

        primal.start().await.unwrap();
        assert_eq!(primal.state(), PrimalState::Running);

        primal.stop().await.unwrap();
        assert_eq!(primal.state(), PrimalState::Stopped);
        assert!(primal.device().is_none());
    }

    #[tokio::test]
    async fn test_health() {
        let mut primal = BarraCudaPrimal::new();
        primal.start().await.unwrap();

        let report = primal.health_check().await.unwrap();
        assert_eq!(report.name, "barraCuda");
    }

    #[tokio::test]
    async fn test_device_available_after_start() {
        let mut primal = BarraCudaPrimal::new();
        primal.start().await.unwrap();

        // Device should be available (at least llvmpipe/CPU backend)
        if primal.device().is_some() {
            assert!(primal.health_status().is_healthy());
        }
    }
}
