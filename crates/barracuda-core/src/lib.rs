// SPDX-License-Identifier: AGPL-3.0-only
//! # barraCuda Core
//!
//! Core primal lifecycle for barraCuda —
//! *BARrier-free Rust Abstracted Cross-platform Unified Dimensional Algebra*.
//!
//! This crate wraps the `barracuda` compute library with primal lifecycle
//! management. On `start()`, it discovers GPU hardware and initializes the
//! device pool. Health checks report device availability.
//!
//! Lifecycle and health types are owned by barraCuda, modeled on the
//! ecoPrimals pattern (sourDough scaffold). Mature primals internalize
//! their lifecycle — no runtime dependency on sourDough.

#![forbid(unsafe_code)]
#![cfg_attr(
    test,
    expect(clippy::unwrap_used, reason = "test code uses unwrap for brevity")
)]
#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![expect(
    clippy::doc_markdown,
    reason = "domain terms like barraCuda, wgpu, WebGPU"
)]
#![expect(
    clippy::module_name_repetitions,
    reason = "primal module names are intentionally descriptive"
)]
#![expect(
    clippy::unused_async,
    reason = "PrimalLifecycle trait requires async for consistency"
)]
#![expect(
    clippy::must_use_candidate,
    reason = "IPC handlers return values consumed by framework"
)]
#![expect(
    clippy::missing_errors_doc,
    reason = "error types are self-documenting via thiserror"
)]
#![expect(
    clippy::redundant_closure_for_method_calls,
    reason = "clarity in IPC handler chains"
)]
#![expect(
    clippy::result_large_err,
    reason = "BarracudaCoreError carries diagnostic context"
)]
#![expect(
    clippy::cast_possible_truncation,
    reason = "tensor dimensions validated upstream"
)]

/// Runtime discovery of peer primals via mDNS and fallback scanning.
pub mod discovery;
/// Primal-specific error types and conversions.
pub mod error;
/// Health-check subsystem — liveness, readiness, and device availability.
pub mod health;
/// JSON-RPC 2.0 IPC transport and method handlers.
pub mod ipc;
/// Primal lifecycle management — start, stop, health, and graceful shutdown.
pub mod lifecycle;
/// `tarpc` binary RPC service definition and handlers.
pub mod rpc;
/// Shared request/response types for the RPC layer.
pub mod rpc_types;

pub use barracuda;

/// Canonical primal identity — single source of truth for self-knowledge.
pub const PRIMAL_NAME: &str = "barraCuda";

/// Lowercase namespace used in IPC wire protocol, filesystem paths, and CLI.
///
/// Derived from [`PRIMAL_NAME`] convention: primals use lowercase for
/// machine-facing identifiers (socket paths, JSON-RPC namespaces) and
/// camelCase for human-facing display names.
pub const PRIMAL_NAMESPACE: &str = "barracuda";

use error::BarracudaCoreError;
use health::{HealthReport, HealthStatus, PrimalHealth};
use lifecycle::{PrimalLifecycle, PrimalState};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// barraCuda primal — sovereign GPU compute engine.
///
/// Manages GPU device discovery, shader compilation, and compute dispatch
/// across any vendor's hardware. Holds a tensor store for IPC-created
/// tensors that persist across requests.
pub struct BarraCudaPrimal {
    state: PrimalState,
    device: Option<barracuda::device::WgpuDevice>,
    tensors: Mutex<HashMap<String, Arc<barracuda::tensor::Tensor>>>,
}

impl BarraCudaPrimal {
    /// Create a new barraCuda primal instance.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: PrimalState::Created,
            device: None,
            tensors: Mutex::new(HashMap::new()),
        }
    }

    /// Access the compute device (available after `start()`).
    /// Returns a clone for sharing across handlers; WgpuDevice is cheap to clone.
    #[must_use]
    pub fn device(&self) -> Option<barracuda::device::WgpuDevice> {
        self.device.clone()
    }

    /// Store a tensor and return its handle ID.
    pub fn store_tensor(&self, tensor: barracuda::tensor::Tensor) -> String {
        let tensor_arc = Arc::new(tensor);
        let id = blake3::hash(
            format!("{:p}:{}", Arc::as_ptr(&tensor_arc), self.tensor_count()).as_bytes(),
        )
        .to_hex()[..16]
            .to_string();
        if let Ok(mut store) = self.tensors.lock() {
            store.insert(id.clone(), tensor_arc);
        }
        id
    }

    /// Look up a tensor by ID.
    pub fn get_tensor(&self, id: &str) -> Option<Arc<barracuda::tensor::Tensor>> {
        self.tensors.lock().ok()?.get(id).cloned()
    }

    /// Number of tensors currently stored.
    pub fn tensor_count(&self) -> usize {
        self.tensors.lock().map(|s| s.len()).unwrap_or(0)
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

    async fn start(&mut self) -> Result<(), BarracudaCoreError> {
        if !self.state.can_start() {
            return Err(BarracudaCoreError::lifecycle(
                "Cannot start from current state",
            ));
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
                self.device = Some(dev.as_ref().clone());
            }
            Err(e) => {
                tracing::warn!("barraCuda: no GPU device available ({e}), running CPU-only");
            }
        }

        self.state = PrimalState::Running;
        Ok(())
    }

    async fn stop(&mut self) -> Result<(), BarracudaCoreError> {
        if !self.state.can_stop() {
            return Err(BarracudaCoreError::lifecycle(
                "Cannot stop from current state",
            ));
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

    async fn health_check(&self) -> Result<HealthReport, BarracudaCoreError> {
        let mut report = HealthReport::new(PRIMAL_NAME, env!("CARGO_PKG_VERSION"))
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

        if primal.device().is_some() {
            assert!(primal.health_status().is_healthy());
        }
    }
}
