// SPDX-License-Identifier: AGPL-3.0-or-later
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
#![expect(
    clippy::unused_async,
    reason = "tarpc trait impl requires async signatures the trait defines"
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

/// Capability-based self-discovery.
///
/// Derives capabilities and provides from the IPC dispatch table. Peer primals
/// are discovered at runtime via discovery files and capability scanning, not
/// hardcoded names.
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

/// Capability domain stem per `PRIMAL_SELF_KNOWLEDGE_STANDARD.md` §3.
///
/// Used for domain-based socket naming (`math.sock` / `math-{fid}.sock`)
/// and the `domain` field in `identity.get` responses.
pub const PRIMAL_DOMAIN: &str = "math";

use error::BarracudaCoreError;
use health::{HealthReport, HealthStatus, PrimalHealth};
use lifecycle::{PrimalLifecycle, PrimalState};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// barraCuda primal — sovereign GPU compute engine.
///
/// Manages GPU device discovery, shader compilation, and compute dispatch
/// across any vendor's hardware. Holds a tensor store for IPC-created
/// tensors that persist across requests.
///
/// Uses `RwLock` for the tensor store: reads (dispatch, lookup) are
/// concurrent; writes (store) take exclusive access. This matches the
/// read-heavy IPC workload where many handlers read tensors concurrently.
pub struct BarraCudaPrimal {
    state: PrimalState,
    device: Option<barracuda::device::WgpuDevice>,
    tensors: RwLock<HashMap<String, Arc<barracuda::tensor::Tensor>>>,
}

impl BarraCudaPrimal {
    /// Create a new barraCuda primal instance.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: PrimalState::Created,
            device: None,
            tensors: RwLock::new(HashMap::new()),
        }
    }

    /// Access the compute device (available after `start()`).
    /// Returns a clone for sharing across handlers; `WgpuDevice` is cheap to clone.
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
        if let Ok(mut store) = self.tensors.write() {
            store.insert(id.clone(), tensor_arc);
        }
        id
    }

    /// Look up a tensor by ID.
    pub fn get_tensor(&self, id: &str) -> Option<Arc<barracuda::tensor::Tensor>> {
        self.tensors.read().ok()?.get(id).cloned()
    }

    /// Number of tensors currently stored.
    pub fn tensor_count(&self) -> usize {
        self.tensors.read().map(|s| s.len()).unwrap_or(0)
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

    #[test]
    fn test_default_impl() {
        let primal = BarraCudaPrimal::default();
        assert_eq!(primal.state(), PrimalState::Created);
        assert!(primal.device().is_none());
        assert_eq!(primal.tensor_count(), 0);
    }

    #[test]
    fn test_health_status_before_start() {
        let primal = BarraCudaPrimal::new();
        let status = primal.health_status();
        assert_eq!(status, HealthStatus::Unknown);
        assert!(!primal.is_ready());
    }

    #[tokio::test]
    async fn test_health_status_running_no_gpu() {
        let mut primal = BarraCudaPrimal::new();
        primal.start().await.unwrap();
        if primal.device().is_none() {
            let status = primal.health_status();
            assert!(
                matches!(status, HealthStatus::Degraded { .. }),
                "running without GPU should be degraded"
            );
            assert!(status.is_serving());
            assert!(primal.is_ready());
        }
    }

    #[tokio::test]
    async fn test_health_status_after_stop() {
        let mut primal = BarraCudaPrimal::new();
        primal.start().await.unwrap();
        primal.stop().await.unwrap();
        assert_eq!(primal.health_status(), HealthStatus::Unknown);
        assert!(!primal.is_ready());
    }

    #[tokio::test]
    async fn test_health_check_report_fields() {
        let mut primal = BarraCudaPrimal::new();
        primal.start().await.unwrap();
        let report = primal.health_check().await.unwrap();
        assert_eq!(report.name, PRIMAL_NAME);
        assert_eq!(report.version, env!("CARGO_PKG_VERSION"));
        if primal.device().is_some() {
            assert!(report.details.contains_key("adapter"));
            assert!(report.details.contains_key("device_type"));
        }
    }

    #[test]
    fn test_tensor_store_empty() {
        let primal = BarraCudaPrimal::new();
        assert_eq!(primal.tensor_count(), 0);
        assert!(primal.get_tensor("nonexistent").is_none());
    }

    #[tokio::test]
    async fn test_cannot_start_while_running() {
        let mut primal = BarraCudaPrimal::new();
        primal.start().await.unwrap();
        let err = primal.start().await;
        assert!(err.is_err(), "should not start while already running");
    }

    #[test]
    fn test_cannot_stop_before_start() {
        let mut primal = BarraCudaPrimal::new();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let err = rt.block_on(primal.stop());
        assert!(err.is_err(), "should not stop before starting");
    }

    #[tokio::test]
    async fn test_restart_after_stop() {
        let mut primal = BarraCudaPrimal::new();
        primal.start().await.unwrap();
        primal.stop().await.unwrap();
        primal.start().await.unwrap();
        assert_eq!(primal.state(), PrimalState::Running);
    }

    #[tokio::test]
    async fn test_reload() {
        let mut primal = BarraCudaPrimal::new();
        primal.start().await.unwrap();
        primal.reload().await.unwrap();
        assert_eq!(primal.state(), PrimalState::Running);
    }

    #[tokio::test]
    async fn test_shutdown() {
        let mut primal = BarraCudaPrimal::new();
        primal.start().await.unwrap();
        primal.shutdown().await.unwrap();
        assert_eq!(primal.state(), PrimalState::Stopped);
    }

    #[test]
    fn test_primal_constants() {
        assert_eq!(PRIMAL_NAME, "barraCuda");
        assert_eq!(PRIMAL_NAMESPACE, "barracuda");
        assert_eq!(PRIMAL_DOMAIN, "math");
    }
}
