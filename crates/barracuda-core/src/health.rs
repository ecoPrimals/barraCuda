// SPDX-License-Identifier: AGPL-3.0-only
//! Health check types and trait for observability.
//!
//! Modeled on the ecoPrimals pattern (sourDough scaffold), owned by barraCuda.
//! Provides health reporting for orchestrators, load balancers, and monitoring.

use std::collections::HashMap;
use std::fmt;

/// Overall health status.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HealthStatus {
    /// Healthy and ready to serve requests.
    Healthy,
    /// Running but degraded (e.g. no GPU, CPU-only mode).
    Degraded {
        /// Reason for degraded status.
        reason: String,
    },
    /// Unhealthy and not serving requests.
    Unhealthy {
        /// Reason for unhealthy status.
        reason: String,
    },
    /// Health unknown (e.g. startup in progress).
    Unknown,
}

impl HealthStatus {
    /// Check if the status is healthy.
    #[must_use]
    pub const fn is_healthy(&self) -> bool {
        matches!(self, Self::Healthy)
    }

    /// Check if the status allows serving requests.
    #[must_use]
    pub const fn is_serving(&self) -> bool {
        matches!(self, Self::Healthy | Self::Degraded { .. })
    }
}

impl fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Healthy => write!(f, "healthy"),
            Self::Degraded { reason } => write!(f, "degraded: {reason}"),
            Self::Unhealthy { reason } => write!(f, "unhealthy: {reason}"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

/// Full health report for the barraCuda primal.
#[derive(Clone, Debug)]
pub struct HealthReport {
    /// Primal name.
    pub name: String,
    /// Primal version.
    pub version: String,
    /// Overall status.
    pub status: HealthStatus,
    /// Additional details (adapter name, device type, etc.).
    pub details: HashMap<String, String>,
}

impl HealthReport {
    /// Create a new health report.
    #[must_use]
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            status: HealthStatus::Unknown,
            details: HashMap::new(),
        }
    }

    /// Set status.
    #[must_use]
    pub fn with_status(mut self, status: HealthStatus) -> Self {
        self.status = status;
        self
    }

    /// Add a detail key-value pair.
    #[must_use]
    pub fn with_detail(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.details.insert(key.into(), value.into());
        self
    }
}

/// Health check trait for the barraCuda primal.
///
/// Modeled on the ecoPrimals health pattern.
pub trait PrimalHealth: Send + Sync {
    /// Get the current health status (cheap, pollable).
    fn health_status(&self) -> HealthStatus;

    /// Perform a full health check (may be expensive).
    ///
    /// # Errors
    ///
    /// Returns an error if the health check itself fails.
    fn health_check(
        &self,
    ) -> impl std::future::Future<Output = Result<HealthReport, crate::error::BarracudaCoreError>> + Send;

    /// Check readiness (can it serve requests?).
    fn is_ready(&self) -> bool {
        self.health_status().is_serving()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn health_status_healthy() {
        let status = HealthStatus::Healthy;
        assert!(status.is_healthy());
        assert!(status.is_serving());
        assert_eq!(status.to_string(), "healthy");
    }

    #[test]
    fn health_status_degraded() {
        let status = HealthStatus::Degraded {
            reason: "no GPU".to_string(),
        };
        assert!(!status.is_healthy());
        assert!(status.is_serving());
        assert_eq!(status.to_string(), "degraded: no GPU");
    }

    #[test]
    fn health_status_unhealthy() {
        let status = HealthStatus::Unhealthy {
            reason: "device lost".to_string(),
        };
        assert!(!status.is_healthy());
        assert!(!status.is_serving());
    }

    #[test]
    fn health_report_builder() {
        let report = HealthReport::new("barraCuda", "0.3.5")
            .with_status(HealthStatus::Healthy)
            .with_detail("adapter", "llvmpipe");

        assert_eq!(report.name, "barraCuda");
        assert!(report.status.is_healthy());
        assert_eq!(report.details.get("adapter"), Some(&"llvmpipe".to_string()));
    }
}
