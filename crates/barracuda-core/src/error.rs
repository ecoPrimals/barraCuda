// SPDX-License-Identifier: AGPL-3.0-only
//! barraCuda-core error types.
//!
//! Unified error handling for primal lifecycle, health, IPC, and GPU
//! device management. Replaces the sourDough scaffold `PrimalError` with
//! barraCuda-owned types.

/// Result type alias for barraCuda-core operations.
pub type Result<T> = std::result::Result<T, BarracudaCoreError>;

/// Errors that can occur during barraCuda primal operations.
#[derive(Debug, thiserror::Error)]
pub enum BarracudaCoreError {
    /// Lifecycle error (start/stop/reload).
    #[error("lifecycle error: {0}")]
    Lifecycle(String),

    /// Health check error.
    #[error("health error: {0}")]
    Health(String),

    /// IPC / transport error.
    #[error("ipc error: {0}")]
    Ipc(String),

    /// GPU device error (delegated from barracuda library).
    #[error("device error: {0}")]
    Device(String),

    /// I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// Serialization error.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// JSON parsing/serialization error.
    #[error(transparent)]
    Json(#[from] serde_json::Error),

    /// Compute library error (delegated from barracuda crate).
    #[error(transparent)]
    Compute(#[from] barracuda::error::BarracudaError),
}

impl BarracudaCoreError {
    /// Create a lifecycle error.
    pub fn lifecycle(msg: impl Into<String>) -> Self {
        Self::Lifecycle(msg.into())
    }

    /// Create a health error.
    pub fn health(msg: impl Into<String>) -> Self {
        Self::Health(msg.into())
    }

    /// Create an IPC error.
    pub fn ipc(msg: impl Into<String>) -> Self {
        Self::Ipc(msg.into())
    }

    /// Create a device error.
    pub fn device(msg: impl Into<String>) -> Self {
        Self::Device(msg.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lifecycle_error() {
        let err = BarracudaCoreError::lifecycle("cannot start");
        assert!(err.to_string().contains("lifecycle"));
        assert!(err.to_string().contains("cannot start"));
    }

    #[test]
    fn test_health_error() {
        let err = BarracudaCoreError::health("check failed");
        assert!(err.to_string().contains("health"));
    }

    #[test]
    fn test_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "not found");
        let err: BarracudaCoreError = io_err.into();
        assert!(matches!(err, BarracudaCoreError::Io(_)));
    }
}
