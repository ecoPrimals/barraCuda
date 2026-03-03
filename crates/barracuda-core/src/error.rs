//! barraCuda error types.
//!
//! Unified error handling for GPU device management, shader compilation,
//! tensor operations, and compute dispatch.

#![allow(clippy::doc_markdown)]

/// Result type alias for barraCuda operations.
pub type Result<T> = std::result::Result<T, BarracudaError>;

/// Errors that can occur during barraCuda operations.
#[derive(Debug, thiserror::Error)]
pub enum BarracudaError {
    /// GPU device was lost (transient — can retry on fresh device).
    #[error("device lost: {0}")]
    DeviceLost(String),

    /// GPU device creation or feature negotiation failed.
    #[error("device error: {0}")]
    Device(String),

    /// Shader compilation failed.
    #[error("shader compilation: {0}")]
    ShaderCompilation(String),

    /// Tensor shape mismatch between operands.
    #[error("shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// Expected shape.
        expected: Vec<usize>,
        /// Actual shape.
        actual: Vec<usize>,
    },

    /// Invalid operation for the given inputs.
    #[error("invalid op '{op}': {reason}")]
    InvalidOp {
        /// Operation name.
        op: String,
        /// Why the operation is invalid.
        reason: String,
    },

    /// Buffer or memory allocation failure.
    #[error("buffer error: {0}")]
    Buffer(String),

    /// Compute dispatch failure.
    #[error("dispatch error: {0}")]
    Dispatch(String),
}

impl BarracudaError {
    /// Returns `true` when this error indicates the GPU device was lost.
    ///
    /// Device loss is a transient hardware failure — the operation can be
    /// retried on a fresh device.
    #[must_use]
    pub fn is_device_lost(&self) -> bool {
        matches!(self, Self::DeviceLost(_)) || {
            let msg = self.to_string();
            msg.contains("device lost") || msg.contains("Device lost")
        }
    }

    /// Create a shape mismatch error.
    #[must_use]
    pub fn shape_mismatch(expected: Vec<usize>, actual: Vec<usize>) -> Self {
        Self::ShapeMismatch { expected, actual }
    }

    /// Create an invalid operation error.
    pub fn invalid_op(op: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::InvalidOp {
            op: op.into(),
            reason: reason.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_lost_detection() {
        let err = BarracudaError::DeviceLost("adapter reset".into());
        assert!(err.is_device_lost());

        let err = BarracudaError::Device("device lost during execution".into());
        assert!(err.is_device_lost());

        let err = BarracudaError::ShaderCompilation("syntax error".into());
        assert!(!err.is_device_lost());
    }

    #[test]
    fn test_shape_mismatch() {
        let err = BarracudaError::shape_mismatch(vec![3, 4], vec![5, 6]);
        assert!(err.to_string().contains("shape mismatch"));
    }

    #[test]
    fn test_invalid_op() {
        let err = BarracudaError::invalid_op("matmul", "inner dimensions don't match");
        assert!(err.to_string().contains("matmul"));
    }
}
