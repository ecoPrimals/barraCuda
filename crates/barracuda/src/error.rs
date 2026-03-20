// SPDX-License-Identifier: AGPL-3.0-or-later
//! Error types for barraCuda
//!
//! **Deep Debt Excellence**: Rich error context, zero panic paths

use std::sync::Arc;
use thiserror::Error;

/// Result type alias for barraCuda operations.
pub type Result<T> = std::result::Result<T, BarracudaError>;

/// Unified error type for barraCuda.
///
/// Use the constructor helpers (`shape_mismatch`, `gpu`, etc.) instead of
/// constructing variants directly. Check `is_device_lost()` for retriable failures.
#[derive(Error, Debug)]
pub enum BarracudaError {
    /// Device initialization or selection failed.
    #[error("Device error: {0}")]
    Device(String),

    /// Tensor shapes don't match for a binary op (e.g. matmul dimensions).
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// Expected shape dimensions.
        expected: Vec<usize>,
        /// Actual shape dimensions received.
        actual: Vec<usize>,
    },

    /// Shape violates constraints (e.g. non-positive dim, wrong rank).
    #[error("Invalid shape: expected {expected:?}, got {actual:?}")]
    InvalidShape {
        /// Expected shape dimensions.
        expected: Vec<usize>,
        /// Actual shape dimensions received.
        actual: Vec<usize>,
    },

    /// Operation rejected due to semantic rules (e.g. incompatible strides).
    #[error("Invalid operation: {op} - {reason}")]
    InvalidOperation {
        /// Name of the operation that failed.
        op: String,
        /// Reason the operation was rejected.
        reason: String,
    },

    /// GPU runtime error (buffer map, submit, general GPU failure).
    #[error("GPU error: {0}")]
    Gpu(String),

    /// GPU device was lost (driver reset, TDR, hardware fault).
    ///
    /// This is a retriable failure: the operation can succeed on a fresh device.
    /// Callers should use `is_device_lost()` to detect this and either retry
    /// with a new device or propagate the error for pool-level recovery.
    #[error("GPU device lost: {0}")]
    DeviceLost(String),

    /// WGSL shader failed to compile (syntax, validation).
    #[error("Shader compilation error: {0}")]
    ShaderCompilation(Arc<str>),

    /// GPU or host memory allocation failed.
    #[error("Out of memory: {0}")]
    OutOfMemory(String),

    /// Operation not supported on the selected device (e.g. f64 on CPU).
    #[error("Operation not supported on device: {op} on {device}")]
    UnsupportedOperation {
        /// Name of the unsupported operation.
        op: String,
        /// Device on which the operation is not supported.
        device: String,
    },

    /// Invalid user input (negative stride, out-of-range index).
    #[error("Invalid input: {message}")]
    InvalidInput {
        /// Description of the invalid input.
        message: String,
    },

    /// Kernel or executor execution failed.
    #[error("Execution error: {message}")]
    ExecutionError {
        /// Description of the execution failure.
        message: String,
    },

    /// Requested device unavailable (no adapter, backend disabled).
    #[error("Device not available: {device} - {reason}")]
    DeviceNotAvailable {
        /// Device identifier that was requested.
        device: String,
        /// Reason the device is unavailable.
        reason: String,
    },

    /// No executor registered for this operation (dispatch failure).
    #[error("No available executor for operation: {operation}")]
    NoAvailableExecutor {
        /// Name of the operation that has no executor.
        operation: String,
    },

    /// Unexpected internal state (logic bug, use when panicking is inappropriate).
    #[error("Internal error: {0}")]
    Internal(String),

    /// Numerical failure (NaN, overflow, singular matrix).
    #[error("Numerical error: {message}")]
    Numerical {
        /// Description of the numerical failure.
        message: String,
    },

    /// Resource quota exceeded (concurrent ops, buffer pool).
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),

    /// Requested allocation exceeds device safe limit.
    #[error(
        "Device limit exceeded: {message} (requested {requested_bytes} bytes, safe limit {safe_limit_bytes} bytes)"
    )]
    DeviceLimitExceeded {
        /// Human-readable description of the limit exceeded.
        message: String,
        /// Number of bytes requested.
        requested_bytes: u64,
        /// Device's safe allocation limit in bytes.
        safe_limit_bytes: u64,
    },

    /// Feature not yet implemented.
    #[error("Not implemented: {feature}")]
    NotImplemented {
        /// Name of the unimplemented feature.
        feature: String,
    },

    /// I/O failure (file read, write).
    #[error("IO error: {context}")]
    Io {
        /// Context describing the I/O operation that failed.
        context: String,
        /// The underlying I/O error.
        #[source]
        source: std::sync::Arc<std::io::Error>,
    },

    /// JSON parse/serialize error (config, checkpoint).
    #[error("JSON error: {context}")]
    Json {
        /// Context describing where the JSON error occurred.
        context: String,
        /// Additional error details.
        detail: String,
    },
}

impl BarracudaError {
    /// Construct a device error.
    pub fn device(msg: impl Into<String>) -> Self {
        Self::Device(msg.into())
    }

    /// Construct a device-not-found error.
    pub fn device_not_found(msg: impl Into<String>) -> Self {
        Self::Device(msg.into())
    }

    /// Construct an execution failure error.
    pub fn execution_failed(msg: impl Into<String>) -> Self {
        Self::ExecutionError {
            message: msg.into(),
        }
    }

    /// Construct a shape mismatch error.
    #[must_use]
    pub fn shape_mismatch(expected: Vec<usize>, actual: Vec<usize>) -> Self {
        Self::ShapeMismatch { expected, actual }
    }

    /// Construct an invalid operation error.
    pub fn invalid_op(op: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::InvalidOperation {
            op: op.into(),
            reason: reason.into(),
        }
    }

    /// Construct a GPU runtime error.
    pub fn gpu(msg: impl Into<String>) -> Self {
        Self::Gpu(msg.into())
    }

    /// Construct a shader compilation error.
    pub fn shader_compilation(msg: impl Into<Arc<str>>) -> Self {
        Self::ShaderCompilation(msg.into())
    }

    /// Construct an out-of-memory error.
    pub fn oom(msg: impl Into<String>) -> Self {
        Self::OutOfMemory(msg.into())
    }

    /// Construct an unsupported operation error.
    pub fn unsupported(op: impl Into<String>, device: impl Into<String>) -> Self {
        Self::UnsupportedOperation {
            op: op.into(),
            device: device.into(),
        }
    }

    /// Construct an invalid shape error.
    #[must_use]
    pub fn invalid_shape(expected: Vec<usize>, actual: Vec<usize>) -> Self {
        Self::InvalidShape { expected, actual }
    }

    /// Construct a resource exhausted error.
    pub fn resource_exhausted(msg: impl Into<String>) -> Self {
        Self::ResourceExhausted(msg.into())
    }

    /// Construct an I/O error with context.
    pub fn io(context: impl Into<String>, source: std::io::Error) -> Self {
        Self::Io {
            context: context.into(),
            source: std::sync::Arc::new(source),
        }
    }

    /// Construct a JSON parse/serialize error.
    pub fn json(context: impl Into<String>, detail: impl Into<String>) -> Self {
        Self::Json {
            context: context.into(),
            detail: detail.into(),
        }
    }

    /// Construct a device-lost error.
    pub fn device_lost(msg: impl Into<String>) -> Self {
        Self::DeviceLost(msg.into())
    }

    /// Returns `true` when this error indicates the GPU device was lost.
    ///
    /// Device loss is a transient hardware failure — the operation can be
    /// retried on a fresh device. Callers (and the test infrastructure)
    /// use this to distinguish retriable failures from logic bugs.
    #[must_use]
    pub fn is_device_lost(&self) -> bool {
        if matches!(self, Self::DeviceLost(_)) {
            return true;
        }
        let msg = self.to_string();
        msg.contains("device lost") || msg.contains("Device lost")
    }

    /// Returns `true` when this error is retriable (device lost or transient GPU failure).
    ///
    /// Includes buffer validation errors that occur transiently under
    /// instrumentation pressure (e.g. llvm-cov on llvmpipe).
    #[must_use]
    pub fn is_retriable(&self) -> bool {
        if self.is_device_lost() {
            return true;
        }
        let msg = self.to_string();
        msg.contains("is invalid") || msg.contains("Validation Error")
    }

    /// Wrap any `Display` error as a GPU error with contextual message.
    ///
    /// Replaces the verbose `map_err(|e| BarracudaError::Gpu(format!("ctx: {e}")))`
    /// pattern that appears across 50+ GPU ops.
    ///
    /// # Example
    /// ```ignore
    /// buffer.slice(..).map_async(MapMode::Read, |_| {})
    ///     .map_err(|e| BarracudaError::gpu_ctx("buffer map", e))?;
    /// ```
    pub fn gpu_ctx(context: &str, err: impl std::fmt::Display) -> Self {
        Self::Gpu(format!("{context}: {err}"))
    }
}

impl From<std::io::Error> for BarracudaError {
    fn from(e: std::io::Error) -> Self {
        Self::io("IO operation failed", e)
    }
}

// ── Safe narrowing cast helpers ─────────────────────────────────────────────

/// Safe `u64` to `u32` conversion.
///
/// Use at RPC boundaries where external input (e.g. `degree: u64`) must be
/// narrowed for GPU dispatch (`u32` shader uniforms).
///
/// # Errors
///
/// Returns [`BarracudaError::InvalidInput`] if `v > u32::MAX`.
#[inline]
pub fn u32_from_u64(v: u64) -> Result<u32> {
    u32::try_from(v).map_err(|_| BarracudaError::InvalidInput {
        message: format!("value {v} exceeds u32::MAX ({max})", max = u32::MAX),
    })
}

/// Safe `usize` to `u32` conversion.
///
/// # Errors
///
/// Returns [`BarracudaError::InvalidInput`] if `v > u32::MAX`.
#[inline]
pub fn u32_from_usize(v: usize) -> Result<u32> {
    u32::try_from(v).map_err(|_| BarracudaError::InvalidInput {
        message: format!("value {v} exceeds u32::MAX ({max})", max = u32::MAX),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_variant_constructs_and_displays() {
        let e = BarracudaError::Device("broken".to_string());
        let s = e.to_string();
        assert!(s.contains("Device error"));
        assert!(s.contains("broken"));
    }

    #[test]
    fn shape_mismatch_variant_constructs_and_displays() {
        let e = BarracudaError::ShapeMismatch {
            expected: vec![2, 3],
            actual: vec![2, 4],
        };
        let s = e.to_string();
        assert!(s.contains("Shape mismatch"));
        assert!(s.contains("[2, 3]"));
        assert!(s.contains("[2, 4]"));
    }

    #[test]
    fn invalid_shape_variant_constructs_and_displays() {
        let e = BarracudaError::InvalidShape {
            expected: vec![1, 2, 3],
            actual: vec![1, 2],
        };
        let s = e.to_string();
        assert!(s.contains("Invalid shape"));
        assert!(s.contains("[1, 2, 3]"));
        assert!(s.contains("[1, 2]"));
    }

    #[test]
    fn invalid_operation_variant_constructs_and_displays() {
        let e = BarracudaError::InvalidOperation {
            op: "matmul".to_string(),
            reason: "incompatible dimensions".to_string(),
        };
        let s = e.to_string();
        assert!(s.contains("Invalid operation"));
        assert!(s.contains("matmul"));
        assert!(s.contains("incompatible dimensions"));
    }

    #[test]
    fn gpu_variant_constructs_and_displays() {
        let e = BarracudaError::Gpu("timeout".to_string());
        let s = e.to_string();
        assert!(s.contains("GPU error"));
        assert!(s.contains("timeout"));
    }

    #[test]
    fn shader_compilation_variant_constructs_and_displays() {
        let e = BarracudaError::ShaderCompilation(Arc::from("syntax error"));
        let s = e.to_string();
        assert!(s.contains("Shader compilation error"));
        assert!(s.contains("syntax error"));
    }

    #[test]
    fn out_of_memory_variant_constructs_and_displays() {
        let e = BarracudaError::OutOfMemory("alloc failed".to_string());
        let s = e.to_string();
        assert!(s.contains("Out of memory"));
        assert!(s.contains("alloc failed"));
    }

    #[test]
    fn unsupported_operation_variant_constructs_and_displays() {
        let e = BarracudaError::UnsupportedOperation {
            op: "fft".to_string(),
            device: "cpu".to_string(),
        };
        let s = e.to_string();
        assert!(s.contains("not supported"));
        assert!(s.contains("fft"));
        assert!(s.contains("cpu"));
    }

    #[test]
    fn invalid_input_variant_constructs_and_displays() {
        let e = BarracudaError::InvalidInput {
            message: "negative stride".to_string(),
        };
        let s = e.to_string();
        assert!(s.contains("Invalid input"));
        assert!(s.contains("negative stride"));
    }

    #[test]
    fn execution_error_variant_constructs_and_displays() {
        let e = BarracudaError::ExecutionError {
            message: "kernel crashed".to_string(),
        };
        let s = e.to_string();
        assert!(s.contains("Execution error"));
        assert!(s.contains("kernel crashed"));
    }

    #[test]
    fn device_not_available_variant_constructs_and_displays() {
        let e = BarracudaError::DeviceNotAvailable {
            device: "wgpu".to_string(),
            reason: "no adapter".to_string(),
        };
        let s = e.to_string();
        assert!(s.contains("Device not available"));
        assert!(s.contains("wgpu"));
        assert!(s.contains("no adapter"));
    }

    #[test]
    fn no_available_executor_variant_constructs_and_displays() {
        let e = BarracudaError::NoAvailableExecutor {
            operation: "conv2d".to_string(),
        };
        let s = e.to_string();
        assert!(s.contains("No available executor"));
        assert!(s.contains("conv2d"));
    }

    #[test]
    fn internal_variant_constructs_and_displays() {
        let e = BarracudaError::Internal("unexpected state".to_string());
        let s = e.to_string();
        assert!(s.contains("Internal error"));
        assert!(s.contains("unexpected state"));
    }

    #[test]
    fn helper_device_produces_device_variant() {
        let e = BarracudaError::device("msg");
        assert!(matches!(e, BarracudaError::Device(_)));
    }

    #[test]
    fn helper_device_not_found_produces_device_variant() {
        let e = BarracudaError::device_not_found("not found");
        assert!(matches!(e, BarracudaError::Device(_)));
        assert!(e.to_string().contains("not found"));
    }

    #[test]
    fn helper_execution_failed_produces_execution_error() {
        let e = BarracudaError::execution_failed("failed");
        assert!(matches!(e, BarracudaError::ExecutionError { .. }));
        assert!(e.to_string().contains("failed"));
    }

    #[test]
    fn helper_shape_mismatch_produces_shape_mismatch() {
        let e = BarracudaError::shape_mismatch(vec![1, 2], vec![3, 4]);
        assert!(matches!(e, BarracudaError::ShapeMismatch { .. }));
    }

    #[test]
    fn helper_invalid_op_produces_invalid_operation() {
        let e = BarracudaError::invalid_op("op", "reason");
        assert!(matches!(e, BarracudaError::InvalidOperation { .. }));
        let s = e.to_string();
        assert!(s.contains("op"));
        assert!(s.contains("reason"));
    }

    #[test]
    fn helper_gpu_produces_gpu_variant() {
        let e = BarracudaError::gpu("err");
        assert!(matches!(e, BarracudaError::Gpu(_)));
    }

    #[test]
    fn helper_shader_compilation_produces_shader_compilation() {
        let e = BarracudaError::shader_compilation("msg");
        assert!(matches!(e, BarracudaError::ShaderCompilation(_)));
    }

    #[test]
    fn helper_oom_produces_out_of_memory() {
        let e = BarracudaError::oom("alloc");
        assert!(matches!(e, BarracudaError::OutOfMemory(_)));
    }

    #[test]
    fn helper_unsupported_produces_unsupported_operation() {
        let e = BarracudaError::unsupported("op", "dev");
        assert!(matches!(e, BarracudaError::UnsupportedOperation { .. }));
    }

    #[test]
    fn helper_invalid_shape_produces_invalid_shape() {
        let e = BarracudaError::invalid_shape(vec![1], vec![2]);
        assert!(matches!(e, BarracudaError::InvalidShape { .. }));
    }

    #[test]
    fn result_ok_works() {
        let r: Result<i32> = Ok(42);
        let Ok(v) = r else { panic!("expected Ok(42)") };
        assert_eq!(v, 42);
    }

    #[test]
    fn result_err_works() {
        let r: Result<i32> = Err(BarracudaError::Internal("test".into()));
        let Err(e) = r else { panic!("expected Err") };
        assert!(e.to_string().contains("Internal error"));
    }

    #[test]
    fn device_lost_variant_constructs_and_is_detected() {
        let e = BarracudaError::device_lost("GPU reset");
        assert!(e.is_device_lost());
        assert!(e.is_retriable());
        assert!(e.to_string().contains("GPU reset"));
    }

    #[test]
    fn device_lost_detected_from_gpu_string() {
        let e = BarracudaError::gpu("GPU device lost during submit");
        assert!(e.is_device_lost());
        assert!(e.is_retriable());
    }

    #[test]
    fn non_device_lost_is_not_retriable() {
        let e = BarracudaError::gpu("timeout");
        assert!(!e.is_device_lost());
        assert!(!e.is_retriable());
    }

    #[test]
    fn result_err_matches_and_propagates() {
        fn may_fail() -> Result<u32> {
            Err(BarracudaError::Device("x".into()))
        }
        let r = may_fail();
        assert!(r.is_err());
        let Err(e) = r else { panic!("expected Err") };
        assert!(matches!(e, BarracudaError::Device(_)));
    }
}
