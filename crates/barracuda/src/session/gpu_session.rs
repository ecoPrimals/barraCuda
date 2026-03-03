// SPDX-License-Identifier: AGPL-3.0-or-later
//! GpuSession Builder API — Pre-warmed GPU sessions (L-004)
//!
//! Builds GPU sessions with optional pipeline pre-warming to eliminate
//! cold-start latency from the critical path.

use std::sync::Arc;

use crate::device::warmup::{warmup_device, WarmupConfig, WarmupOp};
use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};

/// Builder for pre-warmed GPU sessions.
///
/// Use this when you need a GPU device with specific pipelines compiled
/// ahead of time to avoid cold-start latency during workload execution.
///
/// # Example
///
/// ```ignore
/// let session = GpuSessionBuilder::new()
///     .pre_warm("matmul")
///     .pre_warm("softmax")
///     .max_concurrent(64)
///     .build()
///     .await?;
/// ```
#[derive(Debug, Default)]
pub struct GpuSessionBuilder {
    device: Option<Arc<WgpuDevice>>,
    pre_warm_pipelines: Vec<String>,
    max_concurrent_dispatches: usize,
}

impl GpuSessionBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            device: None,
            pre_warm_pipelines: Vec::new(),
            max_concurrent_dispatches: 64,
        }
    }

    /// Set the device to use. If not set, a new device is created at build time.
    #[must_use]
    pub fn device(mut self, device: Arc<WgpuDevice>) -> Self {
        self.device = Some(device);
        self
    }

    /// Add a pipeline name to pre-warm at build time.
    #[must_use]
    pub fn pre_warm(mut self, name: impl Into<String>) -> Self {
        self.pre_warm_pipelines.push(name.into());
        self
    }

    /// Set the maximum concurrent dispatch limit (for future scheduling use).
    #[must_use]
    pub fn max_concurrent(mut self, limit: usize) -> Self {
        self.max_concurrent_dispatches = limit;
        self
    }

    /// Build the GPU session.
    ///
    /// If no device was set, creates a new device via `WgpuDevice::new()`.
    /// Pre-warms the requested pipelines on the device.
    pub async fn build(self) -> Result<GpuSession> {
        let device = match self.device {
            Some(d) => d,
            None => {
                Arc::new(
                    WgpuDevice::new()
                        .await
                        .map_err(|e| BarracudaError::InvalidInput {
                            message: format!("Failed to create WgpuDevice: {e}"),
                        })?,
                )
            }
        };

        let warmup_ops: Vec<WarmupOp> = self
            .pre_warm_pipelines
            .iter()
            .filter_map(|name| match name.as_str() {
                "add" | "Add" => Some(WarmupOp::Add),
                "mul" | "Mul" => Some(WarmupOp::Mul),
                "fma" | "Fma" => Some(WarmupOp::Fma),
                "scale" | "Scale" => Some(WarmupOp::Scale),
                "matmul" | "Matmul" => Some(WarmupOp::Matmul),
                "reduce" | "Reduce" => Some(WarmupOp::Reduce),
                "softmax" | "Softmax" => Some(WarmupOp::Softmax),
                "relu" | "ReLU" => Some(WarmupOp::ReLU),
                "binary_op" | "BinaryOp" => Some(WarmupOp::BinaryOp),
                "unary_op" | "UnaryOp" => Some(WarmupOp::UnaryOp),
                _ => None,
            })
            .collect();

        if !warmup_ops.is_empty() {
            let config = WarmupConfig {
                ops: warmup_ops,
                workgroup_sizes: vec![256],
                verbose: false,
            };
            warmup_device(device.as_ref(), &config).map_err(|e| BarracudaError::InvalidInput {
                message: format!("Warmup failed: {e}"),
            })?;
        }

        Ok(GpuSession {
            device,
            max_concurrent_dispatches: self.max_concurrent_dispatches,
        })
    }
}

/// Pre-warmed GPU session — device with optional pipelines already compiled.
///
/// Use `GpuSessionBuilder` to construct with optional pre-warming.
#[derive(Debug, Clone)]
pub struct GpuSession {
    pub(crate) device: Arc<WgpuDevice>,
    pub(crate) max_concurrent_dispatches: usize,
}

impl GpuSession {
    /// Get the underlying device.
    pub fn device(&self) -> &Arc<WgpuDevice> {
        &self.device
    }

    /// Maximum concurrent dispatch limit (for future scheduling).
    pub fn max_concurrent_dispatches(&self) -> usize {
        self.max_concurrent_dispatches
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_defaults() {
        let builder = GpuSessionBuilder::new();
        assert!(builder.device.is_none());
        assert!(builder.pre_warm_pipelines.is_empty());
        assert_eq!(builder.max_concurrent_dispatches, 64);
    }

    #[test]
    fn builder_chaining() {
        let builder = GpuSessionBuilder::new()
            .pre_warm("matmul")
            .pre_warm("softmax")
            .max_concurrent(128);
        assert_eq!(builder.pre_warm_pipelines, vec!["matmul", "softmax"]);
        assert_eq!(builder.max_concurrent_dispatches, 128);
    }
}
