// SPDX-License-Identifier: AGPL-3.0-or-later
//! TensorSession — `SessionTensor` handle

use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use std::sync::Arc;

/// A lightweight tensor reference within a `TensorSession`.
///
/// Tracks a GPU buffer and shape; actual data lives in the session's buffer
/// registry.  Use `to_vec()` or `to_tensor()` after `session.run()`.
#[derive(Debug, Clone)]
pub struct SessionTensor {
    /// Index into the session's buffer registry
    pub(super) buffer_id: usize,
    pub(super) shape: Vec<usize>,
    pub(super) device: Arc<WgpuDevice>,
    /// The backing buffer (populated at alloc time, available immediately)
    pub(super) buffer: Option<Arc<wgpu::Buffer>>,
}

impl SessionTensor {
    /// Tensor shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// True when the tensor has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read data back to CPU (available immediately after buffer allocation;
    /// reflects computed values only after `session.run()`).
    pub fn to_vec(&self) -> Result<Vec<f32>> {
        let buffer = self.buffer.as_ref().ok_or_else(|| {
            BarracudaError::execution_failed("Session not executed yet — call session.run() first")
        })?;
        self.device.read_buffer_f32(buffer, self.len())
    }

    /// Convert to a standalone `Tensor` (GPU-resident copy).
    ///
    /// Performs a CPU round-trip; use `to_vec()` when only the data is needed.
    pub fn to_tensor(&self) -> Result<Tensor> {
        let data = self.to_vec()?;
        Tensor::from_data(&data, self.shape.clone(), self.device.clone())
    }
}
