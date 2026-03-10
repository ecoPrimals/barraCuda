// SPDX-License-Identifier: AGPL-3.0-only
//! Core compute execution and tensor storage abstractions.
//!
//! These traits define the hardware-agnostic interface that CPU, GPU, TPU,
//! and NPU executors implement.

use crate::error::Result;
use crate::unified_math::{MathOp, TensorDescriptor};
use bytes::Bytes;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use super::types::{HardwareCapabilities, HardwareType};

/// Boxed future for dyn-safe async trait methods.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Universal compute executor — any device that can execute mathematical operations.
pub trait ComputeExecutor: Send + Sync {
    /// Device name for logging and diagnostics.
    fn name(&self) -> &str;
    /// Hardware type (GPU, CPU, NPU, etc.).
    fn hardware_type(&self) -> HardwareType;
    /// Hardware capabilities (precision, memory, etc.).
    fn capabilities(&self) -> &HardwareCapabilities;
    /// Whether this executor can run the given operation.
    fn can_execute(&self, op: &MathOp, inputs: &[TensorDescriptor]) -> bool;
    /// Score for operation routing (higher = better fit).
    fn score_operation(&self, op: &MathOp, inputs: &[TensorDescriptor]) -> f64;

    /// Execute the operation on the given inputs.
    fn execute(
        &self,
        op: &MathOp,
        inputs: Vec<Arc<dyn TensorStorage>>,
    ) -> BoxFuture<'_, Result<Arc<dyn TensorStorage>>>;

    /// Allocate tensor storage for the given descriptor.
    fn allocate(
        &self,
        descriptor: TensorDescriptor,
    ) -> BoxFuture<'_, Result<Arc<dyn TensorStorage>>>;

    /// Transfer tensor to this device.
    fn transfer(
        &self,
        tensor: Arc<dyn TensorStorage>,
    ) -> BoxFuture<'_, Result<Arc<dyn TensorStorage>>>;
}

/// Hardware-agnostic tensor storage — data can live on any device.
pub trait TensorStorage: Send + Sync {
    /// Tensor shape and dtype descriptor.
    fn descriptor(&self) -> &TensorDescriptor;
    /// Hardware where the tensor resides.
    fn hardware_type(&self) -> HardwareType;
    /// Read tensor data to CPU memory.
    fn read_to_cpu(&self) -> BoxFuture<'_, Result<Bytes>>;
    /// Write tensor data from CPU memory.
    fn write_from_cpu(&mut self, data: &[u8]) -> BoxFuture<'_, Result<()>>;

    /// True if tensor is on CPU.
    fn is_cpu(&self) -> bool {
        self.hardware_type() == HardwareType::CPU
    }
    /// True if tensor is on GPU.
    fn is_gpu(&self) -> bool {
        self.hardware_type() == HardwareType::GPU
    }
    /// True if tensor is on TPU.
    fn is_tpu(&self) -> bool {
        self.hardware_type() == HardwareType::TPU
    }

    /// Get underlying wgpu buffer if this is a GPU tensor.
    fn as_wgpu_buffer(&self) -> Option<Arc<::wgpu::Buffer>> {
        None
    }
}
