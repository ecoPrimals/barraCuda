// SPDX-License-Identifier: AGPL-3.0-or-later
//! Core compute execution and tensor storage abstractions.
//!
//! These traits define the hardware-agnostic interface that CPU, GPU, TPU,
//! and NPU executors implement.

use crate::error::Result;
use crate::unified_math::{MathOp, TensorDescriptor};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use super::types::{HardwareCapabilities, HardwareType};

/// Boxed future for dyn-safe async trait methods.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Universal compute executor — any device that can execute mathematical operations.
pub trait ComputeExecutor: Send + Sync {
    fn name(&self) -> &str;
    fn hardware_type(&self) -> HardwareType;
    fn capabilities(&self) -> &HardwareCapabilities;
    fn can_execute(&self, op: &MathOp, inputs: &[TensorDescriptor]) -> bool;
    fn score_operation(&self, op: &MathOp, inputs: &[TensorDescriptor]) -> f64;

    fn execute(
        &self,
        op: &MathOp,
        inputs: Vec<Arc<dyn TensorStorage>>,
    ) -> BoxFuture<'_, Result<Arc<dyn TensorStorage>>>;

    fn allocate(
        &self,
        descriptor: TensorDescriptor,
    ) -> BoxFuture<'_, Result<Arc<dyn TensorStorage>>>;

    fn transfer(
        &self,
        tensor: Arc<dyn TensorStorage>,
    ) -> BoxFuture<'_, Result<Arc<dyn TensorStorage>>>;
}

/// Hardware-agnostic tensor storage — data can live on any device.
pub trait TensorStorage: Send + Sync {
    fn descriptor(&self) -> &TensorDescriptor;
    fn hardware_type(&self) -> HardwareType;
    fn read_to_cpu(&self) -> BoxFuture<'_, Result<Vec<u8>>>;
    fn write_from_cpu(&mut self, data: &[u8]) -> BoxFuture<'_, Result<()>>;

    fn is_cpu(&self) -> bool {
        self.hardware_type() == HardwareType::CPU
    }
    fn is_gpu(&self) -> bool {
        self.hardware_type() == HardwareType::GPU
    }
    fn is_tpu(&self) -> bool {
        self.hardware_type() == HardwareType::TPU
    }

    fn as_wgpu_buffer(&self) -> Option<Arc<::wgpu::Buffer>> {
        None
    }
}
