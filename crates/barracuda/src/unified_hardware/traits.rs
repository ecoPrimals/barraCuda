//! Core compute execution and tensor storage abstractions.
//!
//! These traits define the hardware-agnostic interface that CPU, GPU, TPU,
//! and NPU executors implement.

use crate::error::Result;
use crate::unified_math::{MathOp, TensorDescriptor};
use async_trait::async_trait;
use std::sync::Arc;

use super::types::{HardwareCapabilities, HardwareType};

/// Universal compute executor — any device that can execute mathematical operations.
// NOTE(async-dyn): #[async_trait] required — native async fn in trait is not dyn-compatible
#[async_trait]
pub trait ComputeExecutor: Send + Sync {
    fn name(&self) -> &str;
    fn hardware_type(&self) -> HardwareType;
    fn capabilities(&self) -> &HardwareCapabilities;
    fn can_execute(&self, op: &MathOp, inputs: &[TensorDescriptor]) -> bool;
    fn score_operation(&self, op: &MathOp, inputs: &[TensorDescriptor]) -> f64;

    async fn execute(
        &self,
        op: &MathOp,
        inputs: Vec<Arc<dyn TensorStorage>>,
    ) -> Result<Arc<dyn TensorStorage>>;

    async fn allocate(&self, descriptor: TensorDescriptor) -> Result<Arc<dyn TensorStorage>>;

    async fn transfer(&self, tensor: Arc<dyn TensorStorage>) -> Result<Arc<dyn TensorStorage>>;
}

/// Hardware-agnostic tensor storage — data can live on any device.
// NOTE(async-dyn): #[async_trait] required — native async fn in trait is not dyn-compatible
#[async_trait]
pub trait TensorStorage: Send + Sync {
    fn descriptor(&self) -> &TensorDescriptor;
    fn hardware_type(&self) -> HardwareType;
    async fn read_to_cpu(&self) -> Result<Vec<u8>>;
    async fn write_from_cpu(&mut self, data: &[u8]) -> Result<()>;

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
