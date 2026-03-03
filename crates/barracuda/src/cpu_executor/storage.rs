//! CPU tensor storage implementation

use crate::error::Result;
use crate::unified_hardware::{HardwareType, TensorStorage};
use crate::unified_math::TensorDescriptor;
use async_trait::async_trait;

/// CPU tensor storage implementation
pub(super) struct CpuTensorStorage {
    pub(super) descriptor: TensorDescriptor,
    pub(super) data: Vec<u8>,
}

impl CpuTensorStorage {
    pub(super) fn new(descriptor: TensorDescriptor) -> Self {
        let byte_size = descriptor.numel * descriptor.dtype.size_bytes();
        Self {
            descriptor,
            data: vec![0u8; byte_size],
        }
    }
}

// NOTE(async-dyn): #[async_trait] required — native async fn in trait is not dyn-compatible
#[async_trait]
impl TensorStorage for CpuTensorStorage {
    fn descriptor(&self) -> &TensorDescriptor {
        &self.descriptor
    }

    fn hardware_type(&self) -> HardwareType {
        HardwareType::CPU
    }

    async fn read_to_cpu(&self) -> Result<Vec<u8>> {
        Ok(self.data.clone())
    }

    async fn write_from_cpu(&mut self, data: &[u8]) -> Result<()> {
        if data.len() != self.data.len() {
            return Err(crate::error::BarracudaError::InvalidInput {
                message: format!(
                    "Data size mismatch: expected {}, got {}",
                    self.data.len(),
                    data.len()
                ),
            });
        }
        self.data.copy_from_slice(data);
        Ok(())
    }
}
