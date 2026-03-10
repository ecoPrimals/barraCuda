// SPDX-License-Identifier: AGPL-3.0-only
//! CPU tensor storage implementation

use crate::error::Result;
use crate::unified_hardware::{HardwareType, TensorStorage};
use crate::unified_math::TensorDescriptor;
use bytes::Bytes;

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

impl TensorStorage for CpuTensorStorage {
    fn descriptor(&self) -> &TensorDescriptor {
        &self.descriptor
    }

    fn hardware_type(&self) -> HardwareType {
        HardwareType::CPU
    }

    fn read_to_cpu(
        &self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Bytes>> + Send + '_>> {
        let data = Bytes::from(self.data.clone());
        Box::pin(async move { Ok(data) })
    }

    fn write_from_cpu(
        &mut self,
        data: &[u8],
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + '_>> {
        let data = data.to_vec();
        Box::pin(async move {
            if data.len() != self.data.len() {
                return Err(crate::error::BarracudaError::InvalidInput {
                    message: format!(
                        "Data size mismatch: expected {}, got {}",
                        self.data.len(),
                        data.len()
                    ),
                });
            }
            self.data.copy_from_slice(&data);
            Ok(())
        })
    }
}
