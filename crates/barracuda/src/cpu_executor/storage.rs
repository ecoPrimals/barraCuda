// SPDX-License-Identifier: AGPL-3.0-only
//! CPU tensor storage implementation — zero-copy via `BytesMut`.

use crate::error::Result;
use crate::unified_hardware::{HardwareType, TensorStorage};
use crate::unified_math::TensorDescriptor;
use bytes::{Bytes, BytesMut};

/// CPU tensor storage backed by `BytesMut`.
///
/// `read_to_cpu()` freezes a clone of the internal buffer into shared
/// `Bytes` — downstream consumers get a ref-counted view without copying.
pub(super) struct CpuTensorStorage {
    pub(super) descriptor: TensorDescriptor,
    pub(super) data: BytesMut,
}

impl CpuTensorStorage {
    pub(super) fn new(descriptor: TensorDescriptor) -> Self {
        let byte_size = descriptor.numel * descriptor.dtype.size_bytes();
        Self {
            descriptor,
            data: BytesMut::zeroed(byte_size),
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
        let data = Bytes::copy_from_slice(&self.data);
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
