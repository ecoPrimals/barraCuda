// SPDX-License-Identifier: AGPL-3.0-or-later
//! NPU Bridge — Tensor to dense f32 conversion utilities.
//!
//! Provides conversion between GPU-resident `Tensor` and dense `Vec<f32>` for
//! interoperability with external NPU or event-driven hardware backends.
//! barraCuda does not own NPU hardware drivers; this bridge enables consumers
//! (orchestrators, primals) to extract and inject data.

use crate::device::WgpuDevice;
use crate::error::Result as BarracudaResult;
use crate::tensor::Tensor;
use std::sync::Arc;

/// Whether an NPU backend is available in this process.
///
/// Always returns `false` — barraCuda is a math library and does not manage
/// NPU hardware. Orchestration primals provide NPU routing when present.
#[must_use]
pub fn is_npu_available() -> bool {
    false
}

/// Extract dense f32 data from a tensor for external consumption.
///
/// # Errors
///
/// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
/// readback fails (e.g. device lost or out of memory).
pub fn tensor_to_npu_data(tensor: &Tensor) -> BarracudaResult<Vec<f32>> {
    tensor.to_vec()
}

/// Create a tensor from dense f32 data (e.g. NPU output).
///
/// # Errors
///
/// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
/// readback fails (e.g. device lost or out of memory).
pub async fn npu_data_to_tensor(
    data: Vec<f32>,
    shape: Vec<usize>,
    device: Arc<WgpuDevice>,
) -> BarracudaResult<Tensor> {
    Tensor::from_vec_on(data, shape, device).await
}

/// Whether a tensor should be routed to NPU based on workload analysis.
///
/// Always returns `false` for GPU-resident tensors since readback would be
/// required for sparsity analysis, defeating the purpose.
#[must_use]
pub fn should_route_to_npu(_tensor: &Tensor, _priority: Option<crate::workload::Priority>) -> bool {
    false
}

/// Analyze dense data and determine if NPU routing is beneficial.
///
/// Returns `true` only if an NPU is available AND the workload characteristics
/// favor event-driven execution (high sparsity, energy priority, small batch).
#[must_use]
pub fn should_use_npu(data: &[f32], priority: crate::workload::Priority) -> bool {
    use crate::npu::EventCodec;
    use crate::workload::Priority;

    if !is_npu_available() {
        return false;
    }

    let codec = EventCodec::default();
    let sparsity = codec.measure_sparsity(data);

    match priority {
        Priority::Energy => true,
        Priority::Latency if data.len() < crate::workload::DENSE_CPU_THRESHOLD => true,
        _ => sparsity > crate::workload::NPU_SPARSITY_THRESHOLD,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_npu_not_available() {
        assert!(!is_npu_available());
    }

    #[test]
    fn test_should_not_route_without_npu() {
        use crate::workload::Priority;
        let data = vec![0.0; 100];
        assert!(!should_use_npu(&data, Priority::Energy));
    }

    #[tokio::test]
    async fn test_tensor_data_roundtrip() {
        let device = crate::device::test_pool::get_test_device().await;
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];

        let tensor = Tensor::from_vec_on(data.clone(), shape.clone(), device.clone())
            .await
            .unwrap();

        let extracted = tensor_to_npu_data(&tensor).unwrap();
        assert_eq!(extracted, data);

        let restored = npu_data_to_tensor(extracted, shape.clone(), device)
            .await
            .unwrap();
        assert_eq!(restored.shape(), &shape);
    }
}
