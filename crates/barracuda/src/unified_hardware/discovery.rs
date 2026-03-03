// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hardware discovery — runtime detection of available compute devices.

use crate::device::Device;
use crate::error::Result;
use crate::gpu_executor::GpuExecutor;
use std::sync::Arc;
use tracing::debug;

use super::cpu_executor::CpuExecutor;
use super::traits::ComputeExecutor;

/// Discovers all available compute hardware at runtime.
pub struct HardwareDiscovery;

impl HardwareDiscovery {
    pub async fn discover_all() -> Result<Vec<Arc<dyn ComputeExecutor>>> {
        let mut executors: Vec<Arc<dyn ComputeExecutor>> = Vec::new();

        executors.push(Arc::new(CpuExecutor::new()));

        if let Ok(gpu_executors) = Self::discover_gpus().await {
            executors.extend(gpu_executors);
        }

        #[cfg(feature = "tpu")]
        if let Ok(tpu_executors) = Self::discover_tpus().await {
            executors.extend(tpu_executors);
        }

        if let Ok(npu_executors) = Self::discover_npus().await {
            executors.extend(npu_executors);
        }

        Ok(executors)
    }

    async fn discover_gpus() -> Result<Vec<Arc<dyn ComputeExecutor>>> {
        let available = Device::GPU.is_available();
        debug!("GPU discovery: available={}", available);

        if !available {
            return Ok(Vec::new());
        }

        match GpuExecutor::new().await {
            Ok(executor) => {
                debug!("GPU discovered: {}", executor.name());
                Ok(vec![Arc::new(executor) as Arc<dyn ComputeExecutor>])
            }
            Err(e) => {
                debug!("GPU discovery failed: {}", e);
                Ok(Vec::new())
            }
        }
    }

    #[cfg(feature = "tpu")]
    async fn discover_tpus() -> Result<Vec<Arc<dyn ComputeExecutor>>> {
        debug!("TPU discovery: hardware not available in this environment");
        Ok(Vec::new())
    }

    async fn discover_npus() -> Result<Vec<Arc<dyn ComputeExecutor>>> {
        match crate::npu_executor::NpuExecutor::new() {
            Ok(executor) => {
                debug!(
                    "NPU discovered: {} with {} NPUs",
                    executor.name(),
                    executor.npu_count()
                );
                Ok(vec![Arc::new(executor) as Arc<dyn ComputeExecutor>])
            }
            Err(e) => {
                debug!("NPU discovery failed (no Akida hardware): {}", e);
                Ok(Vec::new())
            }
        }
    }
}
