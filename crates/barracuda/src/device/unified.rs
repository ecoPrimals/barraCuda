// SPDX-License-Identifier: AGPL-3.0-or-later
//! Unified Device Abstraction - Phase 2
//!
//! **EVOLVED**: Single Device enum for ALL hardware types!
//!
//! This module provides a unified interface for all compute devices:
//! - CPU: Pure Rust execution
//! - GPU: WGSL shaders via wgpu
//! - NPU: Akida neuromorphic hardware
//! - TPU: Tensor Processing Units (future)
//! - Auto: Automatic selection based on workload
//!
//! # Philosophy
//!
//! **Hardware does the specialization, not the code!**
//!
//! - One codebase, all hardware
//! - Explicit routing when needed
//! - Automatic selection by default
//! - Flexible fallback chains
//!
//! # Deep Debt Compliance
//!
//! - ✅ **Hardware agnostic**: No assumptions
//! - ✅ **Runtime discovery**: Capability-based
//! - ✅ **Explicit control**: When needed
//! - ✅ **Smart defaults**: Auto selection
//! - ✅ **Flexible routing**: Fallback chains
//!
//! # Example
//!
//! ```rust,ignore
//! use barracuda::prelude::{Device, DeviceInfo, Tensor};
//!
//! // Automatic selection (recommended)
//! let tensor = Tensor::randn(vec![1000, 1000]).await?;
//! let result = tensor.matmul(&other).await?; // Auto-routed!
//!
//! // Explicit routing
//! let gpu_tensor = tensor.on(Device::GPU).await?;
//! let npu_tensor = tensor.on(Device::NPU).await?;
//!
//! // Query capabilities
//! let info = Device::CPU.info();
//! println!("CPU supports: {:?}", info.capabilities);
//! ```

use crate::device::akida::AkidaBoard;
use crate::device::capabilities::build_device_info;
use crate::device::routing::{select_for_workload, select_with_preference};
use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result as BarracudaResult};

// Re-export for backward compatibility (mod.rs and prelude use these)
pub use crate::device::capabilities::{Capability, DeviceInfo};
pub use crate::device::device_types::Device;
pub use crate::device::routing::WorkloadHint;

impl Device {
    /// Get device information and capabilities
    ///
    /// **Runtime discovery** — No hardcoding!
    #[must_use]
    pub fn info(&self) -> DeviceInfo {
        build_device_info(*self)
    }

    /// Check if this device is available
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.info().available
    }

    /// List all available devices
    ///
    /// **Runtime discovery** — No assumptions!
    #[must_use]
    pub fn available_devices() -> Vec<Device> {
        vec![
            Device::CPU,
            Device::GPU,
            Device::NPU,
            Device::TPU,
            Device::Auto,
        ]
        .into_iter()
        .filter(|d| d.is_available())
        .collect()
    }

    /// Select best device for given workload characteristics (auto-routing).
    ///
    /// Routes workloads to the appropriate hardware based on the nature
    /// of the computation. GPUs run arbitrary WGSL shaders, NPUs run
    /// pre-compiled neural network models, CPUs handle everything else.
    ///
    /// This is BarraCuda's recommendation. To override, use
    /// [`Device::select_with_preference`] or construct a [`DeviceContext`] directly.
    #[must_use]
    pub fn select_for_workload(workload: &WorkloadHint) -> Device {
        select_for_workload(workload)
    }

    /// Select device with an explicit user preference.
    ///
    /// If the user requests a specific device and it is available, honour
    /// that choice regardless of what the auto-router would recommend.
    ///
    /// Fallback chain when the preferred device is unavailable:
    /// `preferred → auto-route recommendation → GPU → CPU`
    #[must_use]
    pub fn select_with_preference(preferred: Option<Device>, workload: &WorkloadHint) -> Device {
        select_with_preference(preferred, workload)
    }
}

/// Device context for execution
///
/// **Lazy initialization** — Only create when needed!
pub enum DeviceContext {
    /// CPU context (always available)
    CPU,

    /// GPU context (WGSL via wgpu)
    GPU(WgpuDevice),

    /// NPU context (Akida)
    NPU(AkidaBoard),

    /// Not yet initialized
    Uninitialized,
}

impl DeviceContext {
    /// Create context for device
    ///
    /// **Lazy initialization** — Only when needed!
    pub async fn for_device(device: Device) -> BarracudaResult<Self> {
        match device {
            Device::CPU => Ok(DeviceContext::CPU),

            Device::GPU => {
                let wgpu_device = WgpuDevice::new().await?;
                Ok(DeviceContext::GPU(wgpu_device))
            }

            Device::NPU => {
                let capabilities = crate::device::detect_akida_boards()?;
                if capabilities.boards.is_empty() {
                    return Err(BarracudaError::DeviceNotAvailable {
                        device: "NPU".to_string(),
                        reason: "No Akida boards detected".to_string(),
                    });
                }
                Ok(DeviceContext::NPU(capabilities.boards[0].clone()))
            }

            Device::TPU => Err(BarracudaError::DeviceNotAvailable {
                device: "TPU".to_string(),
                reason: "TPU support not yet implemented".to_string(),
            }),

            Device::Auto => {
                if Device::GPU.is_available() {
                    match WgpuDevice::new().await {
                        Ok(wgpu_device) => Ok(DeviceContext::GPU(wgpu_device)),
                        Err(_) => Ok(DeviceContext::CPU),
                    }
                } else {
                    Ok(DeviceContext::CPU)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_display() {
        assert_eq!(Device::CPU.to_string(), "CPU");
        assert_eq!(Device::GPU.to_string(), "GPU");
        assert_eq!(Device::NPU.to_string(), "NPU");
        assert_eq!(Device::Auto.to_string(), "Auto");
    }

    #[test]
    fn test_cpu_always_available() {
        assert!(Device::CPU.is_available());
    }

    #[test]
    fn test_device_info() {
        let info = Device::CPU.info();
        assert_eq!(info.device, Device::CPU);
        assert!(info.available);
        assert!(!info.name.is_empty());
    }

    #[test]
    fn test_workload_selection_strings() {
        let device = Device::select_for_workload(&WorkloadHint::StringOps);
        assert_eq!(device, Device::CPU);
    }

    #[test]
    fn test_workload_selection_small() {
        let device = Device::select_for_workload(&WorkloadHint::SmallWorkload);
        assert_eq!(device, Device::CPU);
    }

    #[test]
    fn test_available_devices() {
        let devices = Device::available_devices();
        assert!(!devices.is_empty());
        assert!(devices.contains(&Device::CPU));
    }

    #[test]
    fn test_select_with_preference_none_uses_auto() {
        let auto = Device::select_for_workload(&WorkloadHint::SmallWorkload);
        let pref = Device::select_with_preference(None, &WorkloadHint::SmallWorkload);
        assert_eq!(auto, pref);
    }

    #[test]
    fn test_select_with_preference_auto_uses_auto() {
        let auto = Device::select_for_workload(&WorkloadHint::General);
        let pref = Device::select_with_preference(Some(Device::Auto), &WorkloadHint::General);
        assert_eq!(auto, pref);
    }

    #[test]
    fn test_select_with_preference_cpu_always_honoured() {
        let dev = Device::select_with_preference(Some(Device::CPU), &WorkloadHint::LargeMatrices);
        assert_eq!(dev, Device::CPU);
    }

    #[test]
    fn test_select_with_preference_unavailable_falls_back() {
        let auto = Device::select_for_workload(&WorkloadHint::General);
        let pref = Device::select_with_preference(Some(Device::TPU), &WorkloadHint::General);
        assert_eq!(auto, pref);
    }

    #[test]
    fn test_science_workloads_route_to_gpu_or_cpu() {
        let hints = [
            WorkloadHint::PhysicsForce,
            WorkloadHint::FFT,
            WorkloadHint::EigenDecomp,
            WorkloadHint::LinearSolve,
            WorkloadHint::MonteCarlo,
            WorkloadHint::SparseMath,
        ];
        for hint in &hints {
            let dev = Device::select_for_workload(hint);
            assert!(
                dev == Device::GPU || dev == Device::CPU,
                "{:?} should route to GPU or CPU, got {:?}",
                hint,
                dev
            );
        }
    }

    #[test]
    fn test_runtime_device_discovery_report() {
        let available = Device::available_devices();
        println!("=== Runtime Device Discovery ===");
        for dev in &available {
            let info = dev.info();
            println!(
                "  {:?}: available={}, name={:?}, capabilities={:?}, mem={}GB, units={}",
                dev,
                info.available,
                info.name,
                info.capabilities,
                info.memory_gb,
                info.compute_units
            );
        }
        println!("  GPU detected: {}", Device::GPU.is_available());
        println!("  NPU detected: {}", Device::NPU.is_available());
        println!("  TPU detected: {}", Device::TPU.is_available());

        let hints = [
            WorkloadHint::PhysicsForce,
            WorkloadHint::FFT,
            WorkloadHint::EigenDecomp,
            WorkloadHint::LinearSolve,
            WorkloadHint::Training,
            WorkloadHint::Inference,
            WorkloadHint::PreScreen,
            WorkloadHint::SurrogateEval,
            WorkloadHint::MonteCarlo,
            WorkloadHint::SparseMath,
            WorkloadHint::Reservoir,
            WorkloadHint::SparseEvents,
            WorkloadHint::EventProcessing,
            WorkloadHint::LargeMatrices,
            WorkloadHint::SmallWorkload,
            WorkloadHint::StringOps,
            WorkloadHint::General,
        ];
        println!("=== Workload Routing ===");
        for hint in &hints {
            let auto = Device::select_for_workload(hint);
            let forced_cpu = Device::select_with_preference(Some(Device::CPU), hint);
            println!("  {:?}: auto={:?}, forced_cpu={:?}", hint, auto, forced_cpu);
        }

        assert!(available.contains(&Device::CPU));
    }
}
