// SPDX-License-Identifier: AGPL-3.0-or-later
//! ToadStool ↔ BarraCuda Hardware Integration
//!
//! ToadStool discovers hardware. BarraCuda runs math on it.
//!
//! Architecture:
//! - GPU/CPU: Run ANY operation via WGSL shaders (same code, different speed)
//! - NPU: Runs pre-compiled SNN models for inference (not general compute)
//!
//! ToadStool routes workloads to the right hardware based on what each can do.
//! Hardware guides its own performance - same WGSL on GPU vs CPU gives
//! identical results at different speeds.

use super::{DeviceSelection, HardwareWorkload};
use crate::error::{BarracudaError, Result};
use toadstool_core::{HardwareManager, HardwareType};

// ─── Discovery ───────────────────────────────────────────────────────────────

/// Discover all available compute devices via ToadStool
pub fn discover_devices() -> Result<HardwareManager> {
    HardwareManager::discover().map_err(|e| BarracudaError::Device(e.to_string()))
}

/// Check if GPU compute is available
pub fn has_gpu() -> bool {
    HardwareManager::discover()
        .map(|hw| hw.has_gpu())
        .unwrap_or(false)
}

/// Check if NPU compute is available
pub fn has_npu() -> bool {
    HardwareManager::discover()
        .map(|hw| hw.has_npu())
        .unwrap_or(false)
}

// ─── Routing Logic ───────────────────────────────────────────────────────────

/// Select best device for a workload
///
/// The routing logic is honest about what each device can do:
/// - GPU: Best for dense parallel compute (WGSL shaders)
/// - CPU: Fallback for everything (WGSL via software rasterizer)
/// - NPU: Best for sparse SNN inference (pre-compiled models only)
///
/// For general math (matmul, Cholesky, RBF, FFT), always routes to GPU/CPU
/// because NPU cannot run arbitrary compute.
pub fn select_best_device(workload_type: HardwareWorkload) -> Result<DeviceSelection> {
    let hw = HardwareManager::discover().map_err(|e| BarracudaError::Device(e.to_string()))?;

    match workload_type {
        // Dense compute: GPU > CPU (NPU can't run WGSL)
        HardwareWorkload::TensorOps
        | HardwareWorkload::NeuralNetwork
        | HardwareWorkload::ScientificCompute
        | HardwareWorkload::HomomorphicEncryption => {
            if hw.has_gpu() {
                Ok(DeviceSelection::Gpu)
            } else {
                Ok(DeviceSelection::Cpu)
            }
        }

        // Event-driven: NPU > GPU > CPU
        // NPU runs pre-compiled SNN models natively
        // If no NPU, GPU runs SNN simulation via WGSL
        HardwareWorkload::SpikingNetwork | HardwareWorkload::ReservoirComputing => {
            if hw.has_npu() {
                Ok(DeviceSelection::Npu)
            } else if hw.has_gpu() {
                Ok(DeviceSelection::Gpu)
            } else {
                Ok(DeviceSelection::Cpu)
            }
        }

        // Mixed workloads: NPU for sparse filtering, GPU for dense compute
        // Route to NPU if available (sparse patterns), else GPU
        HardwareWorkload::Genomics | HardwareWorkload::Bioinformatics => {
            if hw.has_npu() {
                Ok(DeviceSelection::Npu)
            } else if hw.has_gpu() {
                Ok(DeviceSelection::Gpu)
            } else {
                Ok(DeviceSelection::Cpu)
            }
        }
    }
}

/// Select device with explicit hardware preference
///
/// Tries the requested hardware first, falls back if unavailable.
/// Always returns a usable device - never fails on working systems.
pub fn select_device_prefer(preferred: DeviceSelection) -> Result<DeviceSelection> {
    let hw = HardwareManager::discover().map_err(|e| BarracudaError::Device(e.to_string()))?;

    match preferred {
        DeviceSelection::Gpu => {
            if hw.has_gpu() {
                Ok(DeviceSelection::Gpu)
            } else {
                Ok(DeviceSelection::Cpu)
            }
        }
        DeviceSelection::Npu => {
            if hw.has_npu() {
                Ok(DeviceSelection::Npu)
            } else if hw.has_gpu() {
                Ok(DeviceSelection::Gpu)
            } else {
                Ok(DeviceSelection::Cpu)
            }
        }
        DeviceSelection::Cpu => Ok(DeviceSelection::Cpu),
    }
}

/// Get a full hardware report
///
/// Returns what ToadStool found and what BarraCuda can target.
pub fn hardware_report() -> Result<HardwareReport> {
    let hw = HardwareManager::discover().map_err(|e| BarracudaError::Device(e.to_string()))?;
    let registry = super::registry::DeviceRegistry::global();

    let gpu_count = hw.devices_by_type(HardwareType::Gpu).len();
    let npu_count = hw.devices_by_type(HardwareType::Npu).len();

    // Use registry for deduplicated physical device counts
    let physical_gpu_count = registry.gpus().count();
    let physical_discrete_gpu_count = registry.discrete_gpus().count();
    let f64_capable_count = registry.f64_capable().count();

    // Raw adapter counts (may include duplicates due to multiple backends)
    let raw_adapter_count = registry.all_adapter_infos().len();
    let wgpu_cpu_count = registry
        .all_adapter_infos()
        .iter()
        .filter(|a| a.device_type == wgpu::DeviceType::Cpu)
        .count();

    // Collect device info from registry (deduplicated)
    let device_infos: Vec<_> = registry
        .physical_devices()
        .map(|d| {
            let backends: Vec<_> = d
                .backends
                .iter()
                .map(|b| format!("{:?}", b.backend))
                .collect();
            PhysicalDeviceInfo {
                name: d.name.clone(),
                vendor: d.vendor.name().to_string(),
                device_type: d.device_type,
                f64_capable: d.capabilities.f64_shaders,
                backends,
                preferred_backend: d.backends.first().map(|b| format!("{:?}", b.backend)),
            }
        })
        .collect();

    Ok(HardwareReport {
        toadstool_devices: hw.device_count(),
        gpus_discovered: gpu_count,
        npus_discovered: npu_count,
        // Registry-based counts (deduplicated)
        physical_devices: registry.device_count(),
        physical_gpus: physical_gpu_count,
        physical_discrete_gpus: physical_discrete_gpu_count,
        f64_capable_gpus: f64_capable_count,
        // Raw adapter counts (may have duplicates)
        wgpu_adapters: raw_adapter_count,
        wgpu_cpu_adapters: wgpu_cpu_count,
        // Capabilities
        can_run_wgsl_on_gpu: physical_gpu_count > 0,
        can_run_wgsl_on_cpu: wgpu_cpu_count > 0,
        can_run_npu_inference: npu_count > 0,
        // Device details
        devices: device_infos,
    })
}

/// Information about a physical device
#[derive(Debug, Clone)]
pub struct PhysicalDeviceInfo {
    /// Device name
    pub name: String,
    /// Vendor name
    pub vendor: String,
    /// Device type (discrete, integrated, etc.)
    pub device_type: wgpu::DeviceType,
    /// Supports f64 shaders
    pub f64_capable: bool,
    /// Available backends (Vulkan, Metal, etc.)
    pub backends: Vec<String>,
    /// Preferred backend (first in list)
    pub preferred_backend: Option<String>,
}

/// Summary of available hardware
///
/// **Important**: Physical device counts are deduplicated. The same GPU appearing
/// through multiple backends (Vulkan, OpenCL) is counted ONCE.
#[derive(Debug, Clone)]
pub struct HardwareReport {
    /// Total devices ToadStool found (GPUs + NPUs + CPU)
    pub toadstool_devices: usize,
    /// GPU devices found via sysfs
    pub gpus_discovered: usize,
    /// NPU devices found via PCIe scan
    pub npus_discovered: usize,

    // --- Registry-based counts (deduplicated by physical hardware) ---
    /// Unique physical compute devices (deduplicated across backends)
    pub physical_devices: usize,
    /// Unique physical GPUs (discrete + integrated, deduplicated)
    pub physical_gpus: usize,
    /// Unique discrete GPUs only (deduplicated)
    pub physical_discrete_gpus: usize,
    /// GPUs with f64 shader support
    pub f64_capable_gpus: usize,

    // --- Raw adapter counts (may include duplicates) ---
    /// Total WGPU adapters (may include duplicates from multiple backends)
    pub wgpu_adapters: usize,
    /// WGPU adapters that are CPU software rasterizers
    pub wgpu_cpu_adapters: usize,

    // --- Capabilities ---
    /// Can run WGSL on GPU hardware?
    pub can_run_wgsl_on_gpu: bool,
    /// Can run WGSL on CPU (software rasterizer)?
    pub can_run_wgsl_on_cpu: bool,
    /// Can run pre-compiled SNN models on NPU?
    pub can_run_npu_inference: bool,

    /// Detailed info for each physical device
    pub devices: Vec<PhysicalDeviceInfo>,
}

impl std::fmt::Display for HardwareReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "ToadStool + BarraCuda Hardware Report")?;
        writeln!(f, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")?;
        writeln!(f)?;
        writeln!(f, "ToadStool Discovery:")?;
        writeln!(f, "  Total devices: {}", self.toadstool_devices)?;
        writeln!(f, "    GPUs (sysfs): {}", self.gpus_discovered)?;
        writeln!(f, "    NPUs (PCIe):  {}", self.npus_discovered)?;
        writeln!(f)?;
        writeln!(f, "Physical Devices (deduplicated):")?;
        writeln!(f, "  Total:          {}", self.physical_devices)?;
        writeln!(
            f,
            "  GPUs:           {} ({} discrete)",
            self.physical_gpus, self.physical_discrete_gpus
        )?;
        writeln!(f, "  f64-capable:    {}", self.f64_capable_gpus)?;
        writeln!(f)?;
        writeln!(
            f,
            "Raw WGPU Adapters: {} (may include duplicates)",
            self.wgpu_adapters
        )?;
        writeln!(f)?;
        writeln!(f, "Capabilities:")?;
        writeln!(
            f,
            "  WGSL on GPU:    {}",
            if self.can_run_wgsl_on_gpu {
                "✓"
            } else {
                "✗"
            }
        )?;
        writeln!(
            f,
            "  WGSL on CPU:    {}",
            if self.can_run_wgsl_on_cpu {
                "✓"
            } else {
                "✗"
            }
        )?;
        writeln!(
            f,
            "  NPU inference:  {}",
            if self.can_run_npu_inference {
                "✓"
            } else {
                "✗"
            }
        )?;
        writeln!(f)?;
        writeln!(f, "Devices:")?;
        for (idx, device) in self.devices.iter().enumerate() {
            let f64_mark = if device.f64_capable { "f64" } else { "f32" };
            let preferred = device.preferred_backend.as_deref().unwrap_or("none");
            writeln!(
                f,
                "  [{}] {} ({}, {:?}, {}, backends: {})",
                idx,
                device.name,
                device.vendor,
                device.device_type,
                f64_mark,
                device.backends.join("/")
            )?;
            writeln!(f, "      Preferred backend: {preferred}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_discovery() {
        let hw = discover_devices().expect("Failed to discover devices");
        // CPU always available
        assert!(!hw.devices().is_empty());
    }

    #[test]
    fn test_device_selection_tensor_ops() {
        let selection = select_best_device(HardwareWorkload::TensorOps).unwrap();
        // Should route to GPU or CPU, never NPU (can't run WGSL)
        assert!(selection.supports_wgsl());
    }

    #[test]
    fn test_device_selection_scientific() {
        let selection = select_best_device(HardwareWorkload::ScientificCompute).unwrap();
        assert!(selection.supports_wgsl());
    }

    #[test]
    fn test_device_selection_spiking() {
        let selection = select_best_device(HardwareWorkload::SpikingNetwork);
        // Routes to NPU if available, else GPU/CPU - always succeeds
        assert!(selection.is_ok());
    }

    #[test]
    fn test_wgsl_support() {
        assert!(DeviceSelection::Gpu.supports_wgsl());
        assert!(DeviceSelection::Cpu.supports_wgsl());
        assert!(!DeviceSelection::Npu.supports_wgsl());
    }

    #[test]
    fn test_event_driven() {
        assert!(!DeviceSelection::Gpu.is_event_driven());
        assert!(!DeviceSelection::Cpu.is_event_driven());
        assert!(DeviceSelection::Npu.is_event_driven());
    }

    #[test]
    fn test_prefer_gpu() {
        let selection = select_device_prefer(DeviceSelection::Gpu).unwrap();
        // If GPU exists, get GPU. Otherwise CPU.
        assert!(selection.supports_wgsl());
    }

    #[test]
    fn test_prefer_cpu() {
        let selection = select_device_prefer(DeviceSelection::Cpu).unwrap();
        assert_eq!(selection, DeviceSelection::Cpu);
    }

    #[test]
    fn test_hardware_report() {
        let report = hardware_report().unwrap();
        println!("{}", report);
        assert!(report.toadstool_devices > 0, "Should find at least CPU");
        assert!(
            report.wgpu_adapters > 0,
            "Should find at least one WGPU adapter"
        );
    }
}
