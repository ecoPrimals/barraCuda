// SPDX-License-Identifier: AGPL-3.0-only
//! Unified Device Info — high-level capability querying for Device enum.
//!
//! Answers "what can this Device do?" for routing and selection.
//! Contrast with `DeviceCapabilities` (wgpu limits).

use crate::device::device_types::Device;

/// Fallback system memory estimate (GB) when actual detection fails — 64-bit systems.
const FALLBACK_SYSTEM_MEMORY_GB_64BIT: usize = 8;

/// Fallback system memory estimate (GB) when actual detection fails — 32-bit systems.
const FALLBACK_SYSTEM_MEMORY_GB_32BIT: usize = 2;

/// Device information and capabilities for the unified Device enum.
///
/// **Runtime-discovered** — No hardcoding!
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device type
    pub device: Device,

    /// Human-readable name
    pub name: String,

    /// Is this device available?
    pub available: bool,

    /// Device capabilities
    pub capabilities: Vec<Capability>,

    /// Available memory (GB)
    pub memory_gb: usize,

    /// Number of compute units (cores, SMs, etc.)
    pub compute_units: usize,
}

/// Device capabilities for unified device selection.
///
/// **Capability-based** — Query at runtime!
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Capability {
    /// General compute
    Compute,

    /// WGSL shader execution
    WGSL,

    /// Parallel execution
    ParallelExecution,

    /// Sparse event processing
    SparseEvents,

    /// Low power operation
    LowPower,

    /// Matrix operations
    MatrixOps,

    /// Memory operations
    Memory,

    /// Automatic device selection
    AutoSelection,
}

/// Check if GPU is available.
///
/// Optimistic — assume GPU might be available. Full runtime check happens at
/// `DeviceContext` creation.
#[must_use]
pub fn is_gpu_available() -> bool {
    true
}

/// Check if NPU is available by scanning for Akida device nodes or VFIO groups.
#[must_use]
pub fn is_npu_available() -> bool {
    // Check for /dev/akida* devices (C kernel driver path)
    if (0..16).any(|i| std::path::Path::new(&format!("/dev/akida{i}")).exists()) {
        return true;
    }
    // Check for VFIO-eligible devices (future pure Rust path)
    // Scan IOMMU groups for BrainChip vendor 0x1e7c
    let iommu_groups = std::path::Path::new("/sys/kernel/iommu_groups");
    if iommu_groups.exists() {
        if let Ok(entries) = std::fs::read_dir(iommu_groups) {
            for entry in entries.flatten() {
                let devices_dir = entry.path().join("devices");
                if let Ok(devices) = std::fs::read_dir(devices_dir) {
                    for dev in devices.flatten() {
                        let vendor_path = dev.path().join("vendor");
                        if let Ok(vendor) = std::fs::read_to_string(vendor_path) {
                            // BrainChip vendor ID
                            if vendor.trim() == "0x1e7c" {
                                return true;
                            }
                        }
                    }
                }
            }
        }
    }
    false
}

/// Estimate system memory (GB) via OS-level detection.
///
/// On Linux: reads `/proc/meminfo`. On macOS: `sysctl hw.memsize`.
/// Falls back to conservative defaults if detection fails.
#[must_use]
pub fn estimate_system_memory() -> usize {
    detect_system_memory_gb().unwrap_or(if cfg!(target_pointer_width = "64") {
        FALLBACK_SYSTEM_MEMORY_GB_64BIT
    } else {
        FALLBACK_SYSTEM_MEMORY_GB_32BIT
    })
}

/// Detect total physical memory in bytes. Returns `None` on unsupported platforms.
#[must_use]
pub fn detect_system_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        detect_memory_linux()
    }
    #[cfg(target_os = "macos")]
    {
        detect_memory_macos()
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

fn detect_system_memory_gb() -> Option<usize> {
    detect_system_memory_bytes().map(|bytes| (bytes / (1024 * 1024 * 1024)) as usize)
}

#[cfg(target_os = "linux")]
fn detect_memory_linux() -> Option<u64> {
    let contents = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in contents.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb_str = rest.trim().strip_suffix("kB")?.trim();
            let kb: u64 = kb_str.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

#[cfg(target_os = "macos")]
fn detect_memory_macos() -> Option<u64> {
    let output = std::process::Command::new("sysctl")
        .arg("-n")
        .arg("hw.memsize")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&output.stdout);
    s.trim().parse::<u64>().ok()
}

/// Build `DeviceInfo` for a given Device.
///
/// **Runtime discovery** — No hardcoding!
#[must_use]
pub fn build_device_info(device: Device) -> DeviceInfo {
    match device {
        Device::CPU => DeviceInfo {
            device,
            name: "CPU".to_string(),
            available: true,
            capabilities: vec![Capability::Compute, Capability::Memory],
            memory_gb: estimate_system_memory(),
            compute_units: std::thread::available_parallelism()
                .map(std::num::NonZero::get)
                .unwrap_or(4),
        },

        Device::GPU => DeviceInfo {
            device,
            name: "GPU (wgpu)".to_string(),
            available: is_gpu_available(),
            capabilities: vec![
                Capability::Compute,
                Capability::WGSL,
                Capability::ParallelExecution,
            ],
            memory_gb: 0,
            compute_units: 0,
        },

        Device::NPU => DeviceInfo {
            device,
            name: "NPU (Akida)".to_string(),
            available: is_npu_available(),
            capabilities: vec![
                Capability::Compute,
                Capability::SparseEvents,
                Capability::LowPower,
            ],
            memory_gb: 0,
            compute_units: 0,
        },

        Device::TPU => DeviceInfo {
            device,
            name: "TPU".to_string(),
            available: false,
            capabilities: vec![Capability::Compute, Capability::MatrixOps],
            memory_gb: 0,
            compute_units: 0,
        },

        Device::Sovereign => DeviceInfo {
            device,
            name: "Sovereign (coralReef → DRM)".to_string(),
            available: is_sovereign_available(),
            capabilities: vec![
                Capability::Compute,
                Capability::WGSL,
                Capability::ParallelExecution,
            ],
            memory_gb: 0,
            compute_units: 0,
        },

        Device::Auto => DeviceInfo {
            device,
            name: "Auto (smart selection)".to_string(),
            available: true,
            capabilities: vec![Capability::AutoSelection],
            memory_gb: 0,
            compute_units: 0,
        },
    }
}

/// Check if sovereign dispatch is available (feature + hardware).
fn is_sovereign_available() -> bool {
    #[cfg(feature = "sovereign-dispatch")]
    {
        crate::device::coral_reef_device::CoralReefDevice::with_auto_device()
            .map(|d| d.has_dispatch())
            .unwrap_or(false)
    }
    #[cfg(not(feature = "sovereign-dispatch"))]
    false
}
