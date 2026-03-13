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

/// Information about a GPU bound to `vfio-pci`, suitable for handoff to
/// toadStool or `CoralReefDevice::from_vfio_device`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VfioGpuInfo {
    /// PCI address (e.g. `"0000:01:00.0"`).
    pub pci_address: String,
    /// PCI vendor ID (e.g. `0x10de` for NVIDIA).
    pub vendor_id: u16,
    /// PCI device ID.
    pub device_id: u16,
    /// IOMMU group number.
    pub iommu_group: u32,
}

/// Check if any GPU is bound to `vfio-pci` (available for sovereign VFIO dispatch).
///
/// Scans `/sys/kernel/iommu_groups/*/devices/*/` for devices with a known
/// GPU vendor ID (NVIDIA `0x10de`, AMD `0x1002`, Intel `0x8086`) whose
/// `driver` symlink resolves to `vfio-pci`.
#[must_use]
pub fn is_vfio_gpu_available() -> bool {
    !discover_vfio_gpus().is_empty()
}

/// Discover all GPUs currently bound to `vfio-pci`.
///
/// Returns a `Vec<VfioGpuInfo>` for each GPU found. Empty if no VFIO GPUs
/// are available (no IOMMU, no GPUs bound to `vfio-pci`, or non-Linux).
#[must_use]
pub fn discover_vfio_gpus() -> Vec<VfioGpuInfo> {
    #[cfg(target_os = "linux")]
    {
        discover_vfio_gpus_linux()
    }
    #[cfg(not(target_os = "linux"))]
    {
        Vec::new()
    }
}

#[cfg(target_os = "linux")]
fn discover_vfio_gpus_linux() -> Vec<VfioGpuInfo> {
    const GPU_VENDOR_NVIDIA: &str = "0x10de";
    const GPU_VENDOR_AMD: &str = "0x1002";
    const GPU_VENDOR_INTEL: &str = "0x8086";

    let mut results = Vec::new();
    let iommu_groups = std::path::Path::new("/sys/kernel/iommu_groups");
    let Ok(groups) = std::fs::read_dir(iommu_groups) else {
        return results;
    };
    for group_entry in groups.flatten() {
        let group_name = group_entry.file_name();
        let Ok(iommu_group) = group_name.to_string_lossy().parse::<u32>() else {
            continue;
        };
        let devices_dir = group_entry.path().join("devices");
        let Ok(devices) = std::fs::read_dir(devices_dir) else {
            continue;
        };
        for dev in devices.flatten() {
            let dev_path = dev.path();
            let pci_address = dev.file_name().to_string_lossy().to_string();

            let Ok(vendor_str) = std::fs::read_to_string(dev_path.join("vendor")) else {
                continue;
            };
            let vendor_trimmed = vendor_str.trim();

            if vendor_trimmed != GPU_VENDOR_NVIDIA
                && vendor_trimmed != GPU_VENDOR_AMD
                && vendor_trimmed != GPU_VENDOR_INTEL
            {
                continue;
            }

            let driver_link = dev_path.join("driver");
            if let Ok(driver_target) = std::fs::read_link(&driver_link) {
                let driver_name = driver_target
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default();
                if driver_name != "vfio-pci" {
                    continue;
                }
            } else {
                continue;
            }

            let vendor_id =
                u16::from_str_radix(vendor_trimmed.trim_start_matches("0x"), 16).unwrap_or(0);
            let device_id = std::fs::read_to_string(dev_path.join("device"))
                .ok()
                .and_then(|s| u16::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
                .unwrap_or(0);

            results.push(VfioGpuInfo {
                pci_address,
                vendor_id,
                device_id,
                iommu_group,
            });
        }
    }
    results
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
            name: if is_vfio_gpu_available() {
                "Sovereign (coralReef → VFIO)".to_string()
            } else {
                "Sovereign (coralReef → DRM)".to_string()
            },
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
