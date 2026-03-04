// SPDX-License-Identifier: AGPL-3.0-or-later
//! Device Registry — Physical device tracking with backend preference
//!
//! This module provides deduplication of physical devices that may appear
//! multiple times through different backends (Vulkan, OpenCL, Metal, DX12).
//!
//! **Problem**: wgpu with `Backends::all()` returns the same physical GPU
//! as multiple adapters (e.g., RTX 3090 via Vulkan AND OpenCL).
//!
//! **Solution**: Track physical devices by (vendor, device_id) and aggregate
//! their backend capabilities. Prefer Vulkan for ecoPrimals workloads.
//!
//! # Example
//!
//! ```rust,ignore
//! use barracuda::device::registry::DeviceRegistry;
//!
//! let registry = DeviceRegistry::discover();
//!
//! // Get unique physical devices (deduplicated)
//! for device in registry.physical_devices() {
//!     println!("{}: {} backends available",
//!         device.name,
//!         device.backends.len()
//!     );
//! }
//!
//! // Get the best adapter for a physical device (prefers Vulkan)
//! let device = registry.get_preferred_adapter(0)?;
//! ```

use std::collections::HashMap;
use std::sync::OnceLock;

/// Global device registry (lazily initialized)
static GLOBAL_REGISTRY: OnceLock<DeviceRegistry> = OnceLock::new();

/// Backend preference order for ecoPrimals
/// Vulkan is preferred for cross-platform compatibility and f64 support
const BACKEND_PREFERENCE: &[wgpu::Backend] = &[
    wgpu::Backend::Vulkan,        // Best f64 support, cross-platform
    wgpu::Backend::Metal,         // macOS native
    wgpu::Backend::Dx12,          // Windows native
    wgpu::Backend::Gl,            // Legacy fallback
    wgpu::Backend::BrowserWebGpu, // WASM
];

/// A physical compute device (GPU, NPU, etc.)
///
/// This represents one physical piece of hardware, regardless of how many
/// backends can access it. For example, an RTX 3090 accessible via both
/// Vulkan and OpenGL is ONE PhysicalDevice with two backends.
#[derive(Debug, Clone)]
pub struct PhysicalDevice {
    /// Unique identifier: (vendor_id, device_id)
    pub id: PhysicalDeviceId,

    /// Human-readable device name
    pub name: String,

    /// Vendor (NVIDIA, AMD, Intel, Apple, etc.)
    pub vendor: DeviceVendor,

    /// Device type (discrete, integrated, CPU, etc.)
    pub device_type: wgpu::DeviceType,

    /// Available backends for this device
    pub backends: Vec<BackendInfo>,

    /// Combined capabilities across all backends
    pub capabilities: DeviceCapabilities,
}

/// Unique identifier for a physical device
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PhysicalDeviceId {
    pub vendor_id: u32,
    pub device_id: u32,
    /// Normalized device name (for deduplication when device_id is 0)
    pub name_hash: u64,
}

impl PhysicalDeviceId {
    pub fn from_adapter_info(info: &wgpu::AdapterInfo) -> Self {
        // OpenGL backends sometimes report device_id=0
        // In that case, use the device name for deduplication
        let name_hash = if info.device == 0 {
            // Hash the normalized name (remove backend suffixes like "/PCIe/SSE2")
            use std::hash::{Hash, Hasher};
            let normalized = Self::normalize_device_name(&info.name);
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            normalized.hash(&mut hasher);
            hasher.finish()
        } else {
            0 // Not used when device_id is valid
        };

        Self {
            vendor_id: info.vendor,
            device_id: info.device,
            name_hash,
        }
    }

    /// Normalize device name for comparison
    /// Removes backend-specific suffixes like "/PCIe/SSE2", "(RADV NAVI21)", etc.
    fn normalize_device_name(name: &str) -> String {
        // Remove common suffixes
        let name = name.trim();

        // Remove OpenGL-style suffixes: "NVIDIA GeForce RTX 3090/PCIe/SSE2" -> "NVIDIA GeForce RTX 3090"
        let name = if let Some(idx) = name.find("/PCIe") {
            &name[..idx]
        } else {
            name
        };

        // Remove Vulkan driver info: "AMD Radeon RX 6950 XT (RADV NAVI21)" -> "AMD Radeon RX 6950 XT"
        let name = if let Some(idx) = name.find(" (RADV") {
            &name[..idx]
        } else if let Some(idx) = name.find(" (ACO") {
            &name[..idx]
        } else {
            name
        };

        name.trim().to_string()
    }

    /// Check if two IDs likely refer to the same physical device
    pub fn likely_same_device(&self, other: &Self) -> bool {
        // Same vendor is required
        if self.vendor_id != other.vendor_id {
            return false;
        }

        // If both have valid device_id, compare those
        if self.device_id != 0 && other.device_id != 0 {
            return self.device_id == other.device_id;
        }

        // If one has device_id=0, compare name hashes
        // This handles OpenGL backends that don't report device_id
        if self.device_id == 0 || other.device_id == 0 {
            // Get the valid device_id and check if it matches a known pattern
            // For now, assume same vendor + same name hash = same device
            return self.name_hash == other.name_hash && self.name_hash != 0;
        }

        false
    }
}

/// Information about a specific backend for a device
#[derive(Debug, Clone)]
pub struct BackendInfo {
    pub backend: wgpu::Backend,
    pub adapter_index: usize, // Index in wgpu's adapter list
    pub driver: String,
    pub driver_info: String,
}

/// Aggregated device capabilities
#[derive(Debug, Clone, Default)]
pub struct DeviceCapabilities {
    /// Supports f64 (double precision) in shaders
    pub f64_shaders: bool,

    /// Supports f16 (half precision)
    pub f16_shaders: bool,

    /// Maximum buffer size (bytes)
    pub max_buffer_size: u64,

    /// Maximum workgroup size
    pub max_workgroup_size: u32,

    /// Maximum workgroups per dimension
    pub max_workgroups: [u32; 3],

    /// Estimated VRAM (bytes), 0 if unknown
    pub vram_bytes: u64,

    /// Compute capability (NVIDIA-specific, 0.0 if unknown)
    pub compute_capability: f32,
}

/// Known device vendors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceVendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Qualcomm,
    Arm,
    Software, // CPU/software rasterizer
    Unknown,
}

impl DeviceVendor {
    pub fn from_vendor_id(id: u32) -> Self {
        match id {
            0x10DE => Self::Nvidia,
            0x1002 => Self::Amd,
            0x8086 => Self::Intel,
            0x106B => Self::Apple,
            0x5143 => Self::Qualcomm,
            0x13B5 => Self::Arm,
            0 => Self::Software,
            _ => Self::Unknown,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Nvidia => "NVIDIA",
            Self::Amd => "AMD",
            Self::Intel => "Intel",
            Self::Apple => "Apple",
            Self::Qualcomm => "Qualcomm",
            Self::Arm => "ARM",
            Self::Software => "Software",
            Self::Unknown => "Unknown",
        }
    }
}

/// Device registry — tracks all physical devices and their backends
#[derive(Debug)]
pub struct DeviceRegistry {
    /// Physical devices indexed by their ID
    devices: HashMap<PhysicalDeviceId, PhysicalDevice>,

    /// Ordered list of device IDs (for stable iteration)
    device_order: Vec<PhysicalDeviceId>,

    /// Raw adapter infos (for creating devices)
    adapter_infos: Vec<wgpu::AdapterInfo>,
}

impl DeviceRegistry {
    /// Get or create the global device registry
    pub fn global() -> &'static DeviceRegistry {
        GLOBAL_REGISTRY.get_or_init(Self::discover)
    }

    /// Discover all devices and build the registry
    pub fn discover() -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapters: Vec<wgpu::Adapter> =
            pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
        let adapter_infos: Vec<wgpu::AdapterInfo> = adapters.iter().map(|a| a.get_info()).collect();

        let mut devices: HashMap<PhysicalDeviceId, PhysicalDevice> = HashMap::new();
        let mut device_order: Vec<PhysicalDeviceId> = Vec::new();

        for (idx, info) in adapter_infos.iter().enumerate() {
            let id = PhysicalDeviceId::from_adapter_info(info);

            let backend_info = BackendInfo {
                backend: info.backend,
                adapter_index: idx,
                driver: info.driver.clone(),
                driver_info: info.driver_info.clone(),
            };

            // Check if this adapter matches an existing physical device
            // This handles the case where OpenGL reports device_id=0
            let existing_id = Self::find_matching_device(&devices, &id, info);

            if let Some(existing_id) = existing_id {
                // Add this backend to existing device
                if let Some(device) = devices.get_mut(&existing_id) {
                    device.backends.push(backend_info);
                }
            } else {
                // New physical device
                let vendor = DeviceVendor::from_vendor_id(info.vendor);
                let capabilities = Self::probe_capabilities(info);

                let device = PhysicalDevice {
                    id: id.clone(),
                    name: info.name.clone(),
                    vendor,
                    device_type: info.device_type,
                    backends: vec![backend_info],
                    capabilities,
                };

                device_order.push(id.clone());
                devices.insert(id, device);
            }
        }

        // Sort backends by preference for each device
        for device in devices.values_mut() {
            device.backends.sort_by_key(|b| {
                BACKEND_PREFERENCE
                    .iter()
                    .position(|&pref| pref == b.backend)
                    .unwrap_or(usize::MAX)
            });
        }

        // Sort devices: discrete GPUs first, then by vendor preference
        device_order.sort_by(|a, b| {
            let da = devices
                .get(a)
                .expect("device_order entries are keys of devices");
            let db = devices
                .get(b)
                .expect("device_order entries are keys of devices");

            // Discrete GPUs first
            let type_ord_a = Self::device_type_order(da.device_type);
            let type_ord_b = Self::device_type_order(db.device_type);
            if type_ord_a != type_ord_b {
                return type_ord_a.cmp(&type_ord_b);
            }

            // Then by name (stable ordering)
            da.name.cmp(&db.name)
        });

        Self {
            devices,
            device_order,
            adapter_infos,
        }
    }

    /// Find an existing device that matches this adapter
    ///
    /// This handles deduplication when:
    /// 1. Same vendor+device ID (straightforward match)
    /// 2. Same vendor, one has device_id=0 (OpenGL quirk), names match
    fn find_matching_device(
        devices: &HashMap<PhysicalDeviceId, PhysicalDevice>,
        new_id: &PhysicalDeviceId,
        new_info: &wgpu::AdapterInfo,
    ) -> Option<PhysicalDeviceId> {
        // First, try exact match
        if devices.contains_key(new_id) {
            return Some(new_id.clone());
        }

        // If new adapter has device_id=0 (OpenGL quirk), try name-based matching
        if new_id.device_id == 0 {
            let normalized_name = PhysicalDeviceId::normalize_device_name(&new_info.name);

            for (existing_id, existing_device) in devices {
                // Same vendor
                if existing_id.vendor_id != new_id.vendor_id {
                    continue;
                }

                // Check if names match (after normalization)
                let existing_normalized =
                    PhysicalDeviceId::normalize_device_name(&existing_device.name);
                if normalized_name == existing_normalized {
                    return Some(existing_id.clone());
                }

                // Also check if the new name contains the existing device's base name
                // "NVIDIA GeForce RTX 3090/PCIe/SSE2" contains "NVIDIA GeForce RTX 3090"
                if normalized_name.contains(&existing_normalized)
                    || existing_normalized.contains(&normalized_name)
                {
                    return Some(existing_id.clone());
                }
            }
        }

        // If existing device has device_id=0, check if this new adapter matches
        for (existing_id, existing_device) in devices {
            if existing_id.device_id == 0 && existing_id.vendor_id == new_id.vendor_id {
                let normalized_name = PhysicalDeviceId::normalize_device_name(&new_info.name);
                let existing_normalized =
                    PhysicalDeviceId::normalize_device_name(&existing_device.name);

                if normalized_name == existing_normalized
                    || normalized_name.contains(&existing_normalized)
                    || existing_normalized.contains(&normalized_name)
                {
                    return Some(existing_id.clone());
                }
            }
        }

        None
    }

    fn device_type_order(dt: wgpu::DeviceType) -> u8 {
        match dt {
            wgpu::DeviceType::DiscreteGpu => 0,
            wgpu::DeviceType::IntegratedGpu => 1,
            wgpu::DeviceType::VirtualGpu => 2,
            wgpu::DeviceType::Cpu => 3,
            wgpu::DeviceType::Other => 4,
        }
    }

    fn probe_capabilities(info: &wgpu::AdapterInfo) -> DeviceCapabilities {
        // Basic capability detection from adapter info
        // More detailed probing requires creating a device, which we defer

        let vendor = DeviceVendor::from_vendor_id(info.vendor);

        // NVIDIA GPUs: compute capability from device ID ranges
        let compute_capability = if vendor == DeviceVendor::Nvidia {
            Self::nvidia_compute_capability(info.device)
        } else {
            0.0
        };

        // f64 support: NVIDIA with CC >= 6.0, AMD GCN+, Intel
        let f64_shaders = match vendor {
            DeviceVendor::Nvidia => compute_capability >= 6.0,
            DeviceVendor::Amd => true,    // GCN and later support f64
            DeviceVendor::Intel => true,  // Most Intel GPUs support f64
            DeviceVendor::Apple => false, // Apple GPUs don't support f64
            _ => false,
        };

        DeviceCapabilities {
            f64_shaders,
            f16_shaders: true,       // Most modern GPUs support f16
            max_buffer_size: 0,      // Requires device creation to probe
            max_workgroup_size: 256, // Conservative default
            max_workgroups: [65535, 65535, 65535],
            vram_bytes: 0, // Not available from adapter info
            compute_capability,
        }
    }

    fn nvidia_compute_capability(device_id: u32) -> f32 {
        // Approximate compute capability from NVIDIA device ID ranges
        // This is heuristic - actual CC requires CUDA query
        match device_id >> 4 {
            // RTX 40 series (Ada Lovelace) - CC 8.9
            0x260..=0x28F => 8.9,
            // RTX 30 series (Ampere) - CC 8.6
            0x220..=0x25F => 8.6,
            // RTX 20 series (Turing) - CC 7.5
            0x1E0..=0x21F => 7.5,
            // GTX 10 series (Pascal) - CC 6.1
            0x1B0..=0x1DF => 6.1,
            // GTX 9 series (Maxwell) - CC 5.2
            0x130..=0x17F => 5.2,
            _ => 6.0, // Conservative default for unknown NVIDIA
        }
    }

    /// Get all unique physical devices (deduplicated)
    pub fn physical_devices(&self) -> impl Iterator<Item = &PhysicalDevice> {
        self.device_order.iter().map(|id| &self.devices[id])
    }

    /// Get number of unique physical devices
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Get physical device by index (in discovery order)
    pub fn get_device(&self, index: usize) -> Option<&PhysicalDevice> {
        self.device_order.get(index).map(|id| &self.devices[id])
    }

    /// Get physical device by ID
    pub fn get_device_by_id(&self, id: PhysicalDeviceId) -> Option<&PhysicalDevice> {
        self.devices.get(&id)
    }

    /// Get the preferred adapter index for a physical device
    ///
    /// Returns the adapter index for the best backend (Vulkan preferred)
    pub fn get_preferred_adapter_index(&self, device_index: usize) -> Option<usize> {
        self.get_device(device_index)
            .and_then(|d| d.backends.first())
            .map(|b| b.adapter_index)
    }

    /// Get adapter index for a specific backend on a device
    pub fn get_adapter_for_backend(
        &self,
        device_index: usize,
        backend: wgpu::Backend,
    ) -> Option<usize> {
        self.get_device(device_index).and_then(|d| {
            d.backends
                .iter()
                .find(|b| b.backend == backend)
                .map(|b| b.adapter_index)
        })
    }

    /// Get all GPU devices (discrete and integrated)
    pub fn gpus(&self) -> impl Iterator<Item = &PhysicalDevice> {
        self.physical_devices().filter(|d| {
            matches!(
                d.device_type,
                wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
            )
        })
    }

    /// Get discrete GPUs only
    pub fn discrete_gpus(&self) -> impl Iterator<Item = &PhysicalDevice> {
        self.physical_devices()
            .filter(|d| d.device_type == wgpu::DeviceType::DiscreteGpu)
    }

    /// Get devices that support f64
    pub fn f64_capable(&self) -> impl Iterator<Item = &PhysicalDevice> {
        self.physical_devices()
            .filter(|d| d.capabilities.f64_shaders)
    }

    /// Get raw adapter info by index (for wgpu device creation)
    pub fn get_adapter_info(&self, index: usize) -> Option<&wgpu::AdapterInfo> {
        self.adapter_infos.get(index)
    }

    /// Get all raw adapter infos
    pub fn all_adapter_infos(&self) -> &[wgpu::AdapterInfo] {
        &self.adapter_infos
    }

    /// Generate a hardware report
    pub fn report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== Device Registry Report ===\n\n");

        report.push_str(&format!(
            "Physical devices: {} (from {} adapters)\n\n",
            self.device_count(),
            self.adapter_infos.len()
        ));

        for (idx, device) in self.physical_devices().enumerate() {
            report.push_str(&format!(
                "Device {}: {} ({})\n",
                idx,
                device.name,
                device.vendor.name()
            ));
            report.push_str(&format!("  Type: {:?}\n", device.device_type));
            report.push_str(&format!(
                "  ID: vendor=0x{:04X}, device=0x{:04X}\n",
                device.id.vendor_id, device.id.device_id
            ));
            report.push_str(&format!(
                "  f64 support: {}\n",
                device.capabilities.f64_shaders
            ));

            if device.capabilities.compute_capability > 0.0 {
                report.push_str(&format!(
                    "  Compute capability: {:.1}\n",
                    device.capabilities.compute_capability
                ));
            }

            report.push_str("  Backends:\n");
            for (bidx, backend) in device.backends.iter().enumerate() {
                let preferred = if bidx == 0 { " (preferred)" } else { "" };
                report.push_str(&format!(
                    "    - {:?}{}: adapter[{}]\n",
                    backend.backend, preferred, backend.adapter_index
                ));
            }
            report.push('\n');
        }

        report
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_discovery() {
        let registry = DeviceRegistry::discover();

        // Should have at least discovered something
        // (could be CPU-only in CI environments)
        println!("{}", registry.report());

        // Verify no duplicate physical devices
        let mut seen_ids = std::collections::HashSet::new();
        for device in registry.physical_devices() {
            assert!(
                seen_ids.insert(device.id.clone()),
                "Duplicate physical device: {:?}",
                device.id
            );
        }
    }

    #[test]
    fn test_backend_preference() {
        let registry = DeviceRegistry::discover();

        for device in registry.physical_devices() {
            if device.backends.len() > 1 {
                // If multiple backends, Vulkan should be first (if available)
                let has_vulkan = device
                    .backends
                    .iter()
                    .any(|b| b.backend == wgpu::Backend::Vulkan);
                if has_vulkan {
                    assert_eq!(
                        device.backends[0].backend,
                        wgpu::Backend::Vulkan,
                        "Vulkan should be preferred backend for {:?}",
                        device.name
                    );
                }
            }
        }
    }

    #[test]
    fn test_vendor_detection() {
        assert_eq!(DeviceVendor::from_vendor_id(0x10DE), DeviceVendor::Nvidia);
        assert_eq!(DeviceVendor::from_vendor_id(0x1002), DeviceVendor::Amd);
        assert_eq!(DeviceVendor::from_vendor_id(0x8086), DeviceVendor::Intel);
        assert_eq!(DeviceVendor::from_vendor_id(0x106B), DeviceVendor::Apple);
    }
}
