// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pipeline Cache for BarraCuda
//!
//! Caches shader modules, bind group layouts, and compute pipelines
//! to eliminate redundant GPU object creation.
//!
//! This is critical for achieving CUDA parity - native CUDA only
//! compiles kernels once, while naive wgpu creates them every dispatch.
//!
//! Note: Each wgpu Device has its own cache because GPU objects are
//! not transferable between devices.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};
use wgpu::{BindGroupLayout, ComputePipeline, Device, ShaderModule};

/// Recover from RwLock poison on read. Safe for caches: worst case is a
/// stale entry or cache miss, never data corruption.
fn read_or_recover<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(|e| e.into_inner())
}

/// Recover from RwLock poison on write. Safe for caches: a previously
/// panicked thread may have left partial state, but inserting over it
/// is correct behavior for a cache.
fn write_or_recover<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    lock.write().unwrap_or_else(|e| e.into_inner())
}

/// Unique device identifier that works across wgpu instances
///
/// wgpu's `global_id()` is only unique within a single Instance.
/// Since GpuPool creates devices from different instances, we use
/// a hash of the device's physical characteristics instead.
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct DeviceFingerprint {
    /// Hash of adapter name
    name_hash: u64,
    /// Hash of backend + device type
    backend_hash: u64,
}

impl DeviceFingerprint {
    /// Create fingerprint from adapter info
    pub fn from_device_info(device: &Device, adapter_info: &wgpu::AdapterInfo) -> Self {
        use std::hash::Hasher;

        // Hash the adapter name
        let mut name_hasher = std::collections::hash_map::DefaultHasher::new();
        adapter_info.name.hash(&mut name_hasher);
        let name_hash = name_hasher.finish();

        // Hash backend + device type for additional uniqueness
        let mut backend_hasher = std::collections::hash_map::DefaultHasher::new();
        format!("{:?}:{:?}", adapter_info.backend, adapter_info.device_type)
            .hash(&mut backend_hasher);
        // Also include the wgpu global_id for uniqueness within the same instance
        device.global_id().hash(&mut backend_hasher);
        let backend_hash = backend_hasher.finish();

        Self {
            name_hash,
            backend_hash,
        }
    }

    /// Create from just adapter info (for quick lookups)
    pub fn from_adapter_info(adapter_info: &wgpu::AdapterInfo) -> Self {
        use std::hash::Hasher;

        let mut name_hasher = std::collections::hash_map::DefaultHasher::new();
        adapter_info.name.hash(&mut name_hasher);
        let name_hash = name_hasher.finish();

        let mut backend_hasher = std::collections::hash_map::DefaultHasher::new();
        format!("{:?}:{:?}", adapter_info.backend, adapter_info.device_type)
            .hash(&mut backend_hasher);
        let backend_hash = backend_hasher.finish();

        Self {
            name_hash,
            backend_hash,
        }
    }
}

/// Key for caching shader modules (includes device fingerprint)
#[derive(Clone, Hash, PartialEq, Eq)]
pub struct ShaderKey {
    /// Hash of shader source
    source_hash: u64,
    /// Device fingerprint (unique per physical GPU)
    device_fingerprint: DeviceFingerprint,
}

impl ShaderKey {
    pub fn new(source: &str, device_fingerprint: DeviceFingerprint) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        source.hash(&mut hasher);
        Self {
            source_hash: hasher.finish(),
            device_fingerprint,
        }
    }
}

/// Bind group layout signature (without device - used for creating keys)
#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct BindGroupLayoutSignature {
    /// Number of read-only storage buffers
    pub read_only_buffers: u32,
    /// Number of read-write storage buffers
    pub read_write_buffers: u32,
    /// Number of uniform buffers
    pub uniform_buffers: u32,
}

impl BindGroupLayoutSignature {
    /// Standard elementwise binary op (2 read, 1 write)
    pub fn elementwise_binary() -> Self {
        Self {
            read_only_buffers: 2,
            read_write_buffers: 1,
            uniform_buffers: 0,
        }
    }

    /// Standard unary op (1 read, 1 write)
    pub fn elementwise_unary() -> Self {
        Self {
            read_only_buffers: 1,
            read_write_buffers: 1,
            uniform_buffers: 0,
        }
    }

    /// Reduction op (1 read, 1 write, 1 uniform for params)
    pub fn reduction() -> Self {
        Self {
            read_only_buffers: 1,
            read_write_buffers: 1,
            uniform_buffers: 1,
        }
    }

    /// Matmul (2 read, 1 write, 1 uniform for dimensions)
    pub fn matmul() -> Self {
        Self {
            read_only_buffers: 2,
            read_write_buffers: 1,
            uniform_buffers: 1,
        }
    }

    /// Ternary op like FMA: d = a * b + c (3 read, 1 write)
    pub fn elementwise_ternary() -> Self {
        Self {
            read_only_buffers: 3,
            read_write_buffers: 1,
            uniform_buffers: 0,
        }
    }
}

/// Key for caching bind group layouts (includes device fingerprint)
#[derive(Clone, Hash, PartialEq, Eq)]
pub struct BindGroupLayoutKey {
    signature: BindGroupLayoutSignature,
    device_fingerprint: DeviceFingerprint,
}

impl BindGroupLayoutKey {
    pub fn new(signature: BindGroupLayoutSignature, device_fingerprint: DeviceFingerprint) -> Self {
        Self {
            signature,
            device_fingerprint,
        }
    }
}

/// Key for caching compute pipelines (includes device fingerprint)
#[derive(Clone, Hash, PartialEq, Eq)]
pub struct PipelineKey {
    source_hash: u64,
    layout_signature: BindGroupLayoutSignature,
    entry_point: String,
    device_fingerprint: DeviceFingerprint,
}

impl PipelineKey {
    pub fn new(
        shader_source: &str,
        layout_signature: BindGroupLayoutSignature,
        entry_point: &str,
        device_fingerprint: DeviceFingerprint,
    ) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        shader_source.hash(&mut hasher);
        Self {
            source_hash: hasher.finish(),
            layout_signature,
            entry_point: entry_point.to_string(),
            device_fingerprint,
        }
    }
}

/// Thread-safe pipeline cache — evolved from DashMap to stdlib `RwLock<HashMap>`.
///
/// Access pattern: very frequent reads (cache hits), rare writes (first-use only).
/// RwLock's read-write semantics are ideal: unlimited concurrent readers, exclusive writer.
///
/// Note: Keys include device fingerprint to ensure GPU objects are only used
/// with the device that created them.
pub struct PipelineCache {
    /// Cached shader modules (keyed by source hash + device)
    shaders: RwLock<HashMap<ShaderKey, Arc<ShaderModule>>>,

    /// Cached bind group layouts (keyed by signature + device)
    layouts: RwLock<HashMap<BindGroupLayoutKey, Arc<BindGroupLayout>>>,

    /// Cached compute pipelines (keyed by shader + layout + entry + device)
    pipelines: RwLock<HashMap<PipelineKey, Arc<ComputePipeline>>>,
}

impl PipelineCache {
    pub fn new() -> Self {
        Self {
            shaders: RwLock::new(HashMap::new()),
            layouts: RwLock::new(HashMap::new()),
            pipelines: RwLock::new(HashMap::new()),
        }
    }

    /// Get or compile a shader module
    ///
    /// Uses device + adapter_info to create a fingerprint unique per device instance.
    /// This ensures layouts/pipelines from different wgpu::Device instances don't collide.
    pub fn get_or_compile_shader(
        &self,
        device: &Device,
        adapter_info: &wgpu::AdapterInfo,
        source: &str,
        label: Option<&str>,
    ) -> Arc<ShaderModule> {
        let fingerprint = DeviceFingerprint::from_device_info(device, adapter_info);
        let key = ShaderKey::new(source, fingerprint);

        // Fast path: already compiled.
        if let Some(m) = read_or_recover(&self.shaders).get(&key) {
            return m.clone();
        }
        // Slow path: compile and cache.
        let module = Arc::new(device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label,
            source: wgpu::ShaderSource::Wgsl(source.into()),
        }));
        write_or_recover(&self.shaders)
            .entry(key)
            .or_insert(module)
            .clone()
    }

    /// Get or create a bind group layout
    ///
    /// Uses device + adapter_info to create a fingerprint unique per device instance.
    pub fn get_or_create_layout(
        &self,
        device: &Device,
        adapter_info: &wgpu::AdapterInfo,
        signature: BindGroupLayoutSignature,
        label: Option<&str>,
    ) -> Arc<BindGroupLayout> {
        let fingerprint = DeviceFingerprint::from_device_info(device, adapter_info);
        let key = BindGroupLayoutKey::new(signature, fingerprint);

        // Fast path: already created.
        if let Some(l) = read_or_recover(&self.layouts).get(&key) {
            return l.clone();
        }
        // Slow path: create from signature.
        let layout = {
            let mut entries = Vec::new();
            let mut binding = 0u32;

            // Read-only storage buffers
            for _ in 0..signature.read_only_buffers {
                entries.push(wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                });
                binding += 1;
            }

            // Read-write storage buffers
            for _ in 0..signature.read_write_buffers {
                entries.push(wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                });
                binding += 1;
            }

            // Uniform buffers
            for _ in 0..signature.uniform_buffers {
                entries.push(wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                });
                binding += 1;
            }

            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label,
                entries: &entries,
            });
            Arc::new(layout)
        };
        write_or_recover(&self.layouts)
            .entry(key)
            .or_insert(layout)
            .clone()
    }

    /// Get or create a compute pipeline
    ///
    /// Uses adapter_info to create a device fingerprint that's unique per physical GPU.
    pub fn get_or_create_pipeline(
        &self,
        device: &Device,
        adapter_info: &wgpu::AdapterInfo,
        shader_source: &str,
        layout_signature: BindGroupLayoutSignature,
        entry_point: &str,
        label: Option<&str>,
    ) -> Arc<ComputePipeline> {
        let fingerprint = DeviceFingerprint::from_device_info(device, adapter_info);
        let key = PipelineKey::new(shader_source, layout_signature, entry_point, fingerprint);

        // Fast path: already compiled.
        if let Some(p) = read_or_recover(&self.pipelines).get(&key) {
            return p.clone();
        }
        // Slow path: compile shaders, build pipeline layout, create pipeline.
        let shader = self.get_or_compile_shader(device, adapter_info, shader_source, label);
        let layout = self.get_or_create_layout(device, adapter_info, layout_signature, label);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label,
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = Arc::new(
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label,
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point,
                cache: None,
                compilation_options: Default::default(),
            }),
        );
        write_or_recover(&self.pipelines)
            .entry(key)
            .or_insert(pipeline)
            .clone()
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            shaders: read_or_recover(&self.shaders).len(),
            layouts: read_or_recover(&self.layouts).len(),
            pipelines: read_or_recover(&self.pipelines).len(),
        }
    }

    /// Clear all cached objects
    pub fn clear(&self) {
        write_or_recover(&self.shaders).clear();
        write_or_recover(&self.layouts).clear();
        write_or_recover(&self.pipelines).clear();
    }
}

impl Default for PipelineCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub shaders: usize,
    pub layouts: usize,
    pub pipelines: usize,
}

// Global pipeline cache (singleton pattern via std::sync::LazyLock)
// Evolved from lazy_static to pure std (Rust 1.80+)
pub static GLOBAL_CACHE: std::sync::LazyLock<PipelineCache> =
    std::sync::LazyLock::new(PipelineCache::new);

/// Clear the global pipeline cache (for testing only)
///
/// This clears all cached shaders, layouts, and pipelines.
/// Should only be used in tests to ensure isolation.
pub fn clear_global_cache() {
    GLOBAL_CACHE.clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fingerprint_consistency() {
        let adapter_info = wgpu::AdapterInfo {
            name: "Test GPU".to_string(),
            vendor: 0,
            device: 0,
            device_type: wgpu::DeviceType::DiscreteGpu,
            driver: "test".to_string(),
            driver_info: "1.0".to_string(),
            backend: wgpu::Backend::Vulkan,
        };

        let fp1 = DeviceFingerprint::from_adapter_info(&adapter_info);
        let fp2 = DeviceFingerprint::from_adapter_info(&adapter_info);
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_different_gpus_different_fingerprints() {
        let nvidia = wgpu::AdapterInfo {
            name: "NVIDIA RTX 3090".to_string(),
            vendor: 0,
            device: 0,
            device_type: wgpu::DeviceType::DiscreteGpu,
            driver: "nvidia".to_string(),
            driver_info: "1.0".to_string(),
            backend: wgpu::Backend::Vulkan,
        };

        let amd = wgpu::AdapterInfo {
            name: "AMD RX 6950 XT".to_string(),
            vendor: 0,
            device: 0,
            device_type: wgpu::DeviceType::DiscreteGpu,
            driver: "radv".to_string(),
            driver_info: "1.0".to_string(),
            backend: wgpu::Backend::Vulkan,
        };

        let fp_nvidia = DeviceFingerprint::from_adapter_info(&nvidia);
        let fp_amd = DeviceFingerprint::from_adapter_info(&amd);
        assert_ne!(
            fp_nvidia, fp_amd,
            "Different GPUs should have different fingerprints"
        );
    }

    #[test]
    fn test_layout_signature_presets() {
        let binary = BindGroupLayoutSignature::elementwise_binary();
        assert_eq!(binary.read_only_buffers, 2);
        assert_eq!(binary.read_write_buffers, 1);
    }
}
