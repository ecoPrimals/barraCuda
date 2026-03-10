// SPDX-License-Identifier: AGPL-3.0-only
//! Pipeline Cache for `BarraCuda`
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

/// Recover from `RwLock` poison on read. Safe for caches: worst case is a
/// stale entry or cache miss, never data corruption.
fn read_or_recover<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    lock.read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Recover from `RwLock` poison on write. Safe for caches: a previously
/// panicked thread may have left partial state, but inserting over it
/// is correct behavior for a cache.
fn write_or_recover<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    lock.write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Unique device identifier that works across wgpu instances
///
/// wgpu's `global_id()` is only unique within a single Instance.
/// Since `GpuPool` creates devices from different instances, we use
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
    #[must_use]
    pub fn from_device_info(device: &Device, adapter_info: &wgpu::AdapterInfo) -> Self {
        use std::hash::Hasher;

        let mut name_hasher = std::collections::hash_map::DefaultHasher::new();
        adapter_info.name.hash(&mut name_hasher);
        let name_hash = name_hasher.finish();

        let mut backend_hasher = std::collections::hash_map::DefaultHasher::new();
        std::mem::discriminant(&adapter_info.backend).hash(&mut backend_hasher);
        std::mem::discriminant(&adapter_info.device_type).hash(&mut backend_hasher);
        device.hash(&mut backend_hasher);
        let backend_hash = backend_hasher.finish();

        Self {
            name_hash,
            backend_hash,
        }
    }

    /// Create from just adapter info (for quick lookups)
    #[must_use]
    pub fn from_adapter_info(adapter_info: &wgpu::AdapterInfo) -> Self {
        use std::hash::Hasher;

        let mut name_hasher = std::collections::hash_map::DefaultHasher::new();
        adapter_info.name.hash(&mut name_hasher);
        let name_hash = name_hasher.finish();

        let mut backend_hasher = std::collections::hash_map::DefaultHasher::new();
        std::mem::discriminant(&adapter_info.backend).hash(&mut backend_hasher);
        std::mem::discriminant(&adapter_info.device_type).hash(&mut backend_hasher);
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
    /// Create a shader cache key from source and device fingerprint.
    #[must_use]
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
    #[must_use]
    pub fn elementwise_binary() -> Self {
        Self {
            read_only_buffers: 2,
            read_write_buffers: 1,
            uniform_buffers: 0,
        }
    }

    /// Standard unary op (1 read, 1 write)
    #[must_use]
    pub fn elementwise_unary() -> Self {
        Self {
            read_only_buffers: 1,
            read_write_buffers: 1,
            uniform_buffers: 0,
        }
    }

    /// Reduction op (1 read, 1 write, 1 uniform for params)
    #[must_use]
    pub fn reduction() -> Self {
        Self {
            read_only_buffers: 1,
            read_write_buffers: 1,
            uniform_buffers: 1,
        }
    }

    /// Matmul (2 read, 1 write, 1 uniform for dimensions)
    #[must_use]
    pub fn matmul() -> Self {
        Self {
            read_only_buffers: 2,
            read_write_buffers: 1,
            uniform_buffers: 1,
        }
    }

    /// Ternary op like FMA: d = a * b + c (3 read, 1 write)
    #[must_use]
    pub fn elementwise_ternary() -> Self {
        Self {
            read_only_buffers: 3,
            read_write_buffers: 1,
            uniform_buffers: 0,
        }
    }

    /// Two-input reduction with params: covariance, correlation, cosine similarity
    /// (2 read, 1 rw, 1 uniform). Equivalent to `matmul()` layout.
    #[must_use]
    pub fn two_input_reduction() -> Self {
        Self {
            read_only_buffers: 2,
            read_write_buffers: 1,
            uniform_buffers: 1,
        }
    }

    /// Three-input reduction with params: weighted dot product
    /// (3 read, 1 rw, 1 uniform).
    #[must_use]
    pub fn three_input_reduction() -> Self {
        Self {
            read_only_buffers: 3,
            read_write_buffers: 1,
            uniform_buffers: 1,
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
    /// Create a bind group layout cache key.
    #[must_use]
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
    entry_point_hash: u64,
    device_fingerprint: DeviceFingerprint,
}

impl PipelineKey {
    /// Create a pipeline cache key from shader source, layout, entry point, and device.
    #[must_use]
    pub fn new(
        shader_source: &str,
        layout_signature: BindGroupLayoutSignature,
        entry_point: &str,
        device_fingerprint: DeviceFingerprint,
    ) -> Self {
        let mut src_hasher = std::collections::hash_map::DefaultHasher::new();
        shader_source.hash(&mut src_hasher);
        let mut ep_hasher = std::collections::hash_map::DefaultHasher::new();
        entry_point.hash(&mut ep_hasher);
        Self {
            source_hash: src_hasher.finish(),
            layout_signature,
            entry_point_hash: ep_hasher.finish(),
            device_fingerprint,
        }
    }
}

/// Thread-safe pipeline cache — evolved from `DashMap` to stdlib `RwLock<HashMap>`.
///
/// Access pattern: very frequent reads (cache hits), rare writes (first-use only).
/// `RwLock`'s read-write semantics are ideal: unlimited concurrent readers, exclusive writer.
///
/// Note: Keys include device fingerprint to ensure GPU objects are only used
/// with the device that created them.
#[derive(Default)]
pub struct PipelineCache {
    /// Cached shader modules — f64-canonical, auto-downcast to f32 (keyed by source hash + device)
    shaders: RwLock<HashMap<ShaderKey, Arc<ShaderModule>>>,

    /// Cached shader modules — f64-native, no downcast (keyed by source hash + device)
    shaders_f64: RwLock<HashMap<ShaderKey, Arc<ShaderModule>>>,

    /// Cached bind group layouts (keyed by signature + device)
    layouts: RwLock<HashMap<BindGroupLayoutKey, Arc<BindGroupLayout>>>,

    /// Cached compute pipelines — f64-canonical, auto-downcast (keyed by shader + layout + entry + device)
    pipelines: RwLock<HashMap<PipelineKey, Arc<ComputePipeline>>>,

    /// Cached compute pipelines — f64-native (keyed by shader + layout + entry + device)
    pipelines_f64: RwLock<HashMap<PipelineKey, Arc<ComputePipeline>>>,
}

impl PipelineCache {
    /// Create an empty pipeline cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            shaders: RwLock::new(HashMap::new()),
            shaders_f64: RwLock::new(HashMap::new()),
            layouts: RwLock::new(HashMap::new()),
            pipelines: RwLock::new(HashMap::new()),
            pipelines_f64: RwLock::new(HashMap::new()),
        }
    }

    /// Get or compile a shader module
    ///
    /// Uses device + `adapter_info` to create a fingerprint unique per device instance.
    /// This ensures layouts/pipelines from different `wgpu::Device` instances don't collide.
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
        // Slow path: auto-downcast f64-canonical source to f32 then compile and cache.
        // f64-canonical architecture: shaders are authored in f64 as the source of
        // truth. The cache always downcasts for broad f32 compatibility.
        // Ops that need native f64 must compile via WgpuDevice::compile_shader_f64().
        let resolved: std::borrow::Cow<'_, str> =
            if crate::shaders::precision::compiler::source_is_f64(source) {
                std::borrow::Cow::Owned(
                    crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(source),
                )
            } else {
                std::borrow::Cow::Borrowed(source)
            };
        let module = Arc::new(device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label,
            source: wgpu::ShaderSource::Wgsl((&*resolved).into()),
        }));
        write_or_recover(&self.shaders)
            .entry(key)
            .or_insert(module)
            .clone()
    }

    /// Get or create a bind group layout
    ///
    /// Uses device + `adapter_info` to create a fingerprint unique per device instance.
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
    /// Uses `adapter_info` to create a device fingerprint that's unique per physical GPU.
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
            immediate_size: 0,
        });
        let pipeline = Arc::new(
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label,
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                cache: None,
                compilation_options: Default::default(),
            }),
        );
        write_or_recover(&self.pipelines)
            .entry(key)
            .or_insert(pipeline)
            .clone()
    }

    /// Get or compile a shader module preserving f64 types (no downcast).
    ///
    /// For ops that upload real f64 data and need the shader to read it natively.
    /// Cached separately from the f32-downcast path to avoid key collisions.
    pub fn get_or_compile_shader_f64_native(
        &self,
        device: &Device,
        adapter_info: &wgpu::AdapterInfo,
        source: &str,
        label: Option<&str>,
    ) -> Arc<ShaderModule> {
        let fingerprint = DeviceFingerprint::from_device_info(device, adapter_info);
        let key = ShaderKey::new(source, fingerprint);

        if let Some(m) = read_or_recover(&self.shaders_f64).get(&key) {
            return m.clone();
        }
        let module = Arc::new(device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label,
            source: wgpu::ShaderSource::Wgsl(source.into()),
        }));
        write_or_recover(&self.shaders_f64)
            .entry(key)
            .or_insert(module)
            .clone()
    }

    /// Get or create a compute pipeline preserving f64 types (no downcast).
    ///
    /// For ops that upload real f64 data to GPU buffers. The shader is compiled
    /// as-is, so it must match the buffer data layout. Cached separately from
    /// the f32-downcast path.
    pub fn get_or_create_pipeline_f64_native(
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

        if let Some(p) = read_or_recover(&self.pipelines_f64).get(&key) {
            return p.clone();
        }
        let shader =
            self.get_or_compile_shader_f64_native(device, adapter_info, shader_source, label);
        let layout = self.get_or_create_layout(device, adapter_info, layout_signature, label);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label,
            bind_group_layouts: &[&layout],
            immediate_size: 0,
        });
        let pipeline = Arc::new(
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label,
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                cache: None,
                compilation_options: Default::default(),
            }),
        );
        write_or_recover(&self.pipelines_f64)
            .entry(key)
            .or_insert(pipeline)
            .clone()
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            shaders: read_or_recover(&self.shaders).len()
                + read_or_recover(&self.shaders_f64).len(),
            layouts: read_or_recover(&self.layouts).len(),
            pipelines: read_or_recover(&self.pipelines).len()
                + read_or_recover(&self.pipelines_f64).len(),
        }
    }

    /// Clear all cached objects
    pub fn clear(&self) {
        write_or_recover(&self.shaders).clear();
        write_or_recover(&self.shaders_f64).clear();
        write_or_recover(&self.layouts).clear();
        write_or_recover(&self.pipelines).clear();
        write_or_recover(&self.pipelines_f64).clear();
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of cached shader modules.
    pub shaders: usize,
    /// Number of cached bind group layouts.
    pub layouts: usize,
    /// Number of cached compute pipelines.
    pub pipelines: usize,
}

/// Global pipeline cache (singleton pattern via `std::sync::LazyLock`).
///
/// Evolved from `lazy_static` to pure std (Rust 1.80+).
pub static GLOBAL_CACHE: std::sync::LazyLock<PipelineCache> =
    std::sync::LazyLock::new(PipelineCache::new);

/// Create a pipeline for an op that uploads real f64 data to GPU buffers.
///
/// Auto-selects compilation path based on device capabilities:
/// - Device has `SHADER_F64`: compile shader as-is (f64-native cache)
/// - No `SHADER_F64`: use the f32-downcast cache (caller's `Fp64Strategy`
///   should have already selected a DF64 shader variant)
pub fn create_f64_data_pipeline(
    device: &crate::device::WgpuDevice,
    shader_src: &str,
    layout_sig: BindGroupLayoutSignature,
    entry: &str,
    label: Option<&str>,
) -> std::sync::Arc<wgpu::ComputePipeline> {
    let adapter_info = device.adapter_info();
    let needs_native = device
        .device()
        .features()
        .contains(wgpu::Features::SHADER_F64)
        && crate::shaders::precision::compiler::source_is_f64(shader_src);

    if needs_native {
        GLOBAL_CACHE.get_or_create_pipeline_f64_native(
            device.device(),
            adapter_info,
            shader_src,
            layout_sig,
            entry,
            label,
        )
    } else {
        GLOBAL_CACHE.get_or_create_pipeline(
            device.device(),
            adapter_info,
            shader_src,
            layout_sig,
            entry,
            label,
        )
    }
}

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
            device_pci_bus_id: String::new(),
            subgroup_min_size: 1,
            subgroup_max_size: 128,
            transient_saves_memory: false,
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
            device_pci_bus_id: String::new(),
            subgroup_min_size: 1,
            subgroup_max_size: 128,
            transient_saves_memory: false,
        };

        let amd = wgpu::AdapterInfo {
            name: "AMD RX 6950 XT".to_string(),
            vendor: 0,
            device: 0,
            device_type: wgpu::DeviceType::DiscreteGpu,
            driver: "radv".to_string(),
            driver_info: "1.0".to_string(),
            backend: wgpu::Backend::Vulkan,
            device_pci_bus_id: String::new(),
            subgroup_min_size: 1,
            subgroup_max_size: 128,
            transient_saves_memory: false,
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
