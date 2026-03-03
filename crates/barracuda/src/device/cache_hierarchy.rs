//! Universal Substrate Cache Hierarchy
//!
//! **RUNTIME DISCOVERY, NOT VENDOR HARDCODING**
//!
//! This module discovers cache hierarchies via runtime probing, not hardcoded
//! vendor-specific values. The silicon tells us what it can do.
//!
//! # Design Principles
//!
//! 1. **Probe, don't assume** — Run bandwidth microbenchmarks to find cache boundaries
//! 2. **Adaptive discovery** — Bandwidth inflection points reveal cache levels
//! 3. **Conservative defaults** — If probing fails, use safe generic values
//! 4. **Zero vendor hardcoding** — No "if AMD then 128MB" patterns
//!
//! # Usage
//!
//! ```no_run
//! use barracuda::device::{SubstrateMemoryHierarchy, CacheAwareTiler};
//! use barracuda::prelude::WgpuDevice;
//!
//! # async fn example() -> barracuda::error::Result<()> {
//! let device = WgpuDevice::new().await?;
//! // Discover cache hierarchy via probing
//! let hierarchy = SubstrateMemoryHierarchy::probe(&device).await;
//!
//! // Or use quick estimation from adapter limits
//! let hierarchy = SubstrateMemoryHierarchy::estimate(&device);
//!
//! // Tile workloads to fit cache
//! let tiler = CacheAwareTiler::new(hierarchy);
//! let config = tiler.optimal_tile_size(1024, 8, 3.0);
//! # Ok(())
//! # }
//! ```

use crate::device::{SubstrateType, WgpuDevice};
use std::time::Instant;
use wgpu::util::DeviceExt;

// ── Device memory estimates (conservative defaults; probing refines) ─────────

const DISCRETE_GPU_CACHE_ESTIMATE: u64 = 32 * 1024 * 1024;
const DISCRETE_GPU_BW_GBS: f64 = 500.0;
const DISCRETE_GPU_VRAM_ESTIMATE: u64 = 16 * 1024 * 1024 * 1024;

const INTEGRATED_GPU_CACHE_ESTIMATE: u64 = 16 * 1024 * 1024;
const INTEGRATED_GPU_BW_GBS: f64 = 100.0;
const INTEGRATED_GPU_VRAM_ESTIMATE: u64 = 8 * 1024 * 1024 * 1024;

const CPU_CACHE_ESTIMATE: u64 = 32 * 1024 * 1024;
const CPU_BW_GBS: f64 = 50.0;
const CPU_VRAM_ESTIMATE: u64 = 64 * 1024 * 1024 * 1024;

const FALLBACK_CACHE_ESTIMATE: u64 = 8 * 1024 * 1024;
const FALLBACK_BW_GBS: f64 = 100.0;
const FALLBACK_VRAM_ESTIMATE: u64 = 8 * 1024 * 1024 * 1024;

/// Bandwidth sample point for cache probing
#[derive(Debug, Clone)]
struct BandwidthSample {
    size_bytes: u64,
    bandwidth_gbs: f64,
}

/// Cache level descriptor — discovered at runtime
#[derive(Debug, Clone)]
pub struct CacheLevel {
    /// Cache level name (L1, L2, L3, Cache, etc.)
    pub name: &'static str,

    /// Size in bytes (discovered via probing)
    pub size_bytes: u64,

    /// Observed bandwidth in GB/s
    pub bandwidth_gbs: f64,

    /// Whether this level is shared across compute units
    pub shared: bool,
}

/// Main memory specification
#[derive(Debug, Clone)]
pub struct MainMemory {
    /// Size in bytes (from adapter limits)
    pub size_bytes: u64,

    /// Observed or theoretical bandwidth in GB/s
    pub bandwidth_gbs: f64,
}

/// Discovered memory hierarchy for a substrate
#[derive(Debug, Clone)]
pub struct SubstrateMemoryHierarchy {
    /// Substrate identifier
    pub substrate_name: String,

    /// Substrate type (GPU, CPU, NPU)
    pub substrate_type: SubstrateType,

    /// Discovered cache levels (smallest/fastest to largest/slowest)
    pub cache_levels: Vec<CacheLevel>,

    /// Main memory (DRAM/VRAM)
    pub main_memory: MainMemory,

    /// Computed optimal tile size
    pub optimal_tile_bytes: u64,

    /// Was this hierarchy probed or estimated?
    pub probed: bool,
}

impl SubstrateMemoryHierarchy {
    /// Quick estimation from device info (no probing)
    ///
    /// Uses device type to estimate cache hierarchy.
    /// This is fast but less accurate than probing.
    pub fn estimate(device: &WgpuDevice) -> Self {
        let info = device.adapter_info();

        // Determine substrate type from device info
        let substrate_type = Self::classify_substrate(info);

        // Estimate based on device type, NOT vendor
        // These are conservative defaults — actual probing will refine them
        let (cache_size, main_bw, vram_size) = match info.device_type {
            wgpu::DeviceType::DiscreteGpu => {
                // Discrete GPUs: conservative 32MB cache estimate
                // Real sizes vary from 4MB (old) to 128MB (Infinity Cache)
                // Probing will discover actual boundaries
                (
                    DISCRETE_GPU_CACHE_ESTIMATE,
                    DISCRETE_GPU_BW_GBS,
                    DISCRETE_GPU_VRAM_ESTIMATE,
                )
            }
            wgpu::DeviceType::IntegratedGpu => {
                // iGPUs share system memory, smaller effective cache
                (
                    INTEGRATED_GPU_CACHE_ESTIMATE,
                    INTEGRATED_GPU_BW_GBS,
                    INTEGRATED_GPU_VRAM_ESTIMATE,
                )
            }
            wgpu::DeviceType::Cpu => {
                // CPUs have L3 cache, typically 8-64MB
                (CPU_CACHE_ESTIMATE, CPU_BW_GBS, CPU_VRAM_ESTIMATE)
            }
            _ => {
                // Unknown device type, use conservative defaults
                (
                    FALLBACK_CACHE_ESTIMATE,
                    FALLBACK_BW_GBS,
                    FALLBACK_VRAM_ESTIMATE,
                )
            }
        };

        Self {
            substrate_name: info.name.clone(),
            substrate_type,
            cache_levels: vec![CacheLevel {
                name: "Cache",
                size_bytes: cache_size,
                bandwidth_gbs: main_bw * 2.0, // Cache is typically faster
                shared: true,
            }],
            main_memory: MainMemory {
                size_bytes: vram_size,
                bandwidth_gbs: main_bw,
            },
            optimal_tile_bytes: cache_size / 2, // Target 50% cache utilization
            probed: false,
        }
    }

    /// Probe cache hierarchy via bandwidth microbenchmarks
    ///
    /// This is slower but much more accurate than estimation.
    /// Runs bandwidth tests at increasing data sizes and detects
    /// inflection points that indicate cache boundaries.
    pub async fn probe(device: &WgpuDevice) -> Self {
        let info = device.adapter_info();
        let substrate_type = Self::classify_substrate(info);

        // Probe sizes from 4KB to 512MB (powers of 4)
        let probe_sizes: Vec<u64> = vec![
            4 * 1024,          // 4 KB
            16 * 1024,         // 16 KB
            64 * 1024,         // 64 KB
            256 * 1024,        // 256 KB
            1024 * 1024,       // 1 MB
            4 * 1024 * 1024,   // 4 MB
            16 * 1024 * 1024,  // 16 MB
            64 * 1024 * 1024,  // 64 MB
            256 * 1024 * 1024, // 256 MB
            512 * 1024 * 1024, // 512 MB
        ];

        // Run bandwidth probes
        let mut samples: Vec<BandwidthSample> = Vec::new();
        for &size in &probe_sizes {
            if let Some(bw) = Self::probe_bandwidth(device, size).await {
                samples.push(BandwidthSample {
                    size_bytes: size,
                    bandwidth_gbs: bw,
                });
            }
        }

        // Need at least 3 samples to find inflection points
        if samples.len() < 3 {
            return Self::estimate(device);
        }

        // Find inflection points where bandwidth drops significantly (>20%)
        let mut cache_levels = Vec::new();
        let mut peak_bw = 0.0f64;
        let mut last_cache_bw = samples[0].bandwidth_gbs;

        for i in 0..samples.len() {
            let sample = &samples[i];
            peak_bw = peak_bw.max(sample.bandwidth_gbs);

            // Check for significant bandwidth drop from previous level
            if i > 0 {
                let prev = &samples[i - 1];
                let drop_ratio = sample.bandwidth_gbs / last_cache_bw;

                // Bandwidth drop of >20% indicates cache boundary
                if drop_ratio < 0.8 {
                    // Previous sample was the cache boundary
                    let level_name = match cache_levels.len() {
                        0 => "L2",
                        1 => "L3",
                        _ => "Cache",
                    };

                    cache_levels.push(CacheLevel {
                        name: level_name,
                        size_bytes: prev.size_bytes,
                        bandwidth_gbs: last_cache_bw,
                        shared: cache_levels.is_empty(), // First level typically shared
                    });

                    last_cache_bw = sample.bandwidth_gbs;
                }
            }
        }

        // If no inflection points found, use the highest bandwidth point
        if cache_levels.is_empty() {
            let max_sample = samples.iter().max_by(|a, b| {
                a.bandwidth_gbs
                    .partial_cmp(&b.bandwidth_gbs)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            if let Some(sample) = max_sample {
                cache_levels.push(CacheLevel {
                    name: "Cache",
                    size_bytes: sample.size_bytes,
                    bandwidth_gbs: sample.bandwidth_gbs,
                    shared: true,
                });
            }
        }

        // VRAM bandwidth from largest probe that succeeded
        let main_bw = samples.last().map(|s| s.bandwidth_gbs).unwrap_or(100.0);
        let main_memory = MainMemory {
            size_bytes: device.device().limits().max_buffer_size,
            bandwidth_gbs: main_bw,
        };

        // Optimal tile targets ~70% of largest cache
        let optimal_tile = cache_levels
            .last()
            .map(|c| (c.size_bytes as f64 * 0.7) as u64)
            .unwrap_or(16 * 1024 * 1024);

        Self {
            substrate_name: info.name.clone(),
            substrate_type,
            cache_levels,
            main_memory,
            optimal_tile_bytes: optimal_tile,
            probed: true,
        }
    }

    /// Probe bandwidth at a specific buffer size
    async fn probe_bandwidth(device: &WgpuDevice, size_bytes: u64) -> Option<f64> {
        let elements = (size_bytes / 4) as usize; // f32 = 4 bytes

        // Skip if buffer would exceed device limits
        if size_bytes > device.device().limits().max_buffer_size / 3 {
            return None;
        }

        let shader_src = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) {
        return;
    }
    output[idx] = a[idx] + b[idx];
}
"#;

        let wgpu_device = device.device();
        let queue = device.queue();

        // Create test buffers
        let data: Vec<f32> = (0..elements).map(|i| i as f32 * 0.001).collect();

        let buf_a = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("probe_a"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let buf_b = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("probe_b"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let buf_out = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probe_out"),
            size: size_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let shader = wgpu_device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("probe_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let bgl_entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = wgpu_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[bgl_entry(0, true), bgl_entry(1, true), bgl_entry(2, false)],
        });

        let bufs = [&buf_a, &buf_b, &buf_out];
        let bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &bufs
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });

        let pl = wgpu_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pl),
            module: &shader,
            entry_point: "main",
            cache: None,
            compilation_options: Default::default(),
        });

        let workgroups = (elements as u32).div_ceil(256).min(65_535);

        let run_pass = |poll: bool| {
            let mut enc =
                wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                p.set_pipeline(&pipeline);
                p.set_bind_group(0, &bind_group, &[]);
                p.dispatch_workgroups(workgroups, 1, 1);
            }
            queue.submit(Some(enc.finish()));
            if poll {
                wgpu_device.poll(wgpu::Maintain::Wait);
            }
        };

        for _ in 0..3 {
            run_pass(false);
        }
        wgpu_device.poll(wgpu::Maintain::Wait);

        let iterations = 10;
        let start = Instant::now();
        for _ in 0..iterations {
            run_pass(true);
        }

        let elapsed = start.elapsed();
        let time_per_op = elapsed.as_secs_f64() / iterations as f64;

        // Bandwidth: (read A + read B + write C) = 3 * size
        let bytes = size_bytes * 3;
        let bandwidth_gbs = bytes as f64 / time_per_op / 1e9;

        Some(bandwidth_gbs)
    }

    /// Discover cache hierarchy (alias for estimate)
    pub fn discover(device: &WgpuDevice) -> Self {
        Self::estimate(device)
    }

    /// Classify substrate type from adapter info.
    ///
    /// Strategy: vendor ID first (authoritative, set by driver), then name
    /// substring matching as a fallback for unknown/zero vendor IDs.
    fn classify_substrate(info: &wgpu::AdapterInfo) -> SubstrateType {
        use crate::device::vendor::{
            VENDOR_AMD, VENDOR_APPLE, VENDOR_ARM, VENDOR_INTEL, VENDOR_NVIDIA, VENDOR_QUALCOMM,
        };

        match info.device_type {
            wgpu::DeviceType::Cpu => SubstrateType::Cpu,

            wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu => {
                // Primary: vendor ID is authoritative when the driver reports it.
                match info.vendor {
                    VENDOR_NVIDIA => SubstrateType::NvidiaGpu,
                    VENDOR_AMD => SubstrateType::AmdGpu,
                    VENDOR_INTEL => SubstrateType::IntelGpu,
                    VENDOR_APPLE => SubstrateType::AppleGpu,
                    VENDOR_ARM | VENDOR_QUALCOMM => SubstrateType::Other,
                    _ => {
                        // Fallback: name-based heuristic for drivers that report
                        // vendor_id = 0 (e.g. some Mesa/software configurations).
                        const NAME_TABLE: &[(&[&str], SubstrateType)] = &[
                            (
                                &["nvidia", "geforce", "quadro", "tesla"],
                                SubstrateType::NvidiaGpu,
                            ),
                            (&["amd", "radeon", "rx ", "navi"], SubstrateType::AmdGpu),
                            (&["intel", "arc", "iris"], SubstrateType::IntelGpu),
                            (&["apple", "m1", "m2", "m3"], SubstrateType::AppleGpu),
                        ];
                        let n = info.name.to_lowercase();
                        NAME_TABLE
                            .iter()
                            .find(|(patterns, _)| patterns.iter().any(|p| n.contains(p)))
                            .map(|(_, st)| *st)
                            .unwrap_or(SubstrateType::Other)
                    }
                }
            }

            _ => SubstrateType::Other,
        }
    }

    /// Get the largest cache level
    pub fn largest_cache(&self) -> Option<&CacheLevel> {
        self.cache_levels.iter().max_by_key(|c| c.size_bytes)
    }

    /// Total cache capacity
    pub fn total_cache_bytes(&self) -> u64 {
        self.cache_levels.iter().map(|c| c.size_bytes).sum()
    }
}

/// Cache residency status
#[derive(Debug, Clone)]
pub enum CacheResidency {
    /// Data fits in cache
    Resident {
        cache_level: &'static str,
        utilization: f64,
    },
    /// Data exceeds cache, DRAM-bound
    DramBound { overflow_bytes: u64 },
}

/// Tile configuration for cache-aware execution
#[derive(Debug, Clone)]
pub struct TileConfig {
    /// Elements per tile
    pub tile_elements: usize,
    /// Bytes per tile
    pub tile_bytes: u64,
    /// Number of tiles
    pub num_tiles: usize,
    /// Target cache level
    pub target_cache: &'static str,
    /// Expected bandwidth
    pub expected_bandwidth_gbs: f64,
}

/// Cache-aware workload tiler
pub struct CacheAwareTiler {
    hierarchy: SubstrateMemoryHierarchy,
}

impl CacheAwareTiler {
    /// Create a tiler for a substrate
    pub fn new(hierarchy: SubstrateMemoryHierarchy) -> Self {
        Self { hierarchy }
    }

    /// Compute optimal tile size
    ///
    /// # Arguments
    /// * `total_bytes` - Total data size
    /// * `element_size` - Bytes per element
    /// * `read_write_ratio` - Data reuse factor (e.g., 3.0 for A*B+C)
    pub fn optimal_tile_size(
        &self,
        total_bytes: u64,
        element_size: usize,
        read_write_ratio: f64,
    ) -> TileConfig {
        let working_set_per_element = (element_size as f64 * read_write_ratio) as u64;

        // Find largest cache that can fit useful tiles
        for cache in self.hierarchy.cache_levels.iter().rev() {
            // Target 70% utilization for headroom
            let usable_cache = (cache.size_bytes as f64 * 0.7) as u64;
            let elements_per_tile = usable_cache / working_set_per_element.max(1);

            if elements_per_tile >= 1024 {
                let tile_bytes = elements_per_tile * element_size as u64;
                let num_tiles = total_bytes.div_ceil(tile_bytes) as usize;

                return TileConfig {
                    tile_elements: elements_per_tile as usize,
                    tile_bytes,
                    num_tiles,
                    target_cache: cache.name,
                    expected_bandwidth_gbs: cache.bandwidth_gbs,
                };
            }
        }

        // Fall back to single pass (DRAM-bound)
        let total_elements = (total_bytes / element_size as u64) as usize;
        TileConfig {
            tile_elements: total_elements,
            tile_bytes: total_bytes,
            num_tiles: 1,
            target_cache: "DRAM",
            expected_bandwidth_gbs: self.hierarchy.main_memory.bandwidth_gbs,
        }
    }

    /// Check if data fits in cache
    pub fn is_cache_resident(&self, data_bytes: u64) -> CacheResidency {
        for cache in &self.hierarchy.cache_levels {
            if data_bytes <= cache.size_bytes {
                return CacheResidency::Resident {
                    cache_level: cache.name,
                    utilization: data_bytes as f64 / cache.size_bytes as f64,
                };
            }
        }

        let largest = self
            .hierarchy
            .cache_levels
            .last()
            .map(|c| c.size_bytes)
            .unwrap_or(0);

        CacheResidency::DramBound {
            overflow_bytes: data_bytes.saturating_sub(largest),
        }
    }

    /// Predict effective bandwidth
    pub fn predict_bandwidth(&self, data_bytes: u64) -> f64 {
        match self.is_cache_resident(data_bytes) {
            CacheResidency::Resident { cache_level, .. } => self
                .hierarchy
                .cache_levels
                .iter()
                .find(|c| c.name == cache_level)
                .map(|c| c.bandwidth_gbs)
                .unwrap_or(self.hierarchy.main_memory.bandwidth_gbs),
            CacheResidency::DramBound { .. } => {
                // Expect ~80% of theoretical DRAM bandwidth
                self.hierarchy.main_memory.bandwidth_gbs * 0.8
            }
        }
    }
}

impl std::fmt::Display for SubstrateMemoryHierarchy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Memory Hierarchy: {} ({})",
            self.substrate_name,
            if self.probed { "probed" } else { "estimated" }
        )?;
        writeln!(f, "  Type: {:?}", self.substrate_type)?;
        for cache in &self.cache_levels {
            writeln!(
                f,
                "  {}: {} MB @ {:.0} GB/s",
                cache.name,
                cache.size_bytes / 1024 / 1024,
                cache.bandwidth_gbs
            )?;
        }
        writeln!(
            f,
            "  DRAM: {:.0} GB/s (theoretical)",
            self.main_memory.bandwidth_gbs
        )?;
        writeln!(
            f,
            "  Optimal tile: {} MB",
            self.optimal_tile_bytes / 1024 / 1024
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiling() {
        let hierarchy = SubstrateMemoryHierarchy {
            substrate_name: "Test GPU".to_string(),
            substrate_type: SubstrateType::Other,
            cache_levels: vec![CacheLevel {
                name: "Cache",
                size_bytes: 64 * 1024 * 1024, // 64 MB
                bandwidth_gbs: 1000.0,
                shared: true,
            }],
            main_memory: MainMemory {
                size_bytes: 16 * 1024 * 1024 * 1024,
                bandwidth_gbs: 500.0,
            },
            optimal_tile_bytes: 32 * 1024 * 1024,
            probed: false,
        };

        let tiler = CacheAwareTiler::new(hierarchy);

        // 1 GB workload
        let config = tiler.optimal_tile_size(
            1024 * 1024 * 1024,
            4,   // f32
            3.0, // A*B+C
        );

        assert!(
            config.tile_bytes < 64 * 1024 * 1024,
            "Tiles should fit in cache"
        );
        assert!(config.num_tiles > 1, "Large workload should be tiled");
    }

    #[test]
    fn test_cache_residency() {
        let hierarchy = SubstrateMemoryHierarchy {
            substrate_name: "Test GPU".to_string(),
            substrate_type: SubstrateType::Other,
            cache_levels: vec![CacheLevel {
                name: "Cache",
                size_bytes: 64 * 1024 * 1024,
                bandwidth_gbs: 1000.0,
                shared: true,
            }],
            main_memory: MainMemory {
                size_bytes: 16 * 1024 * 1024 * 1024,
                bandwidth_gbs: 500.0,
            },
            optimal_tile_bytes: 32 * 1024 * 1024,
            probed: false,
        };

        let tiler = CacheAwareTiler::new(hierarchy);

        // 10 MB should fit
        match tiler.is_cache_resident(10 * 1024 * 1024) {
            CacheResidency::Resident { .. } => {}
            _ => panic!("10 MB should be cache-resident"),
        }

        // 100 MB should overflow
        match tiler.is_cache_resident(100 * 1024 * 1024) {
            CacheResidency::DramBound { .. } => {}
            _ => panic!("100 MB should be DRAM-bound"),
        }
    }
}
