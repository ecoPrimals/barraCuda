// SPDX-License-Identifier: AGPL-3.0-or-later
//! Auto-Tuning Runtime for BarraCuda
//!
//! Discovers optimal GPU parameters at runtime through calibration.
//!
//! Philosophy:
//! - Don't assume vendor capabilities - discover ground truth
//! - Handle silicon lottery and generation differences
//! - Work seamlessly with unknown/new hardware
//! - Cache calibrations per GPU for fast startup

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::RwLock;
use std::time::Instant;

use wgpu::util::DeviceExt;

/// Trait for GPU devices that support calibration (submit+poll under lock).
/// Implemented by WgpuDevice to avoid circular dependency.
pub(crate) trait GpuDeviceForCalibration {
    fn device(&self) -> &wgpu::Device;
    fn name(&self) -> &str;
    /// Submit commands and poll, holding the GPU lock. Used during calibration.
    fn submit_and_poll_calibration(&self, commands: impl IntoIterator<Item = wgpu::CommandBuffer>);
}

/// Calibration result for a specific GPU
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GpuCalibration {
    /// Unique device identifier (from wgpu global_id)
    pub device_id: String,
    /// Device name (adapter name)
    pub device_name: String,
    /// Optimal workgroup size discovered
    pub optimal_workgroup_size: u32,
    /// Measured peak bandwidth (GB/s)
    pub peak_bandwidth_gbps: f64,
    /// Measured single-op overhead (μs)
    pub dispatch_overhead_us: f64,
    /// Calibration timestamp (epoch seconds)
    pub calibrated_at: u64,
}

impl Default for GpuCalibration {
    fn default() -> Self {
        Self {
            device_id: String::new(),
            device_name: String::new(),
            optimal_workgroup_size: 256, // Safe default
            peak_bandwidth_gbps: 0.0,
            dispatch_overhead_us: 0.0,
            calibrated_at: 0,
        }
    }
}

/// Auto-tuner for discovering optimal GPU parameters
pub struct AutoTuner {
    /// Cached calibrations by device name
    calibrations: RwLock<HashMap<String, GpuCalibration>>,
    /// Path to persistent cache file
    cache_path: Option<PathBuf>,
}

impl AutoTuner {
    /// Create a new auto-tuner
    pub fn new() -> Self {
        Self {
            calibrations: RwLock::new(HashMap::new()),
            cache_path: None,
        }
    }

    /// Create auto-tuner with persistent cache
    #[must_use]
    pub fn with_cache(cache_path: PathBuf) -> Self {
        let mut tuner = Self::new();
        tuner.cache_path = Some(cache_path.clone());

        // Try to load existing calibrations
        #[cfg(feature = "serde")]
        if let Ok(contents) = std::fs::read_to_string(&cache_path) {
            if let Ok(cals) = serde_json::from_str::<Vec<GpuCalibration>>(&contents) {
                let mut map = tuner
                    .calibrations
                    .write()
                    .expect("calibrations RwLock poisoned");
                for cal in cals {
                    map.insert(cal.device_name.clone(), cal);
                }
            }
        }
        // Without serde, we still check file exists but can't parse
        #[cfg(not(feature = "serde"))]
        let _ = std::fs::read_to_string(&cache_path);

        tuner
    }

    /// Get calibration for a device, or calibrate if needed
    pub(crate) fn get_or_calibrate(
        &self,
        wgpu_device: &impl GpuDeviceForCalibration,
    ) -> GpuCalibration {
        let device_name = wgpu_device.name();
        // Check cache first
        {
            let cals = self
                .calibrations
                .read()
                .expect("calibrations RwLock poisoned");
            if let Some(cal) = cals.get(device_name) {
                return cal.clone();
            }
        }

        // Need to calibrate
        let cal = self.calibrate_device(wgpu_device);

        // Cache it
        {
            let mut cals = self
                .calibrations
                .write()
                .expect("calibrations RwLock poisoned");
            cals.insert(device_name.to_string(), cal.clone());
        }

        // Persist if cache path set
        self.save_cache();

        cal
    }

    /// Perform calibration for a device
    fn calibrate_device(&self, wgpu_device: &impl GpuDeviceForCalibration) -> GpuCalibration {
        let device = wgpu_device.device();
        let device_name = wgpu_device.name();
        // Test workgroup sizes
        let wg_sizes = [32, 64, 128, 256];
        let test_size = 4_000_000usize;

        let mut best_wg = 256u32;
        let mut best_bw = 0.0f64;

        for &wg_size in &wg_sizes {
            if let Some(bw) = self.measure_bandwidth(wgpu_device, wg_size, test_size) {
                if bw > best_bw {
                    best_bw = bw;
                    best_wg = wg_size;
                }
            }
        }

        // Measure dispatch overhead with optimal WG
        let overhead = self.measure_overhead(wgpu_device, best_wg);

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        GpuCalibration {
            device_id: format!("{:?}", device.global_id()),
            device_name: device_name.to_string(),
            optimal_workgroup_size: best_wg,
            peak_bandwidth_gbps: best_bw,
            dispatch_overhead_us: overhead,
            calibrated_at: now,
        }
    }

    /// Measure bandwidth for a given workgroup size
    fn measure_bandwidth(
        &self,
        wgpu_device: &impl GpuDeviceForCalibration,
        workgroup_size: u32,
        size: usize,
    ) -> Option<f64> {
        let device = wgpu_device.device();
        let shader_src = format!(
            r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) {{
        return;
    }}
    output[idx] = a[idx] + b[idx];
}}
"#
        );

        let data: Vec<f32> = (0..size).map(|i| i as f32 * 0.001).collect();

        let buf_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("A"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let buf_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("B"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let buf_out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Out"),
            size: (size * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_out.as_entire_binding(),
                },
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pl),
            module: &shader,
            entry_point: "main",
            cache: None,
            compilation_options: Default::default(),
        });

        let workgroups = (size as u32).div_ceil(workgroup_size).min(65_535);

        // Warmup
        for _ in 0..3 {
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            wgpu_device.submit_and_poll_calibration(Some(encoder.finish()));
        }

        // Measure
        let iterations = 10;
        let start = Instant::now();

        for _ in 0..iterations {
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            wgpu_device.submit_and_poll_calibration(Some(encoder.finish()));
        }

        let elapsed = start.elapsed();
        let time_per_op = elapsed.as_secs_f64() / iterations as f64;

        // Bandwidth: (read A + read B + write C) = 3 * size * 4 bytes
        let bytes = size * 3 * 4;
        let bandwidth_gbps = bytes as f64 / time_per_op / 1e9;

        Some(bandwidth_gbps)
    }

    /// Measure dispatch overhead
    fn measure_overhead(
        &self,
        wgpu_device: &impl GpuDeviceForCalibration,
        workgroup_size: u32,
    ) -> f64 {
        let device = wgpu_device.device();
        // Use tiny workload to measure pure overhead
        let size = 1024usize;

        let shader_src = format!(
            r#"
@group(0) @binding(0) var<storage, read_write> data: array<f32>;

@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;
    if (idx < arrayLength(&data)) {{
        data[idx] = data[idx] + 1.0;
    }}
}}
"#
        );

        let data: Vec<f32> = vec![0.0; size];

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Data"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pl),
            module: &shader,
            entry_point: "main",
            cache: None,
            compilation_options: Default::default(),
        });

        let workgroups = (size as u32).div_ceil(workgroup_size);

        // Warmup
        for _ in 0..10 {
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            wgpu_device.submit_and_poll_calibration(Some(encoder.finish()));
        }

        // Measure many iterations
        let iterations = 100;
        let start = Instant::now();

        for _ in 0..iterations {
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            wgpu_device.submit_and_poll_calibration(Some(encoder.finish()));
        }

        let elapsed = start.elapsed();
        elapsed.as_secs_f64() * 1e6 / iterations as f64
    }

    /// Save calibrations to cache file
    #[cfg(feature = "serde")]
    fn save_cache(&self) {
        if let Some(ref path) = self.cache_path {
            let cals = self
                .calibrations
                .read()
                .expect("calibrations RwLock poisoned");
            let cal_vec: Vec<&GpuCalibration> = cals.values().collect();
            if let Ok(json) = serde_json::to_string_pretty(&cal_vec) {
                let _ = std::fs::write(path, json);
            }
        }
    }

    #[cfg(not(feature = "serde"))]
    fn save_cache(&self) {
        // No-op without serde
    }

    /// Force recalibration
    pub(crate) fn recalibrate(&self, wgpu_device: &impl GpuDeviceForCalibration) -> GpuCalibration {
        let cal = self.calibrate_device(wgpu_device);

        {
            let mut cals = self
                .calibrations
                .write()
                .expect("calibrations RwLock poisoned");
            cals.insert(wgpu_device.name().to_string(), cal.clone());
        }

        self.save_cache();
        cal
    }
}

impl Default for AutoTuner {
    fn default() -> Self {
        Self::new()
    }
}

// Global auto-tuner instance (singleton pattern via std::sync::LazyLock)
// Evolved from lazy_static to pure std (Rust 1.80+)
pub static GLOBAL_TUNER: std::sync::LazyLock<AutoTuner> = std::sync::LazyLock::new(|| {
    // Try to use a standard cache location
    let cache_dir = std::env::var("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::var("HOME")
                .map(|h| PathBuf::from(h).join(".cache"))
                .unwrap_or_else(|_| std::env::temp_dir())
        });

    let cache_path = cache_dir
        .join(env!("CARGO_PKG_NAME"))
        .join("gpu_calibrations.json");

    // Ensure directory exists
    if let Some(parent) = cache_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    AutoTuner::with_cache(cache_path)
});

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_calibration() {
        let cal = GpuCalibration::default();
        assert_eq!(cal.optimal_workgroup_size, 256);
    }

    #[test]
    fn test_tuner_creation() {
        let tuner = AutoTuner::new();
        assert!(tuner
            .calibrations
            .read()
            .expect("calibrations RwLock poisoned")
            .is_empty());
    }
}
