// SPDX-License-Identifier: AGPL-3.0-or-later
//! f64 throughput ratio probing — metalForge discovery
//!
//! Runs FMA microbenchmarks in f32 and f64 to measure the actual throughput
//! ratio, classifying devices into performance tiers. This drives workload
//! routing decisions (native f64 vs quantize-to-f32).
//!
//! Key findings (metalForge validated):
//! - Titan V (SM70): 1:2 ratio (f64 at half f32 speed)
//! - RTX 4070 (Ada Lovelace): 1:64 ratio (f64 heavily throttled)

use super::probe::{adapter_key, lock_cache};
use crate::device::WgpuDevice;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

/// f64-to-f32 throughput ratio discovered by probing.
#[derive(Debug, Clone, Copy)]
pub struct F64ThroughputRatio {
    /// Measured f64 GFLOPS.
    pub f64_gflops: f64,
    /// Measured f32 GFLOPS.
    pub f32_gflops: f64,
    /// `f32_gflops` / `f64_gflops` — higher means worse f64 relative performance.
    pub ratio: f64,
}

impl F64ThroughputRatio {
    /// Classify the f64 capability tier based on measured ratio.
    #[must_use]
    pub fn tier(&self) -> F64Tier {
        match self.ratio {
            r if r <= 2.5 => F64Tier::Native,
            r if r <= 8.0 => F64Tier::Capable,
            r if r <= 32.0 => F64Tier::Consumer,
            _ => F64Tier::Throttled,
        }
    }
}

/// Classification of f64 performance tier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F64Tier {
    /// f64 at 1:2 or better — full physics workloads viable
    Native,
    /// f64 at 1:2 to 1:8 — acceptable for medium precision
    Capable,
    /// f64 at 1:8 to 1:32 — use sparingly, prefer f32 with validation
    Consumer,
    /// f64 at 1:32+ — quantize to f32, validate against CPU f64
    Throttled,
}

impl std::fmt::Display for F64Tier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Native => write!(f, "Native (1:2)"),
            Self::Capable => write!(f, "Capable (1:2–1:8)"),
            Self::Consumer => write!(f, "Consumer (1:8–1:32)"),
            Self::Throttled => write!(f, "Throttled (1:32+)"),
        }
    }
}

static F64_RATIO_CACHE: LazyLock<Mutex<HashMap<String, F64ThroughputRatio>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Probe the f64-to-f32 throughput ratio on this device.
///
/// Runs a trivial FMA loop in both f32 and f64, measures time, computes ratio.
/// Results are cached per adapter.
pub async fn probe_f64_throughput_ratio(device: &WgpuDevice) -> Option<F64ThroughputRatio> {
    let key = adapter_key(device);

    if let Some(cached) = lock_cache(&F64_RATIO_CACHE).get(&key).copied() {
        return Some(cached);
    }

    let f32_time = run_throughput_probe(device, false).await?;
    let f64_time = run_throughput_probe(device, true).await?;

    if f32_time <= 0.0 || f64_time <= 0.0 {
        return None;
    }

    let ops_per_dispatch = 256.0 * 1024.0 * 100.0;
    let f32_gflops = ops_per_dispatch / f32_time / 1e9;
    let f64_gflops = ops_per_dispatch / f64_time / 1e9;

    let result = F64ThroughputRatio {
        f64_gflops,
        f32_gflops,
        ratio: f32_gflops / f64_gflops,
    };

    lock_cache(&F64_RATIO_CACHE).insert(key, result);
    Some(result)
}

/// Read cached f64 throughput ratio.
#[must_use]
pub fn cached_f64_ratio(device: &WgpuDevice) -> Option<F64ThroughputRatio> {
    lock_cache(&F64_RATIO_CACHE)
        .get(&adapter_key(device))
        .copied()
}

/// Run a throughput probe (f32 or f64 FMA loop).
async fn run_throughput_probe(device: &WgpuDevice, use_f64: bool) -> Option<f64> {
    let shader_src = if use_f64 {
        "enable f64;\n\
         @group(0) @binding(0) var<storage, read_write> out: array<f64>;\n\
         @compute @workgroup_size(256)\n\
         fn main(@builtin(global_invocation_id) id: vec3<u32>) {\n\
             var acc = f64(1.0);\n\
             for (var i = 0u; i < 100u; i = i + 1u) {\n\
                 acc = fma(acc, f64(1.000001), f64(0.000001));\n\
             }\n\
             out[id.x] = acc;\n\
         }"
    } else {
        "@group(0) @binding(0) var<storage, read_write> out: array<f32>;\n\
         @compute @workgroup_size(256)\n\
         fn main(@builtin(global_invocation_id) id: vec3<u32>) {\n\
             var acc = f32(1.0);\n\
             for (var i = 0u; i < 100u; i = i + 1u) {\n\
                 acc = fma(acc, f32(1.000001), f32(0.000001));\n\
             }\n\
             out[id.x] = acc;\n\
         }"
    };

    let wgpu_dev = device.device();
    let elem_size: u64 = if use_f64 { 8 } else { 4 };
    let num_elements: u64 = 256 * 1024;
    let buf_size = num_elements * elem_size;

    let scope1 = wgpu_dev.push_error_scope(wgpu::ErrorFilter::Validation);
    let shader = wgpu_dev.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("throughput_probe"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });
    if scope1.pop().await.is_some() {
        return None;
    }

    let out_buf = wgpu_dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tp_out"),
        size: buf_size,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let bgl = wgpu_dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

    let pl = wgpu_dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        immediate_size: 0,
    });

    let scope2 = wgpu_dev.push_error_scope(wgpu::ErrorFilter::Validation);
    let pipeline = wgpu_dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("throughput_probe"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        cache: None,
        compilation_options: Default::default(),
    });
    if scope2.pop().await.is_some() {
        return None;
    }

    let bg = wgpu_dev.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: out_buf.as_entire_binding(),
        }],
    });

    let workgroups = (num_elements / 256) as u32;

    let dispatch = |enc: &mut wgpu::CommandEncoder| {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    };

    // Hold WgpuDevice::lock() for all submit/poll to prevent concurrent access
    let mut warmup_buffers = Vec::with_capacity(3);
    for _ in 0..3 {
        let mut enc =
            wgpu_dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        dispatch(&mut enc);
        warmup_buffers.push(enc.finish());
    }
    device.submit_and_poll_inner(warmup_buffers);

    let iterations = 20;
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let mut enc =
            wgpu_dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        dispatch(&mut enc);
        device.submit_and_poll_inner(Some(enc.finish()));
    }
    let elapsed = start.elapsed().as_secs_f64() / iterations as f64;

    Some(elapsed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f64_tier_classification() {
        let native = F64ThroughputRatio {
            f64_gflops: 7000.0,
            f32_gflops: 14000.0,
            ratio: 2.0,
        };
        assert_eq!(native.tier(), F64Tier::Native);

        let consumer = F64ThroughputRatio {
            f64_gflops: 200.0,
            f32_gflops: 3200.0,
            ratio: 16.0,
        };
        assert_eq!(consumer.tier(), F64Tier::Consumer);

        let throttled = F64ThroughputRatio {
            f64_gflops: 100.0,
            f32_gflops: 6400.0,
            ratio: 64.0,
        };
        assert_eq!(throttled.tier(), F64Tier::Throttled);
    }

    #[tokio::test]
    async fn test_throughput_ratio_probe() {
        use crate::device::test_pool::get_test_device;
        let dev = get_test_device().await;
        let ratio = probe_f64_throughput_ratio(&dev).await;
        if let Some(r) = ratio {
            assert!(r.ratio > 0.0, "Ratio must be positive");
            assert!(r.f32_gflops > 0.0, "f32 GFLOPS must be positive");
            assert!(r.f64_gflops > 0.0, "f64 GFLOPS must be positive");
        }
    }
}
