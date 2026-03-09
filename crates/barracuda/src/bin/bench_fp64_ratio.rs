// SPDX-License-Identifier: AGPL-3.0-or-later

//! FP64:FP32:DF64 ratio micro-benchmark.
//!
//! Measures raw ALU throughput for f32, f64, and df64 (double-float f32 pairs)
//! using a pure FMA chain — no memory bottleneck, no branching, no control flow.
//!
//! The df64 path demonstrates the "core streaming" strategy: using FP32 cores
//! for ~14-digit-precision f64-equivalent work instead of the dedicated FP64 units.
//!
//! # Usage
//!   cargo run --release -p barracuda --bin `bench_fp64_ratio`
//!
//! # Provenance
//! Absorbed from hotSpring `bench_fp64_ratio.rs` (Feb 2026).

use barracuda::device::WgpuDevice;
use std::sync::Arc;
use std::time::Instant;

const CHAIN_LENGTH: u32 = 4096;
const N_THREADS: u32 = 4_194_304;
const WG_SIZE: u32 = 256;
const WARMUP: usize = 3;
const MEASURE: usize = 10;

fn fma_shader_f32() -> String {
    format!(
        r"
@group(0) @binding(0) var<storage, read_write> output: array<f32>;

@compute @workgroup_size({WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= {N_THREADS}u) {{ return; }}

    var acc = f32(i) * 1.0000001;
    let m = f32(1.0000001);
    let a = f32(0.0000001);

    for (var j = 0u; j < {CHAIN_LENGTH}u; j = j + 1u) {{
        acc = acc * m + a;
    }}

    output[i] = acc;
}}
"
    )
}

fn fma_shader_f64() -> String {
    format!(
        r"
@group(0) @binding(0) var<storage, read_write> output: array<f64>;

@compute @workgroup_size({WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= {N_THREADS}u) {{ return; }}

    var acc = f64(i) * 1.0000001;
    let m = f64(1.0000001);
    let a = f64(0.0000001);

    for (var j = 0u; j < {CHAIN_LENGTH}u; j = j + 1u) {{
        acc = acc * m + a;
    }}

    output[i] = acc;
}}
"
    )
}

fn fma_shader_df64() -> String {
    format!(
        r"
struct Df64 {{
    hi: f32,
    lo: f32,
}}

fn two_sum(a: f32, b: f32) -> Df64 {{
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    return Df64(s, e);
}}

fn split(a: f32) -> vec2<f32> {{
    let c = 4097.0 * a;
    let ah = c - (c - a);
    let al = a - ah;
    return vec2<f32>(ah, al);
}}

fn two_prod(a: f32, b: f32) -> Df64 {{
    let p = a * b;
    let sa = split(a);
    let sb = split(b);
    let e = ((sa.x * sb.x - p) + sa.x * sb.y + sa.y * sb.x) + sa.y * sb.y;
    return Df64(p, e);
}}

fn df64_add(a: Df64, b: Df64) -> Df64 {{
    let s = two_sum(a.hi, b.hi);
    let e = a.lo + b.lo;
    return two_sum(s.hi, s.lo + e);
}}

fn df64_mul(a: Df64, b: Df64) -> Df64 {{
    let p = two_prod(a.hi, b.hi);
    let lo = p.lo + (a.hi * b.lo + a.lo * b.hi);
    return two_sum(p.hi, lo);
}}

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<storage, read> constants: array<f32>;

@compute @workgroup_size({WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= {N_THREADS}u) {{ return; }}

    let m = Df64(constants[0], constants[1]);
    let a = Df64(constants[2], constants[3]);
    var acc = Df64(f32(i) * m.hi, a.lo);

    for (var j = 0u; j < {CHAIN_LENGTH}u; j = j + 1u) {{
        acc = df64_add(df64_mul(acc, m), a);
    }}

    output[i * 2u] = acc.hi;
    output[i * 2u + 1u] = acc.lo;
}}
"
    )
}

fn bench_shader(device: &WgpuDevice, shader_src: &str, elem_bytes: u64, label: &str) -> f64 {
    bench_shader_inner(device, shader_src, elem_bytes, label, false)
}

fn bench_shader_df64(device: &WgpuDevice, shader_src: &str, elem_bytes: u64, label: &str) -> f64 {
    bench_shader_inner(device, shader_src, elem_bytes, label, true)
}

fn bench_shader_inner(
    device: &WgpuDevice,
    shader_src: &str,
    elem_bytes: u64,
    label: &str,
    needs_constants: bool,
) -> f64 {
    use wgpu::util::DeviceExt;

    let module = device
        .device()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

    let output_buf = device.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size: u64::from(N_THREADS) * elem_bytes,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let mut entries = vec![wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }];

    let constants_buf;
    if needs_constants {
        let const_data: [f32; 4] = [1.000_000_1, 0.000_000_1, 0.000_000_1, 0.0];
        let bytes: &[u8] = bytemuck::cast_slice(&const_data);
        constants_buf = Some(device.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("constants"),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE,
            },
        ));
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
    } else {
        constants_buf = None;
    }

    let bind_group_layout =
        device
            .device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &entries,
            });

    let pipeline_layout = device
        .device()
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

    let pipeline = device
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    let mut bg_entries = vec![wgpu::BindGroupEntry {
        binding: 0,
        resource: output_buf.as_entire_binding(),
    }];
    if let Some(ref cb) = constants_buf {
        bg_entries.push(wgpu::BindGroupEntry {
            binding: 1,
            resource: cb.as_entire_binding(),
        });
    }

    let bind_group = device
        .device()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &bg_entries,
        });

    let workgroups = N_THREADS.div_ceil(WG_SIZE);

    for _ in 0..WARMUP {
        let mut encoder = device
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        device.submit_and_poll(Some(encoder.finish()));
    }

    let t0 = Instant::now();
    for _ in 0..MEASURE {
        let mut encoder = device
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        device.submit_and_poll(Some(encoder.finish()));
    }
    t0.elapsed().as_secs_f64() / MEASURE as f64
}

#[tokio::main]
async fn main() {
    let device = match WgpuDevice::new().await {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("No GPU adapter: {e}");
            barracuda::validation::exit_no_gpu();
        }
    };

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  FP64:FP32:DF64 — Core Streaming Micro-Benchmark           ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Threads:      {N_THREADS:>12}");
    println!("  FMA chain:    {CHAIN_LENGTH:>12} ops/thread");
    println!(
        "  Total FLOPs:  {:>12} (2 FLOP per FMA × chain × threads)",
        2u64 * u64::from(CHAIN_LENGTH) * u64::from(N_THREADS)
    );
    println!("  Warmup:       {WARMUP:>12} rounds");
    println!("  Measure:      {MEASURE:>12} rounds");
    println!();

    let total_flops = 2.0 * f64::from(CHAIN_LENGTH) * f64::from(N_THREADS);

    println!("── FP32 FMA chain (raw FP32 cores) ──");
    let f32_time = bench_shader(&device, &fma_shader_f32(), 4, "fma_f32");
    let f32_tflops = total_flops / f32_time / 1e12;
    println!("  Time:       {:.3} ms", f32_time * 1e3);
    println!("  Throughput: {f32_tflops:.2} TFLOPS");
    println!();

    println!("── FP64 FMA chain (dedicated FP64 units) ──");
    let f64_time = bench_shader(&device, &fma_shader_f64(), 8, "fma_f64");
    let f64_tflops = total_flops / f64_time / 1e12;
    println!("  Time:       {:.3} ms", f64_time * 1e3);
    println!("  Throughput: {f64_tflops:.2} TFLOPS");
    println!();

    println!("── DF64 FMA chain (f32-pair on FP32 cores, ~14 digit precision) ──");
    let df64_time = bench_shader_df64(&device, &fma_shader_df64(), 8, "fma_df64");
    let df64_equiv_tflops = total_flops / df64_time / 1e12;
    println!("  Time:       {:.3} ms", df64_time * 1e3);
    println!("  Throughput: {df64_equiv_tflops:.2} equivalent-f64 TFLOPS");
    println!(
        "  (each df64 FMA ≈ 10 f32 ops → {:.2} raw f32 TFLOPS)",
        df64_equiv_tflops * 10.0
    );
    println!();

    let ratio_64_32 = f32_tflops / f64_tflops;
    let speedup_df64_vs_f64 = df64_equiv_tflops / f64_tflops;
    println!("══════════════════════════════════════════════════════════════");
    println!("  RESULTS");
    println!();
    println!("  {:>22} {:>10} {:>10} {:>10}", "", "FP32", "FP64", "DF64");
    println!(
        "  {:>22} {:>8.2} T {:>8.3} T {:>8.2} T",
        "Throughput (TFLOPS)", f32_tflops, f64_tflops, df64_equiv_tflops
    );
    println!(
        "  {:>22} {:>10} {:>8.1}× {:>8.1}×",
        "vs FP64",
        format!("{:.0}×", ratio_64_32),
        "1.0",
        format!("{:.1}", speedup_df64_vs_f64)
    );
    println!(
        "  {:>22} {:>10} {:>10} {:>10}",
        "Precision (digits)", "7", "16", "14"
    );
    println!();

    if speedup_df64_vs_f64 > 2.0 {
        println!("  DF64 delivers {speedup_df64_vs_f64:.1}× the f64-equivalent throughput");
        println!("  by streaming to FP32 cores instead of waiting for FP64 units.");
        println!();
        println!("  STRATEGY: Hybrid core streaming");
        println!("    → Bulk matrix ops: DF64 on FP32 cores ({df64_equiv_tflops:.2} TFLOPS)");
        println!("    → Precision-critical accumulations: native f64 ({f64_tflops:.3} TFLOPS)");
    } else {
        println!("  DF64 speedup over native f64 is modest ({speedup_df64_vs_f64:.1}×).");
        println!("  This GPU has strong native FP64 hardware.");
        println!("  Use native f64 for everything — the hardware supports it.");
    }
    println!("══════════════════════════════════════════════════════════════");
}
