//! Unidirectional Pipeline Benchmark
//!
//! Compares traditional bidirectional GPU patterns vs unidirectional streaming.
//!
//! # Running
//!
//! ```bash
//! cargo bench --package barracuda --bench unidirectional_benchmark
//! ```

use barracuda::device::WgpuDevice;
use barracuda::staging::{UnidirectionalConfig, UnidirectionalPipeline};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Work unit size in bytes (8 KB - typical small compute job)
const WORK_UNIT_SIZE: usize = 8 * 1024;

/// Number of work units to process
const WORK_UNIT_COUNT: usize = 1000;

/// Results from a benchmark run
#[derive(Debug)]
struct BenchmarkResult {
    name: &'static str,
    total_time: Duration,
    work_units: usize,
    total_bytes: usize,
    throughput_mbps: f64,
    avg_latency_us: f64,
}

impl BenchmarkResult {
    fn print(&self) {
        println!("\n{}", "=".repeat(60));
        println!("  {}", self.name);
        println!("{}", "=".repeat(60));
        println!("  Total time:     {:>12.2?}", self.total_time);
        println!("  Work units:     {:>12}", self.work_units);
        println!(
            "  Total bytes:    {:>12} ({:.2} MB)",
            self.total_bytes,
            self.total_bytes as f64 / (1024.0 * 1024.0)
        );
        println!("  Throughput:     {:>12.2} MB/s", self.throughput_mbps);
        println!("  Avg latency:    {:>12.2} µs", self.avg_latency_us);
        println!("{}", "=".repeat(60));
    }
}

/// Traditional bidirectional pattern: upload → compute → download (blocking)
fn bench_traditional(device: &WgpuDevice) -> BenchmarkResult {
    let work_data: Vec<u8> = (0..WORK_UNIT_SIZE).map(|i| (i % 256) as u8).collect();
    let mut total_latency = Duration::ZERO;

    let start = Instant::now();

    for _ in 0..WORK_UNIT_COUNT {
        let unit_start = Instant::now();

        // Create buffer
        let buffer = device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Traditional:WorkUnit"),
            size: WORK_UNIT_SIZE as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Upload (blocking)
        device.queue().write_buffer(&buffer, 0, &work_data);
        device.queue().submit(std::iter::empty());
        device.device().poll(wgpu::Maintain::Wait);

        // Simulate compute (empty command buffer for benchmark)
        let encoder = device
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Traditional:Compute"),
            });
        device.queue().submit(Some(encoder.finish()));
        device.device().poll(wgpu::Maintain::Wait);

        // Download (blocking)
        let staging = device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Traditional:Staging"),
            size: WORK_UNIT_SIZE as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Traditional:Download"),
            });
        encoder.copy_buffer_to_buffer(&buffer, 0, &staging, 0, WORK_UNIT_SIZE as u64);
        device.queue().submit(Some(encoder.finish()));
        device.device().poll(wgpu::Maintain::Wait);

        // Map and read (blocking)
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device.device().poll(wgpu::Maintain::Wait);
        let _ = receiver.recv();

        let _data = slice.get_mapped_range();
        drop(_data);
        staging.unmap();

        total_latency += unit_start.elapsed();
    }

    let total_time = start.elapsed();
    let total_bytes = WORK_UNIT_COUNT * WORK_UNIT_SIZE;

    BenchmarkResult {
        name: "Traditional (Bidirectional)",
        total_time,
        work_units: WORK_UNIT_COUNT,
        total_bytes,
        throughput_mbps: (total_bytes as f64 / (1024.0 * 1024.0)) / total_time.as_secs_f64(),
        avg_latency_us: (total_latency.as_micros() as f64) / (WORK_UNIT_COUNT as f64),
    }
}

/// Unidirectional pattern: fire-and-forget streaming
fn bench_unidirectional(device: Arc<WgpuDevice>) -> BenchmarkResult {
    let config = UnidirectionalConfig::with_sizes(16, 4); // 16 MB input, 4 MB output

    let mut pipeline = match UnidirectionalPipeline::new(device.clone(), config) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to create pipeline: {}", e);
            return BenchmarkResult {
                name: "Unidirectional (Fire-and-Forget)",
                total_time: Duration::ZERO,
                work_units: 0,
                total_bytes: 0,
                throughput_mbps: 0.0,
                avg_latency_us: 0.0,
            };
        }
    };

    let work_data: Vec<u8> = (0..WORK_UNIT_SIZE).map(|i| (i % 256) as u8).collect();

    let start = Instant::now();

    // Submit all work units (fire-and-forget)
    let mut handles = Vec::with_capacity(WORK_UNIT_COUNT);
    for _ in 0..WORK_UNIT_COUNT {
        if let Some(handle) = pipeline.try_submit(&work_data) {
            handles.push(handle);
        }
    }

    // Mark all as completed (simulating compute completion)
    for handle in &handles {
        pipeline.mark_completed(handle.id, WORK_UNIT_SIZE / 10);
    }

    let total_time = start.elapsed();
    let stats = pipeline.stats();
    let actual_work_units = handles.len();
    let total_bytes = actual_work_units * WORK_UNIT_SIZE;

    BenchmarkResult {
        name: "Unidirectional (Fire-and-Forget)",
        total_time,
        work_units: actual_work_units,
        total_bytes,
        throughput_mbps: (total_bytes as f64 / (1024.0 * 1024.0)) / total_time.as_secs_f64(),
        avg_latency_us: if actual_work_units > 0 {
            (stats.avg_latency_ns as f64) / 1000.0
        } else {
            0.0
        },
    }
}

/// Batched traditional pattern
fn bench_batched_traditional(device: &WgpuDevice) -> BenchmarkResult {
    let batch_size = 100;
    let work_data: Vec<u8> = (0..WORK_UNIT_SIZE).map(|i| (i % 256) as u8).collect();
    let batch_data: Vec<u8> = work_data
        .iter()
        .cycle()
        .take(WORK_UNIT_SIZE * batch_size)
        .copied()
        .collect();

    let start = Instant::now();
    let batches = WORK_UNIT_COUNT / batch_size;

    for _ in 0..batches {
        let buffer = device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Batched:WorkUnits"),
            size: (WORK_UNIT_SIZE * batch_size) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        device.queue().write_buffer(&buffer, 0, &batch_data);

        let encoder = device
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Batched:Compute"),
            });
        device.queue().submit(Some(encoder.finish()));
        device.device().poll(wgpu::Maintain::Wait);

        let staging = device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Batched:Staging"),
            size: (WORK_UNIT_SIZE * batch_size) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Batched:Download"),
            });
        encoder.copy_buffer_to_buffer(
            &buffer,
            0,
            &staging,
            0,
            (WORK_UNIT_SIZE * batch_size) as u64,
        );
        device.queue().submit(Some(encoder.finish()));
        device.device().poll(wgpu::Maintain::Wait);

        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device.device().poll(wgpu::Maintain::Wait);
        let _ = receiver.recv();

        let _data = slice.get_mapped_range();
        drop(_data);
        staging.unmap();
    }

    let total_time = start.elapsed();
    let total_bytes = WORK_UNIT_COUNT * WORK_UNIT_SIZE;

    BenchmarkResult {
        name: "Batched Traditional (100 per batch)",
        total_time,
        work_units: WORK_UNIT_COUNT,
        total_bytes,
        throughput_mbps: (total_bytes as f64 / (1024.0 * 1024.0)) / total_time.as_secs_f64(),
        avg_latency_us: (total_time.as_micros() as f64) / (batches as f64),
    }
}

fn main() {
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║     UNIDIRECTIONAL PIPELINE BENCHMARK                      ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();
    println!(
        "Config: {} work units × {} KB = {} MB",
        WORK_UNIT_COUNT,
        WORK_UNIT_SIZE / 1024,
        (WORK_UNIT_COUNT * WORK_UNIT_SIZE) / (1024 * 1024)
    );

    let device = tokio::runtime::Runtime::new()
        .expect("runtime")
        .block_on(async { WgpuDevice::new().await.expect("Failed to create device") });

    let device = Arc::new(device);

    println!("\nDevice: {}", device.name());

    println!("\nWarming up...");
    let _ = bench_traditional(&device);

    println!("Running benchmarks...");

    let traditional = bench_traditional(&device);
    let batched = bench_batched_traditional(&device);
    let unidirectional = bench_unidirectional(device.clone());

    traditional.print();
    batched.print();
    unidirectional.print();

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║                    COMPARISON                              ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    if traditional.throughput_mbps > 0.0 {
        let batched_speedup = batched.throughput_mbps / traditional.throughput_mbps;
        let uni_speedup = unidirectional.throughput_mbps / traditional.throughput_mbps;

        println!("\n  vs Traditional:");
        println!("    Batched:        {:.2}x throughput", batched_speedup);
        println!("    Unidirectional: {:.2}x throughput", uni_speedup);
    }

    println!("\nDone.");
}
