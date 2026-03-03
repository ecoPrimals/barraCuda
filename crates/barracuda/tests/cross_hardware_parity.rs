//! Cross-Hardware Parity Tests
//!
//! Proves BarraCuda produces IDENTICAL results across hardware:
//! - Same WGSL shader
//! - Same input data
//! - GPU vs CPU → must match within f32 tolerance
//!
//! This is the core proof that "any math on any hardware" works.

use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use std::sync::Arc;

/// Helper: create device for specific hardware, skip test if unavailable
async fn try_gpu() -> Option<Arc<WgpuDevice>> {
    WgpuDevice::new_gpu().await.ok().map(Arc::new)
}

async fn try_cpu() -> Option<Arc<WgpuDevice>> {
    WgpuDevice::new_cpu().await.ok().map(Arc::new)
}

/// Compare two f32 slices with tolerance
fn assert_close(label: &str, a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(
        a.len(),
        b.len(),
        "{}: length mismatch {} vs {}",
        label,
        a.len(),
        b.len()
    );
    for (i, (va, vb)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (va - vb).abs() < tol,
            "{} mismatch at index {}: GPU={}, CPU={}, diff={}",
            label,
            i,
            va,
            vb,
            (va - vb).abs()
        );
    }
}

// ─── Adapter Enumeration ─────────────────────────────────────────────────────

#[tokio::test]
async fn test_enumerate_all_adapters() {
    let adapters = WgpuDevice::enumerate_adapters();

    println!("WGPU Adapter Report:");
    println!("  Total adapters: {}", adapters.len());

    let mut has_gpu = false;
    let mut has_cpu = false;

    for info in &adapters {
        let hw_type = match info.device_type {
            wgpu::DeviceType::DiscreteGpu => {
                has_gpu = true;
                "GPU (discrete)"
            }
            wgpu::DeviceType::IntegratedGpu => {
                has_gpu = true;
                "GPU (integrated)"
            }
            wgpu::DeviceType::Cpu => {
                has_cpu = true;
                "CPU (software)"
            }
            wgpu::DeviceType::VirtualGpu => "GPU (virtual)",
            wgpu::DeviceType::Other => "Other",
        };
        println!("  - {} | {} | {:?}", info.name, hw_type, info.backend);
    }

    println!("\n  Hardware summary:");
    println!("    GPU available: {}", has_gpu);
    println!("    CPU fallback:  {}", has_cpu);
    println!("    Cross-hw test: {}", has_gpu && has_cpu);

    assert!(!adapters.is_empty(), "Should find at least one adapter");
}

// ─── Matmul Parity ───────────────────────────────────────────────────────────

#[tokio::test]
async fn test_matmul_gpu_cpu_parity() {
    let gpu = try_gpu().await;
    let cpu = try_cpu().await;

    if gpu.is_none() || cpu.is_none() {
        println!("SKIP: Need both GPU and CPU adapters for parity test");
        println!(
            "  GPU: {}",
            if gpu.is_some() {
                "available"
            } else {
                "not found"
            }
        );
        println!(
            "  CPU: {}",
            if cpu.is_some() {
                "available"
            } else {
                "not found"
            }
        );
        return;
    }

    let gpu = gpu.unwrap();
    let cpu = cpu.unwrap();

    println!("Matmul parity: {} vs {}", gpu.name(), cpu.name());

    // Same input data
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let b_data = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2

    // GPU execution (matmul consumes self, so we create fresh tensors)
    let a_gpu = Tensor::from_vec_on(a_data.clone(), vec![2, 3], gpu.clone())
        .await
        .unwrap();
    let b_gpu = Tensor::from_vec_on(b_data.clone(), vec![3, 2], gpu)
        .await
        .unwrap();
    let gpu_data = a_gpu.matmul(&b_gpu).unwrap().to_vec().unwrap();

    // CPU execution
    let a_cpu = Tensor::from_vec_on(a_data, vec![2, 3], cpu.clone())
        .await
        .unwrap();
    let b_cpu = Tensor::from_vec_on(b_data, vec![3, 2], cpu).await.unwrap();
    let cpu_data = a_cpu.matmul(&b_cpu).unwrap().to_vec().unwrap();

    println!("  GPU result: {:?}", gpu_data);
    println!("  CPU result: {:?}", cpu_data);

    assert_close("matmul", &gpu_data, &cpu_data, 1e-4);
    println!("  PASS: GPU and CPU produce identical matmul results");
}

// ─── Element-wise Add Parity ─────────────────────────────────────────────────

#[tokio::test]
async fn test_add_gpu_cpu_parity() {
    let gpu = try_gpu().await;
    let cpu = try_cpu().await;

    if gpu.is_none() || cpu.is_none() {
        println!("SKIP: Need both GPU and CPU adapters");
        return;
    }

    let gpu = gpu.unwrap();
    let cpu = cpu.unwrap();

    println!("Add parity: {} vs {}", gpu.name(), cpu.name());

    let a_data = vec![1.0, 2.0, 3.0, 4.0];
    let b_data = vec![10.0, 20.0, 30.0, 40.0];

    // GPU
    let a_gpu = Tensor::from_vec_on(a_data.clone(), vec![4], gpu.clone())
        .await
        .unwrap();
    let b_gpu = Tensor::from_vec_on(b_data.clone(), vec![4], gpu.clone())
        .await
        .unwrap();
    let gpu_data = a_gpu.add(&b_gpu).unwrap().to_vec().unwrap();

    // CPU
    let a_cpu = Tensor::from_vec_on(a_data.clone(), vec![4], cpu.clone())
        .await
        .unwrap();
    let b_cpu = Tensor::from_vec_on(b_data.clone(), vec![4], cpu.clone())
        .await
        .unwrap();
    let cpu_data = a_cpu.add(&b_cpu).unwrap().to_vec().unwrap();

    assert_close("add", &gpu_data, &cpu_data, 1e-6);
    println!("  PASS: GPU and CPU produce identical add results");
}

// ─── Cholesky Parity ─────────────────────────────────────────────────────────

#[tokio::test]
async fn test_cholesky_gpu_cpu_parity() {
    let gpu = try_gpu().await;
    let cpu = try_cpu().await;

    if gpu.is_none() || cpu.is_none() {
        println!("SKIP: Need both GPU and CPU adapters");
        return;
    }

    let gpu = gpu.unwrap();
    let cpu = cpu.unwrap();

    println!("Cholesky parity: {} vs {}", gpu.name(), cpu.name());

    // SPD matrix
    let a_data = vec![4.0, 2.0, 2.0, 3.0];

    // GPU
    let a_gpu = Tensor::from_vec_on(a_data.clone(), vec![2, 2], gpu.clone())
        .await
        .unwrap();
    let gpu_data = a_gpu.cholesky().unwrap().to_vec().unwrap();

    // CPU
    let a_cpu = Tensor::from_vec_on(a_data.clone(), vec![2, 2], cpu.clone())
        .await
        .unwrap();
    let cpu_data = a_cpu.cholesky().unwrap().to_vec().unwrap();

    println!("  GPU Cholesky: {:?}", gpu_data);
    println!("  CPU Cholesky: {:?}", cpu_data);

    assert_close("cholesky", &gpu_data, &cpu_data, 1e-4);
    println!("  PASS: GPU and CPU produce identical Cholesky results");
}

// ─── Softmax Parity ──────────────────────────────────────────────────────────

#[tokio::test]
async fn test_softmax_gpu_cpu_parity() {
    let gpu = try_gpu().await;
    let cpu = try_cpu().await;

    if gpu.is_none() || cpu.is_none() {
        println!("SKIP: Need both GPU and CPU adapters");
        return;
    }

    let gpu = gpu.unwrap();
    let cpu = cpu.unwrap();

    println!("Softmax parity: {} vs {}", gpu.name(), cpu.name());

    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    // GPU
    let t_gpu = Tensor::from_vec_on(data.clone(), vec![5], gpu.clone())
        .await
        .unwrap();
    let gpu_data = t_gpu.softmax().unwrap().to_vec().unwrap();

    // CPU
    let t_cpu = Tensor::from_vec_on(data.clone(), vec![5], cpu.clone())
        .await
        .unwrap();
    let cpu_data = t_cpu.softmax().unwrap().to_vec().unwrap();

    println!("  GPU softmax: {:?}", gpu_data);
    println!("  CPU softmax: {:?}", cpu_data);

    assert_close("softmax", &gpu_data, &cpu_data, 1e-4);

    // Verify sum to 1.0
    let gpu_sum: f32 = gpu_data.iter().sum();
    let cpu_sum: f32 = cpu_data.iter().sum();
    assert!(
        (gpu_sum - 1.0).abs() < 1e-4,
        "GPU softmax should sum to 1.0, got {}",
        gpu_sum
    );
    assert!(
        (cpu_sum - 1.0).abs() < 1e-4,
        "CPU softmax should sum to 1.0, got {}",
        cpu_sum
    );

    println!("  PASS: GPU and CPU produce identical softmax results");
}

// ─── ToadStool Guided Device Selection ───────────────────────────────────────

#[cfg(feature = "toadstool")]
#[tokio::test]
async fn test_toadstool_guided_device_selection() {
    use barracuda::device::{discover_devices, select_best_device, HardwareWorkload};

    // ToadStool discovers hardware
    let hw = discover_devices().expect("ToadStool discovery failed");
    println!("ToadStool discovered {} devices", hw.device_count());
    for device in hw.devices() {
        println!("  - {} ({:?})", device.name, device.hardware_type);
    }

    // ToadStool recommends device for tensor ops
    let selection =
        select_best_device(HardwareWorkload::TensorOps).expect("Device selection failed");
    println!("ToadStool recommends: {:?} for TensorOps", selection);

    // BarraCuda creates device from recommendation
    let device = WgpuDevice::from_selection(selection).await.unwrap();
    println!(
        "BarraCuda device: {} ({:?})",
        device.name(),
        device.device_type()
    );

    // Verify it works by running a simple operation
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let device_arc = Arc::new(device);
    let tensor = Tensor::from_vec_on(data.clone(), vec![2, 2], device_arc)
        .await
        .unwrap();
    let result = tensor.to_vec().unwrap();
    assert_eq!(result, data);

    println!("  ToadStool -> BarraCuda integration: PASS");
}

// ─── Performance Comparison (not parity) ─────────────────────────────────────

#[tokio::test]
async fn test_performance_comparison() {
    let gpu = try_gpu().await;
    let cpu = try_cpu().await;

    println!("\n=== Cross-Hardware Performance Report ===\n");

    // Test on whatever hardware is available
    let devices: Vec<(String, Arc<WgpuDevice>)> = {
        let mut d = Vec::new();
        if let Some(g) = gpu {
            d.push((format!("GPU ({})", g.name()), g));
        }
        if let Some(c) = cpu {
            d.push((format!("CPU ({})", c.name()), c));
        }
        if d.is_empty() {
            // Fall back to shared test pool
            let auto = barracuda::device::test_pool::get_test_device().await;
            d.push((format!("Auto ({})", auto.name()), auto));
        }
        d
    };

    // Benchmark: 64x64 matmul
    let size = 64;
    let a_data: Vec<f32> = (0..size * size).map(|i| (i as f32) * 0.01).collect();
    let b_data: Vec<f32> = (0..size * size).map(|i| (i as f32) * 0.01 + 0.5).collect();

    for (name, device) in &devices {
        let _warmup = Tensor::from_vec_on(a_data.clone(), vec![size, size], device.clone())
            .await
            .unwrap();
        let b = Tensor::from_vec_on(b_data.clone(), vec![size, size], device.clone())
            .await
            .unwrap();

        let start = std::time::Instant::now();
        let iterations = 10;
        for _ in 0..iterations {
            let a_clone = Tensor::from_vec_on(a_data.clone(), vec![size, size], device.clone())
                .await
                .unwrap();
            let _result = a_clone.matmul(&b).unwrap();
        }
        let elapsed = start.elapsed();

        println!(
            "  {}: {}x{} matmul x{} = {:.2} ms ({:.2} ms/op)",
            name,
            size,
            size,
            iterations,
            elapsed.as_secs_f64() * 1000.0,
            elapsed.as_secs_f64() * 1000.0 / iterations as f64,
        );
    }

    println!("\n  Hardware decides its own performance.");
    println!("  Same WGSL, same math, different speed.\n");
}
