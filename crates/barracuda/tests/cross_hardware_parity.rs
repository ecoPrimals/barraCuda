// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-Hardware Parity Tests
//!
//! Proves `BarraCuda` produces IDENTICAL results across hardware:
//! - Same WGSL shader
//! - Same input data
//! - GPU vs CPU → must match within f32 tolerance
//!
//! This is the core proof that "any math on any hardware" works.

#![expect(clippy::unwrap_used, reason = "tests")]
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
    let adapters = WgpuDevice::enumerate_adapters().await;

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
        let _ = (&info.name, hw_type, &info.backend);
    }

    let _ = (has_gpu, has_cpu);

    assert!(!adapters.is_empty(), "Should find at least one adapter");
}

// ─── Matmul Parity ───────────────────────────────────────────────────────────

#[tokio::test]
async fn test_matmul_gpu_cpu_parity() {
    let gpu = try_gpu().await;
    let cpu = try_cpu().await;

    if gpu.is_none() || cpu.is_none() {
        return;
    }

    let gpu = gpu.unwrap();
    let cpu = cpu.unwrap();

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

    assert_close("matmul", &gpu_data, &cpu_data, 1e-4);
}

// ─── Element-wise Add Parity ─────────────────────────────────────────────────

#[tokio::test]
async fn test_add_gpu_cpu_parity() {
    let gpu = try_gpu().await;
    let cpu = try_cpu().await;

    if gpu.is_none() || cpu.is_none() {
        return;
    }

    let gpu = gpu.unwrap();
    let cpu = cpu.unwrap();

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
}

// ─── Cholesky Parity ─────────────────────────────────────────────────────────

#[tokio::test]
async fn test_cholesky_gpu_cpu_parity() {
    let gpu = try_gpu().await;
    let cpu = try_cpu().await;

    if gpu.is_none() || cpu.is_none() {
        return;
    }

    let gpu = gpu.unwrap();
    let cpu = cpu.unwrap();

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

    assert_close("cholesky", &gpu_data, &cpu_data, 1e-4);
}

// ─── Softmax Parity ──────────────────────────────────────────────────────────

#[tokio::test]
async fn test_softmax_gpu_cpu_parity() {
    let gpu = try_gpu().await;
    let cpu = try_cpu().await;

    if gpu.is_none() || cpu.is_none() {
        return;
    }

    let gpu = gpu.unwrap();
    let cpu = cpu.unwrap();

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

    assert_close("softmax", &gpu_data, &cpu_data, 1e-4);

    // Verify sum to 1.0
    let gpu_sum: f32 = gpu_data.iter().sum();
    let cpu_sum: f32 = cpu_data.iter().sum();
    assert!(
        (gpu_sum - 1.0).abs() < 1e-4,
        "GPU softmax should sum to 1.0, got {gpu_sum}"
    );
    assert!(
        (cpu_sum - 1.0).abs() < 1e-4,
        "CPU softmax should sum to 1.0, got {cpu_sum}"
    );
}

// ─── Performance Comparison (not parity) ─────────────────────────────────────

#[tokio::test]
async fn test_performance_comparison() {
    let gpu = try_gpu().await;
    let cpu = try_cpu().await;

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
    let b_data: Vec<f32> = (0..size * size)
        .map(|i| (i as f32).mul_add(0.01, 0.5))
        .collect();

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

        let _ = (name, elapsed, iterations);
    }
}
