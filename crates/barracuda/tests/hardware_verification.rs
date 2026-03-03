// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hardware Verification Tests
//!
//! Comprehensive tests for heterogeneous hardware setups:
//! - 2 GPUs from different vendors/eras (e.g., NVIDIA RTX + AMD RX)
//! - 2 NPUs (Akida)
//! - Cross-vendor math parity
//! - Routing verification
//!
//! ## Test Categories
//!
//! 1. **Discovery Tests** - Verify all hardware is detected
//! 2. **Parity Tests** - Same math produces same results across vendors
//! 3. **Routing Tests** - barraCuda routes to correct hardware
//! 4. **Performance Tests** - Verify expected performance characteristics
//!
//! ## Running Tests
//!
//! ```bash
//! cargo test -p barracuda --test hardware_verification -- --nocapture
//! ```

mod common;

use barracuda::device::{
    ComputeWorkload, Device, DeviceSelection, HardwareWorkload, KernelRouter, KernelTarget,
    WgpuDevice,
};
use barracuda::tensor::Tensor;
use std::collections::HashMap;
use std::sync::Arc;

// ============================================================================
// Test Infrastructure
// ============================================================================

/// Hardware inventory for test reporting
#[derive(Debug)]
struct HardwareInventory {
    gpus: Vec<GpuInfo>,
    npus: Vec<NpuInfo>,
}

#[derive(Debug)]
struct GpuInfo {
    name: String,
    vendor: String,
    device_type: wgpu::DeviceType,
    backend: wgpu::Backend,
    adapter_index: usize,
}

#[derive(Debug)]
struct NpuInfo {
    name: String,
    device_path: String,
}

impl HardwareInventory {
    fn discover() -> Self {
        let adapters = WgpuDevice::enumerate_adapters();

        let mut gpus = Vec::new();
        for (idx, info) in adapters.iter().enumerate() {
            let vendor = if info.name.to_lowercase().contains("nvidia") {
                "NVIDIA"
            } else if info.name.to_lowercase().contains("amd")
                || info.name.to_lowercase().contains("radeon")
            {
                "AMD"
            } else if info.name.to_lowercase().contains("intel") {
                "Intel"
            } else {
                "Unknown"
            };

            // Only count discrete/integrated GPUs
            if matches!(
                info.device_type,
                wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
            ) || (info.device_type == wgpu::DeviceType::Other
                && (vendor == "NVIDIA" || vendor == "AMD"))
            {
                gpus.push(GpuInfo {
                    name: info.name.clone(),
                    vendor: vendor.to_string(),
                    device_type: info.device_type,
                    backend: info.backend,
                    adapter_index: idx,
                });
            }
        }

        // Scan for NPUs
        let mut npus = Vec::new();
        for i in 0..16 {
            let path = format!("/dev/akida{}", i);
            if std::path::Path::new(&path).exists() {
                npus.push(NpuInfo {
                    name: format!("Akida NPU {}", i),
                    device_path: path,
                });
            }
        }

        Self { gpus, npus }
    }

    fn print_report(&self) {
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║           HARDWARE VERIFICATION INVENTORY                     ║");
        println!("╠══════════════════════════════════════════════════════════════╣");

        println!(
            "║ GPUs Detected: {}                                             ",
            self.gpus.len()
        );
        for gpu in &self.gpus {
            println!(
                "║   [{:2}] {} ({}) - {:?}/{:?}",
                gpu.adapter_index, gpu.name, gpu.vendor, gpu.device_type, gpu.backend
            );
        }

        println!(
            "║ NPUs Detected: {}                                             ",
            self.npus.len()
        );
        for npu in &self.npus {
            println!("║   {} @ {}", npu.name, npu.device_path);
        }

        println!("╚══════════════════════════════════════════════════════════════╝\n");
    }

    fn has_multi_gpu(&self) -> bool {
        self.gpus.len() >= 2
    }

    fn has_npu(&self) -> bool {
        !self.npus.is_empty()
    }
}

/// Runs a cross-vendor GPU test, skipping when adapters lack required features
/// (Validation Error, doesn't support) in addition to NVK driver issues.
fn run_cross_vendor_resilient_async<F, Fut>(f: F) -> bool
where
    F: FnOnce() -> Fut + Send + std::panic::UnwindSafe + 'static,
    Fut: std::future::Future<Output = ()>,
{
    let handle = std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to build test runtime");
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rt.block_on(f())))
    });

    match handle.join().expect("test thread panicked") {
        Ok(()) => true,
        Err(e) => {
            let msg = e
                .downcast_ref::<String>()
                .map(|s| s.as_str())
                .or_else(|| e.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic");
            let skip = msg.contains("does not exist")
                || msg.contains("device lost")
                || msg.contains("Parent device")
                || msg.contains("Validation Error")
                || msg.contains("doesn't support")
                || msg.contains("is no longer alive");
            if skip {
                eprintln!("Cross-vendor test skipped: {msg}");
                false
            } else {
                std::panic::resume_unwind(e);
            }
        }
    }
}

/// Try to create a WgpuDevice from adapter index. Returns None if the adapter
/// doesn't support required features (e.g. BUFFER_STORAGE, COMPUTE_SHADER) or
/// device creation otherwise fails. Catches panics from wgpu validation errors.
async fn try_create_device(adapter_index: usize) -> Option<Arc<WgpuDevice>> {
    let result = tokio::task::spawn_blocking(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to build runtime");
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            rt.block_on(WgpuDevice::from_adapter_index(adapter_index))
        }))
    })
    .await
    .ok();

    match result {
        Some(Ok(Ok(device))) => Some(Arc::new(device)),
        Some(Ok(Err(e))) => {
            eprintln!(
                "Adapter {}: skipped (unsupported features): {}",
                adapter_index, e
            );
            None
        }
        Some(Err(_)) => {
            eprintln!(
                "Adapter {}: skipped (panic during device creation)",
                adapter_index
            );
            None
        }
        None => None,
    }
}

/// Compare f32 slices with tolerance
fn assert_close(label: &str, a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "{}: length mismatch", label);
    let mut max_diff = 0.0f32;
    let mut max_idx = 0;
    for (i, (va, vb)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (va - vb).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
    }
    assert!(
        max_diff < tol,
        "{}: max diff {} at idx {} (tol {})",
        label,
        max_diff,
        max_idx,
        tol
    );
}

// ============================================================================
// Discovery Tests
// ============================================================================

#[test]
fn test_hardware_discovery_report() {
    let inventory = HardwareInventory::discover();
    inventory.print_report();

    // Should find at least one compute device
    assert!(
        !inventory.gpus.is_empty() || Device::CPU.is_available(),
        "No compute hardware detected"
    );
}

#[tokio::test]
async fn test_all_gpus_can_create_device() {
    if !common::run_gpu_resilient_async(|| async {
        let inventory = HardwareInventory::discover();

        println!("Testing device creation for {} GPUs", inventory.gpus.len());

        let mut successful = 0;
        let mut failed = Vec::new();

        for gpu in &inventory.gpus {
            match WgpuDevice::from_adapter_index(gpu.adapter_index).await {
                Ok(_device) => {
                    println!("  ✓ {} - Device created", gpu.name);
                    successful += 1;
                }
                Err(e) => {
                    println!("  ✗ {} - Failed: {}", gpu.name, e);
                    failed.push(gpu.name.clone());
                }
            }
        }

        println!(
            "\nSummary: {}/{} devices created successfully",
            successful,
            inventory.gpus.len()
        );

        // At least one GPU should work
        assert!(successful > 0, "No GPUs could create devices: {:?}", failed);
    }) {
        return;
    }
}

#[test]
fn test_kernel_router_creation() {
    let router = KernelRouter::new();
    assert!(router.is_ok(), "KernelRouter should initialize");

    let router = router.unwrap();
    println!("Available NPU models: {:?}", router.available_npu_models());
}

// ============================================================================
// Cross-Vendor GPU Parity Tests
// ============================================================================

#[tokio::test]
async fn test_cross_vendor_matmul_parity() {
    if !run_cross_vendor_resilient_async(|| async {
        let inventory = HardwareInventory::discover();

        if !inventory.has_multi_gpu() {
            println!("SKIP: Need 2+ GPUs for cross-vendor parity test");
            return;
        }

        println!("\n=== Cross-Vendor Matmul Parity Test ===\n");

        let mut devices: Vec<(String, Arc<WgpuDevice>)> = Vec::new();
        for gpu in &inventory.gpus {
            if let Some(device) = try_create_device(gpu.adapter_index).await {
                devices.push((format!("{} ({})", gpu.name, gpu.vendor), device));
            }
        }

        if devices.len() < 2 {
            println!("SKIP: Need 2+ working GPUs for parity test");
            return;
        }

        let size = 64;
        let a_data: Vec<f32> = (0..size * size).map(|i| (i as f32) * 0.01).collect();
        let b_data: Vec<f32> = (0..size * size).map(|i| (i as f32) * 0.01 + 0.5).collect();

        let mut results: HashMap<String, Vec<f32>> = HashMap::new();

        for (name, device) in &devices {
            let a = Tensor::from_vec_on(a_data.clone(), vec![size, size], device.clone())
                .await
                .unwrap();
            let b = Tensor::from_vec_on(b_data.clone(), vec![size, size], device.clone())
                .await
                .unwrap();

            let result = a.matmul(&b).unwrap().to_vec().unwrap();
            println!("  {} - computed (first 4 values: {:?})", name, &result[..4]);
            results.insert(name.clone(), result);
        }

        let device_names: Vec<_> = results.keys().cloned().collect();
        for i in 0..device_names.len() {
            for j in (i + 1)..device_names.len() {
                let name_a = &device_names[i];
                let name_b = &device_names[j];
                let result_a = &results[name_a];
                let result_b = &results[name_b];

                assert_close(
                    &format!("{} vs {}", name_a, name_b),
                    result_a,
                    result_b,
                    1e-3,
                );
                println!("  ✓ {} matches {}", name_a, name_b);
            }
        }

        println!("\n  PASS: All GPUs produce identical matmul results\n");
    }) {
        return;
    }
}

#[tokio::test]
async fn test_cross_vendor_cholesky_parity() {
    if !run_cross_vendor_resilient_async(|| async {
        let inventory = HardwareInventory::discover();

        if !inventory.has_multi_gpu() {
            println!("SKIP: Need 2+ GPUs for cross-vendor parity test");
            return;
        }

        println!("\n=== Cross-Vendor Cholesky Parity Test ===\n");

        let mut devices: Vec<(String, Arc<WgpuDevice>)> = Vec::new();
        for gpu in &inventory.gpus {
            if let Some(device) = try_create_device(gpu.adapter_index).await {
                devices.push((gpu.name.clone(), device));
            }
        }

        if devices.len() < 2 {
            println!("SKIP: Need 2+ working GPUs");
            return;
        }

        let spd = vec![4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0];

        let mut results: HashMap<String, Vec<f32>> = HashMap::new();

        for (name, device) in &devices {
            let a = Tensor::from_vec_on(spd.clone(), vec![3, 3], device.clone())
                .await
                .unwrap();

            let result = a.cholesky().unwrap().to_vec().unwrap();
            println!("  {} Cholesky L: {:?}", name, result);
            results.insert(name.clone(), result);
        }

        let device_names: Vec<_> = results.keys().cloned().collect();
        for i in 0..device_names.len() {
            for j in (i + 1)..device_names.len() {
                let name_a = &device_names[i];
                let name_b = &device_names[j];
                assert_close(
                    &format!("{} vs {}", name_a, name_b),
                    &results[name_a],
                    &results[name_b],
                    1e-4,
                );
                println!("  ✓ {} matches {}", name_a, name_b);
            }
        }

        println!("\n  PASS: Cross-vendor Cholesky parity verified\n");
    }) {
        return;
    }
}

#[tokio::test]
async fn test_cross_vendor_softmax_parity() {
    if !run_cross_vendor_resilient_async(|| async {
        let inventory = HardwareInventory::discover();

        if !inventory.has_multi_gpu() {
            println!("SKIP: Need 2+ GPUs for cross-vendor parity test");
            return;
        }

        println!("\n=== Cross-Vendor Softmax Parity Test ===\n");

        let mut devices: Vec<(String, Arc<WgpuDevice>)> = Vec::new();
        for gpu in &inventory.gpus {
            if let Some(device) = try_create_device(gpu.adapter_index).await {
                devices.push((gpu.name.clone(), device));
            }
        }

        if devices.len() < 2 {
            return;
        }

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let mut results: HashMap<String, Vec<f32>> = HashMap::new();

        for (name, device) in &devices {
            let t = Tensor::from_vec_on(data.clone(), vec![8], device.clone())
                .await
                .unwrap();

            let result = t.softmax().unwrap().to_vec().unwrap();
            let sum: f32 = result.iter().sum();
            println!("  {} softmax sum: {:.6}", name, sum);
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "{} softmax should sum to 1.0",
                name
            );
            results.insert(name.clone(), result);
        }

        let device_names: Vec<_> = results.keys().cloned().collect();
        for i in 0..device_names.len() {
            for j in (i + 1)..device_names.len() {
                assert_close(
                    &format!("softmax {} vs {}", device_names[i], device_names[j]),
                    &results[&device_names[i]],
                    &results[&device_names[j]],
                    1e-4,
                );
            }
        }

        println!("  PASS: Cross-vendor softmax parity verified\n");
    }) {
        return;
    }
}

// ============================================================================
// Kernel Router Tests
// ============================================================================

#[test]
fn test_kernel_router_dense_workloads_to_wgsl() {
    let router = KernelRouter::default();

    // Dense workloads should ALWAYS go to WGSL (GPU/CPU)
    let dense_workloads = vec![
        ComputeWorkload::DenseMatmul {
            m: 1024,
            n: 1024,
            k: 1024,
        },
        ComputeWorkload::FFT {
            size: 1024,
            batch_count: 10,
        },
        ComputeWorkload::PhysicsForce {
            particle_count: 10000,
            force_type: "lennard_jones".to_string(),
        },
        ComputeWorkload::Eigendecomp { matrix_size: 256 },
        ComputeWorkload::LinearSolve { system_size: 512 },
    ];

    for workload in dense_workloads {
        let target = router.route(&workload).unwrap();
        match target {
            KernelTarget::Wgsl { device, .. } => {
                assert!(
                    device.supports_wgsl(),
                    "Dense workload {:?} should route to WGSL-capable device",
                    workload
                );
            }
            other => {
                panic!(
                    "Dense workload {:?} should route to WGSL, got {:?}",
                    workload, other
                );
            }
        }
    }

    println!("✓ All dense workloads correctly routed to WGSL");
}

#[test]
fn test_kernel_router_small_workloads_to_cpu() {
    let router = KernelRouter::default();

    // Small workloads should prefer CPU (avoid GPU dispatch overhead).
    // Threshold: DenseMatmul uses m*n*k < 1000; Eigendecomp < 128; LinearSolve < 256.
    let small_workloads = vec![
        ComputeWorkload::DenseMatmul { m: 9, n: 9, k: 9 }, // 729 < 1000
        ComputeWorkload::Eigendecomp { matrix_size: 32 },
        ComputeWorkload::LinearSolve { system_size: 64 },
    ];

    for workload in small_workloads {
        let target = router.route(&workload).unwrap();
        if let KernelTarget::Wgsl { device, .. } = target {
            assert_eq!(
                device,
                DeviceSelection::Cpu,
                "Small workload {:?} should route to CPU",
                workload
            );
        }
    }

    println!("✓ Small workloads correctly routed to CPU");
}

#[test]
fn test_kernel_router_npu_fallback() {
    let router = KernelRouter::default();

    // NPU workloads without models should fall back to WGSL
    let npu_workloads = vec![
        ComputeWorkload::SparseInference {
            input_sparsity: 0.95,
            model_name: "nonexistent_model".to_string(),
        },
        ComputeWorkload::ReservoirState {
            reservoir_size: 1000,
            input_dim: 100,
        },
        ComputeWorkload::BinaryPrescreen {
            input_count: 10000,
            threshold: 0.5,
        },
    ];

    for workload in npu_workloads {
        let target = router.route(&workload).unwrap();
        match target {
            KernelTarget::Wgsl { .. } => {
                // Expected fallback to WGSL
            }
            KernelTarget::Npu { .. } => {
                // Also valid if NPU models are present
            }
            other => {
                panic!(
                    "NPU workload {:?} should route to Wgsl or Npu, got {:?}",
                    workload, other
                );
            }
        }
    }

    println!("✓ NPU workloads correctly handle fallback");
}

// ============================================================================
// Performance Characterization
// ============================================================================

#[tokio::test]
async fn test_multi_gpu_performance_characterization() {
    if !common::run_gpu_resilient_async(|| async {
        let inventory = HardwareInventory::discover();

        if !inventory.has_multi_gpu() {
            println!("SKIP: Need 2+ GPUs for performance characterization");
            return;
        }

        println!("\n=== Multi-GPU Performance Characterization ===\n");

        let mut devices: Vec<(String, Arc<WgpuDevice>)> = Vec::new();
        for gpu in &inventory.gpus {
            if let Some(device) = try_create_device(gpu.adapter_index).await {
                devices.push((format!("{} ({})", gpu.name, gpu.vendor), device));
            }
        }

        let size = 256;
        let a_data: Vec<f32> = (0..size * size).map(|i| (i as f32) * 0.001).collect();
        let b_data: Vec<f32> = (0..size * size).map(|i| (i as f32) * 0.001 + 0.5).collect();

        println!("Benchmark: {}x{} matmul x 20 iterations\n", size, size);

        for (name, device) in &devices {
            let _warmup = Tensor::from_vec_on(a_data.clone(), vec![size, size], device.clone())
                .await
                .unwrap();

            let b = Tensor::from_vec_on(b_data.clone(), vec![size, size], device.clone())
                .await
                .unwrap();

            let start = std::time::Instant::now();
            let iterations = 20;
            for _ in 0..iterations {
                let a = Tensor::from_vec_on(a_data.clone(), vec![size, size], device.clone())
                    .await
                    .unwrap();
                let _result = a.matmul(&b).unwrap();
            }
            let elapsed = start.elapsed();

            let total_ms = elapsed.as_secs_f64() * 1000.0;
            let per_op_ms = total_ms / iterations as f64;
            let gflops =
                (2.0 * (size as f64).powi(3) * iterations as f64) / elapsed.as_secs_f64() / 1e9;

            println!(
                "  {}: {:.2} ms total, {:.3} ms/op, {:.2} GFLOP/s",
                name, total_ms, per_op_ms, gflops
            );
        }

        println!("\n  Performance characterization complete.\n");
    }) {
        return;
    }
}

// ============================================================================
// NPU Detection and Routing
// ============================================================================

#[test]
fn test_npu_detection() {
    let inventory = HardwareInventory::discover();

    println!("\n=== NPU Detection ===\n");
    println!("NPUs found: {}", inventory.npus.len());

    for npu in &inventory.npus {
        println!("  - {} @ {}", npu.name, npu.device_path);
    }

    if inventory.has_npu() {
        println!("\n  NPU hardware detected - routing tests enabled\n");
    } else {
        println!("\n  No NPU hardware - NPU routing tests will be skipped\n");
    }
}

#[test]
fn test_kernel_router_npu_capability_check() {
    let router = KernelRouter::default();
    let _inventory = HardwareInventory::discover();

    // Dense workloads can NEVER go to NPU
    let dense = ComputeWorkload::DenseMatmul {
        m: 1024,
        n: 1024,
        k: 1024,
    };
    assert!(
        !router.can_route_to_npu(&dense),
        "Dense matmul should not be NPU-routable"
    );

    // Sparse inference MAY go to NPU if model exists
    let sparse = ComputeWorkload::SparseInference {
        input_sparsity: 0.95,
        model_name: "test".to_string(),
    };
    // This will be false unless we register an NPU model
    let can_route = router.can_route_to_npu(&sparse);
    println!(
        "Sparse inference can route to NPU: {} (expected: depends on model registration)",
        can_route
    );
}
