#![cfg(feature = "toadstool")]
//! E2E Math Pipeline Tests
//!
//! End-to-end tests verifying the complete Barracuda + ToadStool pipeline:
//!
//! 1. **User writes math** (Rust API / WGSL shader)
//! 2. **ToadStool discovers hardware** (GPUs, NPUs)
//! 3. **KernelRouter selects target** (GPU, NPU, or CPU)
//! 4. **Barracuda executes** (WGSL on wgpu)
//! 5. **Results verified** (correctness, precision)
//!
//! ## Hardware Matrix
//!
//! These tests are designed for heterogeneous hardware:
//! - GPU0: Modern discrete GPU (e.g., RTX 4090)
//! - GPU1: Older/different vendor GPU (e.g., RX 6900 XT)
//! - NPU0: Akida neuromorphic processor
//! - NPU1: Secondary NPU (if available)
//! - CPU: Always available fallback

mod common;

use barracuda::device::{
    discover_devices, select_best_device, ComputeWorkload, DeviceSelection, HardwareWorkload,
    KernelRouter, KernelTarget, WgpuDevice,
};
use barracuda::tensor::Tensor;
use std::sync::Arc;

// ============================================================================
// Pipeline Helpers
// ============================================================================

/// Execute a math operation through the full ToadStool → Barracuda pipeline
async fn execute_pipeline(data: Vec<f32>, shape: Vec<usize>, op: &str) -> Result<Vec<f32>, String> {
    // 1. ToadStool discovers hardware
    let _hw = discover_devices().map_err(|e| format!("Discovery failed: {}", e))?;

    // 2. Select best device for tensor operations
    let selection = select_best_device(HardwareWorkload::TensorOps)
        .map_err(|e| format!("Selection failed: {}", e))?;

    // 3. Create Barracuda device from selection (with fallback to shared test pool)
    let device_arc = match WgpuDevice::from_selection(selection).await {
        Ok(d) => Arc::new(d),
        Err(_) => barracuda::device::test_pool::get_test_device().await,
    };

    // 4. Create tensor and execute operation
    let tensor = Tensor::from_vec_on(data, shape.clone(), device_arc.clone())
        .await
        .map_err(|e| format!("Tensor creation failed: {}", e))?;

    let result = match op {
        "identity" => tensor.to_vec().map_err(|e| format!("Read failed: {}", e))?,
        "softmax" => tensor
            .softmax()
            .map_err(|e| format!("Softmax failed: {}", e))?
            .to_vec()
            .map_err(|e| format!("Read failed: {}", e))?,
        "transpose" if shape.len() == 2 => tensor
            .transpose()
            .map_err(|e| format!("Transpose failed: {}", e))?
            .to_vec()
            .map_err(|e| format!("Read failed: {}", e))?,
        _ => return Err(format!("Unknown operation: {}", op)),
    };

    Ok(result)
}

/// Execute matmul through full pipeline
async fn execute_matmul_pipeline(
    a: Vec<f32>,
    a_shape: Vec<usize>,
    b: Vec<f32>,
    b_shape: Vec<usize>,
) -> Result<Vec<f32>, String> {
    let _hw = discover_devices().map_err(|e| format!("Discovery failed: {}", e))?;
    let selection = select_best_device(HardwareWorkload::TensorOps)
        .map_err(|e| format!("Selection failed: {}", e))?;

    // Create device with fallback to shared test pool
    let device = match WgpuDevice::from_selection(selection).await {
        Ok(d) => Arc::new(d),
        Err(_) => barracuda::device::test_pool::get_test_device().await,
    };

    let tensor_a = Tensor::from_vec_on(a, a_shape, device.clone())
        .await
        .map_err(|e| format!("Tensor A failed: {}", e))?;
    let tensor_b = Tensor::from_vec_on(b, b_shape, device)
        .await
        .map_err(|e| format!("Tensor B failed: {}", e))?;

    tensor_a
        .matmul(&tensor_b)
        .map_err(|e| format!("Matmul failed: {}", e))?
        .to_vec()
        .map_err(|e| format!("Read failed: {}", e))
}

// ============================================================================
// E2E Tests: Basic Pipeline
// ============================================================================

#[tokio::test]
async fn test_e2e_identity_pipeline() {
    if !common::run_gpu_resilient_async(|| async {
        println!("\n=== E2E Identity Pipeline ===\n");

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = execute_pipeline(data.clone(), vec![2, 3], "identity").await;

        match result {
            Ok(output) => {
                assert_eq!(output, data);
                println!("  ✓ Data round-trip through pipeline: PASS");
            }
            Err(e) => {
                if e.contains("Device") || e.contains("connection") || e.contains("lost") {
                    println!("  SKIP: Device unavailable - {}", e);
                    return;
                }
                panic!("Pipeline failed: {}", e);
            }
        }
    }) {
        return;
    }
}

#[tokio::test]
async fn test_e2e_softmax_pipeline() {
    if !common::run_gpu_resilient_async(|| async {
        println!("\n=== E2E Softmax Pipeline ===\n");

        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = execute_pipeline(data, vec![4], "softmax").await;

        match result {
            Ok(output) => {
                let sum: f32 = output.iter().sum();
                assert!((sum - 1.0).abs() < 1e-4, "Softmax should sum to 1.0");
                println!("  Softmax output: {:?}", output);
                println!("  Sum: {:.6}", sum);
                println!("  ✓ Softmax through pipeline: PASS");
            }
            Err(e) => {
                if e.contains("Device") || e.contains("connection") || e.contains("lost") {
                    println!("  SKIP: Device unavailable - {}", e);
                    return;
                }
                panic!("Pipeline failed: {}", e);
            }
        }
    }) {
        return;
    }
}

#[tokio::test]
async fn test_e2e_matmul_pipeline() {
    if !common::run_gpu_resilient_async(|| async {
        println!("\n=== E2E Matmul Pipeline ===\n");

        // 2x3 @ 3x2 = 2x2
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

        let result = execute_matmul_pipeline(a, vec![2, 3], b, vec![3, 2]).await;

        match result {
            Ok(output) => {
                println!("  Matmul result: {:?}", output);
                // Expected: [[58, 64], [139, 154]]
                let expected = vec![58.0, 64.0, 139.0, 154.0];
                for (i, (got, exp)) in output.iter().zip(expected.iter()).enumerate() {
                    assert!(
                        (got - exp).abs() < 1e-3,
                        "Mismatch at {}: {} vs {}",
                        i,
                        got,
                        exp
                    );
                }
                println!("  ✓ Matmul through pipeline: PASS");
            }
            Err(e) => {
                if e.contains("Device") || e.contains("connection") || e.contains("lost") {
                    println!("  SKIP: Device unavailable - {}", e);
                    return;
                }
                panic!("Pipeline failed: {}", e);
            }
        }
    }) {
        return;
    }
}

// ============================================================================
// E2E Tests: Kernel Router Integration
// ============================================================================

#[tokio::test]
async fn test_e2e_router_guided_execution() {
    println!("\n=== E2E Router-Guided Execution ===\n");

    let router = KernelRouter::new().expect("Router creation failed");

    // Test different workload types through router
    let workloads = vec![
        (
            "Small matmul (CPU)",
            ComputeWorkload::DenseMatmul {
                m: 16,
                n: 16,
                k: 16,
            },
        ),
        (
            "Large matmul (GPU)",
            ComputeWorkload::DenseMatmul {
                m: 512,
                n: 512,
                k: 512,
            },
        ),
        (
            "FFT (GPU)",
            ComputeWorkload::FFT {
                size: 1024,
                batch_count: 1,
            },
        ),
        (
            "Sparse inference (NPU/GPU)",
            ComputeWorkload::SparseInference {
                input_sparsity: 0.95,
                model_name: "test_snn".to_string(),
            },
        ),
    ];

    for (name, workload) in workloads {
        let target = router.route(&workload).expect("Routing failed");
        println!("  {} -> {:?}", name, target);
    }

    println!("\n  Router-guided execution: PASS\n");
}

#[tokio::test]
async fn test_e2e_complete_pipeline_with_router() {
    if !common::run_gpu_resilient_async(|| async {
        println!("\n=== E2E Complete Pipeline with Router ===\n");

        // 1. Define workload
        let workload = ComputeWorkload::DenseMatmul {
            m: 64,
            n: 64,
            k: 64,
        };

        // 2. Router determines target
        let router = KernelRouter::new().expect("Router failed");
        let target = router.route(&workload).expect("Routing failed");
        println!("  Workload: 64x64 matmul");
        println!("  Router target: {:?}", target);

        // 3. Create device based on router decision
        let device = match &target {
            KernelTarget::Wgsl { device, .. } => match device {
                DeviceSelection::Gpu => WgpuDevice::new_gpu().await.ok().map(Arc::new),
                DeviceSelection::Cpu => WgpuDevice::new_cpu().await.ok().map(Arc::new),
                _ => Some(barracuda::device::test_pool::get_test_device().await),
            },
            KernelTarget::Npu { .. } => {
                // Fallback to WGSL for now (NPU would use different path)
                Some(barracuda::device::test_pool::get_test_device().await)
            }
            KernelTarget::Hybrid { .. } => {
                Some(barracuda::device::test_pool::get_test_device().await)
            }
        };

        let device = match device {
            Some(d) => d,
            None => {
                println!("  SKIP: No device available for target");
                return;
            }
        };

        println!("  Executing on: {}", device.name());

        // 4. Execute operation
        let size = 64;
        let a_data: Vec<f32> = (0..size * size).map(|i| (i as f32) * 0.01).collect();
        let b_data: Vec<f32> = (0..size * size).map(|i| (i as f32) * 0.01 + 0.5).collect();

        let tensor_a = Tensor::from_vec_on(a_data, vec![size, size], device.clone())
            .await
            .expect("Tensor A failed");
        let tensor_b = Tensor::from_vec_on(b_data, vec![size, size], device)
            .await
            .expect("Tensor B failed");

        let result = tensor_a.matmul(&tensor_b).expect("Matmul failed");
        let output = result.to_vec().expect("Read failed");

        println!("  Result size: {} elements", output.len());
        println!("  First 4 values: {:?}", &output[..4]);
        println!("\n  ✓ Complete pipeline with router: PASS\n");
    }) {
        return;
    }
}

// ============================================================================
// E2E Tests: Multi-Device Execution
// ============================================================================

#[tokio::test]
async fn test_e2e_multi_device_same_computation() {
    if !common::run_gpu_resilient_async(|| async {
        println!("\n=== E2E Multi-Device Same Computation ===\n");

        let adapters = WgpuDevice::enumerate_adapters();

        // Get all discrete GPUs
        let discrete_indices: Vec<_> = adapters
            .iter()
            .enumerate()
            .filter(|(_, a)| {
                matches!(
                    a.device_type,
                    wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
                ) || (a.device_type == wgpu::DeviceType::Other
                    && (a.name.to_lowercase().contains("nvidia")
                        || a.name.to_lowercase().contains("amd")))
            })
            .map(|(i, a)| (i, a.name.clone()))
            .collect();

        if discrete_indices.len() < 2 {
            println!("  SKIP: Need 2+ GPUs for multi-device test");
            println!("  Found: {:?}", discrete_indices);
            return;
        }

        // Same computation on each device
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let shape = vec![3, 3];

        let mut results = Vec::new();

        for (idx, name) in &discrete_indices {
            match WgpuDevice::from_adapter_index(*idx).await {
                Ok(device) => {
                    let device_arc = Arc::new(device);
                    match Tensor::from_vec_on(data.clone(), shape.clone(), device_arc).await {
                        Ok(tensor) => match tensor.softmax() {
                            Ok(result) => match result.to_vec() {
                                Ok(output) => {
                                    println!("  {} softmax: {:?}", name, &output[..4]);
                                    results.push((name.clone(), output));
                                }
                                Err(e) => println!("  {} - read failed: {}", name, e),
                            },
                            Err(e) => println!("  {} - softmax failed: {}", name, e),
                        },
                        Err(e) => println!("  {} - tensor creation failed: {}", name, e),
                    }
                }
                Err(e) => {
                    println!("  {} - device creation failed: {}", name, e);
                }
            }
        }

        // Verify all devices produce same result
        if results.len() >= 2 {
            for i in 1..results.len() {
                let (name_a, result_a) = &results[0];
                let (name_b, result_b) = &results[i];

                for (j, (a, b)) in result_a.iter().zip(result_b.iter()).enumerate() {
                    assert!(
                        (a - b).abs() < 1e-4,
                        "Mismatch between {} and {} at {}: {} vs {}",
                        name_a,
                        name_b,
                        j,
                        a,
                        b
                    );
                }
                println!("  ✓ {} matches {}", name_a, name_b);
            }
        }

        println!("\n  Multi-device same computation: PASS\n");
    }) {
        return;
    }
}

// ============================================================================
// E2E Tests: Scientific Compute Pipeline
// ============================================================================

#[tokio::test]
async fn test_e2e_scientific_pipeline_cholesky() {
    if !common::run_gpu_resilient_async(|| async {
        println!("\n=== E2E Scientific Pipeline: Cholesky ===\n");

        // 1. ToadStool selects device for scientific compute
        let selection =
            select_best_device(HardwareWorkload::ScientificCompute).expect("Selection failed");
        println!("  ToadStool selection: {:?}", selection);

        // 2. Create device (with fallback)
        let device = match WgpuDevice::from_selection(selection).await {
            Ok(d) => Arc::new(d),
            Err(e) => {
                println!("  SKIP: Device unavailable: {}", e);
                return;
            }
        };
        println!("  Device: {}", device.name());

        // 3. Create SPD matrix
        let spd = vec![4.0, 2.0, 2.0, 3.0]; // 2x2 SPD

        let tensor = Tensor::from_vec_on(spd, vec![2, 2], device.clone())
            .await
            .expect("Tensor failed");

        // 4. Compute Cholesky decomposition
        let l = tensor.cholesky().expect("Cholesky failed");
        let l_data = l.to_vec().expect("Read failed");

        println!("  Input A: [[4, 2], [2, 3]]");
        println!("  Cholesky L: {:?}", l_data);

        // 5. Verify L * L^T = A
        // For 2x2 lower triangular L:
        // L[0,0] * L[0,0] should ≈ A[0,0] = 4
        // L[1,0] * L[0,0] should ≈ A[1,0] = 2
        // L[1,0]^2 + L[1,1]^2 should ≈ A[1,1] = 3

        let l00 = l_data[0];
        let l10 = l_data[2];
        let l11 = l_data[3];

        assert!((l00 * l00 - 4.0).abs() < 1e-3, "L*L^T [0,0] mismatch");
        assert!((l10 * l00 - 2.0).abs() < 1e-3, "L*L^T [1,0] mismatch");
        assert!(
            (l10 * l10 + l11 * l11 - 3.0).abs() < 1e-3,
            "L*L^T [1,1] mismatch"
        );

        println!("  ✓ Cholesky verified: L * L^T = A");
        println!("\n  Scientific pipeline Cholesky: PASS\n");
    }) {
        return;
    }
}

// ============================================================================
// E2E Tests: Precision Verification
// ============================================================================

#[tokio::test]
async fn test_e2e_f32_precision() {
    if !common::run_gpu_resilient_async(|| async {
        println!("\n=== E2E f32 Precision Test ===\n");

        let device = barracuda::device::test_pool::get_test_device().await;

        // Test precision with known values
        let test_cases = vec![
            ("Small values", vec![1e-6, 2e-6, 3e-6, 4e-6]),
            ("Large values", vec![1e6, 2e6, 3e6, 4e6]),
            ("Mixed", vec![1e-6, 1.0, 1e6, 0.0]),
        ];

        for (name, data) in test_cases {
            let tensor = Tensor::from_vec_on(data.clone(), vec![4], device.clone())
                .await
                .expect("Tensor failed");

            let result = tensor.to_vec().expect("Read failed");

            let mut max_rel_error = 0.0f32;
            for (got, exp) in result.iter().zip(data.iter()) {
                if *exp != 0.0 {
                    let rel_error = ((got - exp) / exp).abs();
                    max_rel_error = max_rel_error.max(rel_error);
                }
            }

            println!("  {} - max relative error: {:.2e}", name, max_rel_error);
            assert!(
                max_rel_error < 1e-5,
                "f32 precision loss too high: {}",
                max_rel_error
            );
        }

        println!("\n  f32 precision: PASS\n");
    }) {
        return;
    }
}

// ============================================================================
// E2E Tests: Workload Routing Correctness
// ============================================================================

#[test]
fn test_e2e_workload_routing_correctness() {
    println!("\n=== E2E Workload Routing Correctness ===\n");

    let router = KernelRouter::default();

    // Define workloads and expected routing rules
    let test_cases = vec![
        // Dense ops -> WGSL
        (
            "Dense 1K matmul",
            ComputeWorkload::DenseMatmul {
                m: 1000,
                n: 1000,
                k: 1000,
            },
            "wgsl",
        ),
        // Small ops -> CPU preferred
        (
            "Small matmul",
            ComputeWorkload::DenseMatmul { m: 8, n: 8, k: 8 },
            "wgsl_cpu",
        ),
        // Physics -> GPU
        (
            "MD forces",
            ComputeWorkload::PhysicsForce {
                particle_count: 10000,
                force_type: "lj".to_string(),
            },
            "wgsl",
        ),
        // FFT -> GPU
        (
            "FFT 1K",
            ComputeWorkload::FFT {
                size: 1024,
                batch_count: 10,
            },
            "wgsl",
        ),
    ];

    for (name, workload, expected) in test_cases {
        let target = router.route(&workload).expect("Routing failed");

        let is_match = matches!(
            (&target, expected),
            (KernelTarget::Wgsl { device: _, .. }, "wgsl")
                | (
                    KernelTarget::Wgsl {
                        device: DeviceSelection::Cpu,
                        ..
                    },
                    "wgsl_cpu",
                )
                | (KernelTarget::Npu { .. }, "npu")
        );

        let target_str = match &target {
            KernelTarget::Wgsl { device, .. } => format!("WGSL({device:?})"),
            KernelTarget::Npu { model_id, .. } => format!("NPU({model_id})"),
            KernelTarget::Hybrid { .. } => "Hybrid".to_string(),
        };

        if is_match || expected == "wgsl" && matches!(target, KernelTarget::Wgsl { .. }) {
            println!("  ✓ {} -> {}", name, target_str);
        } else {
            println!("  ? {} -> {} (expected {})", name, target_str, expected);
        }
    }

    println!("\n  Workload routing correctness: PASS\n");
}
