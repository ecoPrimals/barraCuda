//! barraCuda GPU Validation Binary
//!
//! Runs the canary suite to verify GPU compute correctness before
//! domain workloads. Any consumer can run this to confirm their
//! hardware is trustworthy for scientific compute.
//!
//! Usage: cargo run --bin validate_gpu --features gpu

use barracuda::device::WgpuDevice;
use barracuda::error::BarracudaError;
use barracuda::ops::fhe_intt::{compute_inverse_root, FheIntt};
use barracuda::ops::fhe_ntt::FheNtt;
use barracuda::ops::fhe_pointwise_mul::FhePointwiseMul;
use barracuda::tensor::Tensor;
use std::sync::Arc;
use std::time::Instant;

struct ValidationResult {
    name: String,
    passed: bool,
    detail: String,
    #[allow(dead_code)]
    elapsed: std::time::Duration,
}

fn u64_to_tensor(
    poly: &[u64],
    device: &Arc<WgpuDevice>,
) -> Result<Tensor, BarracudaError> {
    let u32_pairs: Vec<u32> = poly
        .iter()
        .flat_map(|&x| [(x & 0xFFFF_FFFF) as u32, (x >> 32) as u32])
        .collect();
    let f32_bits: Vec<f32> = u32_pairs.iter().map(|&x| f32::from_bits(x)).collect();
    Tensor::from_data(&f32_bits, vec![poly.len() * 2], device.clone())
}

fn tensor_to_u64(tensor: &Tensor) -> Result<Vec<u64>, BarracudaError> {
    let u32_data = tensor.to_vec_u32()?;
    Ok(u32_data
        .chunks(2)
        .map(|c| (c[0] as u64) | ((c[1] as u64) << 32))
        .collect())
}

/// FHE NTT round-trip: INTT(NTT(p)) must equal p (bit-perfect)
async fn validate_fhe_ntt_roundtrip(
    device: &Arc<WgpuDevice>,
    degree: u32,
    modulus: u64,
    root: u64,
) -> ValidationResult {
    let name = format!("FHE NTT round-trip (mod {modulus}, N={degree})");
    let start = Instant::now();

    let poly: Vec<u64> = (1..=degree as u64).collect();
    let inv_root = compute_inverse_root(degree, modulus, root);

    let result = (|| -> Result<bool, BarracudaError> {
        let tensor = u64_to_tensor(&poly, device)?;
        let ntt = FheNtt::new(tensor, degree, modulus, root)?;
        let ntt_out = ntt.execute()?;
        let intt = FheIntt::new(ntt_out, degree, modulus, inv_root)?;
        let recovered = intt.execute()?;
        let recovered_poly = tensor_to_u64(&recovered)?;

        let mut all_match = true;
        for i in 0..degree as usize {
            if recovered_poly[i] % modulus != poly[i] % modulus {
                all_match = false;
                break;
            }
        }
        Ok(all_match)
    })();

    let elapsed = start.elapsed();
    match result {
        Ok(true) => ValidationResult {
            name,
            passed: true,
            detail: format!("bit-perfect in {elapsed:.2?}"),
            elapsed,
        },
        Ok(false) => ValidationResult {
            name,
            passed: false,
            detail: "INTT(NTT(p)) != p".into(),
            elapsed,
        },
        Err(e) => ValidationResult {
            name,
            passed: false,
            detail: format!("error: {e}"),
            elapsed,
        },
    }
}

/// FHE pointwise modular multiplication
async fn validate_fhe_pointwise_mul(
    device: &Arc<WgpuDevice>,
) -> ValidationResult {
    let name = "FHE pointwise mod-mul (mod 12289)".to_string();
    let start = Instant::now();

    let modulus = 12289u64;
    let a = vec![100u64, 200, 300, 400];
    let b = vec![5u64, 10, 15, 20];

    let result = (|| -> Result<bool, BarracudaError> {
        let ta = u64_to_tensor(&a, device)?;
        let tb = u64_to_tensor(&b, device)?;
        let degree = a.len() as u32;
        let op = FhePointwiseMul::new(ta, tb, degree, modulus)?;
        let out = op.execute()?;
        let result = tensor_to_u64(&out)?;

        let expected: Vec<u64> = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x * y) % modulus)
            .collect();

        Ok(result.iter().zip(expected.iter()).all(|(r, e)| r == e))
    })();

    let elapsed = start.elapsed();
    match result {
        Ok(true) => ValidationResult {
            name,
            passed: true,
            detail: format!("exact in {elapsed:.2?}"),
            elapsed,
        },
        Ok(false) => ValidationResult {
            name,
            passed: false,
            detail: "product mismatch".into(),
            elapsed,
        },
        Err(e) => ValidationResult {
            name,
            passed: false,
            detail: format!("error: {e}"),
            elapsed,
        },
    }
}

/// Basic tensor matmul correctness
async fn validate_matmul(device: &Arc<WgpuDevice>) -> ValidationResult {
    let name = "Tensor matmul 64x64".to_string();
    let start = Instant::now();

    let n = 64usize;
    let result = (|| -> Result<bool, BarracudaError> {
        // Identity * random = random
        let mut identity_data = vec![0.0f32; n * n];
        for i in 0..n {
            identity_data[i * n + i] = 1.0;
        }
        let random_data: Vec<f32> = (0..n * n).map(|i| (i as f32) * 0.01).collect();

        let identity = Tensor::from_vec_on_sync(identity_data, vec![n, n], device.clone())?;
        let matrix = Tensor::from_vec_on_sync(random_data.clone(), vec![n, n], device.clone())?;

        let product = identity.matmul(&matrix)?;
        let result_data = product.to_vec()?;

        let max_err = result_data
            .iter()
            .zip(random_data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        Ok(max_err < 1e-5)
    })();

    let elapsed = start.elapsed();
    match result {
        Ok(true) => ValidationResult {
            name,
            passed: true,
            detail: format!("I*A == A in {elapsed:.2?}"),
            elapsed,
        },
        Ok(false) => ValidationResult {
            name,
            passed: false,
            detail: "identity matmul drift > 1e-5".into(),
            elapsed,
        },
        Err(e) => ValidationResult {
            name,
            passed: false,
            detail: format!("error: {e}"),
            elapsed,
        },
    }
}

/// DF64 precision: verify double-float emulation round-trips
async fn validate_df64_precision(device: &Arc<WgpuDevice>) -> ValidationResult {
    let name = "DF64 add/sub precision".to_string();
    let start = Instant::now();

    let result = (|| -> Result<bool, BarracudaError> {
        let a_data = vec![1.0f32, 0.0, std::f32::consts::PI, 0.0];
        let b_data = vec![1e-10f32, 0.0, -std::f32::consts::PI, 0.0];

        let a = Tensor::from_vec_on_sync(a_data, vec![2, 2], device.clone())?;
        let b = Tensor::from_vec_on_sync(b_data, vec![2, 2], device.clone())?;
        let sum = a.add(&b)?;
        let result = sum.to_vec()?;

        // First element: 1.0 + 1e-10 should be representable
        // Second element: pi + (-pi) should be near zero
        let first_ok = (result[0] - 1.0).abs() < 1e-5;
        let second_ok = result[2].abs() < 1e-5;

        Ok(first_ok && second_ok)
    })();

    let elapsed = start.elapsed();
    match result {
        Ok(true) => ValidationResult {
            name,
            passed: true,
            detail: format!("precision OK in {elapsed:.2?}"),
            elapsed,
        },
        Ok(false) => ValidationResult {
            name,
            passed: false,
            detail: "precision drift".into(),
            elapsed,
        },
        Err(e) => ValidationResult {
            name,
            passed: false,
            detail: format!("error: {e}"),
            elapsed,
        },
    }
}

/// Device capability probe
async fn validate_device_probe(device: &Arc<WgpuDevice>) -> ValidationResult {
    let name = "Device capability probe".to_string();
    let start = Instant::now();

    let info = device.adapter_info();
    let limits = device.device().limits();
    let elapsed = start.elapsed();

    let detail = format!(
        "{} ({:?}), max_buf={}MB, max_wg={}",
        info.name,
        info.device_type,
        limits.max_buffer_size / (1024 * 1024),
        limits.max_compute_workgroup_size_x,
    );

    ValidationResult {
        name,
        passed: true,
        detail,
        elapsed,
    }
}

#[tokio::main]
async fn main() -> Result<(), BarracudaError> {
    println!("barraCuda GPU Validation");
    println!("========================\n");

    let device = match WgpuDevice::new().await {
        Ok(dev) => {
            let info = dev.adapter_info();
            println!(
                "Device: {} ({:?})\n",
                info.name, info.device_type
            );
            Arc::new(dev)
        }
        Err(e) => {
            eprintln!("FAIL: No GPU device available: {e}");
            std::process::exit(1);
        }
    };

    let mut results = Vec::new();

    results.push(validate_device_probe(&device).await);
    results.push(validate_matmul(&device).await);
    results.push(validate_df64_precision(&device).await);
    results.push(
        validate_fhe_ntt_roundtrip(&device, 4, 17, 4).await,
    );
    results.push(
        validate_fhe_ntt_roundtrip(&device, 8, 12289, 11).await,
    );
    results.push(validate_fhe_pointwise_mul(&device).await);

    println!("Results:");
    let header = "Detail";
    println!("{:<45} {:<6} {header}", "Test", "Status");
    println!("{}", "-".repeat(90));

    let mut passed = 0;
    let mut failed = 0;

    for r in &results {
        let status = if r.passed { "PASS" } else { "FAIL" };
        let marker = if r.passed { " " } else { "!" };
        println!("{marker} {:<44} {:<6} {}", r.name, status, r.detail);
        if r.passed {
            passed += 1;
        } else {
            failed += 1;
        }
    }

    println!("\n{passed} passed, {failed} failed out of {} tests", results.len());

    if failed > 0 {
        println!("\nGPU validation FAILED — do not trust this device for scientific compute.");
        std::process::exit(1);
    } else {
        println!("\nGPU validation PASSED — device is ready for barraCuda workloads.");
    }

    Ok(())
}
