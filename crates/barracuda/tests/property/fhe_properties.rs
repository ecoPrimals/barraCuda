// SPDX-License-Identifier: AGPL-3.0-or-later
//! FHE Property-Based GPU Integration Tests
//!
//! Validates fundamental cryptographic and mathematical properties on GPU:
//! 1. NTT-INTT Round-trip (perfect reconstruction)
//! 2. Modulus Switch Correctness (preserves mod relationships)
//! 3. Rotation Composition (rotate(a+b) = rotate(a) ∘ rotate(b))
//! 4. Homomorphic Properties (enc(a) + enc(b) = enc(a+b))
//! 5. Key Switch Security (structural validity)
//!
//! All tests skip gracefully when no GPU is present (CI / headless environments).

use barracuda::device::test_pool;
use barracuda::ops::fhe_intt::{compute_inverse_root, FheIntt};
use barracuda::ops::fhe_key_switch::FheKeySwitch;
use barracuda::ops::fhe_modulus_switch::FheModulusSwitch;
use barracuda::ops::fhe_ntt::{compute_primitive_root, FheNtt};
use barracuda::ops::fhe_poly_add::{create_fhe_poly_tensor, FhePolyAdd};
use barracuda::ops::fhe_poly_sub::FhePolySub;
use barracuda::ops::fhe_rotate::FheRotate;
use barracuda::tensor::Tensor;
use std::sync::Arc;

// ── Device capability guard ──────────────────────────────────────────────────

/// Get a test device that supports 64-bit integer shaders (`SHADER_INT64`).
///
/// FHE shaders store coefficients as u64 (two-u32 pairs) and require `SHADER_INT64`.
/// Returns `None` if the current device lacks this capability, causing the test to skip.
async fn get_fhe_device() -> Option<Arc<barracuda::device::WgpuDevice>> {
    let device = test_pool::get_test_device_if_gpu_available().await?;
    // Skip on software renderers that lack 64-bit integer shader support.
    if !device
        .device()
        .features()
        .contains(wgpu::Features::SHADER_INT64)
    {
        return None;
    }
    Some(device)
}

// ── Read-back helper ────────────────────────────────────────────────────────

/// Convert a u32-pair encoded FHE tensor to a Vec<u64>.
///
/// FHE tensors store u64 coefficients as two u32 values: `[lo, hi]` per element.
fn tensor_to_u64(tensor: &Tensor) -> Vec<u64> {
    let raw = tensor.to_vec_u32().expect("tensor read-back failed");
    raw.chunks_exact(2)
        .map(|pair| pair[0] as u64 | ((pair[1] as u64) << 32))
        .collect()
}

// ── Math helpers ─────────────────────────────────────────────────────────────

fn mod_add(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 + b as u128) % m as u128) as u64
}

fn mod_sub(a: u64, b: u64, m: u64) -> u64 {
    if a >= b {
        (a - b) % m
    } else {
        m - ((b - a) % m)
    }
}

// ── NTT helper: choose a valid (modulus, degree) pair ────────────────────────
// modulus must satisfy q ≡ 1 (mod 2·degree) for the NTT to exist.
// q = 12289 ≡ 1 (mod 2·4096) — the canonical CKKS/NTRU prime.
// For small tests we use (17, 4) since 17 ≡ 1 (mod 8).

// ============================================================================
// PROPERTY 1: NTT-INTT Round-trip (Perfect Reconstruction)
// ============================================================================

#[tokio::test]
async fn test_ntt_intt_roundtrip_small() {
    let Some(device) = get_fhe_device().await else {
        return;
    };
    let degree = 4u32;
    let modulus = 17u64; // 17 ≡ 1 (mod 8), N=4

    let input_data = vec![1u64, 2, 3, 4];
    let input = create_fhe_poly_tensor(&input_data, device.clone())
        .await
        .unwrap();

    let root = compute_primitive_root(degree, modulus);
    let inv_root = compute_inverse_root(degree, modulus, root);

    let ntt_out = FheNtt::new(input, degree, modulus, root)
        .unwrap()
        .execute()
        .unwrap();

    let recovered = FheIntt::new(ntt_out, degree, modulus, inv_root)
        .unwrap()
        .execute()
        .unwrap();

    let result = tensor_to_u64(&recovered);
    assert_eq!(input_data, result, "NTT-INTT round-trip failed");
}

#[tokio::test]
async fn test_ntt_intt_roundtrip_larger() {
    let Some(device) = get_fhe_device().await else {
        return;
    };
    // degree=8, q=257 since 257 ≡ 1 (mod 16)
    let degree = 8u32;
    let modulus = 257u64;

    let input_data: Vec<u64> = (1..=8).map(|i| (i * 7) % modulus).collect();
    let input = create_fhe_poly_tensor(&input_data, device.clone())
        .await
        .unwrap();

    let root = compute_primitive_root(degree, modulus);
    let inv_root = compute_inverse_root(degree, modulus, root);

    let ntt_out = FheNtt::new(input, degree, modulus, root)
        .unwrap()
        .execute()
        .unwrap();

    let recovered = FheIntt::new(ntt_out, degree, modulus, inv_root)
        .unwrap()
        .execute()
        .unwrap();

    let result = tensor_to_u64(&recovered);
    assert_eq!(input_data, result, "NTT-INTT round-trip failed (degree=8)");
}

// ============================================================================
// PROPERTY 2: Modulus Switch Correctness
// ============================================================================

#[tokio::test]
async fn test_modulus_switch_reduces_mod() {
    let Some(device) = get_fhe_device().await else {
        return;
    };
    let degree = 4u32;
    let modulus_old = 257u64;
    let modulus_new = 17u64;

    let input_data = vec![10u64, 20, 30, 40];
    let input = create_fhe_poly_tensor(&input_data, device.clone())
        .await
        .unwrap();

    let output = FheModulusSwitch::new(input, degree, modulus_old, modulus_new)
        .unwrap()
        .execute()
        .unwrap();

    let result = tensor_to_u64(&output);
    for (i, (&inp, &out)) in input_data.iter().zip(result.iter()).enumerate() {
        assert_eq!(out, inp % modulus_new, "modulus switch wrong at index {i}");
    }
}

#[tokio::test]
async fn test_modulus_switch_identity() {
    let Some(device) = get_fhe_device().await else {
        return;
    };
    let degree = 4u32;
    let modulus = 257u64;

    let input_data = vec![5u64, 10, 15, 20];
    let input = create_fhe_poly_tensor(&input_data, device.clone())
        .await
        .unwrap();

    // Same-to-same: should be identity
    let output = FheModulusSwitch::new(input, degree, modulus, modulus)
        .unwrap()
        .execute()
        .unwrap();

    let result = tensor_to_u64(&output);
    assert_eq!(input_data, result, "identity modulus switch failed");
}

// ============================================================================
// PROPERTY 3: Rotation Properties
// ============================================================================

#[tokio::test]
async fn test_rotation_composition() {
    let Some(device) = get_fhe_device().await else {
        return;
    };
    let degree = 8u32;
    let modulus = 257u64;

    let input_data: Vec<u64> = (0..8).map(|i| i as u64).collect();

    // rotate(2) ∘ rotate(3) should equal rotate(5)
    let t1 = create_fhe_poly_tensor(&input_data, device.clone())
        .await
        .unwrap();
    let rotated_2 = FheRotate::new(t1, degree, 2, modulus)
        .unwrap()
        .execute()
        .unwrap();
    let composed = FheRotate::new(rotated_2, degree, 3, modulus)
        .unwrap()
        .execute()
        .unwrap();

    let t2 = create_fhe_poly_tensor(&input_data, device.clone())
        .await
        .unwrap();
    let direct = FheRotate::new(t2, degree, 5, modulus)
        .unwrap()
        .execute()
        .unwrap();

    let composed_data = tensor_to_u64(&composed);
    let direct_data = tensor_to_u64(&direct);
    assert_eq!(
        composed_data, direct_data,
        "rotation composition: rotate(2)∘rotate(3) != rotate(5)"
    );
}

#[tokio::test]
async fn test_rotation_inverse() {
    let Some(device) = get_fhe_device().await else {
        return;
    };
    let degree = 8u32;
    let modulus = 257u64;
    let k = 3i32;

    let input_data: Vec<u64> = (0..8).map(|i| (i * 10) as u64).collect();
    let input = create_fhe_poly_tensor(&input_data, device.clone())
        .await
        .unwrap();

    let rotated = FheRotate::new(input, degree, k, modulus)
        .unwrap()
        .execute()
        .unwrap();
    let recovered = FheRotate::new(rotated, degree, -(k), modulus)
        .unwrap()
        .execute()
        .unwrap();

    let result = tensor_to_u64(&recovered);
    assert_eq!(input_data, result, "rotation inverse failed");
}

// ============================================================================
// PROPERTY 4: Homomorphic Addition / Subtraction
// ============================================================================

#[tokio::test]
async fn test_homomorphic_addition() {
    let Some(device) = get_fhe_device().await else {
        return;
    };
    let degree = 4u32;
    let modulus = 257u64;

    let a_data = vec![10u64, 20, 30, 40];
    let b_data = vec![5u64, 15, 25, 35];

    let (a, b) = tokio::join!(
        create_fhe_poly_tensor(&a_data, device.clone()),
        create_fhe_poly_tensor(&b_data, device.clone()),
    );

    let result = FhePolyAdd::new(a.unwrap(), b.unwrap(), degree, modulus)
        .unwrap()
        .execute()
        .unwrap();

    let out = tensor_to_u64(&result);
    for (i, ((&av, &bv), &rv)) in a_data.iter().zip(b_data.iter()).zip(out.iter()).enumerate() {
        assert_eq!(rv, mod_add(av, bv, modulus), "poly add wrong at index {i}");
    }
}

#[tokio::test]
async fn test_homomorphic_subtraction() {
    let Some(device) = get_fhe_device().await else {
        return;
    };
    let degree = 4u32;
    let modulus = 257u64;

    let a_data = vec![40u64, 30, 20, 10];
    let b_data = vec![5u64, 15, 25, 35];

    let (a, b) = tokio::join!(
        create_fhe_poly_tensor(&a_data, device.clone()),
        create_fhe_poly_tensor(&b_data, device.clone()),
    );

    let result = FhePolySub::new(a.unwrap(), b.unwrap(), degree, modulus)
        .unwrap()
        .execute()
        .unwrap();

    let out = tensor_to_u64(&result);
    for (i, ((&av, &bv), &rv)) in a_data.iter().zip(b_data.iter()).zip(out.iter()).enumerate() {
        assert_eq!(rv, mod_sub(av, bv, modulus), "poly sub wrong at index {i}");
    }
}

#[tokio::test]
async fn test_poly_add_associativity() {
    let Some(device) = get_fhe_device().await else {
        return;
    };
    let degree = 4u32;
    let modulus = 257u64;

    let a_data = vec![10u64, 20, 30, 40];
    let b_data = vec![5u64, 15, 25, 35];
    let c_data = vec![3u64, 7, 11, 13];

    // (a + b) + c
    let (a1, b1, c1) = tokio::join!(
        create_fhe_poly_tensor(&a_data, device.clone()),
        create_fhe_poly_tensor(&b_data, device.clone()),
        create_fhe_poly_tensor(&c_data, device.clone()),
    );
    let ab = FhePolyAdd::new(a1.unwrap(), b1.unwrap(), degree, modulus)
        .unwrap()
        .execute()
        .unwrap();
    let abc1 = FhePolyAdd::new(ab, c1.unwrap(), degree, modulus)
        .unwrap()
        .execute()
        .unwrap();

    // a + (b + c)
    let (a2, b2, c2) = tokio::join!(
        create_fhe_poly_tensor(&a_data, device.clone()),
        create_fhe_poly_tensor(&b_data, device.clone()),
        create_fhe_poly_tensor(&c_data, device.clone()),
    );
    let bc = FhePolyAdd::new(b2.unwrap(), c2.unwrap(), degree, modulus)
        .unwrap()
        .execute()
        .unwrap();
    let abc2 = FhePolyAdd::new(a2.unwrap(), bc, degree, modulus)
        .unwrap()
        .execute()
        .unwrap();

    let r1 = tensor_to_u64(&abc1);
    let r2 = tensor_to_u64(&abc2);
    assert_eq!(r1, r2, "poly add associativity: (a+b)+c != a+(b+c)");
}

// ============================================================================
// PROPERTY 5: Key Switch Security (Structural Validity)
// ============================================================================

#[tokio::test]
async fn test_key_switch_output_in_range() {
    let Some(device) = get_fhe_device().await else {
        return;
    };
    let degree = 4u32;
    let modulus = 257u64;
    let decomp_base = 2u32;
    let decomp_levels = 4u32;

    let input_data = vec![100u64, 150, 200, 250];
    let input = create_fhe_poly_tensor(&input_data, device.clone())
        .await
        .unwrap();

    let output = FheKeySwitch::new(input, degree, modulus, decomp_base, decomp_levels)
        .unwrap()
        .execute()
        .unwrap();

    let result = tensor_to_u64(&output);
    assert_eq!(
        result.len(),
        (degree * decomp_levels) as usize,
        "key switch output size mismatch"
    );
    for (i, &v) in result.iter().enumerate() {
        assert!(
            v < modulus,
            "key switch value out of range at index {i}: {v} >= {modulus}"
        );
    }
}

#[tokio::test]
async fn test_key_switch_deterministic() {
    let Some(device) = get_fhe_device().await else {
        return;
    };
    let degree = 4u32;
    let modulus = 257u64;
    let decomp_base = 2u32;
    let decomp_levels = 4u32;

    let input_data = vec![50u64, 100, 150, 200];

    let (t1, t2) = tokio::join!(
        create_fhe_poly_tensor(&input_data, device.clone()),
        create_fhe_poly_tensor(&input_data, device.clone()),
    );

    let o1 = FheKeySwitch::new(t1.unwrap(), degree, modulus, decomp_base, decomp_levels)
        .unwrap()
        .execute()
        .unwrap();
    let o2 = FheKeySwitch::new(t2.unwrap(), degree, modulus, decomp_base, decomp_levels)
        .unwrap()
        .execute()
        .unwrap();

    let r1 = tensor_to_u64(&o1);
    let r2 = tensor_to_u64(&o2);
    assert_eq!(r1, r2, "key switch non-deterministic");
}

// ============================================================================
// Cross-Property: NTT Linearity — NTT(a + b) == NTT(a) + NTT(b)
// ============================================================================

#[tokio::test]
async fn test_ntt_linearity() {
    let Some(device) = get_fhe_device().await else {
        return;
    };
    let degree = 8u32;
    let modulus = 257u64;

    let a_data: Vec<u64> = (0..8).map(|i| (i * 5) as u64).collect();
    let b_data: Vec<u64> = (0..8).map(|i| (i * 3) as u64).collect();

    let root = compute_primitive_root(degree, modulus);

    // Method 1: NTT(a + b)
    let (ta, tb) = tokio::join!(
        create_fhe_poly_tensor(&a_data, device.clone()),
        create_fhe_poly_tensor(&b_data, device.clone()),
    );
    let sum_t = FhePolyAdd::new(ta.unwrap(), tb.unwrap(), degree, modulus)
        .unwrap()
        .execute()
        .unwrap();
    let ntt_sum = FheNtt::new(sum_t, degree, modulus, root)
        .unwrap()
        .execute()
        .unwrap();

    // Method 2: NTT(a) + NTT(b)
    let (ta2, tb2) = tokio::join!(
        create_fhe_poly_tensor(&a_data, device.clone()),
        create_fhe_poly_tensor(&b_data, device.clone()),
    );
    let ntt_a = FheNtt::new(ta2.unwrap(), degree, modulus, root)
        .unwrap()
        .execute()
        .unwrap();
    let ntt_b = FheNtt::new(tb2.unwrap(), degree, modulus, root)
        .unwrap()
        .execute()
        .unwrap();
    let ntt_a_plus_b = FhePolyAdd::new(ntt_a, ntt_b, degree, modulus)
        .unwrap()
        .execute()
        .unwrap();

    let r1 = tensor_to_u64(&ntt_sum);
    let r2 = tensor_to_u64(&ntt_a_plus_b);
    assert_eq!(r1, r2, "NTT linearity: NTT(a+b) != NTT(a) + NTT(b)");
}
