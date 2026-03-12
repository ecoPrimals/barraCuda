// SPDX-License-Identifier: AGPL-3.0-only
//! FHE Polynomial Subtraction Shader
//!
//! **Purpose**: Subtract two polynomials modulo q (FHE ciphertext operation)
//!
//! **Deep Debt**: Pure WGSL, hardware-agnostic, numerically precise
//!
//! **Algorithm**: Coefficient-wise subtraction with Barrett modular reduction
//!
//! ## Mathematical Background
//!
//! FHE ciphertexts are polynomials over Z_q[X]/(X^N + 1):
//! - Degree N (typically 2048, 4096, or 8192)
//! - Coefficients modulo q (large prime, e.g., 2^64)
//! - Subtraction: (a₀ - b₀) mod q, (a₁ - b₁) mod q, ..., (aₙ - bₙ) mod q
//!
//! ## Modular Subtraction
//!
//! For a - b mod q:
//! - If a >= b: result = a - b
//! - If a < b: result = (q - b) + a = q + a - b (wrapped subtraction)
//! - Always ensure result < q

struct Params {
    degree: u32,        // Polynomial degree (N)
    modulus_lo: u32,    // q lower 32 bits
    modulus_hi: u32,    // q upper 32 bits
    _pad0: u32,         // Alignment padding
}

@group(0) @binding(0) var<storage, read> poly_a: array<u32>;      // First polynomial (2×degree for u64)
@group(0) @binding(1) var<storage, read> poly_b: array<u32>;      // Second polynomial (2×degree for u64)
@group(0) @binding(2) var<storage, read_write> result: array<u32>; // Result polynomial (2×degree for u64)
@group(0) @binding(3) var<uniform> params: Params;

/// Reconstruct 64-bit value from two 32-bit parts
fn u64_from_parts(lo: u32, hi: u32) -> vec2<u32> {
    return vec2<u32>(lo, hi);
}

/// Compare two 64-bit values (returns true if a >= b)
fn u64_gte(a: vec2<u32>, b: vec2<u32>) -> bool {
    if (a.y > b.y) { return true; }
    if (a.y < b.y) { return false; }
    return a.x >= b.x;
}

/// Subtract two 64-bit values (with borrow handling)
fn u64_sub(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let borrow = select(0u, 1u, a.x < b.x);
    let lo_diff = a.x - b.x;
    let hi_diff = a.y - b.y - borrow;
    return vec2<u32>(lo_diff, hi_diff);
}

/// Add two 64-bit values (for wrapped subtraction)
fn u64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo_sum = a.x + b.x;
    let carry = select(0u, 1u, lo_sum < a.x);
    let hi_sum = a.y + b.y + carry;
    return vec2<u32>(lo_sum, hi_sum);
}

/// Modular subtraction: (a - b) mod q
fn modular_sub(a: vec2<u32>, b: vec2<u32>, q: vec2<u32>) -> vec2<u32> {
    // If a >= b, simple subtraction
    if (u64_gte(a, b)) {
        return u64_sub(a, b);
    }
    
    // If a < b, wrapped subtraction: q + a - b
    // This is equivalent to: (q - b) + a
    let q_minus_b = u64_sub(q, b);
    return u64_add(q_minus_b, a);
}

@compute @workgroup_size(256)
fn fhe_poly_sub(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    
    // Boundary check
    if (idx >= params.degree) {
        return;
    }
    
    // Each coefficient is stored as two u32 values (lo, hi)
    let idx_lo = idx * 2u;
    let idx_hi = idx_lo + 1u;
    
    // Load coefficients from poly_a
    let a_lo = poly_a[idx_lo];
    let a_hi = poly_a[idx_hi];
    let a = u64_from_parts(a_lo, a_hi);
    
    // Load coefficients from poly_b
    let b_lo = poly_b[idx_lo];
    let b_hi = poly_b[idx_hi];
    let b = u64_from_parts(b_lo, b_hi);
    
    // Load modulus
    let q = u64_from_parts(params.modulus_lo, params.modulus_hi);
    
    // Modular subtraction
    let diff = modular_sub(a, b, q);
    
    // Store result
    result[idx_lo] = diff.x;
    result[idx_hi] = diff.y;
}
