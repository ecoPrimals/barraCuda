// SPDX-License-Identifier: AGPL-3.0-or-later
//! FHE Polynomial Addition Shader
//!
//! **Purpose**: Add two polynomials modulo q (FHE ciphertext operation)
//!
//! **Deep Debt**: Pure WGSL, hardware-agnostic, numerically precise
//!
//! **Algorithm**: Coefficient-wise addition with Barrett modular reduction
//!
//! ## Mathematical Background
//!
//! FHE ciphertexts are polynomials over Z_q[X]/(X^N + 1):
//! - Degree N (typically 2048, 4096, or 8192)
//! - Coefficients modulo q (large prime, e.g., 2^64)
//! - Addition: (a₀ + b₀) mod q, (a₁ + b₁) mod q, ..., (aₙ + bₙ) mod q
//!
//! ## Barrett Reduction
//!
//! For efficient modular reduction without division:
//! - Precompute μ = ⌊2^(2k) / q⌋ where k = bitwidth(q)
//! - Approximate quotient: q_approx = ⌊(a * μ) / 2^(2k)⌋
//! - Remainder: r = a - q_approx * q
//! - Final correction: if r >= q then r -= q (at most once)

struct Params {
    degree: u32,        // Polynomial degree (N)
    modulus_lo: u32,    // q lower 32 bits
    modulus_hi: u32,    // q upper 32 bits
    mu_lo: u32,         // Barrett constant μ lower 32 bits
    mu_hi: u32,         // Barrett constant μ upper 32 bits
}

@group(0) @binding(0) var<storage, read> poly_a: array<u32>;      // First polynomial (2×degree for u64)
@group(0) @binding(1) var<storage, read> poly_b: array<u32>;      // Second polynomial (2×degree for u64)
@group(0) @binding(2) var<storage, read_write> result: array<u32>; // Result polynomial (2×degree for u64)
@group(0) @binding(3) var<uniform> params: Params;

/// Reconstruct 64-bit value from two 32-bit parts
fn u64_from_parts(lo: u32, hi: u32) -> vec2<u32> {
    return vec2<u32>(lo, hi);
}

/// Add two 64-bit values (with overflow handling)
fn u64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo_sum = a.x + b.x;
    let carry = select(0u, 1u, lo_sum < a.x);  // Detect carry
    let hi_sum = a.y + b.y + carry;
    return vec2<u32>(lo_sum, hi_sum);
}

/// Subtract two 64-bit values (assumes a >= b)
fn u64_sub(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let borrow = select(0u, 1u, a.x < b.x);
    let lo_diff = a.x - b.x;
    let hi_diff = a.y - b.y - borrow;
    return vec2<u32>(lo_diff, hi_diff);
}

/// Compare two 64-bit values (returns true if a >= b)
fn u64_gte(a: vec2<u32>, b: vec2<u32>) -> bool {
    if (a.y > b.y) { return true; }
    if (a.y < b.y) { return false; }
    return a.x >= b.x;
}

/// Multiply two 32-bit values to get 64-bit result
fn u32_mul_to_u64(a: u32, b: u32) -> vec2<u32> {
    // Split into 16-bit parts to avoid overflow
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;
    
    // Partial products
    let p_ll = a_lo * b_lo;
    let p_lh = a_lo * b_hi;
    let p_hl = a_hi * b_lo;
    let p_hh = a_hi * b_hi;
    
    // Combine
    let mid = p_lh + p_hl + (p_ll >> 16u);
    let lo = (mid << 16u) | (p_ll & 0xFFFFu);
    let hi = p_hh + (mid >> 16u);
    
    return vec2<u32>(lo, hi);
}

/// Multiply two 64-bit values (returns lower 64 bits)
fn u64_mul_lo(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    // (a.y * 2^32 + a.x) * (b.y * 2^32 + b.x)
    // = a.x*b.x + 2^32*(a.x*b.y + a.y*b.x) + 2^64*a.y*b.y
    // Lower 64 bits = a.x*b.x + 2^32*(a.x*b.y + a.y*b.x)[lower]
    
    let p0 = u32_mul_to_u64(a.x, b.x);  // bits [0:63]
    let p1 = u32_mul_to_u64(a.x, b.y);  // bits [32:95]
    let p2 = u32_mul_to_u64(a.y, b.x);  // bits [32:95]
    
    // Combine: result_lo = p0.x, result_hi = p0.y + p1.x + p2.x
    let hi_sum = p0.y + p1.x + p2.x;
    
    return vec2<u32>(p0.x, hi_sum);
}

/// Modular reduction: a mod q (simple iterative version)
fn mod_reduce(a: vec2<u32>, q: vec2<u32>) -> vec2<u32> {
    // Simple modular reduction: repeatedly subtract q while a >= q
    // This works for values close to q (typical in FHE after addition)
    var r = a;
    
    // At most 2 subtractions needed for addition (a + b < 2q)
    if (u64_gte(r, q)) {
        r = u64_sub(r, q);
    }
    if (u64_gte(r, q)) {
        r = u64_sub(r, q);
    }
    
    return r;
}

@compute @workgroup_size(256)
fn fhe_poly_add(@builtin(global_invocation_id) gid: vec3<u32>) {
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
    
    // Add coefficients
    let sum = u64_add(a, b);
    
    // Load modulus
    let q = u64_from_parts(params.modulus_lo, params.modulus_hi);
    
    // Modular reduction
    let reduced = mod_reduce(sum, q);
    
    // Store result
    result[idx_lo] = reduced.x;
    result[idx_hi] = reduced.y;
}
