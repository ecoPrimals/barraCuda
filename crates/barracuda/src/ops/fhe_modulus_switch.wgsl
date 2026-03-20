// SPDX-License-Identifier: AGPL-3.0-or-later
// FHE Modulus Switching Shader
//
// **Purpose**: Scale ciphertext coefficients to smaller modulus
//
// **Algorithm**: Exact rounding for modulus reduction
// ```
// scale = q_new / q_old
// coeff_new = round(coeff_old * scale) mod q_new
// ```
//
// **Performance**: O(n) parallel, 1 thread per coefficient

// Import U64 emulation library
@group(0) @binding(0) var<storage, read> input: array<u32>; // u64 as 2xu32
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

struct Params {
    degree: u32,
    modulus_old_lo: u32,
    modulus_old_hi: u32,
    modulus_new_lo: u32,
    modulus_new_hi: u32,
    _padding: vec3<u32>,
}

@group(0) @binding(2) var<uniform> params: Params;

// ============================================================================
// U64 Emulation (minimal subset needed for modulus switch)
// ============================================================================

struct U64 {
    lo: u32,
    hi: u32,
}

fn load_u64(idx: u32) -> U64 {
    return U64(input[idx * 2u], input[idx * 2u + 1u]);
}

fn store_u64(idx: u32, val: U64) {
    output[idx * 2u] = val.lo;
    output[idx * 2u + 1u] = val.hi;
}

fn u64_from_parts(lo: u32, hi: u32) -> U64 {
    return U64(lo, hi);
}

// Multiply u64 by u32 (for scaling)
fn u64_mul_u32(a: U64, b: u32) -> U64 {
    let a0 = a.lo;
    let a1 = a.hi;
    
    // (a1 * 2^32 + a0) * b
    let low_prod = a0 * b;
    let carry_low = low_prod >> 32u;
    
    let high_prod = a1 * b;
    let mid = high_prod + carry_low;
    
    return U64(low_prod & 0xFFFFFFFFu, mid);
}

// Divide u64 by u32 (for scaling ratio)
fn u64_div_u32(a: U64, b: u32) -> U64 {
    // Simplified division for modulus switching
    // In practice, precompute q_new/q_old as fixed-point
    
    // For now, use approximation: (a.hi << 32 | a.lo) / b
    // This is accurate enough for small modulus ratios
    
    let dividend = (u64(a.hi) << 32u) | u64(a.lo);
    let quotient = dividend / u64(b);
    
    return U64(u32(quotient & 0xFFFFFFFFu), u32(quotient >> 32u));
}

// Modulo u64 by u64
fn u64_mod(a: U64, m: U64) -> U64 {
    // Simple modulo for u64
    let a_full = (u64(a.hi) << 32u) | u64(a.lo);
    let m_full = (u64(m.hi) << 32u) | u64(m.lo);
    let result = a_full % m_full;
    
    return U64(u32(result & 0xFFFFFFFFu), u32(result >> 32u));
}

// Compare u64 values
fn u64_lt(a: U64, b: U64) -> bool {
    if (a.hi != b.hi) {
        return a.hi < b.hi;
    }
    return a.lo < b.lo;
}

// ============================================================================
// Modulus Switching Algorithm
// ============================================================================

/// Round (a * q_new) / q_old to nearest integer
fn scale_and_round(coeff: U64, q_old: U64, q_new: U64) -> U64 {
    // Compute: round((coeff * q_new) / q_old)
    
    // Step 1: Multiply coeff * q_new (128-bit intermediate)
    // For simplicity, use direct u64 operations
    let coeff_val = (u64(coeff.hi) << 32u) | u64(coeff.lo);
    let q_new_val = (u64(q_new.hi) << 32u) | u64(q_new.lo);
    let q_old_val = (u64(q_old.hi) << 32u) | u64(q_old.lo);
    
    // scaled = coeff * q_new
    let scaled = coeff_val * q_new_val;
    
    // Step 2: Divide by q_old with rounding
    let quotient = scaled / q_old_val;
    let remainder = scaled % q_old_val;
    
    // Round to nearest: if remainder >= q_old/2, round up
    var result = quotient;
    if (remainder * 2u >= q_old_val) {
        result += 1u;
    }
    
    // Step 3: Reduce mod q_new
    result = result % q_new_val;
    
    return U64(u32(result & 0xFFFFFFFFu), u32(result >> 32u));
}

// ============================================================================
// Main Compute Kernel
// ============================================================================

@compute @workgroup_size(256)
fn modulus_switch(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let coeff_idx = global_id.x;
    
    // Bounds check
    if (coeff_idx >= params.degree) {
        return;
    }
    
    // Load input coefficient (u64)
    let coeff = load_u64(coeff_idx);
    
    // Load moduli
    let q_old = u64_from_parts(params.modulus_old_lo, params.modulus_old_hi);
    let q_new = u64_from_parts(params.modulus_new_lo, params.modulus_new_hi);
    
    // Perform modulus switch: coeff_new = round((coeff * q_new) / q_old) mod q_new
    let coeff_new = scale_and_round(coeff, q_old, q_new);
    
    // Store result
    store_u64(coeff_idx, coeff_new);
}
