// SPDX-License-Identifier: AGPL-3.0-only
// FHE Rotation Shader (Galois Automorphism)
//
// **Purpose**: Apply X → X^(2k+1) automorphism for slot rotation
//
// **Algorithm**: Coefficient permutation based on Galois automorphism
// ```
// For rotation by k:
//   galois = 2k + 1
//   new_index[i] = (i * galois) mod N
//   output[new_index[i]] = input[i] * sign
//   where sign = -1 if (i * galois) >= N, else 1
// ```
//
// **Performance**: O(n) parallel, 1 thread per coefficient

@group(0) @binding(0) var<storage, read> input: array<u32>; // u64 as 2xu32
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

struct Params {
    degree: u32,
    rotation: u32, // Normalized [0, degree)
    modulus_lo: u32,
    modulus_hi: u32,
}

@group(0) @binding(2) var<uniform> params: Params;

// ============================================================================
// U64 Helpers (minimal subset)
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

// Negate u64 mod modulus
fn u64_negate_mod(a: U64, m: U64) -> U64 {
    // result = (modulus - a) mod modulus
    let a_val = (u64(a.hi) << 32u) | u64(a.lo);
    let m_val = (u64(m.hi) << 32u) | u64(m.lo);
    
    if (a_val == 0u) {
        return U64(0u, 0u);
    }
    
    let result = m_val - a_val;
    return U64(u32(result & 0xFFFFFFFFu), u32(result >> 32u));
}

// ============================================================================
// Galois Automorphism
// ============================================================================

/// Compute Galois element: 2*rotation + 1
fn compute_galois_element(rotation: u32) -> u32 {
    return 2u * rotation + 1u;
}

/// Apply Galois automorphism: X^i → X^(i*galois) mod (X^N + 1)
fn apply_automorphism(coeff_idx: u32, galois: u32, degree: u32) -> u32 {
    // new_power = (coeff_idx * galois) mod (2 * degree)
    let new_power = (coeff_idx * galois) % (2u * degree);
    
    // If new_power >= degree, wrap around with negation
    // X^N = -1 in ring R = Z[X]/(X^N + 1)
    if (new_power >= degree) {
        return new_power - degree; // Index for negated coefficient
    } else {
        return new_power;
    }
}

/// Check if coefficient should be negated after automorphism
fn should_negate(coeff_idx: u32, galois: u32, degree: u32) -> bool {
    let new_power = (coeff_idx * galois) % (2u * degree);
    return new_power >= degree;
}

// ============================================================================
// Main Compute Kernel
// ============================================================================

@compute @workgroup_size(256)
fn rotate_automorphism(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let coeff_idx = global_id.x;
    
    // Bounds check
    if (coeff_idx >= params.degree) {
        return;
    }
    
    // Compute Galois element for this rotation
    let galois = compute_galois_element(params.rotation);
    
    // Load input coefficient
    let coeff = load_u64(coeff_idx);
    
    // Apply automorphism: determine new position
    let new_idx = apply_automorphism(coeff_idx, galois, params.degree);
    let negate = should_negate(coeff_idx, galois, params.degree);
    
    // Apply negation if needed (for X^N wraparound)
    let modulus = u64_from_parts(params.modulus_lo, params.modulus_hi);
    let output_coeff = select(coeff, u64_negate_mod(coeff, modulus), negate);
    
    // Store at new position
    // NOTE: This is a simplified version. Full implementation would need
    // atomic operations or separate buffers to avoid race conditions.
    // For sequential GPU execution with proper barriers, this is safe.
    store_u64(new_idx, output_coeff);
}
