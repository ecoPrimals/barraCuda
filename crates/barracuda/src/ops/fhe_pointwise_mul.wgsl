// SPDX-License-Identifier: AGPL-3.0-only
// FHE Point-wise Multiplication (in NTT domain)
//
// Purpose: Multiply two polynomials element-wise in NTT domain
// Used in: Fast polynomial multiplication (NTT → pointwise → INTT)
//
// Input:  A[i], B[i] - two polynomials in NTT domain (N elements each)
// Output: C[i] = A[i] * B[i] mod q - element-wise product
//
// Complexity: O(N) - much faster than O(N²) convolution!
//
// This is the heart of fast polynomial multiplication:
//   poly_mul(a, b) = INTT(pointwise_mul(NTT(a), NTT(b)))
//
// Each element is a 64-bit number stored as two u32 values:
//   value = (high << 32) | low

// Input A (first polynomial in NTT domain)
@group(0) @binding(0) var<storage, read> input_a: array<u32>;

// Input B (second polynomial in NTT domain)
@group(0) @binding(1) var<storage, read> input_b: array<u32>;

// Output C = A ⊙ B (element-wise product)
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

// Parameters for modular arithmetic
struct PointwiseMulParams {
    degree: u32,           // Polynomial degree (N)
    modulus_low: u32,      // Modulus q (lower 32 bits)
    modulus_high: u32,     // Modulus q (upper 32 bits)
    barrett_mu_low: u32,   // Barrett reduction constant (lower 32 bits)
    barrett_mu_high: u32,  // Barrett reduction constant (upper 32 bits)
}

@group(0) @binding(3) var<uniform> params: PointwiseMulParams;

// ═══════════════════════════════════════════════════════════════
// Modular Arithmetic Helpers (64-bit using u32 pairs)
// ═══════════════════════════════════════════════════════════════

/// Widening 32×32 → 64-bit multiply using 16-bit limbs.
fn mul32_wide(a: u32, b: u32) -> vec2<u32> {
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;

    let p0 = a_lo * b_lo;
    let p1 = a_lo * b_hi;
    let p2 = a_hi * b_lo;
    let p3 = a_hi * b_hi;

    var lo = p0 + ((p1 & 0xFFFFu) << 16u);
    let c1 = select(0u, 1u, lo < p0);
    let mid1 = (p1 >> 16u) + c1;

    let prev = lo;
    lo = lo + ((p2 & 0xFFFFu) << 16u);
    let c2 = select(0u, 1u, lo < prev);
    let hi = p3 + (p2 >> 16u) + mid1 + c2;

    return vec2<u32>(lo, hi);
}

/// Reduce a 64-bit value mod q (32-bit) using bit-by-bit shift-and-reduce.
///
/// Processes the 64-bit value MSB-first, maintaining acc = (bits so far) mod q.
/// Correct for all q < 2^31.  O(64) iterations — negligible on GPU.
fn reduce64(val_lo: u32, val_hi: u32, q: u32) -> u32 {
    var acc = 0u;
    for (var i = 31i; i >= 0i; i = i - 1i) {
        acc = (acc << 1u) | ((val_hi >> u32(i)) & 1u);
        if (acc >= q) { acc -= q; }
    }
    for (var i = 31i; i >= 0i; i = i - 1i) {
        acc = (acc << 1u) | ((val_lo >> u32(i)) & 1u);
        if (acc >= q) { acc -= q; }
    }
    return acc;
}

/// Multiply two 64-bit numbers modulo q.
///
/// For typical FHE moduli (< 2^31), both inputs have high == 0.
/// Computes the full 64-bit product then reduces with exact arithmetic.
fn mod_mul(a_low: u32, a_high: u32, b_low: u32, b_high: u32) -> vec2<u32> {
    let modulus = vec2<u32>(params.modulus_low, params.modulus_high);

    // Fast path: both values and modulus fit in 32 bits (standard FHE case).
    if (a_high == 0u && b_high == 0u && modulus.y == 0u) {
        let product = mul32_wide(a_low, b_low);
        let r = reduce64(product.x, product.y, modulus.x);
        return vec2<u32>(r, 0u);
    }

    // Full 128-bit path for 64-bit moduli: 4 partial products → reduce.
    let p0 = mul32_wide(a_low, b_low);
    let p1 = mul32_wide(a_low, b_high);
    let p2 = mul32_wide(a_high, b_low);
    let p3 = mul32_wide(a_high, b_high);

    let w0 = p0.x;
    var w1 = p0.y;
    var w2 = p3.x;
    var w3 = p3.y;

    let prev_w1 = w1;
    w1 = w1 + p1.x;
    let c1 = select(0u, 1u, w1 < prev_w1);

    let prev2 = w1;
    w1 = w1 + p2.x;
    let c2 = select(0u, 1u, w1 < prev2);

    let mid_carry = p1.y + p2.y + c1 + c2;
    let prev_w2 = w2;
    w2 = w2 + mid_carry;
    w3 = w3 + select(0u, 1u, w2 < prev_w2);

    let barrett_mu = vec2<u32>(params.barrett_mu_low, params.barrett_mu_high);
    let product = vec2<u32>(w0, w1);
    let q_hat_approx = (w1 * barrett_mu.x) + (w0 * barrett_mu.y >> 32u);
    let q_times_mod_low = q_hat_approx * modulus.x;
    let q_times_mod_high = q_hat_approx * modulus.y;

    var result = vec2<u32>(
        product.x - q_times_mod_low,
        product.y - q_times_mod_high
    );

    for (var i = 0u; i < 3u; i = i + 1u) {
        if (result.y > modulus.y || (result.y == modulus.y && result.x >= modulus.x)) {
            if (result.x < modulus.x) { result.y -= 1u; }
            result.x -= modulus.x;
            result.y -= modulus.y;
        }
    }

    return result;
}

/// Add two 64-bit numbers modulo q
fn mod_add(a_low: u32, a_high: u32, b_low: u32, b_high: u32) -> vec2<u32> {
    let modulus = vec2<u32>(params.modulus_low, params.modulus_high);
    
    // Add with carry
    var result_low = a_low + b_low;
    var carry = select(0u, 1u, result_low < a_low);  // Carry if overflow
    var result_high = a_high + b_high + carry;
    
    // Reduce if result >= modulus
    if (result_high > modulus.y || (result_high == modulus.y && result_low >= modulus.x)) {
        // result -= modulus
        if (result_low < modulus.x) {
            result_high -= 1u;
        }
        result_low -= modulus.x;
        result_high -= modulus.y;
    }
    
    return vec2<u32>(result_low, result_high);
}

/// Subtract two 64-bit numbers modulo q
fn mod_sub(a_low: u32, a_high: u32, b_low: u32, b_high: u32) -> vec2<u32> {
    let modulus = vec2<u32>(params.modulus_low, params.modulus_high);
    
    var result_low = a_low - b_low;
    var borrow = select(0u, 1u, a_low < b_low);  // Borrow if underflow
    var result_high = a_high - b_high - borrow;
    
    // If result is negative, add modulus
    if (a_high < b_high || (a_high == b_high && a_low < b_low)) {
        let carry = select(0u, 1u, result_low + modulus.x < result_low);
        result_low += modulus.x;
        result_high += modulus.y + carry;
    }
    
    return vec2<u32>(result_low, result_high);
}

// ═══════════════════════════════════════════════════════════════
// Point-wise Multiplication Kernel
// ═══════════════════════════════════════════════════════════════

/// Main kernel: C[i] = A[i] * B[i] mod q
///
/// Each thread processes one coefficient
/// Input/output format: pairs of u32 (low, high) representing 64-bit values
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // Bounds check
    if (idx >= params.degree) {
        return;
    }
    
    // Load A[idx] and B[idx] (each is 64-bit = 2 × u32)
    let a_low = input_a[idx * 2u];
    let a_high = input_a[idx * 2u + 1u];
    let b_low = input_b[idx * 2u];
    let b_high = input_b[idx * 2u + 1u];
    
    // Compute C[idx] = A[idx] * B[idx] mod q
    let result = mod_mul(a_low, a_high, b_low, b_high);
    
    // Store result
    output[idx * 2u] = result.x;       // Low 32 bits
    output[idx * 2u + 1u] = result.y;  // High 32 bits
}

// ═══════════════════════════════════════════════════════════════
// Optional: Batch Point-wise Operations
// ═══════════════════════════════════════════════════════════════

/// Alternative kernel: Process multiple coefficients per thread
/// Useful for smaller polynomials to improve GPU occupancy
@compute @workgroup_size(256)
fn batched(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base_idx = global_id.x * 4u;  // Each thread processes 4 coefficients
    
    // Process 4 coefficients per thread (if available)
    for (var i = 0u; i < 4u; i = i + 1u) {
        let idx = base_idx + i;
        
        if (idx >= params.degree) {
            return;
        }
        
        // Load, multiply, store
        let a_low = input_a[idx * 2u];
        let a_high = input_a[idx * 2u + 1u];
        let b_low = input_b[idx * 2u];
        let b_high = input_b[idx * 2u + 1u];
        
        let result = mod_mul(a_low, a_high, b_low, b_high);
        
        output[idx * 2u] = result.x;
        output[idx * 2u + 1u] = result.y;
    }
}

// ═══════════════════════════════════════════════════════════════
// Performance Notes
// ═══════════════════════════════════════════════════════════════
//
// Expected Performance:
//   N=4096: ~3μs (memory-bound, not compute-bound)
//   Bandwidth: ~200 GB/s (3 × 4096 × 8 bytes / 3μs)
//
// Optimization opportunities:
//   1. Vectorized loads (vec4) for better memory coalescing
//   2. Shared memory for cache locality (not needed for O(N))
//   3. Hardware-specific modular multiply (native u64 on some GPUs)
//
// Why so fast:
//   - Simple element-wise operation (no dependencies)
//   - Perfect memory coalescing (sequential access)
//   - High arithmetic intensity (1 multiply per 24 bytes)
