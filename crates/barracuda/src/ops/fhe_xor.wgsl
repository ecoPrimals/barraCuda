// SPDX-License-Identifier: AGPL-3.0-or-later
// FHE XOR Gate - GPU Implementation
//
// Implements Boolean XOR operation on FHE-encrypted data
// For TFHE binary gates: XOR(a,b) = a + b - 2*(a * b) mod q
//
// This is a simplified FHE XOR gate using polynomial representation.
// Implements the Boolean identity: a XOR b = a + b - 2*(a AND b)

@group(0) @binding(0) var<storage, read> input_a: array<u32>;
@group(0) @binding(1) var<storage, read> input_b: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

struct Params {
    degree: u32,        // Polynomial degree (N)
    modulus_lo: u32,    // q lower 32 bits
    modulus_hi: u32,    // q upper 32 bits
    _pad0: u32,         // Alignment padding
}

@group(0) @binding(3) var<uniform> params: Params;

// Helper: Compare two 64-bit values (a >= b)
fn u64_gte(a: vec2<u32>, b: vec2<u32>) -> bool {
    if (a.y > b.y) { return true; }
    if (a.y < b.y) { return false; }
    return a.x >= b.x;
}

// Helper: Subtract two 64-bit values
fn u64_sub(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let borrow = select(0u, 1u, a.x < b.x);
    return vec2<u32>(a.x - b.x, a.y - b.y - borrow);
}

// Helper: Add two 64-bit values
fn u64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo_sum = a.x + b.x;
    let carry = select(0u, 1u, lo_sum < a.x);
    let hi_sum = a.y + b.y + carry;
    return vec2<u32>(lo_sum, hi_sum);
}

// Simple modular reduction
fn simple_mod_reduce(a: vec2<u32>, q: vec2<u32>) -> vec2<u32> {
    var r = a;
    for (var i = 0u; i < 100u; i = i + 1u) {
        if (!u64_gte(r, q)) {
            break;
        }
        r = u64_sub(r, q);
    }
    return r;
}

// Modular addition
fn mod_add(a: u32, b: u32, q: vec2<u32>) -> u32 {
    let sum = u64_add(vec2<u32>(a, 0u), vec2<u32>(b, 0u));
    let reduced = simple_mod_reduce(sum, q);
    return reduced.x;
}

// Modular multiplication for small values
fn mod_mul(a: u32, b: u32, q: vec2<u32>) -> u32 {
    // For small values, use simple 32-bit multiplication
    let product = a * b;
    let product_vec = vec2<u32>(product, 0u);
    let reduced = simple_mod_reduce(product_vec, q);
    return reduced.x;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // Bounds check
    if (idx >= params.degree) {
        return;
    }
    
    // Get inputs
    let a = input_a[idx];
    let b = input_b[idx];
    
    // Modulus as 64-bit value
    let q = vec2<u32>(params.modulus_lo, params.modulus_hi);
    
    // Boolean XOR: result = a + b - 2*(a * b) mod q
    // Identity: a XOR b = a + b - 2*(a AND b)
    // For binary: 0 XOR 0 = 0+0-0 = 0
    //             0 XOR 1 = 0+1-0 = 1
    //             1 XOR 0 = 1+0-0 = 1
    //             1 XOR 1 = 1+1-2 = 0
    
    let sum = mod_add(a, b, q);
    let product = mod_mul(a, b, q);
    
    // Double the product: 2 * product mod q
    let double_product = mod_add(product, product, q);
    
    // result = (sum - double_product) mod q
    // Need to handle underflow
    var result_vec = vec2<u32>(sum, 0u);
    let double_product_vec = vec2<u32>(double_product, 0u);
    
    if (!u64_gte(result_vec, double_product_vec)) {
        result_vec = u64_add(result_vec, q);
    }
    
    result_vec = u64_sub(result_vec, double_product_vec);
    output[idx] = result_vec.x;
}
