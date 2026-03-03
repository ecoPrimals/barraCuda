// FHE Coefficient Extraction Shader
//
// **Purpose**: Mask all coefficients except target index
//
// **Algorithm**: Simple conditional masking
// ```
// if (coeff_idx == target_index)
//     output[idx] = input[idx]
// else
//     output[idx] = 0
// ```
//
// **Performance**: O(n) parallel, 1 thread per coefficient

@group(0) @binding(0) var<storage, read> input: array<u32>; // u64 as 2xu32
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

struct Params {
    degree: u32,
    target_index: u32,
    _padding: vec2<u32>,
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

fn u64_zero() -> U64 {
    return U64(0u, 0u);
}

// ============================================================================
// Main Compute Kernel
// ============================================================================

@compute @workgroup_size(256)
fn extract_coefficient(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let coeff_idx = global_id.x;
    
    // Bounds check
    if (coeff_idx >= params.degree) {
        return;
    }
    
    // ✅ MASKING: Zero all coefficients except target
    if (coeff_idx == params.target_index) {
        // Keep target coefficient
        let coeff = load_u64(coeff_idx);
        store_u64(coeff_idx, coeff);
    } else {
        // Zero all other coefficients
        store_u64(coeff_idx, u64_zero());
    }
}
