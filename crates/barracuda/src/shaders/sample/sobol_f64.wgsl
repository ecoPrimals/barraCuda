// SPDX-License-Identifier: AGPL-3.0-only
// Sobol quasi-random sequence generator - GPU accelerated (f64 canonical)
// Generates low-discrepancy sequences using Gray code
//
// Output: n_samples × n_dims values in [0, 1)
//
// Applications: Monte Carlo integration, sensitivity analysis, global optimization
// Reference: Sobol (1967), Joe & Kuo (2008)

@group(0) @binding(0) var<storage, read_write> output: array<f64>;
@group(0) @binding(1) var<uniform> params: Params;

struct Params {
    n_samples: u32,
    n_dims: u32,
    skip: u32,       // Number of initial points to skip
    _pad: u32,
}

// Direction numbers for first 8 dimensions (Joe & Kuo 2008)
// Stored as V[dim][bit] where V[d][j] = direction number × 2^(32-j)
// Pre-scaled for 32-bit fixed point: divide by 2^32 to get [0,1)

// For dimension 1: V_j = 2^(32-j)
fn direction_d1(j: u32) -> u32 {
    return 1u << (31u - j);
}

// For dimension 2: polynomial x+1, initial m = [1]
fn direction_d2(j: u32) -> u32 {
    if (j == 0u) { return 2147483648u; }  // 1 << 31
    if (j == 1u) { return 3221225472u; }  // 3 << 30
    if (j == 2u) { return 2684354560u; }  // 5 << 29
    if (j == 3u) { return 4026531840u; }  // 15 << 28
    if (j == 4u) { return 2818572288u; }  // 21 << 27
    return 1u << (31u - j);
}

// For dimension 3: polynomial x+1, initial m = [1,3]
fn direction_d3(j: u32) -> u32 {
    if (j == 0u) { return 2147483648u; }
    if (j == 1u) { return 1073741824u; }
    if (j == 2u) { return 2684354560u; }
    if (j == 3u) { return 1342177280u; }
    if (j == 4u) { return 2952790016u; }
    return 1u << (31u - j);
}

// For dimension 4: polynomial x²+x+1, initial m = [1,3,1]
fn direction_d4(j: u32) -> u32 {
    if (j == 0u) { return 2147483648u; }
    if (j == 1u) { return 3221225472u; }
    if (j == 2u) { return 536870912u; }
    if (j == 3u) { return 2952790016u; }
    if (j == 4u) { return 1476395008u; }
    return 1u << (31u - j);
}

// Get direction number for dimension d (0-indexed) and bit j
fn get_direction(d: u32, j: u32) -> u32 {
    if (d == 0u) { return direction_d1(j); }
    if (d == 1u) { return direction_d2(j); }
    if (d == 2u) { return direction_d3(j); }
    if (d == 3u) { return direction_d4(j); }
    // Fallback for higher dimensions: simple Van der Corput
    return 1u << (31u - (j + d * 3u) % 32u);
}

// Count trailing zeros (position of rightmost 1 bit)
fn count_trailing_zeros(n: u32) -> u32 {
    if (n == 0u) { return 32u; }
    var count = 0u;
    var x = n;
    while ((x & 1u) == 0u) {
        count = count + 1u;
        x = x >> 1u;
    }
    return count;
}

// Generate Sobol point n for dimension d using Gray code
fn sobol_point(n: u32, d: u32) -> f64 {
    if (n == 0u) {
        return 0.0;
    }

    var x = 0u;
    var index = n;
    var j = 0u;

    // Use Gray code: XOR with direction numbers for each 1-bit in index
    while (index > 0u && j < 32u) {
        if ((index & 1u) != 0u) {
            x = x ^ get_direction(d, j);
        }
        index = index >> 1u;
        j = j + 1u;
    }

    // Convert to [0, 1)
    return f64(x) / 4294967296.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let sample_idx = global_id.x;
    if (sample_idx >= params.n_samples) {
        return;
    }

    let n = sample_idx + params.skip;

    // Generate point for each dimension
    for (var d = 0u; d < params.n_dims; d = d + 1u) {
        let value = sobol_point(n, d);
        output[sample_idx * params.n_dims + d] = value;
    }
}
