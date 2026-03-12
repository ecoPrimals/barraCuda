// SPDX-License-Identifier: AGPL-3.0-only
// RBF Kernel Evaluation - Fused pairwise distance + kernel function (f64 canonical)
// Computes K[i,j] = φ(‖xᵢ - xⱼ‖) for various radial basis functions
//
// **Deep Debt Principles**:
// - ✅ Pure WGSL implementation (GPU-optimized)
// - ✅ Safe Rust (no unsafe blocks)
// - ✅ Hardware-agnostic via WebGPU
// - ✅ Runtime-configured kernel type
// - ✅ Fused distance + kernel (single pass, no intermediate)
//
// Use Case: RBF surrogate learning (hotSpring physics)
// - Training: K = rbf_kernel(X_train, X_train)
// - Prediction: K = rbf_kernel(X_new, X_train)
//
// Kernel Functions:
// - Thin Plate Spline: φ(r) = r² · log(r)  [default, best for physics]
// - Gaussian:          φ(r) = exp(-ε²r²)
// - Multiquadric:      φ(r) = sqrt(1 + ε²r²)
// - Inverse MQ:        φ(r) = 1/sqrt(1 + ε²r²)
// - Cubic:             φ(r) = r³
// - Quintic:           φ(r) = r⁵
// - Linear:            φ(r) = r

struct Params {
    n_rows: u32,      // Number of rows in X
    n_cols: u32,      // Number of rows in Y
    n_dims: u32,      // Dimension of points
    kernel_type: u32, // 0=TPS, 1=Gaussian, 2=Multiquadric, 3=InverseMQ, 4=Cubic, 5=Quintic, 6=Linear
    epsilon: f64,     // Shape parameter (for Gaussian, MQ, IMQ)
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f64>;     // Points X [n_rows × n_dims]
@group(0) @binding(1) var<storage, read> y: array<f64>;     // Points Y [n_cols × n_dims]
@group(0) @binding(2) var<storage, read_write> output: array<f64>;  // Kernel matrix K [n_rows × n_cols]
@group(0) @binding(3) var<uniform> params: Params;

// Compute Euclidean distance squared between two points
fn distance_squared(x_idx: u32, y_idx: u32) -> f64 {
    var dist_sq = f64(0.0);
    let n_dims = params.n_dims;
    
    for (var d = 0u; d < n_dims; d = d + 1u) {
        let diff = x[x_idx * n_dims + d] - y[y_idx * n_dims + d];
        dist_sq = dist_sq + diff * diff;
    }
    
    return dist_sq;
}

// Apply kernel function to distance
fn apply_kernel(r_squared: f64) -> f64 {
    let r = sqrt_f64(r_squared);
    let epsilon = params.epsilon;
    let eps_sq = epsilon * epsilon;
    
    switch (params.kernel_type) {
        case 0u: {
            // Thin Plate Spline: r² · log(r)
            if (r < f64(1e-10)) {
                return f64(0.0);  // Limit as r→0
            }
            return r_squared * log_f64(r);
        }
        case 1u: {
            // Gaussian: exp(-ε²r²)
            return exp_f64(-eps_sq * r_squared);
        }
        case 2u: {
            // Multiquadric: sqrt(1 + ε²r²)
            return sqrt_f64(f64(1.0) + eps_sq * r_squared);
        }
        case 3u: {
            // Inverse Multiquadric: 1/sqrt(1 + ε²r²)
            return f64(1.0) / sqrt_f64(f64(1.0) + eps_sq * r_squared);
        }
        case 4u: {
            // Cubic: r³
            return r * r * r;
        }
        case 5u: {
            // Quintic: r⁵
            let r3 = r * r * r;
            return r3 * r * r;
        }
        case 6u: {
            // Linear: r
            return r;
        }
        default: {
            // Default to Thin Plate Spline
            if (r < f64(1e-10)) {
                return f64(0.0);
            }
            return r_squared * log_f64(r);
        }
    }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;  // Row index in X
    let j = global_id.y;  // Column index in Y (row in Y)
    
    if (i >= params.n_rows || j >= params.n_cols) {
        return;
    }
    
    // Fused: compute distance + apply kernel in single pass
    let r_sq = distance_squared(i, j);
    let kernel_val = apply_kernel(r_sq);
    
    // Write to output matrix K[i,j]
    output[i * params.n_cols + j] = kernel_val;
}
