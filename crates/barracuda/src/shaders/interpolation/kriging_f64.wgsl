// SPDX-License-Identifier: AGPL-3.0-only
// ============================================================================
// kriging_f64.wgsl — Spatial interpolation (Ordinary Kriging) at f64 precision
// ============================================================================
//
// UNIFIED PATTERN (Feb 16 2026) — Serves multiple springs:
//   - airSpring: Soil moisture mapping from sparse sensors to field grid
//   - wetSpring: Sample interpolation across sampling sites
//   - General: Any spatial data interpolation with uncertainty
//
// ARCHITECTURE:
//   Phase 1: Build variogram-based covariance matrix K
//   Phase 2: Solve kriging system (K + λI)w = k for weights w
//   Phase 3: Interpolate: z* = Σ w_i * z_i
//
// REQUIRES: SHADER_F64 feature
// Date: February 16, 2026
// License: AGPL-3.0-only
// ============================================================================

// ============================================================================
// VARIOGRAM MODELS (selected via params.variogram_model)
// ============================================================================
// 0 = SPHERICAL
// 1 = EXPONENTIAL
// 2 = GAUSSIAN
// 3 = LINEAR

// exp_f64 provided by math_f64.wgsl auto-injection

/// Compute variogram value γ(h) for distance h
fn variogram(h: f64, nugget: f64, sill: f64, range_param: f64, model: u32) -> f64 {
    let zero = h - h;
    let one = zero + 1.0;
    
    if (h <= zero) {
        return zero;  // γ(0) = 0
    }
    
    let c = sill - nugget;  // Partial sill
    let a = range_param;
    
    switch (model) {
        case 0u: {
            // SPHERICAL: γ(h) = c0 + c * (1.5*h/a - 0.5*(h/a)³) for h ≤ a, else c0 + c
            if (h >= a) {
                return nugget + c;
            }
            let ratio = h / a;
            let three_half = zero + 1.5;
            let half = zero + 0.5;
            return nugget + c * (three_half * ratio - half * ratio * ratio * ratio);
        }
        case 1u: {
            // EXPONENTIAL: γ(h) = c0 + c * (1 - exp(-3h/a))
            let three = zero + 3.0;
            return nugget + c * (one - exp_f64(-three * h / a));
        }
        case 2u: {
            // GAUSSIAN: γ(h) = c0 + c * (1 - exp(-3(h/a)²))
            let three = zero + 3.0;
            let ratio = h / a;
            return nugget + c * (one - exp_f64(-three * ratio * ratio));
        }
        case 3u: {
            // LINEAR: γ(h) = c0 + c * h/a for h ≤ a, else c0 + c
            if (h >= a) {
                return nugget + c;
            }
            return nugget + c * h / a;
        }
        default: {
            return nugget + c;
        }
    }
}

/// Compute Euclidean distance between two 2D points
fn distance_2d(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    let dx = x2 - x1;
    let dy = y2 - y1;
    return sqrt(dx * dx + dy * dy);
}

// ============================================================================
// BINDINGS
// ============================================================================

struct KrigingParams {
    n_known: u32,        // Number of known points
    n_target: u32,       // Number of target (interpolation) points
    variogram_model: u32,
    _pad1: u32,
    nugget: f64,         // Variogram nugget (c0)
    sill: f64,           // Variogram sill (c0 + c)
    range_param: f64,    // Variogram range (a)
}

// Known points: [x, y, z] flattened
@group(0) @binding(0) var<storage, read> known_points: array<f64>;
// Target points: [x, y] flattened
@group(0) @binding(1) var<storage, read> target_points: array<f64>;
// Covariance matrix K (n+1 x n+1) for kriging system
@group(0) @binding(2) var<storage, read_write> covariance_matrix: array<f64>;
// Weights output (n+1 per target point)
@group(0) @binding(3) var<storage, read_write> weights: array<f64>;
// Interpolated values and variances [z*, σ²] per target
@group(0) @binding(4) var<storage, read_write> output: array<f64>;
@group(0) @binding(5) var<uniform> params: KrigingParams;

// ============================================================================
// PHASE 1: Build covariance matrix K
// ============================================================================
// K is (n+1) x (n+1) where K[i,j] = C(h_ij) = sill - γ(h_ij)
// Last row/col is for Lagrange multiplier (all 1s, diagonal 0)

@compute @workgroup_size(16, 16)
fn build_covariance_matrix(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let i = global_id.x;
    let j = global_id.y;
    let n = params.n_known;
    let n1 = n + 1u;
    
    if (i >= n1 || j >= n1) {
        return;
    }
    
    let zero = params.nugget - params.nugget;
    let one = zero + 1.0;
    
    var value: f64;
    
    if (i == n && j == n) {
        // K[n,n] = 0 (Lagrange constraint)
        value = zero;
    } else if (i == n || j == n) {
        // Last row/col = 1 (Lagrange multiplier)
        value = one;
    } else {
        // Regular covariance entry
        let xi = known_points[i * 3u + 0u];
        let yi = known_points[i * 3u + 1u];
        let xj = known_points[j * 3u + 0u];
        let yj = known_points[j * 3u + 1u];
        
        let h = distance_2d(xi, yi, xj, yj);
        let gamma = variogram(h, params.nugget, params.sill, params.range_param, params.variogram_model);
        
        // Covariance C(h) = sill - γ(h)
        value = params.sill - gamma;
    }
    
    covariance_matrix[i * n1 + j] = value;
}

// ============================================================================
// PHASE 2: Build RHS vector k for each target point
// ============================================================================
// k[i] = C(h_i,target) for i < n, k[n] = 1

@compute @workgroup_size(256)
fn build_rhs_vector(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let target_idx = workgroup_id.x;
    let i = global_id.x % 256u;
    let n = params.n_known;
    let n1 = n + 1u;
    
    if (target_idx >= params.n_target || i >= n1) {
        return;
    }
    
    let zero = params.nugget - params.nugget;
    let one = zero + 1.0;
    
    let target_x = target_points[target_idx * 2u + 0u];
    let target_y = target_points[target_idx * 2u + 1u];
    
    var value: f64;
    
    if (i == n) {
        // Lagrange constraint
        value = one;
    } else {
        // Covariance with known point i
        let xi = known_points[i * 3u + 0u];
        let yi = known_points[i * 3u + 1u];
        
        let h = distance_2d(xi, yi, target_x, target_y);
        let gamma = variogram(h, params.nugget, params.sill, params.range_param, params.variogram_model);
        
        value = params.sill - gamma;
    }
    
    // Store in weights buffer temporarily (will be overwritten after solve)
    weights[target_idx * n1 + i] = value;
}

// ============================================================================
// PHASE 3: Apply weights to interpolate (after external solve)
// ============================================================================
// z* = Σ w_i * z_i
// σ² = sill - Σ w_i * k_i

@compute @workgroup_size(256)
fn interpolate(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let target_idx = global_id.x;
    let n = params.n_known;
    let n1 = n + 1u;
    
    if (target_idx >= params.n_target) {
        return;
    }
    
    let zero = params.nugget - params.nugget;
    var z_interp = zero;
    var variance_sum = zero;
    
    let target_x = target_points[target_idx * 2u + 0u];
    let target_y = target_points[target_idx * 2u + 1u];
    
    // Weighted sum of known values
    for (var i = 0u; i < n; i = i + 1u) {
        let w = weights[target_idx * n1 + i];
        let z = known_points[i * 3u + 2u];
        z_interp = z_interp + w * z;
        
        // For variance: sum w_i * k_i
        let xi = known_points[i * 3u + 0u];
        let yi = known_points[i * 3u + 1u];
        let h = distance_2d(xi, yi, target_x, target_y);
        let gamma = variogram(h, params.nugget, params.sill, params.range_param, params.variogram_model);
        let k_i = params.sill - gamma;
        variance_sum = variance_sum + w * k_i;
    }
    
    // Lagrange multiplier contribution
    let lambda = weights[target_idx * n1 + n];
    variance_sum = variance_sum + lambda;
    
    // Kriging variance: σ² = sill - Σ w_i * k_i - λ
    let variance = params.sill - variance_sum;
    
    // Output: [z*, σ²] per target
    output[target_idx * 2u + 0u] = z_interp;
    output[target_idx * 2u + 1u] = variance;
}

// ============================================================================
// SIMPLE KRIGING VARIANT (no Lagrange constraint, known mean)
// ============================================================================
// For cases where the mean is known or estimated separately

@compute @workgroup_size(256)
fn simple_kriging_interpolate(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let target_idx = global_id.x;
    let n = params.n_known;
    
    if (target_idx >= params.n_target) {
        return;
    }
    
    let zero = params.nugget - params.nugget;
    var z_interp = zero;
    var variance_sum = zero;
    
    let target_x = target_points[target_idx * 2u + 0u];
    let target_y = target_points[target_idx * 2u + 1u];
    
    // Weighted sum (weights already solved, stored without Lagrange row)
    for (var i = 0u; i < n; i = i + 1u) {
        let w = weights[target_idx * n + i];
        let z = known_points[i * 3u + 2u];
        z_interp = z_interp + w * z;
        
        // For variance
        let xi = known_points[i * 3u + 0u];
        let yi = known_points[i * 3u + 1u];
        let h = distance_2d(xi, yi, target_x, target_y);
        let gamma = variogram(h, params.nugget, params.sill, params.range_param, params.variogram_model);
        let k_i = params.sill - gamma;
        variance_sum = variance_sum + w * k_i;
    }
    
    let variance = params.sill - variance_sum;
    
    output[target_idx * 2u + 0u] = z_interp;
    output[target_idx * 2u + 1u] = variance;
}
