// SPDX-License-Identifier: AGPL-3.0-only
// Crank-Nicolson PDE Solver (f64) - Shader-First Implementation
//
// Full f64 precision for long-time integration stability.
//
// Parallel operations for the Crank-Nicolson scheme:
//   ∂u/∂t = α · ∂²u/∂x²
//
// The scheme:
//   (uⁿ⁺¹ - uⁿ)/Δt = (α/2)[(∂²u/∂x²)ⁿ⁺¹ + (∂²u/∂x²)ⁿ]
//
// This results in a tridiagonal system at each timestep.
// The RHS computation is EMBARRASSINGLY PARALLEL.
// The tridiagonal solve uses CYCLIC REDUCTION (parallel).
//
// Shader-first pattern:
// - RHS computation: fully parallel
// - Tridiagonal solve: cyclic_reduction_f64.wgsl (O(log n) parallel)
// - Time loop: minimal CPU coordination
//
// Applications:
// - TTM (Two-Temperature Model)
// - Heat diffusion
// - Schrödinger equation (with i)

struct CrankNicolsonParams {
    n: u32,           // Number of interior points
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    r: f64,           // Courant number: α·Δt/Δx²
    left_bc: f64,     // Left boundary value
    right_bc: f64,    // Right boundary value
}

@group(0) @binding(0) var<uniform> params: CrankNicolsonParams;
@group(0) @binding(1) var<storage, read> u: array<f64>;           // Current solution (interior)
@group(0) @binding(2) var<storage, read_write> rhs: array<f64>;   // RHS for tridiagonal system

// Compute RHS of Crank-Nicolson system - FULLY PARALLEL
// RHS_i = (1-r)uⁿᵢ + (r/2)(uⁿᵢ₋₁ + uⁿᵢ₊₁) + boundary terms
@compute @workgroup_size(256, 1, 1)
fn compute_rhs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let n = params.n;
    let r = params.r;
    let half = f64(0.5);
    let one = f64(1.0);
    
    if (i >= n) {
        return;
    }
    
    // Get neighbors (with boundary handling)
    var u_left: f64;
    var u_right: f64;
    
    if (i == 0u) {
        u_left = params.left_bc;
    } else {
        u_left = u[i - 1u];
    }
    
    if (i == n - 1u) {
        u_right = params.right_bc;
    } else {
        u_right = u[i + 1u];
    }
    
    // Main explicit part
    rhs[i] = (one - r) * u[i] + (r * half) * (u_left + u_right);
    
    // Add boundary contributions for implicit part
    if (i == 0u) {
        rhs[i] = rhs[i] + (r * half) * params.left_bc;
    }
    if (i == n - 1u) {
        rhs[i] = rhs[i] + (r * half) * params.right_bc;
    }
}

// Build tridiagonal matrix coefficients - parallel initialization
// a[i] = -r/2 (sub-diagonal)
// b[i] = 1+r (main diagonal)
// c[i] = -r/2 (super-diagonal)
@group(0) @binding(0) var<uniform> build_params: CrankNicolsonParams;
@group(0) @binding(1) var<storage, read_write> a_diag: array<f64>;
@group(0) @binding(2) var<storage, read_write> b_diag: array<f64>;
@group(0) @binding(3) var<storage, read_write> c_diag: array<f64>;

@compute @workgroup_size(256, 1, 1)
fn build_matrix(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let n = build_params.n;
    let r = build_params.r;
    let half = f64(0.5);
    let one = f64(1.0);
    
    if (i >= n) {
        return;
    }
    
    // Main diagonal: always 1+r
    b_diag[i] = one + r;
    
    // Sub-diagonal (length n-1, but we allocate n for simplicity)
    if (i < n - 1u) {
        a_diag[i] = -r * half;
    }
    
    // Super-diagonal (length n-1)
    if (i < n - 1u) {
        c_diag[i] = -r * half;
    }
}

// Apply source term (operator splitting approach)
// u = u + Δt · f(x)
struct SourceParams {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    dt: f64,
}

@group(0) @binding(0) var<uniform> source_params: SourceParams;
@group(0) @binding(1) var<storage, read> source: array<f64>;
@group(0) @binding(2) var<storage, read_write> u_src: array<f64>;

@compute @workgroup_size(256, 1, 1)
fn apply_source(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    
    if (i >= source_params.n) {
        return;
    }
    
    u_src[i] = u_src[i] + source_params.dt * source[i];
}

// 2D Crank-Nicolson (ADI method - Alternating Direction Implicit)
// For ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)
// Split into two half-steps, each with a tridiagonal solve
struct ADI2DParams {
    nx: u32,          // Grid points in x
    ny: u32,          // Grid points in y
    _pad0: u32,
    _pad1: u32,
    rx: f64,          // α·Δt/(2·Δx²)
    ry: f64,          // α·Δt/(2·Δy²)
}

@group(0) @binding(0) var<uniform> adi_params: ADI2DParams;
@group(0) @binding(1) var<storage, read> u_2d: array<f64>;           // [ny × nx]
@group(0) @binding(2) var<storage, read_write> rhs_2d: array<f64>;   // [ny × nx]

// Compute RHS for x-sweep (rows are independent - parallel across rows)
// Each row becomes a tridiagonal system
@compute @workgroup_size(16, 16, 1)
fn adi_rhs_x_sweep(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;  // x index (column)
    let j = global_id.y;  // y index (row)
    let nx = adi_params.nx;
    let ny = adi_params.ny;
    let rx = adi_params.rx;
    let ry = adi_params.ry;
    let two = f64(2.0);
    
    if (i >= nx || j >= ny) {
        return;
    }
    
    let idx = j * nx + i;
    
    // Get y-neighbors
    var u_up: f64 = f64(0.0);
    var u_down: f64 = f64(0.0);
    
    if (j > 0u) {
        u_down = u_2d[(j - 1u) * nx + i];
    }
    if (j < ny - 1u) {
        u_up = u_2d[(j + 1u) * nx + i];
    }
    
    // RHS includes explicit y-diffusion
    rhs_2d[idx] = u_2d[idx] + ry * (u_up - two * u_2d[idx] + u_down);
}

// Compute RHS for y-sweep (columns are independent)
@compute @workgroup_size(16, 16, 1)
fn adi_rhs_y_sweep(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let j = global_id.y;
    let nx = adi_params.nx;
    let ny = adi_params.ny;
    let rx = adi_params.rx;
    let ry = adi_params.ry;
    let two = f64(2.0);
    
    if (i >= nx || j >= ny) {
        return;
    }
    
    let idx = j * nx + i;
    
    // Get x-neighbors
    var u_left: f64 = f64(0.0);
    var u_right: f64 = f64(0.0);
    
    if (i > 0u) {
        u_left = u_2d[idx - 1u];
    }
    if (i < nx - 1u) {
        u_right = u_2d[idx + 1u];
    }
    
    // RHS includes explicit x-diffusion
    rhs_2d[idx] = u_2d[idx] + rx * (u_left - two * u_2d[idx] + u_right);
}

// Compute Laplacian (for any use) - parallel
struct LaplacianParams {
    nx: u32,
    ny: u32,
    _pad0: u32,
    _pad1: u32,
    dx: f64,
    dy: f64,
}

@group(0) @binding(0) var<uniform> lap_params: LaplacianParams;
@group(0) @binding(1) var<storage, read> u_lap: array<f64>;
@group(0) @binding(2) var<storage, read_write> laplacian: array<f64>;

@compute @workgroup_size(16, 16, 1)
fn compute_laplacian(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let j = global_id.y;
    let nx = lap_params.nx;
    let ny = lap_params.ny;
    let two = f64(2.0);
    
    if (i == 0u || i >= nx - 1u || j == 0u || j >= ny - 1u) {
        return;  // Skip boundaries
    }
    
    let idx = j * nx + i;
    let dx2 = lap_params.dx * lap_params.dx;
    let dy2 = lap_params.dy * lap_params.dy;
    
    // ∇²u = (u_{i+1,j} - 2u_{i,j} + u_{i-1,j})/Δx² + (u_{i,j+1} - 2u_{i,j} + u_{i,j-1})/Δy²
    let d2u_dx2 = (u_lap[idx + 1u] - two * u_lap[idx] + u_lap[idx - 1u]) / dx2;
    let d2u_dy2 = (u_lap[idx + nx] - two * u_lap[idx] + u_lap[idx - nx]) / dy2;
    
    laplacian[idx] = d2u_dx2 + d2u_dy2;
}
