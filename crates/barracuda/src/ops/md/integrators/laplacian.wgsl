//! Laplacian Stencil (7-point 3D)
//!
//! **Physics**: Diffusion, wave equations, Poisson solver
//! **Algorithm**: Finite difference approximation of ∇²u
//! **Use Case**: Heat diffusion, electrostatics (PPPM), fluid dynamics
//!
//! **3D Laplacian**:
//! ∇²u(x,y,z) ≈ [u(x±h,y,z) + u(x,y±h,z) + u(x,y,z±h) - 6u(x,y,z)] / h²
//!
//! **7-point stencil** (including center):
//!   Front/Back (z±1)
//!   Left/Right (x±1)  
//!   Top/Bottom (y±1)
//!   Center (0,0,0)
//!
//! **Applications**:
//! - PPPM long-range electrostatics (mesh solver)
//! - Reaction-diffusion systems
//! - Quantum mechanics (Schrödinger equation)
//!
//! **Deep Debt Compliance**:
//! - ✅ Pure WGSL shader
//! - ✅ Zero unsafe code
//! - ✅ Periodic boundaries (3D mesh wrap)

@group(0) @binding(0) var<storage, read> field: array<f32>;           // [nx, ny, nz] flattened
@group(0) @binding(1) var<storage, read_write> laplacian: array<f32>; // Output
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    nx: u32,           // Grid size x
    ny: u32,           // Grid size y
    nz: u32,           // Grid size z
    h_squared: f32,    // Grid spacing squared (h²)
}

// Convert 3D index to flat index with periodic wrapping
fn idx(x: i32, y: i32, z: i32, nx: u32, ny: u32, nz: u32) -> u32 {
    // Periodic boundary conditions
    let x_wrap = (x + i32(nx)) % i32(nx);
    let y_wrap = (y + i32(ny)) % i32(ny);
    let z_wrap = (z + i32(nz)) % i32(nz);
    
    return u32(x_wrap) + u32(y_wrap) * nx + u32(z_wrap) * nx * ny;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);
    let z = i32(global_id.z);
    
    if (global_id.x >= params.nx || global_id.y >= params.ny || global_id.z >= params.nz) {
        return;
    }
    
    // Load center and 6 neighbors
    let center = field[idx(x, y, z, params.nx, params.ny, params.nz)];
    
    let left   = field[idx(x-1, y, z, params.nx, params.ny, params.nz)];
    let right  = field[idx(x+1, y, z, params.nx, params.ny, params.nz)];
    
    let bottom = field[idx(x, y-1, z, params.nx, params.ny, params.nz)];
    let top    = field[idx(x, y+1, z, params.nx, params.ny, params.nz)];
    
    let back   = field[idx(x, y, z-1, params.nx, params.ny, params.nz)];
    let front  = field[idx(x, y, z+1, params.nx, params.ny, params.nz)];
    
    // 7-point Laplacian stencil
    // ∇²u ≈ (sum_neighbors - 6*center) / h²
    let lap = (left + right + bottom + top + back + front - 6.0 * center) / params.h_squared;
    
    // Write result
    let out_idx = idx(x, y, z, params.nx, params.ny, params.nz);
    laplacian[out_idx] = lap;
}
