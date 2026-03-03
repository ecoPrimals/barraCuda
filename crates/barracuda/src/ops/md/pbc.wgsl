// Periodic Boundary Conditions (PBC) Distance Calculation
//
// **Purpose**: Compute minimum image convention distances for MD simulations
// **Input**: Two sets of particle positions, box dimensions
// **Output**: Pairwise distances accounting for periodic boundaries
//
// **Deep Debt Compliance**:
// - Pure WGSL (universal GPU compute)
// - Hardware-agnostic (no vendor-specific code)
// - Mathematically sound (minimum image convention)
//
// **Algorithm**: Minimum Image Convention
// For each pair (i,j):
//   1. Compute delta = pos[j] - pos[i]
//   2. Apply periodic wrapping: delta - box * round(delta / box)
//   3. Compute distance: ||delta||

@group(0) @binding(0) var<storage, read> input_a: array<f32>;     // Positions A [M * D]
@group(0) @binding(1) var<storage, read> input_b: array<f32>;     // Positions B [N * D]
@group(0) @binding(2) var<storage, read> box_dims: array<f32>;    // Box dimensions [D]
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // Distances [M * N]
@group(0) @binding(4) var<uniform> params: Params;

struct Params {
    m: u32,        // Number of particles in A
    n: u32,        // Number of particles in B
    d: u32,        // Dimensionality (typically 3)
    metric: u32,   // 0=L2 (Euclidean), 1=L1 (Manhattan)
}

// Apply minimum image convention to a single component
fn apply_pbc(delta: f32, box_size: f32) -> f32 {
    // delta - box * round(delta / box)
    // This wraps delta to [-box/2, box/2]
    let ratio = delta / box_size;
    let rounded = round(ratio);
    return delta - box_size * rounded;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;  // Index in A
    let j = global_id.y;  // Index in B
    
    if (i >= params.m || j >= params.n) {
        return;
    }
    
    let d = params.d;
    var sum = 0.0;
    
    // Compute distance with PBC
    for (var k = 0u; k < d; k = k + 1u) {
        let a_val = input_a[i * d + k];
        let b_val = input_b[j * d + k];
        let box_k = box_dims[k];
        
        // Apply PBC to delta
        var delta = b_val - a_val;
        delta = apply_pbc(delta, box_k);
        
        // Accumulate based on metric
        if (params.metric == 0u) {
            // L2 (Euclidean): sum of squares
            sum = sum + delta * delta;
        } else {
            // L1 (Manhattan): sum of absolute values
            sum = sum + abs(delta);
        }
    }
    
    // Finalize distance
    var distance = sum;
    if (params.metric == 0u) {
        // L2: take square root
        distance = sqrt(sum);
    }
    
    // Write to output [M x N] matrix
    output[i * params.n + j] = distance;
}
