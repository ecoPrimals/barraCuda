// SPDX-License-Identifier: AGPL-3.0-only
// Meshgrid - Create coordinate grids (complete implementation) (f64 canonical)
// Generates coordinate matrices from coordinate vectors
//
// Example: meshgrid([1,2,3], [4,5]) → [[1,2,3],[1,2,3]], [[4,4,4],[5,5,5]]
//
// Algorithm:
// For each output grid, repeat vectors along appropriate dimensions

struct Params {
    size_x: u32,
    size_y: u32,
    indexing: u32,   // 0 = 'xy' (Cartesian), 1 = 'ij' (matrix)
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> vec_x: array<f64>;
@group(0) @binding(2) var<storage, read> vec_y: array<f64>;
@group(0) @binding(3) var<storage, read_write> grid_x: array<f64>;  // [size_y, size_x] or [size_x, size_y]
@group(0) @binding(4) var<storage, read_write> grid_y: array<f64>;  // [size_y, size_x] or [size_x, size_y]

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.y;
    let j = global_id.x;
    
    if (params.indexing == 0u) {
        // 'xy' indexing (Cartesian)
        if (i >= params.size_y || j >= params.size_x) {
            return;
        }
        let idx = i * params.size_x + j;
        grid_x[idx] = vec_x[j];
        grid_y[idx] = vec_y[i];
        
    } else {
        // 'ij' indexing (matrix)
        if (i >= params.size_x || j >= params.size_y) {
            return;
        }
        let idx = i * params.size_y + j;
        grid_x[idx] = vec_x[i];
        grid_y[idx] = vec_y[j];
    }
}
