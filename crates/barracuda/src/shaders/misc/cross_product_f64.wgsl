// cross_product_f64.wgsl - Vector cross product (f64 canonical)
//
// Computes cross product of 3D vectors
// a × b = (a_y*b_z - a_z*b_y, a_z*b_x - a_x*b_z, a_x*b_y - a_y*b_x)

struct Params {
    num_vectors: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
    _pad5: u32,
    _pad6: u32,
}

@group(0) @binding(0) var<storage, read> input_a: array<f64>;  // [N, 3]
@group(0) @binding(1) var<storage, read> input_b: array<f64>;  // [N, 3]
@group(0) @binding(2) var<storage, read_write> output: array<f64>; // [N, 3]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.num_vectors) {
        return;
    }

    // Load vectors
    let base = idx * 3u;
    let ax = input_a[base + 0u];
    let ay = input_a[base + 1u];
    let az = input_a[base + 2u];

    let bx = input_b[base + 0u];
    let by = input_b[base + 1u];
    let bz = input_b[base + 2u];

    // Cross product
    output[base + 0u] = ay * bz - az * by;
    output[base + 1u] = az * bx - ax * bz;
    output[base + 2u] = ax * by - ay * bx;
}
