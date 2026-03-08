// RFFT extract — copy first N/2+1 complex values from full spectrum.
// Exploits conjugate symmetry: X[k] = conj(X[N-k]) for real inputs.
// f64 canonical; coralReef handles precision lowering.

@group(0) @binding(0) var<storage, read>       spectrum: array<f64>;
@group(0) @binding(1) var<storage, read_write> output:   array<f64>;
@group(0) @binding(2) var<uniform>             params:   ExtractParams;

struct ExtractParams {
    unique_points: u32,
    _p1:           u32,
    _p2:           u32,
    _p3:           u32,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.unique_points { return; }
    output[idx * 2u]       = spectrum[idx * 2u];
    output[idx * 2u + 1u]  = spectrum[idx * 2u + 1u];
}
