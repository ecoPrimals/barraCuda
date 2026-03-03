// batch_ipr.wgsl — Batch Inverse Participation Ratio (f64 canonical)
//
// Each thread computes IPR = sum(|psi_i|^4) for one eigenvector.
// Extended states have IPR ~ 1/dim; localized states have IPR >> 1/dim.
//
// Complements barracuda::spectral (Anderson localization, Hofstadter butterfly).
//
// Provenance: neuralSpring metalForge (Feb 21, 2026) → ToadStool absorption

struct Params {
    dim: u32,
    n_vectors: u32,
}

@group(0) @binding(0) var<storage, read> eigenvectors: array<f64>;
@group(0) @binding(1) var<storage, read_write> ipr_out: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn batch_ipr(@builtin(global_invocation_id) gid: vec3<u32>) {
    let vec_idx = gid.x;
    if vec_idx >= params.n_vectors { return; }

    let offset = vec_idx * params.dim;
    var sum_p4: f64 = 0.0;

    for (var i: u32 = 0u; i < params.dim; i = i + 1u) {
        let val = eigenvectors[offset + i];
        let p2 = val * val;
        sum_p4 = p2 * p2 + sum_p4;
    }

    ipr_out[vec_idx] = sum_p4;
}
