// SPDX-License-Identifier: AGPL-3.0-or-later
// ssf_f64.wgsl — Static Structure Factor S(k)
//
// **Physics**: S(k) = |Σ_j exp(ik·r_j)|² / N = (1/N) * [(Σ cos(k·r_j))² + (Σ sin(k·r_j))²]
//
// **Algorithm**: Each thread handles one k-vector. Loop over all N particles:
//   cos_sum += cos(kx*rx + ky*ry + kz*rz)
//   sin_sum += sin(kx*rx + ky*ry + kz*rz)
// Then S(k) = (cos_sum² + sin_sum²) / N
//
// **Precision**: f64 via bitcast<f64>(vec2<u32>) — same byte layout as array<f64>
// **Workgroup**: @compute @workgroup_size(64) — matches ssf_gpu dispatch
//
// Bindings (matches ssf_gpu.rs): 0=params uniform, 1=positions, 2=k_vectors, 3=output
// Uses array<f64> (compile_shader_f64 enables f64; ssf_gpu reads f64)
//
// Reference: Hansen & McDonald "Theory of Simple Liquids"

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<f64>;
@group(0) @binding(2) var<storage, read> k_vectors: array<f64>;
@group(0) @binding(3) var<storage, read_write> ssf_out: array<f64>;

struct Params {
    n_particles: u32,
    n_k_vectors: u32,
    _pad1: u32,
    _pad2: u32,
}

// sin_f64 / cos_f64 provided by math_f64.wgsl auto-injection
// (Cody-Waite range reduction + fdlibm minimax kernels, ~1 ULP accuracy)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let k_idx = global_id.x;
    let n = params.n_particles;
    let n_k = params.n_k_vectors;
    if (k_idx >= n_k) {
        return;
    }

    let k_base = k_idx * 3u;
    let kx = k_vectors[k_base];
    let ky = k_vectors[k_base + 1u];
    let kz = k_vectors[k_base + 2u];

    var cos_sum: f64 = 0.0;
    var sin_sum: f64 = 0.0;

    for (var j = 0u; j < n; j = j + 1u) {
        let r_base = j * 3u;
        let rx = positions[r_base];
        let ry = positions[r_base + 1u];
        let rz = positions[r_base + 2u];

        let kr = kx * rx + ky * ry + kz * rz;
        cos_sum = cos_sum + cos_f64(kr);
        sin_sum = sin_sum + sin_f64(kr);
    }

    let ssf = (cos_sum * cos_sum + sin_sum * sin_sum) / f64(n);
    ssf_out[k_idx] = ssf;
}
