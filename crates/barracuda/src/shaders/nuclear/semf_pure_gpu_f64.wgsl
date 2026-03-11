// SPDX-License-Identifier: AGPL-3.0-only
// Pure-GPU SEMF: all math done on GPU via math_f64 library
// Input:  nuclei[3*i] = Z, nuclei[3*i+1] = N, nuclei[3*i+2] = A
// Input:  nmp[0..3] = a_v, r0, a_a, e2
// Output: energies[i] = B(Z,N) in MeV

@group(0) @binding(0) var<storage, read> nuclei: array<f64>;
@group(0) @binding(1) var<storage, read> nmp: array<f64>;
@group(0) @binding(2) var<storage, read_write> energies: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&energies)) {
        return;
    }

    let base = 3u * idx;
    let z = nuclei[base + 0u];
    let n = nuclei[base + 1u];
    let a = nuclei[base + 2u];

    if (a < 1.0) {
        energies[idx] = a - a;  // 0.0 as f64
        return;
    }

    let a_v = nmp[0];
    let r0  = nmp[1];
    let a_a = nmp[2];
    let e2  = nmp[3];

    // ALL transcendentals computed on GPU via math_f64 library
    // Constants must be f64: construct from 'a' which is f64
    let two_thirds = (a - a + 2.0) / (a - a + 3.0);
    let one_third  = (a - a + 1.0) / (a - a + 3.0);
    let a_23   = pow_f64(a, two_thirds);    // A^(2/3) — GPU computed
    let a_13   = pow_f64(a, one_third);     // A^(1/3) — GPU computed
    let sqrt_a = sqrt_f64(a);               // sqrt(A) — GPU computed

    let a_s = a_v * 1.1;
    let a_c = 3.0 * e2 / (5.0 * r0);
    let a_p = 12.0 / sqrt_a;

    // Bethe-Weizsacker mass formula
    var b = a_v * a;
    b = b - a_s * a_23;
    b = b - a_c * z * (z - 1.0) / a_13;
    b = b - a_a * (n - z) * (n - z) / a;

    // Pairing: even-even +delta, odd-odd -delta
    let z_i = i32(z);
    let n_i = i32(n);
    if ((z_i & 1) == 0 && (n_i & 1) == 0) {
        b = b + a_p;
    } else if ((z_i & 1) == 1 && (n_i & 1) == 1) {
        b = b - a_p;
    }

    energies[idx] = max_f64(b, b - b);  // max(b, 0) via math_f64 library
}
