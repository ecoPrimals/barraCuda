// SPDX-License-Identifier: AGPL-3.0-or-later
//
// structure_violation_f64.wgsl — Steric clash + bond geometry violations (AlphaFold2)
//
// Clash: violation[i] = max(0, d_vdw - dist(i,j)) for all j != i
// Bond: deviation from ideal length per bond pair.
//
// Bindings: @0 positions[N*3], @1 bond_pairs[B*2], @2 out_clash[N], @3 out_bond[B],
//          @4 uniform{n_atoms, n_bonds, d_vdw}

struct StructureViolationParams {
    n_atoms: u32,
    n_bonds: u32,
    _pad: [u32; 2],
    d_vdw: f64,
}

@group(0) @binding(0) var<storage, read>       positions: array<f64>;   // [N*3]
@group(0) @binding(1) var<storage, read>       bond_pairs: array<u32>; // [B*2]
@group(0) @binding(2) var<storage, read_write> out_clash: array<f64>;  // [N]
@group(0) @binding(3) var<storage, read_write> out_bond: array<f64>;   // [B]
@group(0) @binding(4) var<uniform>             params: StructureViolationParams;

fn dist_atoms(i: u32, j: u32) -> f64 {
    let pi = i * 3u;
    let pj = j * 3u;
    let dx = positions[pi] - positions[pj];
    let dy = positions[pi + 1u] - positions[pj + 1u];
    let dz = positions[pi + 2u] - positions[pj + 2u];
    return sqrt_f64(dx * dx + dy * dy + dz * dz);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = params.n_atoms;
    let B = params.n_bonds;
    let d_vdw = params.d_vdw;

    let idx = gid.x;

    // Clash: threads 0..N-1
    if idx < N {
        var max_clash = f64(0.0);
        for (var j = 0u; j < N; j = j + 1u) {
            if idx == j { continue; }
            let d = dist_atoms(idx, j);
            let v = max(0.0, d_vdw - d);
            max_clash = max(max_clash, v);
        }
        out_clash[idx] = max_clash;
    }

    // Bond: threads N..N+B-1
    if idx >= N && idx < N + B {
        let b_idx = idx - N;
        let a = bond_pairs[b_idx * 2u];
        let b = bond_pairs[b_idx * 2u + 1u];
        let d = dist_atoms(a, b);
        out_bond[b_idx] = abs(d - d_vdw);
    }
}
