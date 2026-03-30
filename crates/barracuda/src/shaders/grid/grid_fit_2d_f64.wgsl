// SPDX-License-Identifier: AGPL-3.0-or-later
//
// grid_fit_2d_f64.wgsl — 2D least-squares surface fit on a structured grid
//
// Fits polynomial surface z = a + bx + cy + dxy to observed data.
// Single-workgroup reduction computes sums for normal equations,
// then solves the 4x4 system X'X β = X'z.
//
// Design matrix row per point: [1, x, y, xy]
//
// Provenance: groundSpring → ToadStool absorption

struct GridFitParams {
    nx: u32,
    ny: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> data: array<f64>;              // [NX*NY]
@group(0) @binding(1) var<storage, read> x_coords: array<f64>;          // [NX]
@group(0) @binding(2) var<storage, read> y_coords: array<f64>;          // [NY]
@group(0) @binding(3) var<storage, read_write> out_coefficients: array<f64>;  // [4] -> a,b,c,d
@group(0) @binding(4) var<uniform> params: GridFitParams;

var<workgroup> shared_sums: array<f64, 64>;  // Scratch for reduction

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let nx = params.nx;
    let ny = params.ny;
    let n_total = nx * ny;
    let tid = lid.x;

    // Each thread accumulates partial sums over its chunk
    var s0: f64 = 0.0;   // sum(1)
    var s1: f64 = 0.0;   // sum(x)
    var s2: f64 = 0.0;   // sum(y)
    var s3: f64 = 0.0;   // sum(xy)
    var s4: f64 = 0.0;   // sum(x^2)
    var s5: f64 = 0.0;   // sum(y^2)
    var s6: f64 = 0.0;   // sum(x^2 y)
    var s7: f64 = 0.0;   // sum(xy^2)
    var s8: f64 = 0.0;   // sum(x^2 y^2)
    var sz: f64 = 0.0;   // sum(z)
    var sxz: f64 = 0.0;  // sum(xz)
    var syz: f64 = 0.0;  // sum(yz)
    var sxyz: f64 = 0.0; // sum(xyz)

    let num_threads = 64u;
    for (var idx = tid; idx < n_total; idx = idx + num_threads) {
        let ix = idx / ny;
        let iy = idx % ny;
        let x = x_coords[ix];
        let y = y_coords[iy];
        let z = data[idx];
        let xy = x * y;
        let x2 = x * x;
        let y2 = y * y;

        s0 += 1.0;
        s1 += x;
        s2 += y;
        s3 += xy;
        s4 += x2;
        s5 += y2;
        s6 += x2 * y;
        s7 += x * y2;
        s8 += x2 * y2;
        sz += z;
        sxz += x * z;
        syz += y * z;
        sxyz += xy * z;
    }

    // Tree reduction across threads: 14 passes, one per sum
    shared_sums[tid] = s0;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride / 2u) {
        if tid < stride {
            shared_sums[tid] = shared_sums[tid] + shared_sums[tid + stride];
        }
        workgroupBarrier();
    }
    let sum0 = shared_sums[0u];

    shared_sums[tid] = s1;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride / 2u) {
        if tid < stride {
            shared_sums[tid] = shared_sums[tid] + shared_sums[tid + stride];
        }
        workgroupBarrier();
    }
    let sum1 = shared_sums[0u];

    shared_sums[tid] = s2;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride / 2u) {
        if tid < stride {
            shared_sums[tid] = shared_sums[tid] + shared_sums[tid + stride];
        }
        workgroupBarrier();
    }
    let sum2 = shared_sums[0u];

    shared_sums[tid] = s3;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride / 2u) {
        if tid < stride {
            shared_sums[tid] = shared_sums[tid] + shared_sums[tid + stride];
        }
        workgroupBarrier();
    }
    let sum3 = shared_sums[0u];

    shared_sums[tid] = s4;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride / 2u) {
        if tid < stride {
            shared_sums[tid] = shared_sums[tid] + shared_sums[tid + stride];
        }
        workgroupBarrier();
    }
    let sum4 = shared_sums[0u];

    shared_sums[tid] = s5;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride / 2u) {
        if tid < stride {
            shared_sums[tid] = shared_sums[tid] + shared_sums[tid + stride];
        }
        workgroupBarrier();
    }
    let sum5 = shared_sums[0u];

    shared_sums[tid] = s6;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride / 2u) {
        if tid < stride {
            shared_sums[tid] = shared_sums[tid] + shared_sums[tid + stride];
        }
        workgroupBarrier();
    }
    let sum6 = shared_sums[0u];

    shared_sums[tid] = s7;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride / 2u) {
        if tid < stride {
            shared_sums[tid] = shared_sums[tid] + shared_sums[tid + stride];
        }
        workgroupBarrier();
    }
    let sum7 = shared_sums[0u];

    shared_sums[tid] = s8;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride / 2u) {
        if tid < stride {
            shared_sums[tid] = shared_sums[tid] + shared_sums[tid + stride];
        }
        workgroupBarrier();
    }
    let sum8 = shared_sums[0u];

    shared_sums[tid] = sz;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride / 2u) {
        if tid < stride {
            shared_sums[tid] = shared_sums[tid] + shared_sums[tid + stride];
        }
        workgroupBarrier();
    }
    let sumz = shared_sums[0u];

    shared_sums[tid] = sxz;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride / 2u) {
        if tid < stride {
            shared_sums[tid] = shared_sums[tid] + shared_sums[tid + stride];
        }
        workgroupBarrier();
    }
    let sumxz = shared_sums[0u];

    shared_sums[tid] = syz;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride / 2u) {
        if tid < stride {
            shared_sums[tid] = shared_sums[tid] + shared_sums[tid + stride];
        }
        workgroupBarrier();
    }
    let sumyz = shared_sums[0u];

    shared_sums[tid] = sxyz;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride / 2u) {
        if tid < stride {
            shared_sums[tid] = shared_sums[tid] + shared_sums[tid + stride];
        }
        workgroupBarrier();
    }
    let sumxyz = shared_sums[0u];

    // Solve 4x4 system: X'X * [a,b,c,d]' = [sumz, sumxz, sumyz, sumxyz]'
    // X'X = [[sum0, sum1, sum2, sum3], [sum1, sum4, sum3, sum6], [sum2, sum3, sum5, sum7], [sum3, sum6, sum7, sum8]]
    if tid == 0u {
        let m = array<f64, 16>(
            sum0, sum1, sum2, sum3,
            sum1, sum4, sum3, sum6,
            sum2, sum3, sum5, sum7,
            sum3, sum6, sum7, sum8
        );
        let rhs = array<f64, 4>(sumz, sumxz, sumyz, sumxyz);

        // 4x4 determinant and inverse (Cramer-style)
        let s0_m = m[0]*m[5] - m[1]*m[4];
        let s1_m = m[0]*m[6] - m[2]*m[4];
        let s2_m = m[0]*m[7] - m[3]*m[4];
        let s3_m = m[1]*m[6] - m[2]*m[5];
        let s4_m = m[1]*m[7] - m[3]*m[5];
        let s5_m = m[2]*m[7] - m[3]*m[6];
        let c5 = s0_m*m[10] - s1_m*m[9] + s3_m*m[8];
        let c4 = s0_m*m[11] - s2_m*m[9] + s4_m*m[8];
        let c3 = s1_m*m[11] - s2_m*m[10] + s5_m*m[8];
        let c2 = s3_m*m[11] - s4_m*m[10] + s5_m*m[9];
        let det = c5*m[15] - c4*m[14] + c3*m[13] - c2*m[12];

        if abs(det) > 1e-15 {
            let inv_det = 1.0 / det;
            for (var col: u32 = 0u; col < 4u; col = col + 1u) {
                var cof: f64 = 0.0;
                for (var row: u32 = 0u; row < 4u; row = row + 1u) {
                    let sign = select(-1.0, 1.0, (row + col) % 2u == 0u);
                    var minor: array<f64, 9>;
                    var mi: u32 = 0u;
                    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                            if i != row && j != col {
                                minor[mi] = m[i*4u+j];
                                mi = mi + 1u;
                            }
                        }
                    }
                    let d = minor[0]*(minor[4]*minor[8]-minor[5]*minor[7])
                          - minor[1]*(minor[3]*minor[8]-minor[5]*minor[6])
                          + minor[2]*(minor[3]*minor[7]-minor[4]*minor[6]);
                    cof += sign * d * rhs[row];
                }
                out_coefficients[col] = cof * inv_det;
            }
        } else {
            out_coefficients[0u] = 0.0;
            out_coefficients[1u] = 0.0;
            out_coefficients[2u] = 0.0;
            out_coefficients[3u] = 0.0;
        }
    }
}
