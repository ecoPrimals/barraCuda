// SPDX-License-Identifier: AGPL-3.0-only
//
// backbone_update_f64.wgsl — Rigid body backbone frame update (AlphaFold2 structure module)
//
// T_i = T_i * delta_T: compose current frame with delta (rotation quaternion + translation).
// Quaternion multiply: q1*q2 = (w1w2-x1x2-y1y2-z1z2, w1x2+x1w2+y1z2-z1y2, ...)
// Then rotate delta_trans by q1 and add to t1.
//
// Bindings: @0 quaternions[N*4], @1 translations[N*3], @2 delta_quat[N*4], @3 delta_trans[N*3],
//          @4 out_quat[N*4], @5 out_trans[N*3], @6 uniform{n: u32}
//
// Provenance: neuralSpring → ToadStool absorption

enable f64;

struct BackboneParams {
    n: u32,
}

@group(0) @binding(0) var<storage, read>       quaternions:   array<f64>;  // [N*4] (w,x,y,z)
@group(0) @binding(1) var<storage, read>       translations:  array<f64>;  // [N*3]
@group(0) @binding(2) var<storage, read>       delta_quat:    array<f64>;  // [N*4]
@group(0) @binding(3) var<storage, read>       delta_trans:   array<f64>;  // [N*3]
@group(0) @binding(4) var<storage, read_write> out_quat:     array<f64>;  // [N*4]
@group(0) @binding(5) var<storage, read_write> out_trans:    array<f64>;  // [N*3]
@group(0) @binding(6) var<uniform>             params:       BackboneParams;

fn quat_mul(w1: f64, x1: f64, y1: f64, z1: f64, w2: f64, x2: f64, y2: f64, z2: f64) -> vec4<f64> {
    return vec4<f64>(
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    );
}

fn quat_rotate_vector(qw: f64, qx: f64, qy: f64, qz: f64, vx: f64, vy: f64, vz: f64) -> vec3<f64> {
    let r = vec3<f64>(qx, qy, qz);
    let v = vec3<f64>(vx, vy, vz);
    let t = 2.0 * cross(r, cross(r, v) + qw * v);
    return v + t;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n { return; }

    let base_q = idx * 4u;
    let base_t = idx * 3u;

    let w1 = quaternions[base_q + 0u];
    let x1 = quaternions[base_q + 1u];
    let y1 = quaternions[base_q + 2u];
    let z1 = quaternions[base_q + 3u];

    let w2 = delta_quat[base_q + 0u];
    let x2 = delta_quat[base_q + 1u];
    let y2 = delta_quat[base_q + 2u];
    let z2 = delta_quat[base_q + 3u];

    let q_out = quat_mul(w1, x1, y1, z1, w2, x2, y2, z2);
    out_quat[base_q + 0u] = q_out.x;
    out_quat[base_q + 1u] = q_out.y;
    out_quat[base_q + 2u] = q_out.z;
    out_quat[base_q + 3u] = q_out.w;

    let dt_x = delta_trans[base_t + 0u];
    let dt_y = delta_trans[base_t + 1u];
    let dt_z = delta_trans[base_t + 2u];

    let rotated = quat_rotate_vector(w1, x1, y1, z1, dt_x, dt_y, dt_z);

    out_trans[base_t + 0u] = translations[base_t + 0u] + rotated.x;
    out_trans[base_t + 1u] = translations[base_t + 1u] + rotated.y;
    out_trans[base_t + 2u] = translations[base_t + 2u] + rotated.z;
}
