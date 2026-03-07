// SPDX-License-Identifier: AGPL-3.0-or-later
//! NAK/NVK stress tests — patterns from hotSpring Yukawa handoff.
//! Extracted from `tests.rs` for maintainability (wateringHole 1000-line convention).

use super::*;

/// Yukawa-pattern: compound assignments (+=, -=) on f64 accumulators.
/// hotSpring found that naga-guided rewrite of compound assignments
/// produced invalid SPIR-V on NVK/NAK. Verify our rewriter handles them.
#[test]
fn test_nak_compound_assignment() {
    let wgsl = r"
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> n: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
let i = gid.x;
if i >= n { return; }
var acc: f64 = 0.0;
for (var j = 0u; j < n; j = j + 1u) {
    if i == j { continue; }
    let dx = input[j] - input[i];
    let r2 = dx * dx;
    acc += dx / r2;
}
output[i] = acc;
}
";
    let result = rewrite_f64_infix_full(wgsl);
    assert!(
        result.is_ok(),
        "Yukawa compound-assign pattern should rewrite: {result:?}"
    );
    let src = result.unwrap();
    assert!(
        src.contains("_df64_sub_f64(") || src.contains("_df64_add_f64("),
        "should contain bridge calls, got:\n{src}"
    );
}

/// Yukawa-pattern: f64 comparisons (>, <) with continue in loops.
#[test]
fn test_nak_comparison_with_continue() {
    let wgsl = r"
@group(0) @binding(0) var<storage, read> pos: array<f64>;
@group(0) @binding(1) var<storage, read_write> forces: array<f64>;
@group(0) @binding(2) var<uniform> n: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
let i = gid.x;
if i >= n { return; }
var fx: f64 = 0.0;
let cutoff_sq: f64 = 9.0;
for (var j = 0u; j < n; j = j + 1u) {
    if i == j { continue; }
    let dx = pos[j] - pos[i];
    let r2 = dx * dx;
    if r2 > cutoff_sq { continue; }
    let r = sqrt(r2);
    let force = exp(-r) / r2;
    fx += force * dx;
}
forces[i] = fx;
}
";
    let result = rewrite_f64_infix_full(wgsl);
    assert!(
        result.is_ok(),
        "comparison+continue pattern should rewrite: {result:?}"
    );
    let src = result.unwrap();
    assert!(
        src.contains("_df64_gt_f64(") || src.contains("_df64_mul_f64("),
        "should contain comparison or arithmetic bridges, got:\n{src}"
    );
}

/// Full Yukawa-like force kernel with all NAK-problematic patterns:
/// compound +=/-=, f64 comparisons, continue, sqrt, exp, nested ops.
/// Must produce valid WGSL when combined with `df64_core` + `df64_transcendentals`.
#[test]
fn test_nak_yukawa_full_validates() {
    let wgsl = r"
struct Params {
n: u32,
_pad0: u32,
cutoff: f64,
cutoff_sq: f64,
}

@group(0) @binding(0) var<storage, read> positions: array<f64>;
@group(0) @binding(1) var<storage, read_write> forces: array<f64>;
@group(0) @binding(2) var<storage, read_write> pe_buf: array<f64>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn yukawa_force(@builtin(global_invocation_id) gid: vec3<u32>) {
let i = gid.x;
if i >= params.n { return; }

let xi = positions[i * 3u];
let yi = positions[i * 3u + 1u];
let zi = positions[i * 3u + 2u];

var fx: f64 = 0.0;
var fy: f64 = 0.0;
var fz: f64 = 0.0;
var pe: f64 = 0.0;

for (var j = 0u; j < params.n; j = j + 1u) {
    if i == j { continue; }

    let dx = positions[j * 3u] - xi;
    let dy = positions[j * 3u + 1u] - yi;
    let dz = positions[j * 3u + 2u] - zi;
    let r2 = dx * dx + dy * dy + dz * dz;

    if r2 > params.cutoff_sq { continue; }

    let r = sqrt(r2);
    let exp_r = exp(-r);
    let inv_r = 1.0 / r;
    let force_mag = exp_r * inv_r * inv_r * (1.0 + r);

    fx += force_mag * dx * inv_r;
    fy += force_mag * dy * inv_r;
    fz += force_mag * dz * inv_r;
    pe += exp_r * inv_r;
}

forces[i * 3u] = fx;
forces[i * 3u + 1u] = fy;
forces[i * 3u + 2u] = fz;
pe_buf[i] = pe * 0.5;
}
";
    let rewritten = rewrite_f64_infix_full(wgsl);
    assert!(
        rewritten.is_ok(),
        "Yukawa force kernel should rewrite successfully: {rewritten:?}"
    );

    let src = rewritten.unwrap();
    assert!(src.contains("_df64_mul_f64("), "should have mul bridges");
    assert!(src.contains("_df64_add_f64("), "should have add bridges");
    assert!(src.contains("_df64_sub_f64("), "should have sub bridges");
    assert!(src.contains("_df64_div_f64("), "should have div bridges");
    assert!(
        src.contains("_df64_gt_f64("),
        "should have comparison bridges"
    );

    // Validate the rewritten source compiles with df64 core
    const DF64_CORE: &str = include_str!("../../../shaders/math/df64_core.wgsl");
    const DF64_TRANS: &str = include_str!("../../../shaders/math/df64_transcendentals.wgsl");
    let full = format!("{DF64_CORE}\n{DF64_TRANS}\n{src}");

    let module = naga::front::wgsl::parse_str(&full)
        .unwrap_or_else(|e| panic!("should parse: {e}\n\nSource:\n{full}"));
    let mut v = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    v.validate(&module)
        .unwrap_or_else(|e| panic!("should validate: {e}\n\nSource:\n{full}"));
}

#[test]
fn test_nak_cg_solver_validates() {
    let wgsl = r"
struct CgParams {
n: u32,
_pad0: u32,
_pad1: u32,
_pad2: u32,
}

@group(0) @binding(0) var<storage, read> r: array<f64>;
@group(0) @binding(1) var<storage, read> p: array<f64>;
@group(0) @binding(2) var<storage, read_write> x: array<f64>;
@group(0) @binding(3) var<uniform> params: CgParams;

@compute @workgroup_size(256)
fn cg_update_xr(@builtin(global_invocation_id) gid: vec3<u32>) {
let i = gid.x;
if i >= params.n { return; }

let alpha: f64 = 0.5;
let beta: f64 = 0.3;

x[i] = x[i] + alpha * p[i];

let r_new = r[i] - alpha * p[i];
let p_new = r_new + beta * p[i];
let dot = r_new * r_new;
let norm = sqrt(dot);

x[i] = x[i] + norm * 0.001;
}
";
    let rewritten = rewrite_f64_infix_full(wgsl);
    assert!(
        rewritten.is_ok(),
        "CG solver kernel should rewrite successfully: {rewritten:?}"
    );

    let src = rewritten.unwrap();
    assert!(src.contains("_df64_add_f64("), "should have add bridges");
    assert!(src.contains("_df64_mul_f64("), "should have mul bridges");
    assert!(src.contains("_df64_sub_f64("), "should have sub bridges");

    const DF64_CORE: &str = include_str!("../../../shaders/math/df64_core.wgsl");
    const DF64_TRANS: &str = include_str!("../../../shaders/math/df64_transcendentals.wgsl");
    let full = format!("{DF64_CORE}\n{DF64_TRANS}\n{src}");

    let module = naga::front::wgsl::parse_str(&full)
        .unwrap_or_else(|e| panic!("should parse: {e}\n\nSource:\n{full}"));
    let mut v = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    v.validate(&module)
        .unwrap_or_else(|e| panic!("should validate: {e}\n\nSource:\n{full}"));
}

#[test]
fn test_nak_yukawa_celllist_validates() {
    let wgsl = r"
struct CellParams {
n: u32,
n_cells: u32,
cutoff: f64,
cutoff_sq: f64,
box_x: f64,
box_y: f64,
box_z: f64,
}

@group(0) @binding(0) var<storage, read> positions: array<f64>;
@group(0) @binding(1) var<storage, read> cell_list: array<u32>;
@group(0) @binding(2) var<storage, read> cell_start: array<u32>;
@group(0) @binding(3) var<storage, read_write> forces: array<f64>;
@group(0) @binding(4) var<uniform> params: CellParams;

fn pbc_wrap(dx: f64, box_len: f64) -> f64 {
var d = dx;
if d > box_len * 0.5 { d -= box_len; }
if d < -box_len * 0.5 { d += box_len; }
return d;
}

@compute @workgroup_size(256)
fn yukawa_celllist(@builtin(global_invocation_id) gid: vec3<u32>) {
let i = gid.x;
if i >= params.n { return; }

let xi = positions[i * 3u];
let yi = positions[i * 3u + 1u];
let zi = positions[i * 3u + 2u];

var fx: f64 = 0.0;
var fy: f64 = 0.0;
var fz: f64 = 0.0;

for (var c = 0u; c < params.n_cells; c++) {
    let start = cell_start[c];
    let end = cell_start[c + 1u];

    for (var idx = start; idx < end; idx++) {
        let j = cell_list[idx];
        if i == j { continue; }

        var dx = pbc_wrap(positions[j * 3u] - xi, params.box_x);
        var dy = pbc_wrap(positions[j * 3u + 1u] - yi, params.box_y);
        var dz = pbc_wrap(positions[j * 3u + 2u] - zi, params.box_z);
        let r2 = dx * dx + dy * dy + dz * dz;

        if r2 > params.cutoff_sq { continue; }

        let r = sqrt(r2);
        let exp_r = exp(-r);
        let inv_r = 1.0 / r;
        let f_mag = exp_r * inv_r * inv_r * (1.0 + r);

        fx += f_mag * dx * inv_r;
        fy += f_mag * dy * inv_r;
        fz += f_mag * dz * inv_r;
    }
}

forces[i * 3u] = fx;
forces[i * 3u + 1u] = fy;
forces[i * 3u + 2u] = fz;
}
";
    let rewritten = rewrite_f64_infix_full(wgsl);
    assert!(
        rewritten.is_ok(),
        "Yukawa cell-list kernel should rewrite: {rewritten:?}"
    );

    let src = rewritten.unwrap();
    const DF64_CORE: &str = include_str!("../../../shaders/math/df64_core.wgsl");
    const DF64_TRANS: &str = include_str!("../../../shaders/math/df64_transcendentals.wgsl");
    let full = format!("{DF64_CORE}\n{DF64_TRANS}\n{src}");

    let module = naga::front::wgsl::parse_str(&full)
        .unwrap_or_else(|e| panic!("should parse: {e}\n\nSource:\n{full}"));
    let mut v = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    v.validate(&module)
        .unwrap_or_else(|e| panic!("should validate: {e}\n\nSource:\n{full}"));
}
