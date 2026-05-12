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
    const DF64_CORE_CL: &str = include_str!("../../../shaders/math/df64_core.wgsl");
    const DF64_TRANS_CL: &str = include_str!("../../../shaders/math/df64_transcendentals.wgsl");
    let full = format!("{DF64_CORE_CL}\n{DF64_TRANS_CL}\n{src}");

    let module = naga::front::wgsl::parse_str(&full)
        .unwrap_or_else(|e| panic!("should parse: {e}\n\nSource:\n{full}"));
    let mut v = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    v.validate(&module)
        .unwrap_or_else(|e| panic!("should validate: {e}\n\nSource:\n{full}"));
}

/// Validate the **production** `yukawa_df64.wgsl` shader parses and validates
/// when combined with the DF64 preamble. This shader uses `df64_*` functions
/// directly (pre-rewritten) and is the actual code dispatched on NVK hardware.
#[test]
fn test_production_yukawa_df64_validates() {
    const DF64_CORE: &str = include_str!("../../../shaders/math/df64_core.wgsl");
    const DF64_TRANS: &str = include_str!("../../../shaders/math/df64_transcendentals.wgsl");
    const YUKAWA_DF64: &str = include_str!("../../../ops/md/forces/yukawa_df64.wgsl");

    // The production shader needs `round_f64` from the math_f64 preamble.
    // Inject a minimal stub — the real preamble is injected by `compile_shader_f64`.
    let round_stub = "fn round_f64(x: f64) -> f64 { return round(x); }\n";

    let full = format!("{DF64_CORE}\n{DF64_TRANS}\n{round_stub}\n{YUKAWA_DF64}");

    let module = naga::front::wgsl::parse_str(&full)
        .unwrap_or_else(|e| panic!("production yukawa_df64.wgsl should parse: {e}"));
    let mut v = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    v.validate(&module)
        .unwrap_or_else(|e| panic!("production yukawa_df64.wgsl should validate: {e}"));
}

/// CPU reference implementation for Yukawa DF64 force/PE verification.
/// Computes the same quantities as `yukawa_df64.wgsl` using exact f64 arithmetic.
/// Use this to validate GPU results when NVK hardware is available.
#[test]
fn test_yukawa_cpu_reference_two_particles() {
    let kappa = 2.0_f64;
    let prefactor = 1.0_f64;
    let box_side = 10.0_f64;

    // Two particles along x-axis separated by distance 1.0
    let positions = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let n = 2usize;

    let mut forces = [0.0f64; 6];
    let mut pe = [0.0f64; 2];

    for i in 0..n {
        let xi = positions[i * 3];
        let yi = positions[i * 3 + 1];
        let zi = positions[i * 3 + 2];

        let mut fx = 0.0f64;
        let mut fy = 0.0f64;
        let mut fz = 0.0f64;
        let mut pe_i = 0.0f64;

        for j in 0..n {
            if i == j {
                continue;
            }
            let dx = positions[j * 3] - xi;
            let dy = positions[j * 3 + 1] - yi;
            let dz = positions[j * 3 + 2] - zi;

            // PBC minimum image
            let dx = dx - box_side * (dx / box_side).round();
            let dy = dy - box_side * (dy / box_side).round();
            let dz = dz - box_side * (dz / box_side).round();

            let r_sq = dx * dx + dy * dy + dz * dz;
            let r = r_sq.sqrt();
            let screening = (-kappa * r).exp();
            let kappa_r = kappa * r;
            let force_mag = prefactor * screening * (1.0 + kappa_r) / r_sq;
            let inv_r = 1.0 / r;

            // Force: repulsive (subtract, same sign as shader)
            fx -= force_mag * dx * inv_r;
            fy -= force_mag * dy * inv_r;
            fz -= force_mag * dz * inv_r;

            // PE: U = prefactor * exp(-kappa*r) / r * 0.5 (half-counting)
            pe_i += 0.5 * prefactor * screening * inv_r;
        }
        forces[i * 3] = fx;
        forces[i * 3 + 1] = fy;
        forces[i * 3 + 2] = fz;
        pe[i] = pe_i;
    }

    // Analytical check: r=1, kappa=2, prefactor=1
    // |F| = prefactor * exp(-kappa*r) * (1 + kappa*r) / r^2 = exp(-2) * 3 ≈ 0.4060
    // Force is repulsive: particle 0 pushed away from particle 1 (negative x)
    let expected_force_mag = (-2.0f64).exp() * 3.0;
    assert!(
        (forces[0] + expected_force_mag).abs() < 1e-14,
        "particle 0 fx should be -mag (repulsive): {} vs expected {}",
        forces[0],
        -expected_force_mag
    );
    assert!(
        (forces[3] - expected_force_mag).abs() < 1e-14,
        "particle 1 fx should be +mag (repulsive opposite): {} vs {}",
        forces[3],
        expected_force_mag
    );

    // PE: 0.5 * exp(-2) / 1 = 0.5 * exp(-2) ≈ 0.0677
    let expected_pe = 0.5 * (-2.0f64).exp();
    assert!(
        (pe[0] - expected_pe).abs() < 1e-14,
        "pe[0]: {} vs expected {}",
        pe[0],
        expected_pe
    );
    assert!((pe[0] - pe[1]).abs() < 1e-14, "pe should be symmetric");

    // Newton's 3rd law
    assert!((forces[0] + forces[3]).abs() < 1e-14, "Newton 3rd law x");
    assert!((forces[1] + forces[4]).abs() < 1e-14, "Newton 3rd law y");
    assert!((forces[2] + forces[5]).abs() < 1e-14, "Newton 3rd law z");
}
