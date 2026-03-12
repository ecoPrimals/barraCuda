// SPDX-License-Identifier: AGPL-3.0-only
// batch_tolerance_search_f64.wgsl — PFAS ion batch tolerance search
//
// **WetSpring handoff absorption** — Exp018 Jones Lab PFAS library screening.
//
// Problem: Match S environmental sample ion masses against R reference ions
// from the PFAS library simultaneously.  For S=10,000 samples and R=259
// reference ions this is 2.59 M comparisons — ideal for GPU.
//
// Match criterion: |m_sample - m_ref| / m_ref < ppm_tol × 1e-6
//   OR             |m_sample - m_ref| < da_tol (absolute tolerance)
//
// Output match_matrix[s * n_refs + r] is a score in [0,1]:
//   1.0 = exact match (Δm = 0)
//   0.0 = outside tolerance
//   linear interpolation between 0 and tolerance limit
// This allows downstream weighted ranking rather than hard cutoffs.
//
// Bindings:
//   0: config uniform
//   1: sample_masses  [S]     f64 — measured m/z values
//   2: ref_masses     [R]     f64 — PFAS library reference m/z values
//   3: match_matrix   [S × R] f32 — match scores (f32 sufficient for scores)

struct TolSearchConfig {
    n_samples:  u32,
    n_refs:     u32,
    _pad0:      u32,
    _pad1:      u32,
    ppm_tol:    f64,   // PPM tolerance (e.g., 5.0 for 5 ppm)
    da_tol:     f64,   // Absolute Da tolerance (fallback for low-mass ions)
}

@group(0) @binding(0) var<uniform>             config:       TolSearchConfig;
@group(0) @binding(1) var<storage, read>       sample_masses: array<f64>;
@group(0) @binding(2) var<storage, read>       ref_masses:    array<f64>;
@group(0) @binding(3) var<storage, read_write> match_matrix:  array<f32>;

// Each thread handles one (sample, reference) pair.
// Dispatch: ceil(S/16) × ceil(R/16) workgroups, 16×16 threads each.
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let s = global_id.x;
    let r = global_id.y;

    if (s >= config.n_samples || r >= config.n_refs) { return; }

    let m_sample = sample_masses[s];
    let m_ref    = ref_masses[r];

    // Use the more permissive of PPM and Da tolerances.
    let ppm_window = m_ref * config.ppm_tol * 1e-6;
    let tol        = max(ppm_window, config.da_tol);
    let delta      = abs(m_sample - m_ref);

    var score: f32;
    if (delta >= tol) {
        score = 0.0f;
    } else {
        // Linear score: 1 at exact match, 0 at tolerance boundary.
        score = f32(1.0 - delta / tol);
    }

    match_matrix[s * config.n_refs + r] = score;
}
