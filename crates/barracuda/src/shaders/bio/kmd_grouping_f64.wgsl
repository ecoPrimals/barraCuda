// SPDX-License-Identifier: AGPL-3.0-or-later
// kmd_grouping_f64.wgsl — Kendrick Mass Defect homologue grouping
//
// **WetSpring handoff absorption** — Exp018 Jones Lab PFAS library analysis.
//
// Kendrick Mass Defect (KMD) identifies homologous series in mass spec data
// by mapping measured m/z values to a Kendrick mass scale defined by a
// repeating unit (e.g., CH₂ = 14.01565 Da, CF₂ = 49.9969 Da).
//
// Algorithm per ion i:
//   KM_i = m_i × (nominal_unit / exact_unit)   (Kendrick mass)
//   NKM_i = round(KM_i)                         (nominal Kendrick mass)
//   KMD_i = NKM_i − KM_i                        (Kendrick mass defect)
//
// Ions with the same NKM and KMD within a tolerance window belong to the
// same homologous series (differ by integer multiples of the repeat unit).
//
// This shader computes [KM, NKM, KMD] for each ion (Pass 1).
// Grouping (Pass 2) is a pairwise comparison and is handled by
// BatchPairReduceF64 or a CPU post-pass, depending on N.
//
// Bindings:
//   0: config uniform
//   1: masses       [N] f64 — measured exact masses
//   2: kendrick_out [N × 3] f64 — [KM, NKM, KMD] per ion

struct KmdConfig {
    n_ions:      u32,
    _pad:        u32,
    exact_unit:  f64,   // Exact mass of repeat unit (e.g., CH₂: 14.01565)
    nominal_unit: f64,  // Nominal (integer) mass   (e.g., CH₂: 14)
}

@group(0) @binding(0) var<uniform>             config:       KmdConfig;
@group(0) @binding(1) var<storage, read>       masses:       array<f64>;
@group(0) @binding(2) var<storage, read_write> kendrick_out: array<f64>;  // [N × 3]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= config.n_ions) { return; }

    let m  = masses[i];
    let km = m * (config.nominal_unit / config.exact_unit);
    let nkm = round(km);
    let kmd = nkm - km;

    kendrick_out[i * 3u + 0u] = km;
    kendrick_out[i * 3u + 1u] = nkm;
    kendrick_out[i * 3u + 2u] = kmd;
}
