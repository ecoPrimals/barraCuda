// SPDX-License-Identifier: AGPL-3.0-only
// dnds_batch_f64.wgsl — Batch Pairwise dN/dS (Nei-Gojobori 1986)
//
// One thread per coding sequence pair. Each thread walks codons,
// classifies synonymous/nonsynonymous sites and differences using
// a GPU-resident genetic code lookup table, then applies
// Jukes-Cantor correction.
//
// GPU dispatch: ceil(n_pairs / 64) workgroups, 64 threads each.
// (Smaller workgroup — each thread does significant per-codon work.)
//
// Provenance: wetSpring handoff v5 (Feb 20, 2026) → ToadStool absorption

struct DnDsParams {
    n_pairs:  u32,
    n_codons: u32,
}

@group(0) @binding(0) var<uniform>             params:  DnDsParams;
// Bases encoded as u32: A=0, C=1, G=2, T=3, gap=4
@group(0) @binding(1) var<storage, read>       seq_a:   array<u32>;  // [n_pairs * n_codons * 3]
@group(0) @binding(2) var<storage, read>       seq_b:   array<u32>;  // [n_pairs * n_codons * 3]
// Genetic code: 64 entries, index = b0*16+b1*4+b2, value = amino acid ID (20 = stop)
@group(0) @binding(3) var<storage, read>       genetic_code: array<u32>; // [64]
@group(0) @binding(4) var<storage, read_write> dn_out:    array<f64>;  // [n_pairs]
@group(0) @binding(5) var<storage, read_write> ds_out:    array<f64>;  // [n_pairs]
@group(0) @binding(6) var<storage, read_write> omega_out: array<f64>;  // [n_pairs]

const STOP_AA: u32 = 20u;

fn translate(b0: u32, b1: u32, b2: u32) -> u32 {
    return genetic_code[b0 * 16u + b1 * 4u + b2];
}

// Nei-Gojobori synonymous site count for one codon.
// For each of 3 positions, try 3 alternative bases, count synonymous.
fn codon_syn_sites(b0: u32, b1: u32, b2: u32) -> f64 {
    let orig_aa = translate(b0, b1, b2);
    if orig_aa == STOP_AA { return f64(0); }

    var bases = array<u32, 4>(0u, 1u, 2u, 3u);
    var syn: f64 = f64(0);

    var s0: u32 = 0u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        if bases[i] != b0 {
            let aa = translate(bases[i], b1, b2);
            if aa != STOP_AA && aa == orig_aa { s0 = s0 + 1u; }
        }
    }
    syn = syn + f64(s0) / f64(3);

    var s1: u32 = 0u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        if bases[i] != b1 {
            let aa = translate(b0, bases[i], b2);
            if aa != STOP_AA && aa == orig_aa { s1 = s1 + 1u; }
        }
    }
    syn = syn + f64(s1) / f64(3);

    var s2: u32 = 0u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        if bases[i] != b2 {
            let aa = translate(b0, b1, bases[i]);
            if aa != STOP_AA && aa == orig_aa { s2 = s2 + 1u; }
        }
    }
    syn = syn + f64(s2) / f64(3);

    return syn;
}

// Returns 1.0 if a single-base mutation is synonymous, 0.0 if nonsynonymous.
fn step_syn(c0: u32, c1: u32, c2: u32, pos: u32, new_base: u32) -> f64 {
    var m0 = c0; var m1 = c1; var m2 = c2;
    if pos == 0u { m0 = new_base; }
    if pos == 1u { m1 = new_base; }
    if pos == 2u { m2 = new_base; }
    let old_aa = translate(c0, c1, c2);
    let new_aa = translate(m0, m1, m2);
    if old_aa == new_aa && old_aa != STOP_AA { return f64(1); }
    return f64(0);
}

// Count (syn_diffs, nonsyn_diffs) for a codon pair with pathway averaging.
fn count_diffs(a0: u32, a1: u32, a2: u32, b0: u32, b1: u32, b2: u32) -> vec2<f64> {
    var d0: u32 = 0u; var d1: u32 = 0u; var d2: u32 = 0u;
    if a0 != b0 { d0 = 1u; }
    if a1 != b1 { d1 = 1u; }
    if a2 != b2 { d2 = 1u; }
    let n_diffs = d0 + d1 + d2;

    if n_diffs == 0u { return vec2<f64>(f64(0), f64(0)); }

    if n_diffs == 1u {
        var syn_d: f64;
        if d0 == 1u { syn_d = step_syn(a0, a1, a2, 0u, b0); }
        else if d1 == 1u { syn_d = step_syn(a0, a1, a2, 1u, b1); }
        else { syn_d = step_syn(a0, a1, a2, 2u, b2); }
        return vec2<f64>(syn_d, f64(1) - syn_d);
    }

    var total_syn: f64 = f64(0);
    var total_non: f64 = f64(0);
    var n_pathways: f64 = f64(0);

    var dp = array<u32, 3>(0u, 0u, 0u);
    var tb = array<u32, 3>(b0, b1, b2);
    var n_dp: u32 = 0u;
    if d0 == 1u { dp[n_dp] = 0u; n_dp = n_dp + 1u; }
    if d1 == 1u { dp[n_dp] = 1u; n_dp = n_dp + 1u; }
    if d2 == 1u { dp[n_dp] = 2u; n_dp = n_dp + 1u; }

    if n_diffs == 2u {
        for (var first: u32 = 0u; first < 2u; first = first + 1u) {
            var c0 = a0; var c1 = a1; var c2 = a2;
            var order0: u32; var order1: u32;
            if first == 0u { order0 = dp[0]; order1 = dp[1]; }
            else { order0 = dp[1]; order1 = dp[0]; }

            let s1 = step_syn(c0, c1, c2, order0, tb[order0]);
            if order0 == 0u { c0 = b0; } else if order0 == 1u { c1 = b1; } else { c2 = b2; }
            total_syn = total_syn + s1;
            total_non = total_non + (f64(1) - s1);

            let s2 = step_syn(c0, c1, c2, order1, tb[order1]);
            total_syn = total_syn + s2;
            total_non = total_non + (f64(1) - s2);

            n_pathways = n_pathways + f64(1);
        }
    } else {
        var perms = array<vec3<u32>, 6>(
            vec3<u32>(0u, 1u, 2u), vec3<u32>(0u, 2u, 1u),
            vec3<u32>(1u, 0u, 2u), vec3<u32>(1u, 2u, 0u),
            vec3<u32>(2u, 0u, 1u), vec3<u32>(2u, 1u, 0u)
        );
        for (var p: u32 = 0u; p < 6u; p = p + 1u) {
            var c0 = a0; var c1 = a1; var c2 = a2;
            for (var step: u32 = 0u; step < 3u; step = step + 1u) {
                let pos = dp[perms[p][step]];
                let s = step_syn(c0, c1, c2, pos, tb[pos]);
                if pos == 0u { c0 = b0; } else if pos == 1u { c1 = b1; } else { c2 = b2; }
                total_syn = total_syn + s;
                total_non = total_non + (f64(1) - s);
            }
            n_pathways = n_pathways + f64(1);
        }
    }

    return vec2<f64>(total_syn / n_pathways, total_non / n_pathways);
}

// Jukes-Cantor correction: d = -(3/4) * ln(1 - 4p/3)
fn jukes_cantor(p: f64) -> f64 {
    if p <= f64(0) { return f64(0); }
    let arg = f64(1) - f64(4) * p / f64(3);
    if arg <= f64(0) { return f64(100); }
    return f64(-1) * f64(3) / f64(4) * log(arg);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair_idx = gid.x;
    if pair_idx >= params.n_pairs { return; }

    let base = pair_idx * params.n_codons * 3u;
    var total_syn_sites: f64 = f64(0);
    var total_non_sites: f64 = f64(0);
    var total_syn_diffs: f64 = f64(0);
    var total_non_diffs: f64 = f64(0);

    for (var cod: u32 = 0u; cod < params.n_codons; cod = cod + 1u) {
        let off = base + cod * 3u;
        let a0 = seq_a[off]; let a1 = seq_a[off + 1u]; let a2 = seq_a[off + 2u];
        let b0 = seq_b[off]; let b1 = seq_b[off + 1u]; let b2 = seq_b[off + 2u];

        if a0 >= 4u || a1 >= 4u || a2 >= 4u { continue; }
        if b0 >= 4u || b1 >= 4u || b2 >= 4u { continue; }

        let s_a = codon_syn_sites(a0, a1, a2);
        let s_b = codon_syn_sites(b0, b1, b2);
        total_syn_sites = total_syn_sites + (s_a + s_b) / f64(2);
        total_non_sites = total_non_sites + ((f64(3) - s_a) + (f64(3) - s_b)) / f64(2);

        let diffs = count_diffs(a0, a1, a2, b0, b1, b2);
        total_syn_diffs = total_syn_diffs + diffs.x;
        total_non_diffs = total_non_diffs + diffs.y;
    }

    var p_s: f64 = f64(0);
    var p_n: f64 = f64(0);
    if total_syn_sites > f64(0) { p_s = total_syn_diffs / total_syn_sites; }
    if total_non_sites > f64(0) { p_n = total_non_diffs / total_non_sites; }

    let ds = jukes_cantor(p_s);
    let dn = jukes_cantor(p_n);

    ds_out[pair_idx] = ds;
    dn_out[pair_idx] = dn;

    if ds > f64(0) {
        omega_out[pair_idx] = dn / ds;
    } else {
        omega_out[pair_idx] = f64(-1);
    }
}
