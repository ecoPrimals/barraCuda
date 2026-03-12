// SPDX-License-Identifier: AGPL-3.0-only
// su3.wgsl — SU(3) 3×3 complex matrix algebra for lattice gauge fields
//
// Prepend complex_f64.wgsl before this file.
//
// Storage layout:
//   One SU(3) matrix occupies 18 consecutive f64 values in a storage buffer.
//   Row-major: element (i, j) at flat index i*3+j, stored as two f64 (re, im).
//   Buffer layout per matrix:  re00 im00 re01 im01 re02 im02
//                               re10 im10 re11 im11 re12 im12
//                               re20 im20 re21 im21 re22 im22
//
// In-register representation: array<vec2<f64>, 9>
//   Element (i,j) = arr[i*3+j]  where .x=real, .y=imag.
//
// Performance:
//   su3_mul has 27 complex FMA pairs.
//
// Naga/wgpu note:
//   WGSL function parameters of array type are value-typed (like `let`).
//   Naga requires that array parameters are copied into a local `var` before
//   any runtime (non-constant) indexing.  All functions below that index
//   with loop variables do this copy-to-var at entry.
//
// hotSpring absorption: lattice/su3.rs (v0.5.16, Feb 2026)
// CPU-validated against standard SU(3) algebra test vectors.

// ── Multiply: C = A × B ───────────────────────────────────────────────────────

fn su3_mul(a: array<vec2<f64>, 9>, b: array<vec2<f64>, 9>) -> array<vec2<f64>, 9> {
    // Copy params to var — Naga requires var for runtime array indexing.
    var av = a;
    var bv = b;
    var r: array<vec2<f64>, 9>;
    for (var i = 0u; i < 3u; i = i + 1u) {
        for (var j = 0u; j < 3u; j = j + 1u) {
            var acc = c64_zero();
            for (var k = 0u; k < 3u; k = k + 1u) {
                acc = c64_add(acc, c64_mul(av[i * 3u + k], bv[k * 3u + j]));
            }
            r[i * 3u + j] = acc;
        }
    }
    return r;
}

// ── Adjoint (conjugate transpose): B = A† ────────────────────────────────────

fn su3_adjoint(a: array<vec2<f64>, 9>) -> array<vec2<f64>, 9> {
    var av = a;
    var r: array<vec2<f64>, 9>;
    for (var i = 0u; i < 3u; i = i + 1u) {
        for (var j = 0u; j < 3u; j = j + 1u) {
            r[j * 3u + i] = c64_conj(av[i * 3u + j]);
        }
    }
    return r;
}

// ── Trace ─────────────────────────────────────────────────────────────────────
// Constant indices: no var copy needed.

fn su3_trace(a: array<vec2<f64>, 9>) -> vec2<f64> {
    return c64_add(c64_add(a[0], a[4]), a[8]);
}

fn su3_re_trace(a: array<vec2<f64>, 9>) -> f64 {
    return a[0].x + a[4].x + a[8].x;
}

// ── Add / scale ───────────────────────────────────────────────────────────────

fn su3_add(a: array<vec2<f64>, 9>, b: array<vec2<f64>, 9>) -> array<vec2<f64>, 9> {
    var av = a;
    var bv = b;
    var r: array<vec2<f64>, 9>;
    for (var i = 0u; i < 9u; i = i + 1u) { r[i] = c64_add(av[i], bv[i]); }
    return r;
}

fn su3_scale(a: array<vec2<f64>, 9>, s: f64) -> array<vec2<f64>, 9> {
    var av = a;
    var r: array<vec2<f64>, 9>;
    for (var i = 0u; i < 9u; i = i + 1u) { r[i] = c64_scale(av[i], s); }
    return r;
}

// ── Plaquette product: U_mu(x) * U_nu(x+mu) * U_mu†(x+nu) * U_nu†(x) ────────
// All four links supplied as pre-loaded matrices.

fn su3_plaquette(u_mu: array<vec2<f64>, 9>,
                 u_nu_fwd: array<vec2<f64>, 9>,
                 u_mu_fwd_nu: array<vec2<f64>, 9>,
                 u_nu: array<vec2<f64>, 9>) -> array<vec2<f64>, 9> {
    return su3_mul(
        su3_mul(u_mu, u_nu_fwd),
        su3_mul(su3_adjoint(u_mu_fwd_nu), su3_adjoint(u_nu)),
    );
}

// ── Identity ──────────────────────────────────────────────────────────────────

fn su3_identity() -> array<vec2<f64>, 9> {
    var r: array<vec2<f64>, 9>;
    for (var i = 0u; i < 9u; i = i + 1u) { r[i] = c64_zero(); }
    r[0] = c64_one();
    r[4] = c64_one();
    r[8] = c64_one();
    return r;
}
