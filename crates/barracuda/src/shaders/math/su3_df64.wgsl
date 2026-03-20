// SPDX-License-Identifier: AGPL-3.0-or-later
// su3_df64.wgsl — SU(3) 3×3 complex matrix algebra using DF64 (f32-pair) arithmetic
//
// Prepend: df64_core.wgsl before this file.
//
// Routes all SU(3) matrix multiplications through the FP32 core array via
// double-float (Df64) arithmetic, achieving ~10x throughput vs native f64
// on consumer GPUs (RTX 3090: 10,496 FP32 cores vs 164 FP64 units).
//
// Precision: ~48-bit mantissa (~14 decimal digits) — sufficient for
// gauge force staple sums and plaquette products where the final result
// is projected back to f64 for accumulation.
//
// Storage: buffers remain f64. Conversion happens at load/store boundaries.
// This preserves all existing buffer layouts and reduction infrastructure.
//
// hotSpring core-streaming discovery (Feb 2026). Wired by toadStool.

// ── DF64 complex number ──────────────────────────────────────────────────────
// Each element of an SU(3) matrix is a complex number (re, im).
// In DF64, each component is a Df64 (hi: f32, lo: f32).

struct Cdf64 {
    re: Df64,
    im: Df64,
}

fn cdf64_zero() -> Cdf64 {
    return Cdf64(df64_zero(), df64_zero());
}

fn cdf64_from_f64(re: f64, im: f64) -> Cdf64 {
    return Cdf64(df64_from_f64(re), df64_from_f64(im));
}

fn cdf64_to_f64(c: Cdf64) -> vec2<f64> {
    return vec2<f64>(df64_to_f64(c.re), df64_to_f64(c.im));
}

fn cdf64_add(a: Cdf64, b: Cdf64) -> Cdf64 {
    return Cdf64(df64_add(a.re, b.re), df64_add(a.im, b.im));
}

fn cdf64_sub(a: Cdf64, b: Cdf64) -> Cdf64 {
    return Cdf64(df64_sub(a.re, b.re), df64_sub(a.im, b.im));
}

// (a + bi)(c + di) = (ac - bd) + (ad + bc)i
fn cdf64_mul(a: Cdf64, b: Cdf64) -> Cdf64 {
    let re = df64_sub(df64_mul(a.re, b.re), df64_mul(a.im, b.im));
    let im = df64_add(df64_mul(a.re, b.im), df64_mul(a.im, b.re));
    return Cdf64(re, im);
}

fn cdf64_conj(a: Cdf64) -> Cdf64 {
    return Cdf64(a.re, df64_neg(a.im));
}

// ── DF64 SU(3) matrix operations ─────────────────────────────────────────────
// SU(3) matrix: array<Cdf64, 9> — 9 complex entries, row-major.
// Element (i,j) at index i*3+j.

fn su3_mul_df64(a: array<Cdf64, 9>, b: array<Cdf64, 9>) -> array<Cdf64, 9> {
    var av = a;
    var bv = b;
    var r: array<Cdf64, 9>;
    for (var i = 0u; i < 3u; i = i + 1u) {
        for (var j = 0u; j < 3u; j = j + 1u) {
            var acc = cdf64_zero();
            for (var k = 0u; k < 3u; k = k + 1u) {
                acc = cdf64_add(acc, cdf64_mul(av[i * 3u + k], bv[k * 3u + j]));
            }
            r[i * 3u + j] = acc;
        }
    }
    return r;
}

fn su3_adjoint_df64(a: array<Cdf64, 9>) -> array<Cdf64, 9> {
    var av = a;
    var r: array<Cdf64, 9>;
    for (var i = 0u; i < 3u; i = i + 1u) {
        for (var j = 0u; j < 3u; j = j + 1u) {
            r[j * 3u + i] = cdf64_conj(av[i * 3u + j]);
        }
    }
    return r;
}

fn su3_add_df64(a: array<Cdf64, 9>, b: array<Cdf64, 9>) -> array<Cdf64, 9> {
    var av = a;
    var bv = b;
    var r: array<Cdf64, 9>;
    for (var i = 0u; i < 9u; i = i + 1u) {
        r[i] = cdf64_add(av[i], bv[i]);
    }
    return r;
}

fn su3_re_trace_df64(a: array<Cdf64, 9>) -> Df64 {
    return df64_add(df64_add(a[0].re, a[4].re), a[8].re);
}

fn su3_identity_df64() -> array<Cdf64, 9> {
    var r: array<Cdf64, 9>;
    let one = Cdf64(df64_from_f32(1.0), df64_zero());
    let zero = cdf64_zero();
    for (var i = 0u; i < 9u; i = i + 1u) { r[i] = zero; }
    r[0] = one;
    r[4] = one;
    r[8] = one;
    return r;
}

fn su3_zero_df64() -> array<Cdf64, 9> {
    var r: array<Cdf64, 9>;
    let z = cdf64_zero();
    for (var i = 0u; i < 9u; i = i + 1u) { r[i] = z; }
    return r;
}

// U_p = U_mu(x) * U_nu(x+mu) * U_mu†(x+nu) * U_nu†(x)
fn su3_plaquette_df64(u_mu: array<Cdf64, 9>,
                      u_nu_fwd: array<Cdf64, 9>,
                      u_mu_fwd_nu: array<Cdf64, 9>,
                      u_nu: array<Cdf64, 9>) -> array<Cdf64, 9> {
    return su3_mul_df64(
        su3_mul_df64(u_mu, u_nu_fwd),
        su3_mul_df64(su3_adjoint_df64(u_mu_fwd_nu), su3_adjoint_df64(u_nu)),
    );
}

// ── Boundary conversion: DF64 SU(3) ↔ f64 SU(3) ────────────────────────────

fn su3_df64_to_f64(m: array<Cdf64, 9>) -> array<vec2<f64>, 9> {
    var mv = m;
    var r: array<vec2<f64>, 9>;
    for (var i = 0u; i < 9u; i = i + 1u) {
        r[i] = cdf64_to_f64(mv[i]);
    }
    return r;
}

fn su3_f64_to_df64(m: array<vec2<f64>, 9>) -> array<Cdf64, 9> {
    var mv = m;
    var r: array<Cdf64, 9>;
    for (var i = 0u; i < 9u; i = i + 1u) {
        r[i] = cdf64_from_f64(mv[i].x, mv[i].y);
    }
    return r;
}
