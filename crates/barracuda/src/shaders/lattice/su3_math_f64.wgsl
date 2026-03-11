// SPDX-License-Identifier: AGPL-3.0-only
// SU(3) pure-math library (no buffer I/O — safe for shader composition).
// Depends on complex_f64.wgsl (prepend before this file).
//
// Note: array function parameters are copied to `var` locals before dynamic
// indexing — naga requires constant indices for value-type array params.

fn su3_idx(row: u32, col: u32) -> u32 {
    return row * 3u + col;
}

fn su3_mul(a_in: array<Complex64, 9>, b_in: array<Complex64, 9>) -> array<Complex64, 9> {
    var a = a_in;
    var b = b_in;
    var r: array<Complex64, 9>;
    for (var i = 0u; i < 3u; i++) {
        for (var j = 0u; j < 3u; j++) {
            var s = c64_zero();
            for (var k = 0u; k < 3u; k++) {
                s = c64_add(s, c64_mul(a[su3_idx(i, k)], b[su3_idx(k, j)]));
            }
            r[su3_idx(i, j)] = s;
        }
    }
    return r;
}

fn su3_adjoint(a_in: array<Complex64, 9>) -> array<Complex64, 9> {
    var a = a_in;
    var r: array<Complex64, 9>;
    for (var i = 0u; i < 3u; i++) {
        for (var j = 0u; j < 3u; j++) {
            r[su3_idx(i, j)] = c64_conj(a[su3_idx(j, i)]);
        }
    }
    return r;
}

fn su3_trace(a_in: array<Complex64, 9>) -> Complex64 {
    var a = a_in;
    return c64_add(c64_add(a[0], a[4]), a[8]);
}

fn su3_re_trace(a_in: array<Complex64, 9>) -> f64 {
    var a = a_in;
    return a[0].re + a[4].re + a[8].re;
}

fn su3_add(a_in: array<Complex64, 9>, b_in: array<Complex64, 9>) -> array<Complex64, 9> {
    var a = a_in;
    var b = b_in;
    var r: array<Complex64, 9>;
    for (var i = 0u; i < 9u; i++) {
        r[i] = c64_add(a[i], b[i]);
    }
    return r;
}

fn su3_scale(a_in: array<Complex64, 9>, s: f64) -> array<Complex64, 9> {
    var a = a_in;
    var r: array<Complex64, 9>;
    for (var i = 0u; i < 9u; i++) {
        r[i] = c64_scale(a[i], s);
    }
    return r;
}

fn su3_identity() -> array<Complex64, 9> {
    var m: array<Complex64, 9>;
    m[0] = c64_one();
    m[4] = c64_one();
    m[8] = c64_one();
    return m;
}
