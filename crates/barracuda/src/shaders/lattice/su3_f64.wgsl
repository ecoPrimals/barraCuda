// SPDX-License-Identifier: AGPL-3.0-or-later
// SU(3) WGSL library: 3x3 complex matrix operations for lattice QCD
// Depends on complex_f64.wgsl (prepend before this file).

// SU(3) link variable: 3x3 complex matrix stored as 18 f64 values
// Layout: row-major [row0_col0.re, row0_col0.im, row0_col1.re, ...]

fn su3_load(data: ptr<storage, array<f64>>, offset: u32) -> array<Complex64, 9> {
    var m: array<Complex64, 9>;
    for (var i = 0u; i < 9u; i++) {
        let base = offset + i * 2u;
        m[i] = c64_new(data[base], data[base + 1u]);
    }
    return m;
}

fn su3_store(data: ptr<storage, array<f64>, read_write>, offset: u32, m: array<Complex64, 9>) {
    for (var i = 0u; i < 9u; i++) {
        let base = offset + i * 2u;
        data[base] = m[i].re;
        data[base + 1u] = m[i].im;
    }
}

fn su3_idx(row: u32, col: u32) -> u32 {
    return row * 3u + col;
}

fn su3_mul(a: array<Complex64, 9>, b: array<Complex64, 9>) -> array<Complex64, 9> {
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

fn su3_adjoint(a: array<Complex64, 9>) -> array<Complex64, 9> {
    var r: array<Complex64, 9>;
    for (var i = 0u; i < 3u; i++) {
        for (var j = 0u; j < 3u; j++) {
            r[su3_idx(i, j)] = c64_conj(a[su3_idx(j, i)]);
        }
    }
    return r;
}

fn su3_trace(a: array<Complex64, 9>) -> Complex64 {
    return c64_add(c64_add(a[0], a[4]), a[8]);
}

fn su3_re_trace(a: array<Complex64, 9>) -> f64 {
    return a[0].re + a[4].re + a[8].re;
}

fn su3_add(a: array<Complex64, 9>, b: array<Complex64, 9>) -> array<Complex64, 9> {
    var r: array<Complex64, 9>;
    for (var i = 0u; i < 9u; i++) {
        r[i] = c64_add(a[i], b[i]);
    }
    return r;
}

fn su3_scale(a: array<Complex64, 9>, s: f64) -> array<Complex64, 9> {
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
