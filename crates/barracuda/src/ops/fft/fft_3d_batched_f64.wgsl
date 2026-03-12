// SPDX-License-Identifier: AGPL-3.0-only
// Batched 3D FFT — processes ALL pencils along one axis in a single dispatch.
//
// Instead of 192 individual 1D FFT dispatches for an 8×8×8 mesh,
// this shader processes all pencils for one axis simultaneously:
//   Z-axis: 64 pencils × 4 elements each = 256 invocations per stage
//   Total: 3 axes × (1 bit-reverse + log2(N) butterfly stages) ≈ 12 dispatches
//
// Uses strided addressing to access pencil elements within row-major 3D layout
// without any CPU-side data rearrangement.
//
// CRITICAL: math_f64.wgsl is prepended by compile_shader_f64().

// ============================================================================
// COMPLEX f64 ARITHMETIC (same as fft_1d_f64.wgsl)
// ============================================================================

struct Complex64 {
    re: f64,
    im: f64,
}

fn complex64_add(a: Complex64, b: Complex64) -> Complex64 {
    return Complex64(a.re + b.re, a.im + b.im);
}

fn complex64_sub(a: Complex64, b: Complex64) -> Complex64 {
    return Complex64(a.re - b.re, a.im - b.im);
}

fn complex64_mul(a: Complex64, b: Complex64) -> Complex64 {
    return Complex64(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

// ============================================================================
// BATCHED FFT BINDINGS
// ============================================================================

@group(0) @binding(0) var<storage, read>       data_in:    array<f64>;
@group(0) @binding(1) var<storage, read_write> data_out:   array<f64>;
@group(0) @binding(2) var<storage, read>       twiddle_re: array<f64>;
@group(0) @binding(3) var<storage, read>       twiddle_im: array<f64>;
@group(0) @binding(4) var<uniform>             params:     BatchedFftParams;

struct BatchedFftParams {
    degree:         u32,
    stage:          u32,
    inverse:        u32,
    element_stride: u32,
    dim1_stride:    u32,
    dim2_stride:    u32,
    dim1_count:     u32,
    dim2_count:     u32,
}

// ============================================================================
// STRIDED ADDRESSING
// ============================================================================

fn pencil_base(pencil_flat: u32) -> u32 {
    let i = pencil_flat / params.dim2_count;
    let j = pencil_flat % params.dim2_count;
    return i * params.dim1_stride + j * params.dim2_stride;
}

fn f64_addr(pencil: u32, element: u32) -> u32 {
    return (pencil_base(pencil) + element * params.element_stride) * 2u;
}

fn load_in(pencil: u32, element: u32) -> Complex64 {
    let a = f64_addr(pencil, element);
    return Complex64(data_in[a], data_in[a + 1u]);
}

fn store_out(pencil: u32, element: u32, v: Complex64) {
    let a = f64_addr(pencil, element);
    data_out[a] = v.re;
    data_out[a + 1u] = v.im;
}

// ============================================================================
// BIT-REVERSE HELPERS
// ============================================================================

fn bit_reverse_index(index: u32, log_n: u32) -> u32 {
    var reversed = 0u;
    var idx = index;
    for (var i = 0u; i < log_n; i = i + 1u) {
        reversed = (reversed << 1u) | (idx & 1u);
        idx = idx >> 1u;
    }
    return reversed;
}

// ============================================================================
// KERNELS
// ============================================================================

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total_pencils = params.dim1_count * params.dim2_count;
    let n_butterflies = params.degree / 2u;
    let flat = gid.x;
    let pencil = flat / n_butterflies;
    let local = flat % n_butterflies;

    if pencil >= total_pencils {
        return;
    }

    let stage = params.stage;
    let stride = 1u << stage;
    let block_size = stride * 2u;
    let block_idx = local / stride;
    let local_idx = local % stride;

    let idx_a = block_idx * block_size + local_idx;
    let idx_b = idx_a + stride;

    let a = load_in(pencil, idx_a);
    let b = load_in(pencil, idx_b);

    let tw_stride = params.degree / (2u * stride);
    let tw_idx = local_idx * tw_stride;
    var tw = Complex64(twiddle_re[tw_idx], twiddle_im[tw_idx]);
    if params.inverse == 1u {
        tw.im = -tw.im;
    }

    let tb = complex64_mul(tw, b);
    store_out(pencil, idx_a, complex64_add(a, tb));
    store_out(pencil, idx_b, complex64_sub(a, tb));
}

@compute @workgroup_size(256)
fn bit_reverse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total_pencils = params.dim1_count * params.dim2_count;
    let flat = gid.x;
    let pencil = flat / params.degree;
    let elem = flat % params.degree;

    if pencil >= total_pencils {
        return;
    }

    let log_n = u32(log2(f32(params.degree)));
    let rev = bit_reverse_index(elem, log_n);

    if elem < rev {
        let a = load_in(pencil, elem);
        let b = load_in(pencil, rev);
        store_out(pencil, elem, b);
        store_out(pencil, rev, a);
    } else if elem == rev {
        store_out(pencil, elem, load_in(pencil, elem));
    }
}
