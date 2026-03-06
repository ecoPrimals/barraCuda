// SPDX-License-Identifier: AGPL-3.0-or-later
//
// fft_radix2_f64.wgsl — Radix-2 FFT stage for real-valued input (f64)
//
// Performs one butterfly stage of a radix-2 Cooley-Tukey FFT.
// Called log2(N) times with increasing stride.  The Rust driver
// orchestrates the multi-pass dispatch.
//
// For real-valued FFT: input is packed as complex with zero imaginary part,
// output is Hermitian-symmetric (only N/2+1 unique complex values).
//
// Complex layout: [re_0, im_0, re_1, im_1, ...] (interleaved f64 pairs)
//
// Dispatch: (N/2, 1, 1) — one thread per butterfly pair
//
// Provenance: groundSpring P1 request for real-valued FFT for spectral methods.

enable f64;

struct FftParams {
    n: u32,
    stage: u32,
    direction: i32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> params: FftParams;
@group(0) @binding(1) var<storage, read_write> data: array<f64>;

const PI: f64 = 3.14159265358979323846;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let half_n = params.n / 2u;
    if (idx >= half_n) { return; }

    let stage = params.stage;
    let block_size = 1u << (stage + 1u);
    let half_block = 1u << stage;

    let block_idx = idx / half_block;
    let within = idx % half_block;

    let top = block_idx * block_size + within;
    let bot = top + half_block;

    // Twiddle factor: W_N^k = exp(-2πi·k/N) (forward) or exp(+2πi·k/N) (inverse)
    let angle = f64(params.direction) * f64(2.0) * PI * f64(within) / f64(block_size);
    let tw_re = cos(angle);
    let tw_im = sin(angle);

    // Load butterfly pair (complex values)
    let top_re = data[top * 2u];
    let top_im = data[top * 2u + 1u];
    let bot_re = data[bot * 2u];
    let bot_im = data[bot * 2u + 1u];

    // Multiply bottom by twiddle
    let tw_bot_re = bot_re * tw_re - bot_im * tw_im;
    let tw_bot_im = bot_re * tw_im + bot_im * tw_re;

    // Butterfly
    data[top * 2u] = top_re + tw_bot_re;
    data[top * 2u + 1u] = top_im + tw_bot_im;
    data[bot * 2u] = top_re - tw_bot_re;
    data[bot * 2u + 1u] = top_im - tw_bot_im;
}
