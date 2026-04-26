// SPDX-License-Identifier: AGPL-3.0-or-later
//! Spectral analysis handlers for JSON-RPC IPC.
//!
//! FFT, power spectrum, and STFT — inline-data CPU paths using
//! Cooley-Tukey radix-2 DIT.

use super::super::jsonrpc::{INVALID_PARAMS, JsonRpcResponse};
use super::params::extract_f64_array;
use serde_json::Value;

/// `spectral.fft` — 1D complex FFT (Cooley-Tukey radix-2, zero-padded).
///
/// Input: real signal as `data` array. Output: complex spectrum as
/// interleaved `[re0, im0, re1, im1, ...]` plus separate `real`/`imag` arrays.
pub(super) fn spectral_fft(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(data) = extract_f64_array(params, "data") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: data (array)");
    };
    if data.is_empty() {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "data must be non-empty");
    }
    let n = data.len().next_power_of_two();
    let mut re: Vec<f64> = data;
    re.resize(n, 0.0);
    let mut im = vec![0.0; n];
    fft_in_place(&mut re, &mut im);
    JsonRpcResponse::success(
        id,
        serde_json::json!({ "result": re, "real": re, "imag": im, "n": n }),
    )
}

/// `spectral.power_spectrum` — power spectral density |X(k)|²/N.
pub(super) fn spectral_power_spectrum(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(data) = extract_f64_array(params, "data") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: data (array)");
    };
    if data.is_empty() {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "data must be non-empty");
    }
    let n = data.len().next_power_of_two();
    let mut re: Vec<f64> = data;
    re.resize(n, 0.0);
    let mut im = vec![0.0; n];
    fft_in_place(&mut re, &mut im);
    let inv_n = 1.0 / n as f64;
    let psd: Vec<f64> = re
        .iter()
        .zip(&im)
        .map(|(&r, &i)| r.mul_add(r, i * i) * inv_n)
        .collect();
    JsonRpcResponse::success(id, serde_json::json!({ "result": psd, "n": n }))
}

/// `spectral.stft` — inline CPU short-time Fourier transform.
///
/// Params: `data` (signal array), `n_fft` (window size, default 256),
/// `hop_length` (default n_fft/4). Returns 2D magnitude spectrogram.
#[expect(
    clippy::cast_possible_truncation,
    reason = "n_fft and hop_length are user-provided small counts"
)]
pub(super) fn spectral_stft(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(data) = extract_f64_array(params, "data") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: data (array)");
    };
    if data.is_empty() {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "data must be non-empty");
    }
    let n_fft = params.get("n_fft").and_then(|v| v.as_u64()).unwrap_or(256) as usize;
    if n_fft == 0 || !n_fft.is_power_of_two() {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "n_fft must be a positive power of 2");
    }
    let hop_length = params
        .get("hop_length")
        .and_then(|v| v.as_u64())
        .map_or(n_fft / 4, |v| v as usize);
    if hop_length == 0 {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "hop_length must be > 0");
    }
    if data.len() < n_fft {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            format!("data length ({}) must be >= n_fft ({n_fft})", data.len()),
        );
    }

    let n_frames = (data.len() - n_fft) / hop_length + 1;
    let freq_bins = n_fft / 2 + 1;
    let mut magnitude: Vec<Vec<f64>> = Vec::with_capacity(n_frames);

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop_length;
        let mut re: Vec<f64> = data[start..start + n_fft].to_vec();
        // Hann window
        for (i, sample) in re.iter_mut().enumerate() {
            let w = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / n_fft as f64).cos());
            *sample *= w;
        }
        let mut im = vec![0.0; n_fft];
        fft_in_place(&mut re, &mut im);
        let frame_mag: Vec<f64> = re[..freq_bins]
            .iter()
            .zip(&im[..freq_bins])
            .map(|(&r, &i)| r.hypot(i))
            .collect();
        magnitude.push(frame_mag);
    }

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "magnitude": magnitude,
            "n_fft": n_fft,
            "hop_length": hop_length,
            "n_frames": n_frames,
            "freq_bins": freq_bins,
        }),
    )
}

/// Cooley-Tukey radix-2 DIT FFT (in-place, power-of-2 length).
fn fft_in_place(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    debug_assert!(n.is_power_of_two());
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = -2.0 * std::f64::consts::PI / len as f64;
        let (wn_im, wn_re) = angle.sin_cos();
        let mut i = 0;
        while i < n {
            let mut wr = 1.0;
            let mut wi = 0.0;
            for k in 0..half {
                let tr = wr * re[i + k + half] - wi * im[i + k + half];
                let ti = wr.mul_add(im[i + k + half], wi * re[i + k + half]);
                re[i + k + half] = re[i + k] - tr;
                im[i + k + half] = im[i + k] - ti;
                re[i + k] += tr;
                im[i + k] += ti;
                let new_wr = wr.mul_add(wn_re, -(wi * wn_im));
                wi = wr.mul_add(wn_im, wi * wn_re);
                wr = new_wr;
            }
            i += len;
        }
        len <<= 1;
    }
}
