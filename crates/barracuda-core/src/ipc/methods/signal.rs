// SPDX-License-Identifier: AGPL-3.0-or-later
//! Signal processing IPC handlers (`signal.*` namespace).
//!
//! Wires healthSpring-derived signal processing primitives to the IPC surface.
//! Includes peak detection, bandpass filtering, and derivative computation.

use super::super::jsonrpc::{INVALID_PARAMS, JsonRpcResponse};
use super::params::extract_f64_array;
use serde_json::Value;

/// `signal.detect_peaks` — find local maxima with minimum distance and height.
///
/// Accepts `signal` (f64 array), `distance` (min samples between peaks),
/// optional `min_height`, optional `min_prominence`.
pub(super) fn signal_detect_peaks(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(signal) = extract_f64_array(params, "signal") else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: signal (array)",
        );
    };
    let distance = params.get("distance").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
    let min_height = params.get("min_height").and_then(|v| v.as_f64());
    let min_prominence = params.get("min_prominence").and_then(|v| v.as_f64());
    let peaks = barracuda::ops::peak_detect_f64::find_peaks_cpu(
        &signal,
        distance,
        min_height,
        min_prominence,
    );
    let indices: Vec<usize> = peaks.iter().map(|p| p.index).collect();
    let heights: Vec<f64> = peaks.iter().map(|p| p.height).collect();
    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "indices": indices,
            "heights": heights,
            "count": peaks.len(),
        }),
    )
}

/// `signal.bandpass` — frequency-domain bandpass filter (zeros outside [low, high] Hz).
///
/// Accepts `signal`, `sample_rate` (Hz), `low_hz`, `high_hz`.
pub(super) fn signal_bandpass(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(signal) = extract_f64_array(params, "signal") else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: signal (array)",
        );
    };
    let Some(sample_rate) = params.get("sample_rate").and_then(|v| v.as_f64()) else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: sample_rate (f64)",
        );
    };
    let Some(low_hz) = params.get("low_hz").and_then(|v| v.as_f64()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: low_hz (f64)");
    };
    let Some(high_hz) = params.get("high_hz").and_then(|v| v.as_f64()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: high_hz (f64)");
    };
    if sample_rate <= 0.0 || low_hz < 0.0 || high_hz <= low_hz {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Invalid filter params: need sample_rate > 0, 0 ≤ low_hz < high_hz",
        );
    }
    let filtered = bandpass_filter_cpu(&signal, sample_rate, low_hz, high_hz);
    JsonRpcResponse::success(id, serde_json::json!({ "result": filtered }))
}

/// CPU frequency-domain bandpass: FFT → zero outside band → IFFT.
fn bandpass_filter_cpu(signal: &[f64], fs: f64, low_hz: f64, high_hz: f64) -> Vec<f64> {
    let n = signal.len();
    if n == 0 {
        return vec![];
    }
    let n_freq = n / 2 + 1;
    let mut re = vec![0.0; n_freq];
    let mut im = vec![0.0; n_freq];

    for k in 0..n_freq {
        let mut sum_re = 0.0;
        let mut sum_im = 0.0;
        let angle_base = -2.0 * std::f64::consts::PI * (k as f64) / (n as f64);
        for (j, &s) in signal.iter().enumerate() {
            let angle = angle_base * (j as f64);
            sum_re = s.mul_add(angle.cos(), sum_re);
            sum_im = s.mul_add(angle.sin(), sum_im);
        }
        let freq = (k as f64) * fs / (n as f64);
        if freq >= low_hz && freq <= high_hz {
            re[k] = sum_re;
            im[k] = sum_im;
        }
    }

    let mut output = vec![0.0; n];
    let norm = 1.0 / (n as f64);
    for (j, out) in output.iter_mut().enumerate() {
        let mut sum = 0.0;
        for k in 0..n_freq {
            let angle = 2.0 * std::f64::consts::PI * (k as f64) * (j as f64) / (n as f64);
            let contrib = re[k].mul_add(angle.cos(), -(im[k] * angle.sin()));
            sum += contrib;
            if k > 0 && k < n_freq - 1 {
                sum += contrib;
            }
        }
        *out = sum * norm;
    }
    output
}

/// `signal.derivative` — 5-point derivative filter (Pan-Tompkins).
///
/// d[i] = (-x[i-2] - 2*x[i-1] + 2*x[i+1] + x[i+2]) / 8
pub(super) fn signal_derivative(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(signal) = extract_f64_array(params, "signal") else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: signal (array)",
        );
    };
    let n = signal.len();
    let mut d = vec![0.0; n];
    for i in 2..n.saturating_sub(2) {
        d[i] = (2.0f64.mul_add(
            signal[i + 1],
            2.0f64.mul_add(-signal[i - 1], -signal[i - 2]),
        ) + signal[i + 2])
            / 8.0;
    }
    JsonRpcResponse::success(id, serde_json::json!({ "result": d }))
}
