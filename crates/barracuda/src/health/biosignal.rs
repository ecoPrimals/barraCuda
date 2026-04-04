// SPDX-License-Identifier: AGPL-3.0-or-later

//! Biosignal processing — EDA, stress detection, beat classification.
//!
//! Absorbed from healthSpring V19 (Exp081–082, Exp084).

// ── Signal Processing Primitives ─────────────────────────────────────────────

/// Efficient O(n) rolling average using a running sum accumulator.
///
/// For each output sample, computes the mean of the surrounding `window_size`
/// input samples (centered). 21x faster than naive convolution for large windows
/// (healthSpring P1 optimization request).
#[must_use]
pub fn rolling_average(signal: &[f64], window_size: usize) -> Vec<f64> {
    if signal.is_empty() || window_size == 0 {
        return vec![];
    }
    let n = signal.len();
    let ws = window_size.min(n);
    let half = ws / 2;
    let mut result = Vec::with_capacity(n);
    let mut sum: f64 = signal[..ws.min(n)].iter().sum();
    let mut left = 0usize;
    let mut right = ws.min(n);

    for i in 0..n {
        let target_left = i.saturating_sub(half);
        let target_right = (i + half + 1).min(n);

        while left < target_left {
            sum -= signal[left];
            left += 1;
        }
        while right < target_right {
            sum += signal[right];
            right += 1;
        }
        while left > target_left && left > 0 {
            left -= 1;
            sum += signal[left];
        }

        let count = right - left;
        result.push(if count > 0 { sum / count as f64 } else { 0.0 });
    }
    result
}

/// 1D convolution of signal with kernel (valid output).
///
/// Returns a vector of length `signal.len() - kernel.len() + 1`.
#[must_use]
pub fn convolve_1d(signal: &[f64], kernel: &[f64]) -> Vec<f64> {
    if kernel.is_empty() || signal.len() < kernel.len() {
        return vec![];
    }

    #[cfg(feature = "cpu-shader")]
    {
        if let Ok(out) = convolve_1d_shader(signal, kernel) {
            return out;
        }
    }

    #[expect(deprecated, reason = "fallback retained until cpu-shader is default")]
    convolve_1d_cpu(signal, kernel)
}

#[deprecated(
    since = "0.4.0",
    note = "use `cpu-shader` feature for WGSL-backed convolve_1d"
)]
fn convolve_1d_cpu(signal: &[f64], kernel: &[f64]) -> Vec<f64> {
    let out_len = signal.len() - kernel.len() + 1;
    (0..out_len)
        .map(|i| {
            signal[i..i + kernel.len()]
                .iter()
                .zip(kernel.iter())
                .map(|(&s, &k)| s * k)
                .sum()
        })
        .collect()
}

#[cfg(feature = "cpu-shader")]
fn convolve_1d_shader(signal: &[f64], kernel: &[f64]) -> crate::error::Result<Vec<f64>> {
    use crate::unified_hardware::{CpuShaderDispatch, ShaderBinding, ShaderDispatch};

    let wgsl = include_str!("../shaders/health/convolve_1d_f64.wgsl");
    let dispatcher = CpuShaderDispatch::new();
    let out_len = signal.len() - kernel.len() + 1;

    let mut sig_buf: Vec<u8> = signal.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut kern_buf: Vec<u8> = kernel.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut out_buf = vec![0u8; out_len * 8];

    // Params { signal_len: u32, kernel_len: u32 } = 8 bytes
    let mut params_buf = vec![0u8; 8];
    params_buf[..4].copy_from_slice(&(signal.len() as u32).to_le_bytes());
    params_buf[4..8].copy_from_slice(&(kernel.len() as u32).to_le_bytes());

    let mut bindings = vec![
        ShaderBinding {
            group: 0,
            binding: 0,
            data: &mut sig_buf,
            read_only: true,
        },
        ShaderBinding {
            group: 0,
            binding: 1,
            data: &mut kern_buf,
            read_only: true,
        },
        ShaderBinding {
            group: 0,
            binding: 2,
            data: &mut out_buf,
            read_only: false,
        },
        ShaderBinding {
            group: 0,
            binding: 3,
            data: &mut params_buf,
            read_only: true,
        },
    ];

    let workgroups = (
        (out_len as u32).div_ceil(crate::device::capabilities::WORKGROUP_SIZE_1D),
        1,
        1,
    );
    dispatcher.dispatch_wgsl(wgsl, "main", &mut bindings, workgroups)?;

    Ok(out_buf
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect())
}

// ── EDA (Electrodermal Activity) ─────────────────────────────────────────────

/// Tonic skin conductance level via moving-window integration.
#[must_use]
pub fn eda_scl(signal: &[f64], window_size: usize) -> Vec<f64> {
    if signal.is_empty() || window_size == 0 {
        return vec![];
    }
    let ws = window_size.min(signal.len());
    let mut result = Vec::with_capacity(signal.len());
    let mut sum: f64 = signal[..ws].iter().sum();
    result.push(sum / ws as f64);

    for i in 1..signal.len() {
        if i + ws - 1 < signal.len() {
            sum += signal[i + ws - 1];
        }
        if i >= 1 && i - 1 + ws <= signal.len() {
            sum -= signal[i - 1];
        }
        let count = ws.min(signal.len() - i);
        result.push(sum / count as f64);
    }
    result
}

/// Phasic SCR component: signal minus tonic SCL.
#[must_use]
pub fn eda_phasic(signal: &[f64], window_size: usize) -> Vec<f64> {
    let scl = eda_scl(signal, window_size);
    signal
        .iter()
        .zip(scl.iter())
        .map(|(&s, &t)| (s - t).max(0.0))
        .collect()
}

/// Detect SCR peaks in the phasic component.
///
/// Returns indices where the phasic component exceeds `threshold`.
#[must_use]
pub fn eda_detect_scr(phasic: &[f64], threshold: f64) -> Vec<usize> {
    let mut peaks = Vec::new();
    for i in 1..phasic.len().saturating_sub(1) {
        if phasic[i] > threshold && phasic[i] > phasic[i - 1] && phasic[i] >= phasic[i + 1] {
            peaks.push(i);
        }
    }
    peaks
}

// ── Stress Detection ─────────────────────────────────────────────────────────

/// Stress assessment result.
#[derive(Debug, Clone)]
pub struct StressAssessment {
    /// Composite stress index in `[0, 1]`.
    pub stress_index: f64,
    /// SCR events per minute.
    pub scr_rate: f64,
    /// Categorical stress level.
    pub level: StressLevel,
}

/// Categorical stress level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StressLevel {
    /// Low stress (`< 0.25`).
    Low,
    /// Moderate stress (`0.25..0.50`).
    Moderate,
    /// High stress (`0.50..0.75`).
    High,
    /// Extreme stress (`>= 0.75`).
    Extreme,
}

/// SCR rate (events per minute).
#[must_use]
pub fn scr_rate(n_scr: usize, duration_s: f64) -> f64 {
    if duration_s <= 0.0 {
        return 0.0;
    }
    n_scr as f64 / (duration_s / 60.0)
}

/// Composite stress index from multiple physiological markers.
///
/// Combines SCR rate, mean SCL, and SCL variability into a single index.
#[must_use]
pub fn compute_stress_index(scr_rate_val: f64, mean_scl: f64, scl_variance: f64) -> f64 {
    0.4 * scr_rate_val.min(20.0) / 20.0
        + 0.3 * mean_scl.min(30.0) / 30.0
        + 0.3 * scl_variance.sqrt().min(5.0) / 5.0
}

/// Assess stress from EDA-derived features.
#[must_use]
pub fn assess_stress(scr_rate_val: f64, mean_scl: f64, scl_variance: f64) -> StressAssessment {
    let si = compute_stress_index(scr_rate_val, mean_scl, scl_variance);
    let level = match () {
        () if si < 0.25 => StressLevel::Low,
        () if si < 0.50 => StressLevel::Moderate,
        () if si < 0.75 => StressLevel::High,
        () => StressLevel::Extreme,
    };
    StressAssessment {
        stress_index: si,
        scr_rate: scr_rate_val,
        level,
    }
}

// ── Beat Classification ──────────────────────────────────────────────────────

/// Beat morphology class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeatClass {
    /// Normal sinus rhythm.
    Normal,
    /// Premature ventricular contraction.
    PVC,
    /// Premature atrial contraction.
    PAC,
    /// Unclassified beat.
    Unknown,
}

/// Beat template for template-matching classification.
#[derive(Debug, Clone)]
pub struct BeatTemplate {
    /// Expected morphology class for this template.
    pub class: BeatClass,
    /// Template waveform (same length as beat windows).
    pub waveform: Vec<f64>,
}

/// Beat classification result.
#[derive(Debug, Clone)]
pub struct BeatResult {
    /// Best-matching template class.
    pub class: BeatClass,
    /// Normalized cross-correlation with the best template.
    pub correlation: f64,
}

/// Normalized cross-correlation between two equal-length signals.
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
#[must_use]
pub fn normalized_correlation(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    let mean_a: f64 = a.iter().sum::<f64>() / n;
    let mean_b: f64 = b.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for (ai, bi) in a.iter().zip(b.iter()) {
        let da = ai - mean_a;
        let db = bi - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let denom = (var_a * var_b).sqrt();
    if denom < 1e-14 { 0.0 } else { cov / denom }
}

/// Classify a single beat window against a set of templates.
///
/// # Panics
/// Panics if `beat` and template waveforms have different lengths (skipped via `continue`).
#[must_use]
pub fn classify_beat(beat: &[f64], templates: &[BeatTemplate]) -> BeatResult {
    let mut best = BeatResult {
        class: BeatClass::Unknown,
        correlation: -1.0,
    };
    for tmpl in templates {
        if tmpl.waveform.len() != beat.len() {
            continue;
        }
        let corr = normalized_correlation(beat, &tmpl.waveform);
        if corr > best.correlation {
            best.correlation = corr;
            best.class = tmpl.class;
        }
    }
    best
}

/// Classify all beats in a batch.
#[must_use]
pub fn classify_all_beats(beats: &[Vec<f64>], templates: &[BeatTemplate]) -> Vec<BeatResult> {
    beats.iter().map(|b| classify_beat(b, templates)).collect()
}

/// Generate a synthetic normal sinus beat template.
#[must_use]
pub fn generate_normal_template(window_size: usize) -> Vec<f64> {
    let mut t = vec![0.0; window_size];
    let mid = window_size / 2;
    let sigma = window_size as f64 / 8.0;
    for (i, val) in t.iter_mut().enumerate() {
        let x = (i as f64 - mid as f64) / sigma;
        *val = (-0.5 * x * x).exp();
    }
    t
}

/// Generate a synthetic PVC (premature ventricular contraction) template.
#[must_use]
pub fn generate_pvc_template(window_size: usize) -> Vec<f64> {
    let mut t = vec![0.0; window_size];
    let mid = window_size / 2;
    let sigma = window_size as f64 / 6.0;
    for (i, val) in t.iter_mut().enumerate() {
        let x = (i as f64 - mid as f64) / sigma;
        *val = -1.2 * (-0.5 * x * x).exp();
    }
    t
}

/// Generate a synthetic PAC (premature atrial contraction) template.
#[must_use]
pub fn generate_pac_template(window_size: usize) -> Vec<f64> {
    let mut t = vec![0.0; window_size];
    let mid = window_size / 2;
    let sigma = window_size as f64 / 7.0;
    for (i, val) in t.iter_mut().enumerate() {
        let x = (i as f64 - mid as f64) / sigma;
        *val = 0.7 * (-0.5 * x * x).exp();
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eda_scl() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let scl = eda_scl(&signal, 3);
        assert_eq!(scl.len(), 5);
        assert!((scl[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_eda_phasic() {
        let signal = vec![1.0, 5.0, 1.0, 1.0, 1.0];
        let phasic = eda_phasic(&signal, 3);
        assert_eq!(phasic.len(), 5);
        assert!(phasic[1] > 0.0, "spike should produce phasic component");
    }

    #[test]
    fn test_scr_detection() {
        let phasic = vec![0.0, 0.1, 0.5, 0.3, 0.0, 0.0, 0.8, 0.2, 0.0];
        let peaks = eda_detect_scr(&phasic, 0.2);
        assert!(peaks.contains(&2));
        assert!(peaks.contains(&6));
    }

    #[test]
    fn test_stress_assessment() {
        let low = assess_stress(2.0, 5.0, 1.0);
        let high = assess_stress(15.0, 25.0, 20.0);
        assert!(low.stress_index < high.stress_index);
        assert_eq!(low.level, StressLevel::Low);
    }

    #[test]
    fn test_normalized_correlation_self() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let corr = normalized_correlation(&a, &a);
        assert!((corr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_beat_classification() {
        let ws = 50;
        let normal = generate_normal_template(ws);
        let pvc = generate_pvc_template(ws);
        let templates = vec![
            BeatTemplate {
                class: BeatClass::Normal,
                waveform: normal.clone(),
            },
            BeatTemplate {
                class: BeatClass::PVC,
                waveform: pvc.clone(),
            },
        ];

        let result = classify_beat(&normal, &templates);
        assert_eq!(result.class, BeatClass::Normal);
        assert!(result.correlation > 0.9);

        let result_pvc = classify_beat(&pvc, &templates);
        assert_eq!(result_pvc.class, BeatClass::PVC);
    }

    #[test]
    fn test_generate_templates() {
        let n = generate_normal_template(50);
        let pvc = generate_pvc_template(50);
        let pac = generate_pac_template(50);
        assert_eq!(n.len(), 50);
        assert!(n[25] > 0.0);
        assert!(pvc[25] < 0.0);
        assert!(pac[25] > 0.0 && pac[25] < n[25]);
    }

    #[test]
    fn test_classify_all_beats() {
        let ws = 30;
        let normal = generate_normal_template(ws);
        let pvc = generate_pvc_template(ws);
        let templates = vec![
            BeatTemplate {
                class: BeatClass::Normal,
                waveform: normal.clone(),
            },
            BeatTemplate {
                class: BeatClass::PVC,
                waveform: pvc.clone(),
            },
        ];
        let beats = vec![normal, pvc];
        let results = classify_all_beats(&beats, &templates);
        assert_eq!(results[0].class, BeatClass::Normal);
        assert_eq!(results[1].class, BeatClass::PVC);
    }

    #[test]
    fn test_rolling_average_constant() {
        let signal = vec![5.0; 10];
        let avg = rolling_average(&signal, 3);
        assert_eq!(avg.len(), 10);
        for &v in &avg {
            assert!((v - 5.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_rolling_average_ramp() {
        let signal: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let avg = rolling_average(&signal, 3);
        assert_eq!(avg.len(), 10);
        assert!((avg[1] - 1.0).abs() < 1e-10);
        assert!((avg[5] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_convolve_1d_identity() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kernel = vec![1.0];
        let result = convolve_1d(&signal, &kernel);
        assert_eq!(result.len(), 5);
        assert!((result[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_convolve_1d_averaging() {
        let signal = vec![0.0, 0.0, 1.0, 0.0, 0.0];
        let kernel = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        let result = convolve_1d(&signal, &kernel);
        assert_eq!(result.len(), 3);
        assert!((result[1] - 1.0 / 3.0).abs() < 1e-10);
    }
}
