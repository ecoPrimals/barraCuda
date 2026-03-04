// SPDX-License-Identifier: AGPL-3.0-or-later
//! CPU implementations of Conv2D, MaxPool2D, and AvgPool2D.
//!
//! Extracted from `cpu_executor.rs` for file-size compliance and logical separation.
//! These are pure-Rust, im2col-free implementations suitable for CPU fallback.

use crate::error::Result;

/// 2D convolution (im2col-free direct convolution).
/// Input `[N, C_in, H, W]`, kernel `[C_out, C_in, kH, kW]`.
#[expect(clippy::too_many_arguments, reason = "API")]
pub fn conv2d(
    input: &[f32],
    kernel: &[f32],
    n: usize,
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    k_h: usize,
    k_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    dil_h: usize,
    dil_w: usize,
) -> Result<Vec<f32>> {
    let eff_k_h = (k_h - 1) * dil_h + 1;
    let eff_k_w = (k_w - 1) * dil_w + 1;
    let h_out = (h + 2 * pad_h - eff_k_h) / stride_h + 1;
    let w_out = (w + 2 * pad_w - eff_k_w) / stride_w + 1;

    let out_size = n * c_out * h_out * w_out;
    let mut output = vec![0.0f32; out_size];

    for b in 0..n {
        for oc in 0..c_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut sum = 0.0f32;

                    for ic in 0..c_in {
                        for kh in 0..k_h {
                            for kw in 0..k_w {
                                let ih = (oh * stride_h + kh * dil_h) as isize - pad_h as isize;
                                let iw = (ow * stride_w + kw * dil_w) as isize - pad_w as isize;

                                if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                    let ih = ih as usize;
                                    let iw = iw as usize;

                                    let input_idx = b * (c_in * h * w) + ic * (h * w) + ih * w + iw;
                                    let kernel_idx =
                                        oc * (c_in * k_h * k_w) + ic * (k_h * k_w) + kh * k_w + kw;

                                    sum += input[input_idx] * kernel[kernel_idx];
                                }
                            }
                        }
                    }

                    let out_idx =
                        b * (c_out * h_out * w_out) + oc * (h_out * w_out) + oh * w_out + ow;
                    output[out_idx] = sum;
                }
            }
        }
    }

    Ok(output)
}

/// 2D max pooling. Input `[N, C, H, W]`.
#[expect(clippy::too_many_arguments, reason = "API")]
pub fn max_pool2d(
    input: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    k_h: usize,
    k_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> Result<Vec<f32>> {
    let h_out = (h + 2 * pad_h - k_h) / stride_h + 1;
    let w_out = (w + 2 * pad_w - k_w) / stride_w + 1;

    let out_size = n * c * h_out * w_out;
    let mut output = vec![f32::NEG_INFINITY; out_size];

    for b in 0..n {
        for ch in 0..c {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut max_val = f32::NEG_INFINITY;

                    for ph in 0..k_h {
                        for pw in 0..k_w {
                            let ih = (oh * stride_h + ph) as isize - pad_h as isize;
                            let iw = (ow * stride_w + pw) as isize - pad_w as isize;

                            if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                let ih = ih as usize;
                                let iw = iw as usize;

                                let input_idx = b * (c * h * w) + ch * (h * w) + ih * w + iw;
                                max_val = max_val.max(input[input_idx]);
                            }
                        }
                    }

                    let out_idx = b * (c * h_out * w_out) + ch * (h_out * w_out) + oh * w_out + ow;
                    output[out_idx] = max_val;
                }
            }
        }
    }

    Ok(output)
}

/// 2D average pooling. Input `[N, C, H, W]`.
#[expect(clippy::too_many_arguments, reason = "API")]
pub fn avg_pool2d(
    input: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    k_h: usize,
    k_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> Result<Vec<f32>> {
    let h_out = (h + 2 * pad_h - k_h) / stride_h + 1;
    let w_out = (w + 2 * pad_w - k_w) / stride_w + 1;

    let out_size = n * c * h_out * w_out;
    let mut output = vec![0.0f32; out_size];

    for b in 0..n {
        for ch in 0..c {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut sum = 0.0f32;
                    let mut count = 0usize;

                    for ph in 0..k_h {
                        for pw in 0..k_w {
                            let ih = (oh * stride_h + ph) as isize - pad_h as isize;
                            let iw = (ow * stride_w + pw) as isize - pad_w as isize;

                            if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                let ih = ih as usize;
                                let iw = iw as usize;

                                let input_idx = b * (c * h * w) + ch * (h * w) + ih * w + iw;
                                sum += input[input_idx];
                                count += 1;
                            }
                        }
                    }

                    let out_idx = b * (c * h_out * w_out) + ch * (h_out * w_out) + oh * w_out + ow;
                    output[out_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
                }
            }
        }
    }

    Ok(output)
}
