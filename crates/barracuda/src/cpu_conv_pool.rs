// SPDX-License-Identifier: AGPL-3.0-or-later
//! CPU implementations of `Conv2D`, `MaxPool2D`, and `AvgPool2D`.
//!
//! Extracted from `cpu_executor.rs` for file-size compliance and logical separation.
//! These are pure-Rust, im2col-free implementations suitable for CPU fallback.

use crate::error::Result;

/// Input tensor shape `[N, C, H, W]`.
#[derive(Debug, Clone, Copy)]
pub struct TensorShape {
    /// Batch size.
    pub n: usize,
    /// Channel count.
    pub c: usize,
    /// Height.
    pub h: usize,
    /// Width.
    pub w: usize,
}

/// Convolution kernel and stride/padding/dilation parameters.
#[derive(Debug, Clone, Copy)]
pub struct Conv2dConfig {
    /// Output channel count.
    pub c_out: usize,
    /// Kernel height.
    pub k_h: usize,
    /// Kernel width.
    pub k_w: usize,
    /// Stride `[h, w]` (default `[1, 1]`).
    pub stride: [usize; 2],
    /// Padding `[h, w]` (default `[0, 0]`).
    pub padding: [usize; 2],
    /// Dilation `[h, w]` (default `[1, 1]`).
    pub dilation: [usize; 2],
}

impl Conv2dConfig {
    /// Create with output channels and kernel size; stride=1, padding=0, dilation=1.
    #[must_use]
    pub fn new(c_out: usize, k_h: usize, k_w: usize) -> Self {
        Self {
            c_out,
            k_h,
            k_w,
            stride: [1, 1],
            padding: [0, 0],
            dilation: [1, 1],
        }
    }

    /// Set stride `[h, w]`.
    #[must_use]
    pub fn stride(mut self, h: usize, w: usize) -> Self {
        self.stride = [h, w];
        self
    }

    /// Set padding `[h, w]`.
    #[must_use]
    pub fn padding(mut self, h: usize, w: usize) -> Self {
        self.padding = [h, w];
        self
    }

    /// Set dilation `[h, w]`.
    #[must_use]
    pub fn dilation(mut self, h: usize, w: usize) -> Self {
        self.dilation = [h, w];
        self
    }
}

/// Pooling kernel and stride/padding parameters.
#[derive(Debug, Clone, Copy)]
pub struct Pool2dConfig {
    /// Kernel height.
    pub k_h: usize,
    /// Kernel width.
    pub k_w: usize,
    /// Stride `[h, w]` (defaults to kernel size).
    pub stride: [usize; 2],
    /// Padding `[h, w]` (default `[0, 0]`).
    pub padding: [usize; 2],
}

impl Pool2dConfig {
    /// Create with kernel size; stride defaults to kernel size, padding to zero.
    #[must_use]
    pub fn new(k_h: usize, k_w: usize) -> Self {
        Self {
            k_h,
            k_w,
            stride: [k_h, k_w],
            padding: [0, 0],
        }
    }

    /// Set stride `[h, w]`.
    #[must_use]
    pub fn stride(mut self, h: usize, w: usize) -> Self {
        self.stride = [h, w];
        self
    }

    /// Set padding `[h, w]`.
    #[must_use]
    pub fn padding(mut self, h: usize, w: usize) -> Self {
        self.padding = [h, w];
        self
    }
}

/// 2D convolution (im2col-free direct convolution).
/// Input `[N, C_in, H, W]`, kernel `[C_out, C_in, kH, kW]`.
///
/// # Errors
///
/// Returns [`Err`] if the operation fails (e.g., dimension overflow or invalid output size).
pub fn conv2d(
    input: &[f32],
    kernel: &[f32],
    shape: TensorShape,
    cfg: Conv2dConfig,
) -> Result<Vec<f32>> {
    let TensorShape { n, c: c_in, h, w } = shape;
    let Conv2dConfig {
        c_out,
        k_h,
        k_w,
        stride: [stride_h, stride_w],
        padding: [pad_h, pad_w],
        dilation: [dil_h, dil_w],
    } = cfg;

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
///
/// # Errors
///
/// Returns [`Err`] if the operation fails (e.g., dimension overflow or invalid output size).
pub fn max_pool2d(input: &[f32], shape: TensorShape, cfg: Pool2dConfig) -> Result<Vec<f32>> {
    let TensorShape { n, c, h, w } = shape;
    let Pool2dConfig {
        k_h,
        k_w,
        stride: [stride_h, stride_w],
        padding: [pad_h, pad_w],
    } = cfg;

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
///
/// # Errors
///
/// Returns [`Err`] if the operation fails (e.g., dimension overflow or invalid output size).
pub fn avg_pool2d(input: &[f32], shape: TensorShape, cfg: Pool2dConfig) -> Result<Vec<f32>> {
    let TensorShape { n, c, h, w } = shape;
    let Pool2dConfig {
        k_h,
        k_w,
        stride: [stride_h, stride_w],
        padding: [pad_h, pad_w],
    } = cfg;

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
