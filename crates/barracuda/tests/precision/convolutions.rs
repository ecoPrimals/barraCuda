// SPDX-License-Identifier: AGPL-3.0-or-later
//! Precision Tests: Convolutions
//!
//! Validate FP32 precision for Conv2D, pooling operations
//! **Deep Debt**: Spatial operations maintain numerical accuracy

use barracuda::device::test_pool::get_test_device;
use barracuda::ops::*;

/// CPU reference for maxpool2d (naive but correct)
fn cpu_maxpool2d_reference(
    input: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> Vec<f32> {
    let out_h = (height - kernel_h) / stride_h + 1;
    let out_w = (width - kernel_w) / stride_w + 1;
    let mut output = Vec::new();

    for b in 0..batch {
        for c in 0..channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let h_start = oh * stride_h;
                    let w_start = ow * stride_w;
                    
                    let mut max_val = f32::NEG_INFINITY;
                    for kh in 0..kernel_h {
                        for kw in 0..kernel_w {
                            let h = h_start + kh;
                            let w = w_start + kw;
                            let idx = ((b * channels + c) * height + h) * width + w;
                            max_val = max_val.max(input[idx]);
                        }
                    }
                    output.push(max_val);
                }
            }
        }
    }

    output
}

#[tokio::test]
async fn test_maxpool2d_precision() {
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let batch = 2;
    let channels = 4;
    let height = 8;
    let width = 8;
    let kernel_h = 2;
    let kernel_w = 2;
    let stride_h = 2;
    let stride_w = 2;

    let input: Vec<f32> = (0..batch * channels * height * width)
        .map(|i| (i as f32) * 0.01)
        .collect();

    // GPU result
    let gpu_result = maxpool2d(
        &dev.device,
        &dev.queue,
        &input,
        batch,
        channels,
        height,
        width,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        0,
        0,
    )
    .await
    .expect("GPU maxpool2d failed");

    // CPU reference
    let cpu_result = cpu_maxpool2d_reference(
        &input, batch, channels, height, width, kernel_h, kernel_w, stride_h, stride_w,
    );

    // Compare
    let max_error = gpu_result
        .iter()
        .zip(cpu_result.iter())
        .map(|(gpu, cpu)| (gpu - cpu).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_error < 1e-6,
        "MaxPool2D should be exact, got max error {}",
        max_error
    );
}
