// SPDX-License-Identifier: AGPL-3.0-or-later
//! Coverage tests for all elementwise WGSL GPU operations.
//!
//! Exercises every `*_wgsl` activation/elementwise op on the `Tensor` API
//! to verify correct GPU dispatch and numerical sanity. Gracefully skips
//! when no GPU is available.

#![expect(clippy::unwrap_used, reason = "tests")]

mod common;

use barracuda::tensor::Tensor;
use std::sync::Arc;

async fn device() -> Option<Arc<barracuda::device::WgpuDevice>> {
    barracuda::device::test_pool::get_test_device_if_gpu_available().await
}

fn tensor(dev: &Arc<barracuda::device::WgpuDevice>, data: &[f32]) -> Tensor {
    Tensor::from_data(data, vec![data.len()], dev.clone()).unwrap()
}

macro_rules! elementwise_test {
    ($name:ident, $method:ident, $input:expr, $check:expr) => {
        #[tokio::test]
        async fn $name() {
            let Some(dev) = device().await else { return };
            let t = tensor(&dev, &$input);
            let result = t.$method().unwrap().to_vec().unwrap();
            assert_eq!(
                result.len(),
                $input.len(),
                "{}: length mismatch",
                stringify!($name)
            );
            let check: Box<dyn Fn(&[f32])> = Box::new($check);
            check(&result);
        }
    };
}

elementwise_test!(
    test_abs_wgsl,
    abs_wgsl,
    [-3.0f32, -1.0, 0.0, 1.0, 3.0],
    |r: &[f32]| {
        assert!((r[0] - 3.0).abs() < 1e-5);
        assert!((r[2]).abs() < 1e-5);
        assert!((r[4] - 3.0).abs() < 1e-5);
    }
);

elementwise_test!(
    test_ceil_wgsl,
    ceil_wgsl,
    [1.2f32, 2.7, -0.3, -1.8, 0.0],
    |r: &[f32]| {
        assert!((r[0] - 2.0).abs() < 1e-5);
        assert!((r[1] - 3.0).abs() < 1e-5);
        assert!((r[2] - 0.0).abs() < 1e-5);
    }
);

elementwise_test!(
    test_floor_wgsl,
    floor_wgsl,
    [1.7f32, 2.3, -0.7, -1.2, 0.0],
    |r: &[f32]| {
        assert!((r[0] - 1.0).abs() < 1e-5);
        assert!((r[1] - 2.0).abs() < 1e-5);
        assert!((r[2] - (-1.0)).abs() < 1e-5);
    }
);

elementwise_test!(
    test_round_wgsl,
    round_wgsl,
    [1.4f32, 1.6, -0.4, -0.6, 0.5],
    |r: &[f32]| {
        assert!((r[0] - 1.0).abs() < 1e-5);
        assert!((r[1] - 2.0).abs() < 1e-5);
    }
);

elementwise_test!(
    test_trunc_wgsl,
    trunc_wgsl,
    [1.7f32, -1.7, 2.3, -2.3, 0.0],
    |r: &[f32]| {
        assert!((r[0] - 1.0).abs() < 1e-5);
        assert!((r[1] - (-1.0)).abs() < 1e-5);
    }
);

elementwise_test!(
    test_frac_wgsl,
    frac_wgsl,
    [1.75f32, 2.25, 3.5, 0.0, -1.75],
    |r: &[f32]| {
        assert!((r[0] - 0.75).abs() < 1e-5);
        assert!((r[1] - 0.25).abs() < 1e-5);
    }
);

elementwise_test!(
    test_neg_wgsl,
    neg_wgsl,
    [1.0f32, -2.0, 0.0, 3.5, -0.5],
    |r: &[f32]| {
        assert!((r[0] - (-1.0)).abs() < 1e-5);
        assert!((r[1] - 2.0).abs() < 1e-5);
        assert!((r[2]).abs() < 1e-5);
    }
);

elementwise_test!(
    test_sign_wgsl,
    sign_wgsl,
    [-5.0f32, -0.1, 0.0, 0.1, 5.0],
    |r: &[f32]| {
        assert!((r[0] - (-1.0)).abs() < 1e-5);
        assert!((r[4] - 1.0).abs() < 1e-5);
    }
);

elementwise_test!(
    test_reciprocal_wgsl,
    reciprocal_wgsl,
    [1.0f32, 2.0, 4.0, 0.5, -2.0],
    |r: &[f32]| {
        assert!((r[0] - 1.0).abs() < 1e-5);
        assert!((r[1] - 0.5).abs() < 1e-5);
        assert!((r[2] - 0.25).abs() < 1e-5);
    }
);

elementwise_test!(
    test_sqrt_wgsl,
    sqrt_wgsl,
    [1.0f32, 4.0, 9.0, 16.0, 25.0],
    |r: &[f32]| {
        assert!((r[0] - 1.0).abs() < 1e-5);
        assert!((r[1] - 2.0).abs() < 1e-5);
        assert!((r[2] - 3.0).abs() < 1e-5);
    }
);

elementwise_test!(
    test_rsqrt_wgsl,
    rsqrt_wgsl,
    [1.0f32, 4.0, 16.0, 25.0, 100.0],
    |r: &[f32]| {
        assert!((r[0] - 1.0).abs() < 1e-5);
        assert!((r[1] - 0.5).abs() < 1e-5);
    }
);

elementwise_test!(
    test_exp_wgsl,
    exp_wgsl,
    [0.0f32, 1.0, -1.0, 2.0, -2.0],
    |r: &[f32]| {
        assert!((r[0] - 1.0).abs() < 1e-4);
        assert!((r[1] - std::f32::consts::E).abs() < 1e-4);
    }
);

elementwise_test!(
    test_log_wgsl,
    log_wgsl,
    [1.0f32, std::f32::consts::E, 10.0, 100.0, 0.5],
    |r: &[f32]| {
        assert!((r[0]).abs() < 1e-4);
        assert!((r[1] - 1.0).abs() < 1e-4);
    }
);

elementwise_test!(
    test_sin_wgsl,
    sin_wgsl,
    [
        0.0f32,
        std::f32::consts::FRAC_PI_2,
        std::f32::consts::PI,
        1.0,
        -1.0
    ],
    |r: &[f32]| {
        assert!((r[0]).abs() < 1e-4);
        assert!((r[1] - 1.0).abs() < 1e-4);
    }
);

elementwise_test!(
    test_cos_wgsl,
    cos_wgsl,
    [
        0.0f32,
        std::f32::consts::FRAC_PI_2,
        std::f32::consts::PI,
        1.0,
        -1.0
    ],
    |r: &[f32]| {
        assert!((r[0] - 1.0).abs() < 1e-4);
        assert!((r[1]).abs() < 1e-4);
    }
);

elementwise_test!(
    test_tan_wgsl,
    tan_wgsl,
    [0.0f32, 0.5, 1.0, -0.5, -1.0],
    |r: &[f32]| {
        assert!((r[0]).abs() < 1e-4);
        assert!((r[1] - 0.5_f32.tan()).abs() < 1e-4);
    }
);

// Activation functions

elementwise_test!(
    test_sigmoid_op,
    sigmoid,
    [0.0f32, 1.0, -1.0, 5.0, -5.0],
    |r: &[f32]| {
        assert!((r[0] - 0.5).abs() < 1e-4);
        assert!(r[3] > 0.99);
        assert!(r[4] < 0.01);
    }
);

elementwise_test!(
    test_silu_wgsl,
    silu_wgsl,
    [0.0f32, 1.0, -1.0, 2.0, -2.0],
    |r: &[f32]| {
        assert!((r[0]).abs() < 1e-4);
        assert!((r[1] - 0.7311).abs() < 1e-3);
    }
);

elementwise_test!(
    test_mish_wgsl,
    mish_wgsl,
    [0.0f32, 1.0, -1.0, 2.0, -2.0],
    |r: &[f32]| {
        assert!((r[0]).abs() < 1e-4);
        assert!(r[1] > 0.8 && r[1] < 0.9);
    }
);

elementwise_test!(
    test_softsign_wgsl,
    softsign_wgsl,
    [0.0f32, 1.0, -1.0, 10.0, -10.0],
    |r: &[f32]| {
        assert!((r[0]).abs() < 1e-4);
        assert!((r[1] - 0.5).abs() < 1e-4);
    }
);

elementwise_test!(
    test_logsigmoid_wgsl,
    logsigmoid_wgsl,
    [0.0f32, 1.0, -1.0, 5.0, -5.0],
    |r: &[f32]| {
        assert!((r[0] - (-0.6931)).abs() < 1e-3);
        assert!(r[3] > -0.01);
    }
);

elementwise_test!(
    test_hardswish_wgsl,
    hardswish_wgsl,
    [0.0f32, 1.0, -1.0, 4.0, -4.0],
    |r: &[f32]| {
        assert!((r[0]).abs() < 1e-4);
        assert!((r[3] - 4.0).abs() < 1e-4);
        assert!((r[4]).abs() < 1e-4);
    }
);

elementwise_test!(
    test_hardsigmoid_wgsl,
    hardsigmoid_wgsl,
    [0.0f32, 1.0, -1.0, 4.0, -4.0],
    |r: &[f32]| {
        assert!((r[0] - 0.5).abs() < 1e-4);
        assert!((r[3] - 1.0).abs() < 1e-4);
        assert!((r[4]).abs() < 1e-4);
    }
);

elementwise_test!(
    test_hardtanh_wgsl,
    hardtanh_wgsl,
    [0.0f32, 0.5, -0.5, 2.0, -2.0],
    |r: &[f32]| {
        assert!((r[0]).abs() < 1e-4);
        assert!((r[3] - 1.0).abs() < 1e-4);
        assert!((r[4] - (-1.0)).abs() < 1e-4);
    }
);

elementwise_test!(
    test_tanhshrink_wgsl,
    tanhshrink_wgsl,
    [0.0f32, 1.0, -1.0, 2.0, -2.0],
    |r: &[f32]| {
        assert!((r[0]).abs() < 1e-4);
        assert!((r[1] - (1.0 - 1.0_f32.tanh())).abs() < 1e-4);
    }
);

elementwise_test!(
    test_gelu_approximate_wgsl,
    gelu_approximate_wgsl,
    [0.0f32, 1.0, -1.0, 2.0, -2.0],
    |r: &[f32]| {
        assert!((r[0]).abs() < 1e-4);
        assert!(r[1] > 0.8 && r[1] < 0.9);
    }
);

elementwise_test!(
    test_selu_wgsl,
    selu_wgsl,
    [0.0f32, 1.0, -1.0, 2.0, -2.0],
    |r: &[f32]| {
        assert!((r[0]).abs() < 1e-4);
        assert!((r[1] - 1.0507).abs() < 1e-3);
    }
);

#[tokio::test]
async fn test_elu_wgsl() {
    let Some(dev) = device().await else { return };
    let t = tensor(&dev, &[0.0, 1.0, -1.0, 2.0, -2.0]);
    let result = t.elu_wgsl().unwrap().to_vec().unwrap();
    assert_eq!(result.len(), 5);
    assert!((result[0]).abs() < 1e-4);
    assert!((result[1] - 1.0).abs() < 1e-4);
    assert!(result[2] < 0.0);
}

#[tokio::test]
async fn test_leaky_relu_wgsl() {
    let Some(dev) = device().await else { return };
    let t = tensor(&dev, &[0.0, 1.0, -1.0, 2.0, -2.0]);
    let result = t.leaky_relu_wgsl().unwrap().to_vec().unwrap();
    assert_eq!(result.len(), 5);
    assert!((result[0]).abs() < 1e-4);
    assert!((result[1] - 1.0).abs() < 1e-4);
    assert!(result[2] < 0.0 && result[2] > -0.1);
}

#[tokio::test]
async fn test_celu_wgsl() {
    let Some(dev) = device().await else { return };
    let t = tensor(&dev, &[0.0, 1.0, -1.0, 2.0, -2.0]);
    let result = t.celu_wgsl().unwrap().to_vec().unwrap();
    assert_eq!(result.len(), 5);
    assert!((result[0]).abs() < 1e-4);
    assert!((result[1] - 1.0).abs() < 1e-4);
    assert!(result[2] < 0.0);
}

#[tokio::test]
async fn test_softshrink_wgsl() {
    let Some(dev) = device().await else { return };
    let t = tensor(&dev, &[0.0, 1.0, -1.0, 0.3, -0.3]);
    let result = t.softshrink_wgsl().unwrap().to_vec().unwrap();
    assert_eq!(result.len(), 5);
    assert!((result[0]).abs() < 1e-4);
    assert!((result[1] - 0.5).abs() < 1e-4);
}

#[tokio::test]
async fn test_hardshrink_wgsl() {
    let Some(dev) = device().await else { return };
    let t = tensor(&dev, &[0.0, 1.0, -1.0, 0.3, -0.3]);
    let result = t.hardshrink_wgsl().unwrap().to_vec().unwrap();
    assert_eq!(result.len(), 5);
    assert!((result[0]).abs() < 1e-4);
    assert!((result[1] - 1.0).abs() < 1e-4);
}

#[tokio::test]
async fn test_pow_wgsl() {
    let Some(dev) = device().await else { return };
    let t = tensor(&dev, &[1.0, 2.0, 3.0, 4.0, 5.0]);
    let result = t.pow_wgsl(2.0).unwrap().to_vec().unwrap();
    assert_eq!(result.len(), 5);
    assert!((result[0] - 1.0).abs() < 1e-4);
    assert!((result[1] - 4.0).abs() < 1e-4);
    assert!((result[2] - 9.0).abs() < 1e-4);
}

#[tokio::test]
async fn test_max_wgsl() {
    let Some(dev) = device().await else { return };
    let t = tensor(&dev, &[1.0, 5.0, 3.0, 2.0, 4.0]);
    let result = t.max_wgsl(None).unwrap().to_vec().unwrap();
    assert!(!result.is_empty());
    assert!((result[0] - 5.0).abs() < 1e-4);
}

#[tokio::test]
async fn test_min_wgsl() {
    let Some(dev) = device().await else { return };
    let t = tensor(&dev, &[3.0, 1.0, 5.0, 2.0, 4.0]);
    let result = t.min_wgsl(None).unwrap().to_vec().unwrap();
    assert!(!result.is_empty());
    assert!((result[0] - 1.0).abs() < 1e-4);
}
