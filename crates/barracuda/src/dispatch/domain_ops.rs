//! Domain operation dispatch wrappers — neuralSpring pattern.
//!
//! Routes operations through GPU (Tensor) or CPU based on dispatch config
//! thresholds. Each wrapper checks device availability and input size before
//! choosing the execution path.

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use crate::device::WgpuDevice;
use crate::dispatch::config::{global_config, DispatchConfig};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use std::sync::Arc;

/// Small constant to avoid division by zero (matches neuralSpring primitives).
const LOG_GUARD: f64 = 1e-300;

// -----------------------------------------------------------------------------
// CPU primitives (pure f64, no device)
// -----------------------------------------------------------------------------

fn matmul_cpu(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += a_ip * b[p * n + j];
            }
        }
    }
    c
}

fn frobenius_norm_cpu(a: &[f64]) -> f64 {
    a.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

fn transpose_cpu(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut t = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            t[j * rows + i] = a[i * cols + j];
        }
    }
    t
}

fn softmax_cpu(x: &[f64]) -> Vec<f64> {
    let max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = x.iter().map(|&v| (v - max).exp()).collect();
    let sum: f64 = exp.iter().sum();
    if sum < LOG_GUARD {
        return vec![0.0; x.len()];
    }
    exp.iter().map(|&v| v / sum).collect()
}

fn gelu_cpu(x: &[f64]) -> Vec<f64> {
    use std::f64::consts::PI;
    x.iter()
        .map(|&xi| {
            let inner = (2.0 / PI).sqrt() * 0.044_715f64.mul_add(xi.powi(3), xi);
            0.5 * xi * (1.0 + inner.tanh())
        })
        .collect()
}

fn l2_distance_cpu(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let sum_sq: f64 = a
        .iter()
        .take(n)
        .zip(b.iter().take(n))
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum();
    sum_sq.sqrt()
}

fn mean_cpu(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

fn variance_cpu(data: &[f64]) -> f64 {
    let n = data.len();
    if n <= 1 {
        return 0.0;
    }
    let mean = mean_cpu(data);
    let var_sum: f64 = data.iter().map(|x| (x - mean).powi(2)).sum();
    var_sum / n as f64
}

fn hmm_forward_step_cpu(
    alpha_prev: &[f64],
    transition: &[f64],
    emission_col: &[f64],
    n_states: usize,
) -> (Vec<f64>, f64) {
    let mut raw = vec![0.0; n_states];
    for j in 0..n_states {
        let mut sum = 0.0;
        for i in 0..n_states {
            sum += alpha_prev[i] * transition[i * n_states + j];
        }
        raw[j] = sum * emission_col[j];
    }
    let scale = raw.iter().sum::<f64>().max(LOG_GUARD);
    let alpha_new: Vec<f64> = raw.iter().map(|&x| x / scale).collect();
    (alpha_new, scale)
}

// -----------------------------------------------------------------------------
// GPU paths (via Tensor)
// -----------------------------------------------------------------------------

fn matmul_gpu(
    a: &[f64],
    b: &[f64],
    m: usize,
    k: usize,
    n: usize,
    device: &Arc<WgpuDevice>,
) -> Result<Vec<f64>> {
    let a_f32: Vec<f32> = a.iter().map(|&x| x as f32).collect();
    let b_f32: Vec<f32> = b.iter().map(|&x| x as f32).collect();

    let a_t = Tensor::from_data(&a_f32, vec![m, k], device.clone())
        .map_err(|e| BarracudaError::Gpu(format!("matmul A upload: {e}")))?;
    let b_t = Tensor::from_data(&b_f32, vec![k, n], device.clone())
        .map_err(|e| BarracudaError::Gpu(format!("matmul B upload: {e}")))?;

    let c_t = a_t
        .matmul(&b_t)
        .map_err(|e| BarracudaError::Gpu(format!("matmul: {e}")))?;

    let c_f32 = c_t
        .to_vec()
        .map_err(|e| BarracudaError::Gpu(format!("matmul readback: {e}")))?;

    Ok(c_f32.into_iter().map(f64::from).collect())
}

fn frobenius_norm_gpu(a: &[f64], device: &Arc<WgpuDevice>) -> Result<f64> {
    let a_f32: Vec<f32> = a.iter().map(|&x| x as f32).collect();
    let n = a_f32.len();

    let a_t = Tensor::from_data(&a_f32, vec![n], device.clone())
        .map_err(|e| BarracudaError::Gpu(format!("frobenius_norm upload: {e}")))?;

    let norm_t = a_t
        .norm()
        .map_err(|e| BarracudaError::Gpu(format!("frobenius_norm: {e}")))?;

    let result = norm_t
        .to_vec()
        .map_err(|e| BarracudaError::Gpu(format!("frobenius_norm readback: {e}")))?;

    Ok(f64::from(result[0]))
}

fn transpose_gpu(
    a: &[f64],
    rows: usize,
    cols: usize,
    device: &Arc<WgpuDevice>,
) -> Result<Vec<f64>> {
    let a_f32: Vec<f32> = a.iter().map(|&x| x as f32).collect();

    let a_t = Tensor::from_data(&a_f32, vec![rows, cols], device.clone())
        .map_err(|e| BarracudaError::Gpu(format!("transpose upload: {e}")))?;

    let t_t = a_t
        .transpose()
        .map_err(|e| BarracudaError::Gpu(format!("transpose: {e}")))?;

    let t_f32 = t_t
        .to_vec()
        .map_err(|e| BarracudaError::Gpu(format!("transpose readback: {e}")))?;

    Ok(t_f32.into_iter().map(f64::from).collect())
}

fn softmax_gpu(x: &[f64], device: &Arc<WgpuDevice>) -> Result<Vec<f64>> {
    let x_f32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
    let n = x_f32.len();

    let x_t = Tensor::from_data(&x_f32, vec![n], device.clone())
        .map_err(|e| BarracudaError::Gpu(format!("softmax upload: {e}")))?;

    let sm = x_t
        .softmax()
        .map_err(|e| BarracudaError::Gpu(format!("softmax: {e}")))?;

    let out = sm
        .to_vec()
        .map_err(|e| BarracudaError::Gpu(format!("softmax readback: {e}")))?;

    Ok(out.into_iter().map(f64::from).collect())
}

fn gelu_gpu(x: &[f64], device: &Arc<WgpuDevice>) -> Result<Vec<f64>> {
    let x_f32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
    let n = x_f32.len();

    let x_t = Tensor::from_data(&x_f32, vec![n], device.clone())
        .map_err(|e| BarracudaError::Gpu(format!("gelu upload: {e}")))?;

    let out_t = x_t
        .gelu_wgsl()
        .map_err(|e| BarracudaError::Gpu(format!("gelu: {e}")))?;

    let out = out_t
        .to_vec()
        .map_err(|e| BarracudaError::Gpu(format!("gelu readback: {e}")))?;

    Ok(out.into_iter().map(f64::from).collect())
}

fn l2_distance_gpu(a: &[f64], b: &[f64], device: &Arc<WgpuDevice>) -> Result<f64> {
    let a_f32: Vec<f32> = a.iter().map(|&x| x as f32).collect();
    let b_f32: Vec<f32> = b.iter().map(|&x| x as f32).collect();
    let n = a_f32.len();

    let a_t = Tensor::from_data(&a_f32, vec![n], device.clone())
        .map_err(|e| BarracudaError::Gpu(format!("l2_distance A: {e}")))?;
    let b_t = Tensor::from_data(&b_f32, vec![n], device.clone())
        .map_err(|e| BarracudaError::Gpu(format!("l2_distance B: {e}")))?;

    let diff = a_t
        .sub(&b_t)
        .map_err(|e| BarracudaError::Gpu(format!("l2_distance sub: {e}")))?;

    let norm = diff
        .norm()
        .map_err(|e| BarracudaError::Gpu(format!("l2_distance norm: {e}")))?;

    let result = norm
        .to_vec()
        .map_err(|e| BarracudaError::Gpu(format!("l2_distance readback: {e}")))?;

    Ok(f64::from(result[0]))
}

fn mean_gpu(data: &[f64], device: &Arc<WgpuDevice>) -> Result<f64> {
    let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    let n = data_f32.len();

    let t = Tensor::from_data(&data_f32, vec![n], device.clone())
        .map_err(|e| BarracudaError::Gpu(format!("mean upload: {e}")))?;

    let m = t
        .mean()
        .map_err(|e| BarracudaError::Gpu(format!("mean: {e}")))?;

    let result = m
        .to_vec()
        .map_err(|e| BarracudaError::Gpu(format!("mean readback: {e}")))?;

    Ok(f64::from(result[0]))
}

fn variance_gpu(data: &[f64], device: &Arc<WgpuDevice>) -> Result<f64> {
    let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    let n = data_f32.len();

    let t = Tensor::from_data(&data_f32, vec![n], device.clone())
        .map_err(|e| BarracudaError::Gpu(format!("variance upload: {e}")))?;

    let mean_t = t
        .mean()
        .map_err(|e| BarracudaError::Gpu(format!("variance mean: {e}")))?;
    let mean_val = mean_t
        .to_vec()
        .map_err(|e| BarracudaError::Gpu(format!("variance mean readback: {e}")))?[0];

    let mean_vec = vec![mean_val; n];
    let mean_broadcast = Tensor::from_data(&mean_vec, vec![n], device.clone())
        .map_err(|e| BarracudaError::Gpu(format!("variance mean_vec: {e}")))?;

    let diff = t
        .sub(&mean_broadcast)
        .map_err(|e| BarracudaError::Gpu(format!("variance sub: {e}")))?;
    let sq = diff
        .mul(&diff)
        .map_err(|e| BarracudaError::Gpu(format!("variance sq: {e}")))?;
    let var = sq
        .mean()
        .map_err(|e| BarracudaError::Gpu(format!("variance: {e}")))?;

    let result = var
        .to_vec()
        .map_err(|e| BarracudaError::Gpu(format!("variance readback: {e}")))?;

    Ok(f64::from(result[0]))
}

fn hmm_forward_step_gpu(
    alpha_prev: &[f64],
    transition: &[f64],
    emission_col: &[f64],
    n_states: usize,
    device: &Arc<WgpuDevice>,
) -> Result<(Vec<f64>, f64)> {
    let a_f32: Vec<f32> = alpha_prev.iter().map(|&x| x as f32).collect();
    let t_f32: Vec<f32> = transition.iter().map(|&x| x as f32).collect();
    let e_f32: Vec<f32> = emission_col.iter().map(|&x| x as f32).collect();

    let alpha_t = Tensor::from_data(&a_f32, vec![1, n_states], device.clone())
        .map_err(|e| BarracudaError::Gpu(format!("hmm_fwd alpha: {e}")))?;
    let trans_t = Tensor::from_data(&t_f32, vec![n_states, n_states], device.clone())
        .map_err(|e| BarracudaError::Gpu(format!("hmm_fwd trans: {e}")))?;

    let propagated = alpha_t
        .matmul(&trans_t)
        .map_err(|e| BarracudaError::Gpu(format!("hmm_fwd matmul: {e}")))?;

    let emit_t = Tensor::from_data(&e_f32, vec![1, n_states], device.clone())
        .map_err(|e| BarracudaError::Gpu(format!("hmm_fwd emit: {e}")))?;

    let raw = propagated
        .mul(&emit_t)
        .map_err(|e| BarracudaError::Gpu(format!("hmm_fwd mul: {e}")))?;

    let scale_t = raw
        .sum()
        .map_err(|e| BarracudaError::Gpu(format!("hmm_fwd sum: {e}")))?;
    let scale_val = scale_t
        .to_vec()
        .map_err(|e| BarracudaError::Gpu(format!("hmm_fwd scale_read: {e}")))?[0];

    let raw_vec = raw
        .to_vec()
        .map_err(|e| BarracudaError::Gpu(format!("hmm_fwd raw_read: {e}")))?;

    let scale = f64::from(scale_val).max(LOG_GUARD);
    let alpha_new: Vec<f64> = raw_vec.iter().map(|&x| f64::from(x) / scale).collect();

    Ok((alpha_new, scale))
}

// -----------------------------------------------------------------------------
// Dispatch API
// -----------------------------------------------------------------------------

/// Matrix multiply dispatch: C = A[m×k] × B[k×n].
///
/// Uses GPU when device is provided and input size exceeds threshold.
pub fn matmul_dispatch(
    a: &[f64],
    b: &[f64],
    m: usize,
    k: usize,
    n: usize,
    device: Option<&Arc<WgpuDevice>>,
) -> Result<Vec<f64>> {
    let input_size = m.max(n).max(k);
    let config = global_config();
    let use_gpu = device.is_some() && config.should_use_gpu(input_size, "matmul");

    if use_gpu {
        if let Some(dev) = device {
            return matmul_gpu(a, b, m, k, n, dev);
        }
    }
    Ok(matmul_cpu(a, b, m, k, n))
}

/// Frobenius norm dispatch: sqrt(sum of squares).
pub fn frobenius_norm_dispatch(a: &[f64], device: Option<&Arc<WgpuDevice>>) -> Result<f64> {
    let input_size = a.len();
    let config = global_config();
    let use_gpu = device.is_some() && config.should_use_gpu(input_size, "frobenius_norm");

    if use_gpu {
        if let Some(dev) = device {
            return frobenius_norm_gpu(a, dev);
        }
    }
    Ok(frobenius_norm_cpu(a))
}

/// Transpose dispatch: [rows×cols] → [cols×rows].
pub fn transpose_dispatch(
    a: &[f64],
    rows: usize,
    cols: usize,
    device: Option<&Arc<WgpuDevice>>,
) -> Result<Vec<f64>> {
    let input_size = rows * cols;
    let config = global_config();
    let use_gpu = device.is_some() && config.should_use_gpu(input_size, "transpose");

    if use_gpu {
        if let Some(dev) = device {
            return transpose_gpu(a, rows, cols, dev);
        }
    }
    Ok(transpose_cpu(a, rows, cols))
}

/// Softmax dispatch: exp(x) / sum(exp(x)).
pub fn softmax_dispatch(x: &[f64], device: Option<&Arc<WgpuDevice>>) -> Result<Vec<f64>> {
    let input_size = x.len();
    let config = global_config();
    let use_gpu = device.is_some() && config.should_use_gpu(input_size, "softmax");

    if use_gpu {
        if let Some(dev) = device {
            return softmax_gpu(x, dev);
        }
    }
    Ok(softmax_cpu(x))
}

/// GELU activation dispatch.
pub fn gelu_dispatch(x: &[f64], device: Option<&Arc<WgpuDevice>>) -> Result<Vec<f64>> {
    let input_size = x.len();
    let config = global_config();
    let use_gpu = device.is_some() && config.should_use_gpu(input_size, "gelu");

    if use_gpu {
        if let Some(dev) = device {
            return gelu_gpu(x, dev);
        }
    }
    Ok(gelu_cpu(x))
}

/// L2 (Euclidean) distance dispatch between two vectors.
pub fn l2_distance_dispatch(a: &[f64], b: &[f64], device: Option<&Arc<WgpuDevice>>) -> Result<f64> {
    let input_size = a.len().min(b.len());
    let config = global_config();
    let use_gpu = device.is_some() && config.should_use_gpu(input_size, "l2_distance");

    if use_gpu {
        if let Some(dev) = device {
            return l2_distance_gpu(a, b, dev);
        }
    }
    Ok(l2_distance_cpu(a, b))
}

/// Mean reduction dispatch.
pub fn mean_dispatch(data: &[f64], device: Option<&Arc<WgpuDevice>>) -> Result<f64> {
    let input_size = data.len();
    let config = global_config();
    let use_gpu = device.is_some() && config.should_use_gpu(input_size, "mean");

    if use_gpu {
        if let Some(dev) = device {
            return mean_gpu(data, dev);
        }
    }
    Ok(mean_cpu(data))
}

/// Variance (population) dispatch.
pub fn variance_dispatch(data: &[f64], device: Option<&Arc<WgpuDevice>>) -> Result<f64> {
    let input_size = data.len();
    let config = global_config();
    let use_gpu = device.is_some() && config.should_use_gpu(input_size, "variance");

    if use_gpu {
        if let Some(dev) = device {
            return variance_gpu(data, dev);
        }
    }
    Ok(variance_cpu(data))
}

/// HMM forward step dispatch: alpha[t] = normalize(emit * (trans^T @ alpha[t-1])).
///
/// Returns (alpha_new, scale).
pub fn hmm_forward_dispatch(
    alpha_prev: &[f64],
    transition: &[f64],
    emission_col: &[f64],
    n_states: usize,
    device: Option<&Arc<WgpuDevice>>,
) -> Result<(Vec<f64>, f64)> {
    let input_size = n_states * n_states + n_states;
    let config = global_config();
    let use_gpu = device.is_some() && config.should_use_gpu(input_size, "hmm");

    if use_gpu {
        if let Some(dev) = device {
            return hmm_forward_step_gpu(alpha_prev, transition, emission_col, n_states, dev);
        }
    }
    Ok(hmm_forward_step_cpu(
        alpha_prev,
        transition,
        emission_col,
        n_states,
    ))
}

// -----------------------------------------------------------------------------
// Dispatch with custom config (for testing)
// -----------------------------------------------------------------------------

/// Matmul dispatch using custom config.
pub fn matmul_dispatch_with_config(
    a: &[f64],
    b: &[f64],
    m: usize,
    k: usize,
    n: usize,
    device: Option<&Arc<WgpuDevice>>,
    config: &DispatchConfig,
) -> Result<Vec<f64>> {
    let input_size = m.max(n).max(k);
    let use_gpu = device.is_some() && config.should_use_gpu(input_size, "matmul");

    if use_gpu {
        if let Some(dev) = device {
            return matmul_gpu(a, b, m, k, n, dev);
        }
    }
    Ok(matmul_cpu(a, b, m, k, n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_dispatch_none_device_cpu_path() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let result = matmul_dispatch(&a, &b, 2, 2, 2, None).expect("matmul");
        assert_eq!(result.len(), 4);
        // [1,2; 3,4] * [5,6; 7,8] = [19,22; 43,50]
        assert!((result[0] - 19.0).abs() < 1e-10);
        assert!((result[1] - 22.0).abs() < 1e-10);
        assert!((result[2] - 43.0).abs() < 1e-10);
        assert!((result[3] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn softmax_dispatch_cpu_path() {
        let x = vec![1.0, 2.0, 3.0];
        let result = softmax_dispatch(&x, None).expect("softmax");
        assert_eq!(result.len(), 3);
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "softmax sum = {sum}");
        assert!(result.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn l2_distance_dispatch_cpu_path() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let d = l2_distance_dispatch(&a, &b, None).expect("l2_distance");
        assert!((d - 5.0).abs() < 1e-10, "L2([0,0],[3,4]) = 5, got {d}");
    }
}
