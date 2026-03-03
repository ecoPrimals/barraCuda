// SPDX-License-Identifier: AGPL-3.0-or-later
//! Batched GPU Nelder-Mead — N independent Nelder-Mead optimizations in parallel.

use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use crate::optimize::batched_nelder_mead_pipeline::{
    apply_nm_step, create_centroid_bgl, create_contract_bgl, create_f64_buffer, create_shrink_bgl,
    create_storage_buffer, run_centroid_reflect, run_contract, run_shrink, BatchedContractParams,
    BatchedShrinkParams, BatchedSimplexParams,
};

/// Configuration for a single Nelder-Mead problem in the batch.
#[derive(Clone)]
pub struct BatchNelderMeadConfig {
    pub dims: usize,
    pub max_iters: usize,
    pub tol: f64,
    pub alpha: f64,
    pub gamma: f64,
    pub rho: f64,
    pub sigma: f64,
}

impl Default for BatchNelderMeadConfig {
    fn default() -> Self {
        Self {
            dims: 2,
            max_iters: 1000,
            tol: 1e-8,
            alpha: 1.0,
            gamma: 2.0,
            rho: 0.5,
            sigma: 0.5,
        }
    }
}

/// Result for one problem in the batch.
#[derive(Debug, Clone)]
pub struct NelderMeadResult {
    pub best_point: Vec<f64>,
    pub best_value: f64,
    pub iterations: usize,
    pub converged: bool,
}

const WGSL_SIMPLEX: &str = include_str!("../shaders/optimizer/simplex_ops_f64.wgsl");

/// Run N independent Nelder-Mead optimizations in parallel on the GPU.
pub async fn batched_nelder_mead_gpu(
    device: &WgpuDevice,
    config: &BatchNelderMeadConfig,
    n_problems: usize,
    initial_simplices: &[f64],
    mut f_values: impl FnMut(&[f64]) -> Vec<f64>,
) -> Result<Vec<NelderMeadResult>> {
    let n = config.dims;
    let n_points = n + 1;
    let expected = n_problems * n_points * n;
    if initial_simplices.len() != expected {
        return Err(BarracudaError::InvalidInput {
            message: format!(
                "initial_simplices len {} != n_problems*(dims+1)*dims = {}",
                initial_simplices.len(),
                expected
            ),
        });
    }

    let shader = device.compile_shader_f64(WGSL_SIMPLEX, Some("Batched NM simplex"));
    let mut simplex: Vec<f64> = initial_simplices.to_vec();
    let mut f_vals = f_values(&simplex);

    let simplex_buf = create_f64_buffer(device, "nm_simplex", &simplex);
    let f_vals_buf = create_f64_buffer(device, "nm_f_vals", &f_vals);
    let worst_idx_buf =
        create_storage_buffer(device, "nm_worst_idx", (n_problems * 4) as u64, false);
    let best_idx_buf = create_storage_buffer(device, "nm_best_idx", (n_problems * 4) as u64, false);
    let centroid_buf =
        create_storage_buffer(device, "nm_centroid", (n_problems * n * 8) as u64, true);
    let output_buf = create_storage_buffer(device, "nm_output", (n_problems * n * 8) as u64, true);
    let inside_buf = create_storage_buffer(device, "nm_inside", (n_problems * 4) as u64, false);
    let contract_out_buf =
        create_storage_buffer(device, "nm_contract_out", (n_problems * n * 8) as u64, true);

    let bgl_centroid = create_centroid_bgl(device);
    let bgl_contract = create_contract_bgl(device);
    let bgl_shrink = create_shrink_bgl(device);

    let mut converged = vec![false; n_problems];
    let mut iterations = vec![0usize; n_problems];
    let mut results = vec![
        NelderMeadResult {
            best_point: vec![0.0; n],
            best_value: f64::INFINITY,
            iterations: 0,
            converged: false,
        };
        n_problems
    ];

    for iter in 0..config.max_iters {
        let mut worst_idx = vec![0u32; n_problems];
        let mut best_idx = vec![0u32; n_problems];
        let mut inside_contract = vec![0u32; n_problems];
        let mut need_contract = vec![false; n_problems];

        for p in 0..n_problems {
            if converged[p] {
                continue;
            }
            let base = p * n_points;
            let mut indices: Vec<usize> = (0..n_points).collect();
            indices.sort_by(|&a, &b| {
                f_vals[base + a]
                    .partial_cmp(&f_vals[base + b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            worst_idx[p] = indices[n] as u32;
            best_idx[p] = indices[0] as u32;
            let f_mean: f64 =
                (0..n_points).map(|i| f_vals[base + i]).sum::<f64>() / n_points as f64;
            let f_std = ((0..n_points)
                .map(|i| (f_vals[base + i] - f_mean).powi(2))
                .sum::<f64>()
                / n_points as f64)
                .sqrt();
            if f_std < config.tol {
                converged[p] = true;
                results[p] = NelderMeadResult {
                    best_point: (0..n)
                        .map(|j| simplex[p * n_points * n + indices[0] * n + j])
                        .collect(),
                    best_value: f_vals[base + indices[0]],
                    iterations: iter,
                    converged: true,
                };
            }
            iterations[p] = iter;
        }

        let active: Vec<usize> = (0..n_problems).filter(|&p| !converged[p]).collect();
        if active.is_empty() {
            break;
        }

        device
            .queue
            .write_buffer(&worst_idx_buf, 0, bytemuck::cast_slice(&worst_idx));
        device
            .queue
            .write_buffer(&best_idx_buf, 0, bytemuck::cast_slice(&best_idx));
        device
            .queue
            .write_buffer(&simplex_buf, 0, bytemuck::cast_slice(&simplex));
        device
            .queue
            .write_buffer(&f_vals_buf, 0, bytemuck::cast_slice(&f_vals));

        let params = BatchedSimplexParams {
            n_problems: n_problems as u32,
            n: n as u32,
            n_points: n_points as u32,
            _pad: 0,
            alpha: config.alpha,
            gamma: config.gamma,
            rho: config.rho,
            sigma: config.sigma,
        };
        let params_buf = device.create_uniform_buffer("nm_params", &params);

        run_centroid_reflect(
            device,
            &shader,
            &bgl_centroid,
            &params_buf,
            &simplex_buf,
            &f_vals_buf,
            &worst_idx_buf,
            &centroid_buf,
            &output_buf,
            n_problems,
            n,
            "batched_compute_centroid",
        );
        run_centroid_reflect(
            device,
            &shader,
            &bgl_centroid,
            &params_buf,
            &simplex_buf,
            &f_vals_buf,
            &worst_idx_buf,
            &centroid_buf,
            &output_buf,
            n_problems,
            n,
            "batched_reflect",
        );

        let reflect_pts: Vec<f64> = device.read_f64_buffer(&output_buf, n_problems * n)?;
        let f_reflect: Vec<f64> = f_values(&reflect_pts);
        let centroid = device.read_f64_buffer(&centroid_buf, n_problems * n)?;
        apply_nm_step(
            &mut simplex,
            &mut f_vals,
            &reflect_pts,
            &f_reflect,
            &centroid,
            &best_idx,
            &worst_idx,
            &mut need_contract,
            &mut inside_contract,
            &active,
            n,
            n_points,
            config.gamma,
            &mut f_values,
            n_problems,
        );

        let contract_needed: Vec<usize> = (0..n_problems).filter(|&p| need_contract[p]).collect();

        if !contract_needed.is_empty() {
            device
                .queue
                .write_buffer(&inside_buf, 0, bytemuck::cast_slice(&inside_contract));
            let contract_params = BatchedContractParams {
                n_problems: n_problems as u32,
                n: n as u32,
                rho: config.rho,
                _pad: [0, 0],
            };
            let contract_params_buf =
                device.create_uniform_buffer("nm_contract_params", &contract_params);
            run_contract(
                device,
                &shader,
                &bgl_contract,
                &contract_params_buf,
                &simplex_buf,
                &worst_idx_buf,
                &centroid_buf,
                &output_buf,
                &inside_buf,
                &contract_out_buf,
                n_problems,
                n,
            );

            let contract_pts = device.read_f64_buffer(&contract_out_buf, n_problems * n)?;
            let f_contract = f_values(&contract_pts);
            let mut need_shrink = Vec::new();
            for &p in &contract_needed {
                let base = p * n_points;
                let worst_idx_p = worst_idx[p] as usize;
                if f_contract[p] < f_vals[base + worst_idx_p] {
                    for j in 0..n {
                        simplex[p * n_points * n + worst_idx_p * n + j] = contract_pts[p * n + j];
                    }
                    f_vals[base + worst_idx_p] = f_contract[p];
                } else {
                    need_shrink.push(p);
                }
            }
            if !need_shrink.is_empty() {
                device
                    .queue
                    .write_buffer(&simplex_buf, 0, bytemuck::cast_slice(&simplex));
                device
                    .queue
                    .write_buffer(&best_idx_buf, 0, bytemuck::cast_slice(&best_idx));
                let shrink_params = BatchedShrinkParams {
                    n_problems: n_problems as u32,
                    n: n as u32,
                    n_points: n_points as u32,
                    _pad: 0,
                    sigma: config.sigma,
                };
                let shrink_params_buf =
                    device.create_uniform_buffer("nm_shrink_params", &shrink_params);
                run_shrink(
                    device,
                    &shader,
                    &bgl_shrink,
                    &shrink_params_buf,
                    &best_idx_buf,
                    &simplex_buf,
                    n_problems,
                    n,
                    n_points,
                );
                simplex = device.read_f64_buffer(&simplex_buf, n_problems * n_points * n)?;
                f_vals = f_values(&simplex);
            }
        }
    }

    for p in 0..n_problems {
        if !converged[p] {
            let base = p * n_points;
            let best_i = (0..n_points)
                .min_by(|&a, &b| {
                    f_vals[base + a]
                        .partial_cmp(&f_vals[base + b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            results[p] = NelderMeadResult {
                best_point: (0..n)
                    .map(|j| simplex[p * n_points * n + best_i * n + j])
                    .collect(),
                best_value: f_vals[base + best_i],
                iterations: iterations[p],
                converged: false,
            };
        }
    }
    Ok(results)
}

#[cfg(test)]
#[path = "batched_nelder_mead_gpu_tests.rs"]
mod tests;
