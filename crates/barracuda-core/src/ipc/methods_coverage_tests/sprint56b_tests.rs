// SPDX-License-Identifier: AGPL-3.0-or-later
//! Coverage tests for Sprint 56b — untested IPC handlers.
//!
//! Tests cover: ode.step, stats.covariance, stats.spearman, stats.fit_linear,
//! stats.empirical_spectral_density, spectral.fft, spectral.power_spectrum,
//! linalg.solve, nautilus.*, ml.mlp_train, ml.esn_predict.

use super::super::linalg::linalg_solve;
use super::super::math::ode_step;
use super::super::stats::{
    stats_covariance, stats_empirical_spectral_density, stats_fit_linear, stats_spearman,
};
use super::super::ml::{ml_esn_predict, ml_mlp_train};
use super::super::nautilus::{
    nautilus_create, nautilus_export, nautilus_import, nautilus_observe, nautilus_predict,
    nautilus_train,
};
use super::super::spectral::{spectral_fft, spectral_power_spectrum};
use crate::ipc::jsonrpc;

// ── ode.step ────────────────────────────────────────────────────────

#[test]
fn ode_step_missing_state() {
    let resp = ode_step(
        &serde_json::json!({"a": [[1.0]], "dt": 0.1}),
        serde_json::json!(1),
    );
    let err = resp.error.expect("missing state should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
}

#[test]
fn ode_step_missing_a_matrix() {
    let resp = ode_step(
        &serde_json::json!({"state": [1.0], "dt": 0.1}),
        serde_json::json!(2),
    );
    let err = resp.error.expect("missing a matrix should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
}

#[test]
fn ode_step_empty_state() {
    let resp = ode_step(
        &serde_json::json!({"state": [], "a": [], "dt": 0.1}),
        serde_json::json!(3),
    );
    let err = resp.error.expect("empty state should fail");
    assert!(err.message.contains("non-empty"));
}

#[test]
fn ode_step_dt_negative() {
    let resp = ode_step(
        &serde_json::json!({"state": [1.0], "a": [[0.0]], "dt": -0.1}),
        serde_json::json!(4),
    );
    let err = resp.error.expect("negative dt should fail");
    assert!(err.message.contains("dt"));
}

#[test]
fn ode_step_dimension_mismatch() {
    let resp = ode_step(
        &serde_json::json!({"state": [1.0, 2.0], "a": [[1.0]], "dt": 0.1}),
        serde_json::json!(5),
    );
    let err = resp.error.expect("dimension mismatch should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
}

#[test]
fn ode_step_b_length_mismatch() {
    let resp = ode_step(
        &serde_json::json!({"state": [1.0], "a": [[0.0]], "b": [1.0, 2.0], "dt": 0.1}),
        serde_json::json!(6),
    );
    let err = resp.error.expect("b length mismatch should fail");
    assert!(err.message.contains("length"));
}

#[test]
fn ode_step_happy_path_decay() {
    // dy/dt = -y (exponential decay): y(t) = y0 * exp(-t)
    let resp = ode_step(
        &serde_json::json!({"state": [1.0], "a": [[-1.0]], "dt": 0.01, "n_steps": 100}),
        serde_json::json!(7),
    );
    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    let state = result["state"].as_array().unwrap();
    let y_final = state[0].as_f64().unwrap();
    // exp(-1) ≈ 0.3679; RK4 should be very close
    assert!((y_final - (-1.0f64).exp()).abs() < 1e-6);
    assert!((result["t_final"].as_f64().unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn ode_step_with_forcing() {
    // dy/dt = -y + 1 (approaches y=1 from y=0)
    let resp = ode_step(
        &serde_json::json!({"state": [0.0], "a": [[-1.0]], "b": [1.0], "dt": 0.01, "n_steps": 500}),
        serde_json::json!(8),
    );
    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    let y_final = result["state"][0].as_f64().unwrap();
    // y(5) = 1 - exp(-5) ≈ 0.9933
    assert!((y_final - (1.0 - (-5.0f64).exp())).abs() < 1e-4);
}

// ── stats.covariance ────────────────────────────────────────────────

#[test]
fn stats_covariance_missing_x() {
    let resp = stats_covariance(&serde_json::json!({"y": [1.0, 2.0]}), serde_json::json!(10));
    assert_eq!(resp.error.unwrap().code, jsonrpc::INVALID_PARAMS);
}

#[test]
fn stats_covariance_happy_path() {
    let resp = stats_covariance(
        &serde_json::json!({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [2.0, 4.0, 6.0, 8.0, 10.0]}),
        serde_json::json!(11),
    );
    assert!(resp.error.is_none());
    let cov = resp.result.unwrap()["result"].as_f64().unwrap();
    // cov(x, 2x) = 2*var(x) = 2*2.5 = 5.0
    assert!((cov - 5.0).abs() < 1e-10);
}

// ── stats.spearman ──────────────────────────────────────────────────

#[test]
fn stats_spearman_missing_y() {
    let resp = stats_spearman(&serde_json::json!({"x": [1.0, 2.0]}), serde_json::json!(20));
    assert_eq!(resp.error.unwrap().code, jsonrpc::INVALID_PARAMS);
}

#[test]
fn stats_spearman_perfect_correlation() {
    let resp = stats_spearman(
        &serde_json::json!({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [10.0, 20.0, 30.0, 40.0, 50.0]}),
        serde_json::json!(21),
    );
    assert!(resp.error.is_none());
    let rho = resp.result.unwrap()["result"].as_f64().unwrap();
    assert!((rho - 1.0).abs() < 1e-10);
}

// ── stats.fit_linear ────────────────────────────────────────────────

#[test]
fn stats_fit_linear_missing_params() {
    let resp = stats_fit_linear(&serde_json::json!({}), serde_json::json!(30));
    assert_eq!(resp.error.unwrap().code, jsonrpc::INVALID_PARAMS);
}

#[test]
fn stats_fit_linear_perfect_line() {
    // y = 2x + 1
    let resp = stats_fit_linear(
        &serde_json::json!({"x": [1.0, 2.0, 3.0, 4.0], "y": [3.0, 5.0, 7.0, 9.0]}),
        serde_json::json!(31),
    );
    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    let slope = result["slope"].as_f64().unwrap();
    let intercept = result["intercept"].as_f64().unwrap();
    let r_sq = result["r_squared"].as_f64().unwrap();
    assert!((slope - 2.0).abs() < 1e-10);
    assert!((intercept - 1.0).abs() < 1e-10);
    assert!((r_sq - 1.0).abs() < 1e-10);
}

#[test]
fn stats_fit_linear_single_point_fails() {
    let resp = stats_fit_linear(
        &serde_json::json!({"x": [1.0], "y": [2.0]}),
        serde_json::json!(32),
    );
    assert_eq!(resp.error.unwrap().code, jsonrpc::INVALID_PARAMS);
}

// ── stats.empirical_spectral_density ────────────────────────────────

#[test]
fn stats_esd_missing_eigenvalues() {
    let resp = stats_empirical_spectral_density(&serde_json::json!({}), serde_json::json!(40));
    assert_eq!(resp.error.unwrap().code, jsonrpc::INVALID_PARAMS);
}

#[test]
fn stats_esd_happy_path() {
    let resp = stats_empirical_spectral_density(
        &serde_json::json!({"eigenvalues": [1.0, 2.0, 3.0, 4.0, 5.0], "n_bins": 3}),
        serde_json::json!(41),
    );
    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    assert_eq!(result["n_bins"].as_u64().unwrap(), 3);
    let density = result["density"].as_array().unwrap();
    assert_eq!(density.len(), 3);
}

// ── spectral.fft ────────────────────────────────────────────────────

#[test]
fn spectral_fft_missing_data() {
    let resp = spectral_fft(&serde_json::json!({}), serde_json::json!(50));
    assert_eq!(resp.error.unwrap().code, jsonrpc::INVALID_PARAMS);
}

#[test]
fn spectral_fft_empty_data() {
    let resp = spectral_fft(&serde_json::json!({"data": []}), serde_json::json!(51));
    assert!(resp.error.unwrap().message.contains("non-empty"));
}

#[test]
fn spectral_fft_dc_signal() {
    // Constant signal [1,1,1,1] -> FFT has DC = 4, rest ≈ 0
    let resp = spectral_fft(
        &serde_json::json!({"data": [1.0, 1.0, 1.0, 1.0]}),
        serde_json::json!(52),
    );
    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    let real = result["real"].as_array().unwrap();
    assert!((real[0].as_f64().unwrap() - 4.0).abs() < 1e-10);
    // Other bins should be ~0
    for r in &real[1..] {
        assert!(r.as_f64().unwrap().abs() < 1e-10);
    }
}

// ── spectral.power_spectrum ─────────────────────────────────────────

#[test]
fn spectral_power_spectrum_missing_data() {
    let resp = spectral_power_spectrum(&serde_json::json!({}), serde_json::json!(60));
    assert_eq!(resp.error.unwrap().code, jsonrpc::INVALID_PARAMS);
}

#[test]
fn spectral_power_spectrum_happy_path() {
    let resp = spectral_power_spectrum(
        &serde_json::json!({"data": [1.0, 0.0, -1.0, 0.0]}),
        serde_json::json!(61),
    );
    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    let psd = result["result"].as_array().unwrap();
    assert_eq!(psd.len(), 4);
    // Parseval: sum(|X[k]|^2/N) = sum(|x[n]|^2) = 1+0+1+0 = 2
    let total: f64 = psd.iter().map(|v| v.as_f64().unwrap()).sum();
    assert!((total - 2.0).abs() < 1e-10);
}

// ── linalg.solve ────────────────────────────────────────────────────

#[test]
fn linalg_solve_missing_matrix() {
    let resp = linalg_solve(&serde_json::json!({"b": [1.0]}), serde_json::json!(70));
    assert_eq!(resp.error.unwrap().code, jsonrpc::INVALID_PARAMS);
}

#[test]
fn linalg_solve_missing_b() {
    let resp = linalg_solve(
        &serde_json::json!({"matrix": [[1.0]]}),
        serde_json::json!(71),
    );
    assert_eq!(resp.error.unwrap().code, jsonrpc::INVALID_PARAMS);
}

#[test]
fn linalg_solve_dimension_mismatch() {
    let resp = linalg_solve(
        &serde_json::json!({"matrix": [[1.0, 2.0], [3.0, 4.0]], "b": [1.0]}),
        serde_json::json!(72),
    );
    assert_eq!(resp.error.unwrap().code, jsonrpc::INVALID_PARAMS);
}

#[test]
fn linalg_solve_identity_system() {
    // I * x = b -> x = b
    let resp = linalg_solve(
        &serde_json::json!({"matrix": [[1.0, 0.0], [0.0, 1.0]], "b": [3.0, 7.0]}),
        serde_json::json!(73),
    );
    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    let x = result["result"].as_array().unwrap();
    assert!((x[0].as_f64().unwrap() - 3.0).abs() < 1e-10);
    assert!((x[1].as_f64().unwrap() - 7.0).abs() < 1e-10);
}

#[test]
fn linalg_solve_2x2_system() {
    // [2 1; 5 7] * x = [11; 13] -> x = [59/9, -1/9] ≈ [6.556, -0.111]
    // Actually: 2x+y=11, 5x+7y=13 -> from first: y=11-2x, sub: 5x+7(11-2x)=13 -> 5x+77-14x=13 -> -9x=-64 -> x=64/9
    // y = 11 - 128/9 = (99-128)/9 = -29/9
    let resp = linalg_solve(
        &serde_json::json!({"matrix": [[2.0, 1.0], [5.0, 7.0]], "b": [11.0, 13.0]}),
        serde_json::json!(74),
    );
    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    let x = result["result"].as_array().unwrap();
    assert!((x[0].as_f64().unwrap() - 64.0 / 9.0).abs() < 1e-10);
    assert!((x[1].as_f64().unwrap() - (-29.0 / 9.0)).abs() < 1e-10);
}

// ── nautilus.* (session lifecycle) ──────────────────────────────────

#[test]
fn nautilus_create_defaults() {
    let resp = nautilus_create(&serde_json::json!({}), serde_json::json!(80));
    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    assert!(
        result["session_id"]
            .as_str()
            .unwrap()
            .starts_with("nautilus-")
    );
    assert_eq!(result["pop_size"].as_u64().unwrap(), 8);
}

#[test]
fn nautilus_create_custom_params() {
    let resp = nautilus_create(
        &serde_json::json!({"name": "test-brain", "pop_size": 4, "generations_per_train": 10}),
        serde_json::json!(81),
    );
    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    assert_eq!(result["name"].as_str().unwrap(), "test-brain");
    assert_eq!(result["pop_size"].as_u64().unwrap(), 4);
}

#[test]
fn nautilus_observe_invalid_session() {
    let resp = nautilus_observe(
        &serde_json::json!({"session_id": "nonexistent", "beta": 1.0, "observables": [1.0]}),
        serde_json::json!(82),
    );
    let err = resp.error.expect("nonexistent session should fail");
    assert!(err.message.contains("session_id") || err.message.contains("Unknown"));
}

#[test]
fn nautilus_train_invalid_session() {
    let resp = nautilus_train(
        &serde_json::json!({"session_id": "nonexistent"}),
        serde_json::json!(83),
    );
    assert!(resp.error.is_some());
}

#[test]
fn nautilus_predict_invalid_session() {
    let resp = nautilus_predict(
        &serde_json::json!({"session_id": "nonexistent", "beta": 1.0}),
        serde_json::json!(84),
    );
    assert!(resp.error.is_some());
}

#[test]
fn nautilus_export_invalid_session() {
    let resp = nautilus_export(
        &serde_json::json!({"session_id": "nonexistent"}),
        serde_json::json!(85),
    );
    assert!(resp.error.is_some());
}

#[test]
fn nautilus_import_missing_data() {
    let resp = nautilus_import(&serde_json::json!({}), serde_json::json!(86));
    assert_eq!(resp.error.unwrap().code, jsonrpc::INVALID_PARAMS);
}

#[test]
fn nautilus_full_lifecycle() {
    // Create
    let resp = nautilus_create(
        &serde_json::json!({"name": "lifecycle-test", "pop_size": 4, "min_observations": 2}),
        serde_json::json!(90),
    );
    assert!(resp.error.is_none());
    let session_id = resp.result.unwrap()["session_id"]
        .as_str()
        .unwrap()
        .to_owned();

    // Observe with physics-specific fields
    for (i, beta) in [0.5, 1.0, 2.0, 4.0].iter().enumerate() {
        let resp = nautilus_observe(
            &serde_json::json!({
                "session_id": session_id,
                "beta": beta,
                "plaquette": 0.5 + beta * 0.1,
                "cg_iters": (i as f64).mul_add(10.0, 100.0),
                "acceptance": 0.8,
                "delta_h_abs": 0.01 * beta,
            }),
            serde_json::json!(91),
        );
        assert!(
            resp.error.is_none(),
            "observe at beta={beta} failed: {:?}",
            resp.error
        );
    }

    // Train (should succeed with >= min_observations)
    let resp = nautilus_train(
        &serde_json::json!({"session_id": session_id}),
        serde_json::json!(92),
    );
    assert!(resp.error.is_none(), "train failed: {:?}", resp.error);
    let result = resp.result.unwrap();
    assert!(result["trained"].as_bool().unwrap());

    // Predict (may return null predictions if model not yet converged)
    let resp = nautilus_predict(
        &serde_json::json!({"session_id": session_id, "beta": 3.0}),
        serde_json::json!(93),
    );
    assert!(resp.error.is_none(), "predict failed: {:?}", resp.error);
    let result = resp.result.unwrap();
    assert_eq!(result["session_id"].as_str().unwrap(), session_id);

    // Export
    let resp = nautilus_export(
        &serde_json::json!({"session_id": session_id}),
        serde_json::json!(94),
    );
    assert!(resp.error.is_none());
    let exported = resp.result.unwrap();
    assert!(exported["brain_json"].is_string());

    // Import (round-trip)
    let resp = nautilus_import(
        &serde_json::json!({"brain_json": exported["brain_json"].as_str().unwrap()}),
        serde_json::json!(95),
    );
    assert!(resp.error.is_none());
    let imported_session = resp.result.unwrap()["session_id"]
        .as_str()
        .unwrap()
        .to_owned();
    assert_ne!(imported_session, session_id);
}

// ── spectral.fft numeric validation ─────────────────────────────────

#[test]
fn spectral_fft_sine_wave() {
    use std::f64::consts::PI;
    let n = 8;
    let data: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * f64::from(i) / f64::from(n)).sin())
        .collect();
    let resp = spectral_fft(&serde_json::json!({"data": data}), serde_json::json!(55));
    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    let real = result["real"].as_array().unwrap();
    let imag = result["imag"].as_array().unwrap();
    // For a pure sine at freq 1, DC should be ~0, bin 1 should have the energy
    assert!(real[0].as_f64().unwrap().abs() < 1e-10);
    // Bin 1 imaginary should be -N/2 = -4
    assert!((imag[1].as_f64().unwrap() + 4.0).abs() < 1e-10);
}

// ── ml.mlp_train ────────────────────────────────────────────────────

#[test]
fn ml_mlp_train_missing_layers() {
    let resp = ml_mlp_train(&serde_json::json!({}), serde_json::json!(100));
    assert_eq!(resp.error.unwrap().code, jsonrpc::INVALID_PARAMS);
}

#[test]
fn ml_mlp_train_empty_layers() {
    let resp = ml_mlp_train(&serde_json::json!({"layers": []}), serde_json::json!(101));
    assert!(resp.error.unwrap().message.contains("non-empty"));
}

#[test]
fn ml_mlp_train_missing_inputs() {
    let resp = ml_mlp_train(
        &serde_json::json!({
            "layers": [{"weights": [[0.1, 0.2]], "biases": [0.0], "activation": "relu"}],
        }),
        serde_json::json!(102),
    );
    assert_eq!(resp.error.unwrap().code, jsonrpc::INVALID_PARAMS);
}

#[test]
fn ml_mlp_train_simple_identity() {
    // Train a 1->1 MLP on identity function (trivial convergence)
    let resp = ml_mlp_train(
        &serde_json::json!({
            "layers": [
                {"weights": [[0.5]], "biases": [0.0], "activation": "identity"}
            ],
            "inputs": [[1.0], [2.0], [3.0], [4.0]],
            "targets": [[1.0], [2.0], [3.0], [4.0]],
            "learning_rate": 0.01,
            "epochs": 200
        }),
        serde_json::json!(103),
    );
    assert!(resp.error.is_none(), "mlp_train failed: {:?}", resp.error);
    let result = resp.result.unwrap();
    assert!(result["mse"].is_number());
    assert!(result["layers"].is_array());
    assert_eq!(result["epochs"].as_u64().unwrap(), 200);
}

#[test]
fn ml_mlp_train_unknown_activation() {
    let resp = ml_mlp_train(
        &serde_json::json!({
            "layers": [{"weights": [[1.0]], "biases": [0.0], "activation": "swish"}],
            "inputs": [[1.0]],
            "targets": [[1.0]],
        }),
        serde_json::json!(104),
    );
    let err = resp.error.unwrap();
    assert!(err.message.contains("activation"));
}

#[test]
fn ml_mlp_train_shape_mismatch() {
    let resp = ml_mlp_train(
        &serde_json::json!({
            "layers": [{"weights": [[1.0, 2.0]], "biases": [0.0, 0.0, 0.0], "activation": "relu"}],
            "inputs": [[1.0]],
            "targets": [[1.0]],
        }),
        serde_json::json!(105),
    );
    assert_eq!(resp.error.unwrap().code, jsonrpc::INVALID_PARAMS);
}

// ── ml.esn_predict ──────────────────────────────────────────────────

#[test]
fn ml_esn_predict_missing_weights() {
    let resp = ml_esn_predict(&serde_json::json!({"input": [1.0]}), serde_json::json!(110));
    assert_eq!(resp.error.unwrap().code, jsonrpc::INVALID_PARAMS);
}

#[test]
fn ml_esn_predict_missing_input() {
    let resp = ml_esn_predict(
        &serde_json::json!({"weights_json": "{}"}),
        serde_json::json!(111),
    );
    assert_eq!(resp.error.unwrap().code, jsonrpc::INVALID_PARAMS);
}

#[test]
fn ml_esn_predict_invalid_weights_json() {
    let resp = ml_esn_predict(
        &serde_json::json!({"weights_json": "not valid json", "input": [1.0]}),
        serde_json::json!(112),
    );
    let err = resp.error.unwrap();
    assert!(err.message.contains("parse") || err.message.contains("Failed"));
}

// ── tensor.matmul_inline ────────────────────────────────────────────

#[test]
fn tensor_matmul_inline_missing_params() {
    use super::super::tensor::tensor_matmul_inline;
    let resp = tensor_matmul_inline(&serde_json::json!({}), serde_json::json!(200));
    let err = resp.error.unwrap();
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("lhs") || err.message.contains("rhs"));
}

#[test]
fn tensor_matmul_inline_empty_lhs() {
    use super::super::tensor::tensor_matmul_inline;
    let resp = tensor_matmul_inline(
        &serde_json::json!({"lhs": [], "rhs": [[1.0]]}),
        serde_json::json!(201),
    );
    let err = resp.error.unwrap();
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("non-empty"));
}

#[test]
fn tensor_matmul_inline_shape_mismatch() {
    use super::super::tensor::tensor_matmul_inline;
    let resp = tensor_matmul_inline(
        &serde_json::json!({"lhs": [[1.0, 2.0]], "rhs": [[1.0], [2.0], [3.0]]}),
        serde_json::json!(202),
    );
    let err = resp.error.unwrap();
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("Shape mismatch") || err.message.contains("shape"));
}

#[test]
fn tensor_matmul_inline_happy_path_2x2() {
    use super::super::tensor::tensor_matmul_inline;
    let resp = tensor_matmul_inline(
        &serde_json::json!({
            "lhs": [[1.0, 2.0], [3.0, 4.0]],
            "rhs": [[5.0, 6.0], [7.0, 8.0]]
        }),
        serde_json::json!(203),
    );
    assert!(resp.error.is_none(), "Expected success: {:?}", resp.error);
    let result = resp.result.unwrap();
    let mat = result["result"].as_array().unwrap();
    assert_eq!(mat.len(), 2);
    let row0: Vec<f64> = mat[0]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let row1: Vec<f64> = mat[1]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert_eq!(row0, vec![19.0, 22.0]);
    assert_eq!(row1, vec![43.0, 50.0]);
    assert_eq!(result["shape"], serde_json::json!([2, 2]));
}

#[test]
fn tensor_matmul_inline_non_square() {
    use super::super::tensor::tensor_matmul_inline;
    let resp = tensor_matmul_inline(
        &serde_json::json!({
            "lhs": [[1.0, 0.0, 2.0]],
            "rhs": [[1.0], [0.0], [3.0]]
        }),
        serde_json::json!(204),
    );
    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    let mat = result["result"].as_array().unwrap();
    assert_eq!(mat.len(), 1);
    let row0: Vec<f64> = mat[0]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert_eq!(row0, vec![7.0]);
    assert_eq!(result["shape"], serde_json::json!([1, 1]));
}

// ── linalg.graph_laplacian ──────────────────────────────────────────

#[test]
fn graph_laplacian_missing_adjacency() {
    use super::super::graph::linalg_graph_laplacian;
    let resp = linalg_graph_laplacian(&serde_json::json!({"n": 2}), serde_json::json!(210));
    let err = resp.error.unwrap();
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("adjacency"));
}

#[test]
fn graph_laplacian_missing_n() {
    use super::super::graph::linalg_graph_laplacian;
    let resp = linalg_graph_laplacian(
        &serde_json::json!({"adjacency": [0.0, 1.0, 1.0, 0.0]}),
        serde_json::json!(211),
    );
    let err = resp.error.unwrap();
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains('n'));
}

#[test]
fn graph_laplacian_size_mismatch() {
    use super::super::graph::linalg_graph_laplacian;
    let resp = linalg_graph_laplacian(
        &serde_json::json!({"adjacency": [0.0, 1.0, 1.0], "n": 2}),
        serde_json::json!(212),
    );
    let err = resp.error.unwrap();
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("length") || err.message.contains("n*n"));
}

#[test]
fn graph_laplacian_happy_path_triangle() {
    use super::super::graph::linalg_graph_laplacian;
    #[rustfmt::skip]
    let adj = vec![
        0.0, 1.0, 1.0,
        1.0, 0.0, 1.0,
        1.0, 1.0, 0.0,
    ];
    let resp = linalg_graph_laplacian(
        &serde_json::json!({"adjacency": adj, "n": 3}),
        serde_json::json!(213),
    );
    assert!(resp.error.is_none(), "Expected success: {:?}", resp.error);
    let result = resp.result.unwrap();
    let rows = result["result"].as_array().unwrap();
    assert_eq!(rows.len(), 3);
    let row0: Vec<f64> = rows[0]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert_eq!(row0, vec![2.0, -1.0, -1.0]);
    let row1: Vec<f64> = rows[1]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert_eq!(row1, vec![-1.0, 2.0, -1.0]);
    assert_eq!(result["n"], 3);
}
