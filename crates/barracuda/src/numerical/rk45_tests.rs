// SPDX-License-Identifier: AGPL-3.0-or-later
use super::*;

#[test]
fn test_exponential_decay() {
    let f = |_t: f64, y: &[f64]| vec![-y[0]];
    let config = Rk45Config::new(1e-8, 1e-10);

    let result = rk45_solve(&f, 0.0, 1.0, &[1.0], &config).unwrap();

    let expected = (-1.0_f64).exp();
    assert!(
        (result.y_final[0] - expected).abs() < 1e-6,
        "y(1) = {}, expected {}",
        result.y_final[0],
        expected
    );
}

#[test]
fn test_exponential_growth() {
    let f = |_t: f64, y: &[f64]| vec![y[0]];
    let config = Rk45Config::new(1e-8, 1e-10);

    let result = rk45_solve(&f, 0.0, 2.0, &[1.0], &config).unwrap();

    let expected = 2.0_f64.exp();
    assert!(
        (result.y_final[0] - expected).abs() < 1e-5,
        "y(2) = {}, expected {}",
        result.y_final[0],
        expected
    );
}

#[test]
fn test_harmonic_oscillator() {
    let f = |_t: f64, y: &[f64]| vec![y[1], -y[0]];
    let config = Rk45Config::new(1e-8, 1e-10);

    let result = rk45_solve(&f, 0.0, std::f64::consts::PI, &[1.0, 0.0], &config).unwrap();

    assert!(
        (result.y_final[0] - (-1.0)).abs() < 1e-5,
        "x(π) = {}, expected -1",
        result.y_final[0]
    );
    assert!(
        result.y_final[1].abs() < 1e-5,
        "v(π) = {}, expected 0",
        result.y_final[1]
    );
}

#[test]
fn test_lotka_volterra() {
    let alpha = 1.5;
    let beta = 1.0;
    let delta = 1.0;
    let gamma = 3.0;

    let f = move |_t: f64, y: &[f64]| {
        let x = y[0];
        let prey = y[1];
        vec![alpha * x - beta * x * prey, delta * x * prey - gamma * prey]
    };

    let config = Rk45Config::new(1e-6, 1e-8);
    let result = rk45_solve(&f, 0.0, 10.0, &[10.0, 5.0], &config).unwrap();

    assert!(result.y_final[0] > 0.0);
    assert!(result.y_final[1] > 0.0);
    assert!(result.y_final[0] < 1e6);
    assert!(result.y_final[1] < 1e6);
}

#[test]
fn test_step_rejection() {
    let f = |_t: f64, y: &[f64]| vec![-50.0 * y[0]];
    let config = Rk45Config::new(1e-8, 1e-10).with_h_init(1.0);

    let result = rk45_solve(&f, 0.0, 0.1, &[1.0], &config).unwrap();

    assert!(result.n_steps > 0);

    let expected = (-50.0 * 0.1_f64).exp();
    assert!((result.y_final[0] - expected).abs() < 1e-6);
}

#[test]
fn test_history() {
    let f = |_t: f64, y: &[f64]| vec![-y[0]];
    let config = Rk45Config::new(1e-6, 1e-9);

    let result = rk45_solve(&f, 0.0, 1.0, &[1.0], &config).unwrap();

    assert!(!result.t_history.is_empty());
    assert!(!result.y_history.is_empty());
    assert_eq!(result.t_history.len(), result.y_history.len());

    assert!((result.t_history[0] - 0.0).abs() < 1e-14);
    assert!((result.t_history.last().unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_rk45_at() {
    let f = |_t: f64, y: &[f64]| vec![-y[0]];
    let config = Rk45Config::default();

    let y = rk45_at(&f, 0.0, 1.0, &[1.0], &config).unwrap();

    assert_eq!(y.len(), 1);
    assert!((y[0] - (-1.0_f64).exp()).abs() < 1e-5);
}

#[test]
fn test_config_builder() {
    let config = Rk45Config::new(1e-4, 1e-6)
        .with_h_init(0.001)
        .with_step_bounds(1e-10, 0.5);

    assert!((config.rtol - 1e-4).abs() < 1e-14);
    assert!((config.h_init - 0.001).abs() < 1e-14);
    assert!((config.h_min - 1e-10).abs() < 1e-14);
    assert!((config.h_max - 0.5).abs() < 1e-14);
}

#[test]
fn test_max_steps_exceeded() {
    let f = |_t: f64, y: &[f64]| vec![-y[0]];
    let config = Rk45Config::new(1e-12, 1e-14)
        .with_max_steps(5)
        .with_h_init(0.001);

    let result = rk45_solve(&f, 0.0, 10.0, &[1.0], &config);
    assert!(result.is_err());
    if let Err(BarracudaError::Numerical { message }) = result {
        assert!(message.contains("Max steps"));
    }
}

#[test]
fn test_config_default() {
    let config = Rk45Config::default();
    assert!((config.rtol - 1e-6).abs() < 1e-14);
    assert!((config.atol - 1e-9).abs() < 1e-14);
    assert!((config.h_init - 0.01).abs() < 1e-14);
    assert!((config.h_min - 1e-12).abs() < 1e-14);
    assert!((config.h_max - 1.0).abs() < 1e-14);
    assert!((config.safety - 0.9).abs() < 1e-14);
    assert_eq!(config.max_steps, 100_000);
}

#[test]
fn test_config_with_safety() {
    let config = Rk45Config::default().with_safety(0.8);
    assert!((config.safety - 0.8).abs() < 1e-14);
}

#[test]
fn test_config_with_max_steps() {
    let config = Rk45Config::default().with_max_steps(500);
    assert_eq!(config.max_steps, 500);
}

#[test]
fn test_linear_ode() {
    let f = |t: f64, _y: &[f64]| vec![2.0 * t];
    let config = Rk45Config::new(1e-8, 1e-10);

    let result = rk45_solve(&f, 0.0, 1.0, &[0.0], &config).unwrap();
    let expected = 1.0;
    assert!(
        (result.y_final[0] - expected).abs() < 1e-6,
        "y(1) = {}, expected {}",
        result.y_final[0],
        expected
    );
}

#[test]
fn test_multi_dimension() {
    let f = |_t: f64, y: &[f64]| vec![-y[0], -2.0 * y[1], -3.0 * y[2]];
    let config = Rk45Config::new(1e-8, 1e-10);

    let result = rk45_solve(&f, 0.0, 1.0, &[1.0, 1.0, 1.0], &config).unwrap();

    assert!((result.y_final[0] - (-1.0_f64).exp()).abs() < 1e-6);
    assert!((result.y_final[1] - (-2.0_f64).exp()).abs() < 1e-6);
    assert!((result.y_final[2] - (-3.0_f64).exp()).abs() < 1e-6);
}

#[test]
fn test_zero_interval_error() {
    let f = |_t: f64, y: &[f64]| vec![y[0]];
    let config = Rk45Config::default();

    let result = rk45_solve(&f, 0.0, 0.0, &[1.0], &config);
    assert!(result.is_err());
}

#[test]
fn test_backward_integration_error() {
    let f = |_t: f64, y: &[f64]| vec![-y[0]];
    let config = Rk45Config::new(1e-8, 1e-10);

    let result = rk45_solve(&f, 1.0, 0.0, &[(-1.0_f64).exp()], &config);
    assert!(result.is_err());
}

#[test]
fn test_variable_trajectory_3d() {
    let f = |_t: f64, y: &[f64]| vec![-y[0], -2.0 * y[1], -3.0 * y[2]];
    let config = Rk45Config::new(1e-8, 1e-10);

    let result = rk45_solve(&f, 0.0, 1.0, &[1.0, 1.0, 1.0], &config).unwrap();

    assert_eq!(result.n_vars(), 3);

    let traj_0 = result.variable_trajectory(0);
    let traj_1 = result.variable_trajectory(1);
    let traj_2 = result.variable_trajectory(2);

    assert_eq!(traj_0.len(), result.y_history.len());
    assert_eq!(traj_1.len(), result.y_history.len());
    assert_eq!(traj_2.len(), result.y_history.len());

    assert!((traj_0[0] - 1.0).abs() < 1e-10, "initial y0=1.0");

    if let Some(&last) = traj_0.last() {
        assert!((last - (-1.0_f64).exp()).abs() < 1e-6);
    }
    if let Some(&last) = traj_1.last() {
        assert!((last - (-2.0_f64).exp()).abs() < 1e-6);
    }
}

#[test]
fn test_variable_trajectory_out_of_bounds() {
    let f = |_t: f64, y: &[f64]| vec![-y[0]];
    let config = Rk45Config::new(1e-8, 1e-10);

    let result = rk45_solve(&f, 0.0, 1.0, &[1.0], &config).unwrap();

    assert_eq!(result.n_vars(), 1);
    assert!(!result.variable_trajectory(0).is_empty());
    assert!(result.variable_trajectory(5).is_empty());
}
