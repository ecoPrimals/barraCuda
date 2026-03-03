//! Basic unit tests: Fused Map-Reduce and Kriging primitives.

use super::*;
use barracuda::ops::fused_map_reduce_f64::FusedMapReduceF64;
use barracuda::ops::kriging_f64::{KrigingF64, VariogramModel};

mod fused_map_reduce_unit {
    use super::*;

    #[test]
    fn test_shannon_wetspring_reference() {
        let device = match create_device_sync() {
            Some(d) => d,
            None => {
                println!("SKIP: No f64-capable device");
                return;
            }
        };

        let fmr = FusedMapReduceF64::new(device).unwrap();
        let counts = vec![10.0, 20.0, 30.0, 40.0];
        let result = fmr.shannon_entropy(&counts).unwrap();
        let expected = cpu_shannon(&counts);
        let error = (result - expected).abs();
        assert!(
            error < 1e-10,
            "Shannon entropy error {} exceeds 1e-10 (got {}, expected {})",
            error,
            result,
            expected
        );
        println!("✓ Shannon wetSpring reference: {} (error: {:.2e})", result, error);
    }

    #[test]
    fn test_shannon_uniform_distribution() {
        let device = match create_device_sync() {
            Some(d) => d,
            None => return,
        };
        let fmr = FusedMapReduceF64::new(device).unwrap();
        let counts = vec![25.0; 4];
        let result = fmr.shannon_entropy(&counts).unwrap();
        let expected = 4.0_f64.ln();
        let error = (result - expected).abs();
        assert!(error < 1e-10, "Uniform Shannon error: {}", error);
        println!("✓ Shannon uniform: {} (expected ln(4) = {})", result, expected);
    }

    #[test]
    fn test_shannon_single_element() {
        let device = match create_device_sync() {
            Some(d) => d,
            None => return,
        };
        let fmr = FusedMapReduceF64::new(device).unwrap();
        let counts = vec![100.0];
        let result = fmr.shannon_entropy(&counts).unwrap();
        assert!(
            result.abs() < 1e-10,
            "Single element Shannon should be 0, got {}",
            result
        );
        println!("✓ Shannon single element: {} (expected 0)", result);
    }

    #[test]
    fn test_simpson_basic() {
        let device = match create_device_sync() {
            Some(d) => d,
            None => return,
        };
        let fmr = FusedMapReduceF64::new(device).unwrap();
        let counts = vec![10.0, 20.0, 30.0, 40.0];
        let result = fmr.simpson_index(&counts).unwrap();
        let expected = cpu_simpson(&counts);
        let error = (result - expected).abs();
        assert!(error < 1e-12, "Simpson error: {}", error);
        println!("✓ Simpson index: {} (error: {:.2e})", result, error);
    }

    #[test]
    fn test_simpson_uniform() {
        let device = match create_device_sync() {
            Some(d) => d,
            None => return,
        };
        let fmr = FusedMapReduceF64::new(device).unwrap();
        let counts = vec![25.0; 4];
        let result = fmr.simpson_index(&counts).unwrap();
        let expected = 0.25;
        let error = (result - expected).abs();
        assert!(error < 1e-12, "Uniform Simpson error: {}", error);
        println!("✓ Simpson uniform: {} (expected 0.25)", result);
    }

    #[test]
    fn test_sum_reduction() {
        let device = match create_device_sync() {
            Some(d) => d,
            None => return,
        };
        let fmr = FusedMapReduceF64::new(device).unwrap();
        let data: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let result = fmr.sum(&data).unwrap();
        let expected: f64 = data.iter().sum();
        let error = (result - expected).abs();
        assert!(error < 1e-10, "Sum error: {}", error);
        println!("✓ Sum 1..100: {} (expected {})", result, expected);
    }

    #[test]
    fn test_max_reduction() {
        let device = match create_device_sync() {
            Some(d) => d,
            None => return,
        };
        let fmr = FusedMapReduceF64::new(device).unwrap();
        let data = vec![1.0, 5.0, 3.0, 9.0, 2.0, 7.0];
        let result = fmr.max(&data).unwrap();
        assert!(
            (result - 9.0).abs() < 1e-10,
            "Max should be 9.0, got {}",
            result
        );
        println!("✓ Max: {} (expected 9.0)", result);
    }

    #[test]
    fn test_min_reduction() {
        let device = match create_device_sync() {
            Some(d) => d,
            None => return,
        };
        let fmr = FusedMapReduceF64::new(device).unwrap();
        let data = vec![5.0, 3.0, 9.0, 1.0, 7.0];
        let result = fmr.min(&data).unwrap();
        assert!(
            (result - 1.0).abs() < 1e-10,
            "Min should be 1.0, got {}",
            result
        );
        println!("✓ Min: {} (expected 1.0)", result);
    }

    #[test]
    fn test_sum_of_squares() {
        let device = match create_device_sync() {
            Some(d) => d,
            None => return,
        };
        let fmr = FusedMapReduceF64::new(device).unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = fmr.sum_of_squares(&data).unwrap();
        let expected: f64 = data.iter().map(|x| x * x).sum();
        let error = (result - expected).abs();
        assert!(error < 1e-10, "Sum of squares error: {}", error);
        println!("✓ Sum of squares: {} (expected {})", result, expected);
    }
}

mod kriging_unit {
    use super::*;

    #[test]
    fn test_variogram_spherical() {
        let model = VariogramModel::Spherical {
            nugget: 0.0,
            sill: 1.0,
            range: 10.0,
        };
        assert!((model.gamma(0.0) - 0.0).abs() < 1e-10, "γ(0) should be 0");
        assert!(
            (model.gamma(10.0) - 1.0).abs() < 1e-10,
            "γ(range) should equal sill"
        );
        assert!(
            (model.gamma(20.0) - 1.0).abs() < 1e-10,
            "γ(h > range) should equal sill"
        );
        println!("✓ Spherical variogram: γ(0)=0, γ(10)=1, γ(20)=1");
    }

    #[test]
    fn test_variogram_exponential() {
        let model = VariogramModel::Exponential {
            nugget: 0.1,
            sill: 1.0,
            range: 10.0,
        };
        assert!(model.gamma(0.0).abs() < 1e-10, "γ(0) should be 0");
        let gamma_at_range = model.gamma(10.0);
        let expected = 0.1 + 0.9 * (1.0 - (-3.0_f64).exp());
        assert!(
            (gamma_at_range - expected).abs() < 1e-6,
            "Exponential γ(a) incorrect: {} vs {}",
            gamma_at_range,
            expected
        );
        println!("✓ Exponential variogram: γ(0)=0, γ(10)≈{:.4}", gamma_at_range);
    }

    #[test]
    fn test_kriging_at_known_point() {
        let device = match create_device_sync() {
            Some(d) => d,
            None => return,
        };
        let kriging = KrigingF64::new(device).unwrap();
        let known = vec![
            (0.0, 0.0, 1.0),
            (10.0, 0.0, 2.0),
            (0.0, 10.0, 3.0),
            (10.0, 10.0, 4.0),
        ];
        let targets = vec![(0.0, 0.0)];
        let model = VariogramModel::Spherical {
            nugget: 0.0,
            sill: 1.0,
            range: 15.0,
        };
        let result = kriging.interpolate(&known, &targets, model).unwrap();
        assert!(
            (result.values[0] - 1.0).abs() < 1e-6,
            "Kriging at known point should reproduce value: got {} expected 1.0",
            result.values[0]
        );
        assert!(
            result.variances[0] < 0.01,
            "Variance at known point should be ~0: got {}",
            result.variances[0]
        );
        println!(
            "✓ Kriging at known point: value={:.6}, variance={:.6}",
            result.values[0], result.variances[0]
        );
    }

    #[test]
    fn test_kriging_center_interpolation() {
        let device = match create_device_sync() {
            Some(d) => d,
            None => return,
        };
        let kriging = KrigingF64::new(device).unwrap();
        let known = vec![
            (0.0, 0.0, 1.0),
            (10.0, 0.0, 1.0),
            (0.0, 10.0, 1.0),
            (10.0, 10.0, 1.0),
        ];
        let targets = vec![(5.0, 5.0)];
        let model = VariogramModel::Spherical {
            nugget: 0.0,
            sill: 0.5,
            range: 15.0,
        };
        let result = kriging.interpolate(&known, &targets, model).unwrap();
        assert!(
            (result.values[0] - 1.0).abs() < 0.01,
            "Center interpolation should be ~1.0: got {}",
            result.values[0]
        );
        println!(
            "✓ Kriging center: value={:.6}, variance={:.6}",
            result.values[0], result.variances[0]
        );
    }

    #[test]
    fn test_kriging_linear_trend() {
        let device = match create_device_sync() {
            Some(d) => d,
            None => return,
        };
        let kriging = KrigingF64::new(device).unwrap();
        let known = vec![
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 1.0),
            (0.0, 10.0, 1.0),
            (10.0, 10.0, 2.0),
        ];
        let targets = vec![(5.0, 5.0)];
        let model = VariogramModel::Gaussian {
            nugget: 0.0,
            sill: 0.5,
            range: 15.0,
        };
        let result = kriging.interpolate(&known, &targets, model).unwrap();
        assert!(
            (result.values[0] - 1.0).abs() < 0.15,
            "Linear trend center should be ~1.0: got {}",
            result.values[0]
        );
        println!(
            "✓ Kriging linear trend: value={:.6} (expected ~1.0)",
            result.values[0]
        );
    }

    #[test]
    fn test_simple_kriging() {
        let device = match create_device_sync() {
            Some(d) => d,
            None => return,
        };
        let kriging = KrigingF64::new(device).unwrap();
        let known = vec![
            (0.0, 0.0, 10.0),
            (10.0, 0.0, 12.0),
            (0.0, 10.0, 11.0),
            (10.0, 10.0, 13.0),
        ];
        let targets = vec![(5.0, 5.0)];
        let model = VariogramModel::Exponential {
            nugget: 0.0,
            sill: 2.0,
            range: 10.0,
        };
        let mean = 11.5;
        let result = kriging
            .interpolate_simple(&known, &targets, model, mean)
            .unwrap();
        assert!(
            result.values[0] > 10.0 && result.values[0] < 14.0,
            "Simple kriging value out of range: {}",
            result.values[0]
        );
        println!(
            "✓ Simple kriging: value={:.6}, variance={:.6}",
            result.values[0], result.variances[0]
        );
    }

    #[test]
    fn test_variogram_fitting() {
        let _device = match create_device_sync() {
            Some(d) => d,
            None => return,
        };
        let known: Vec<(f64, f64, f64)> = (0..10)
            .flat_map(|i| {
                (0..10).map(move |j| {
                    let x = i as f64 * 2.0;
                    let y = j as f64 * 2.0;
                    let z = x * 0.1 + y * 0.1 + (x * y * 0.01);
                    (x, y, z)
                })
            })
            .collect();
        let (lag_distances, lag_semivariances) =
            KrigingF64::fit_variogram(&known, 10, 20.0).unwrap();
        assert_eq!(lag_distances.len(), 10);
        assert_eq!(lag_semivariances.len(), 10);
        let non_empty_lags: Vec<_> = lag_semivariances.iter().filter(|&&v| v > 0.0).collect();
        assert!(!non_empty_lags.is_empty(), "Should have computed some lag values");
        println!(
            "✓ Variogram fitting: {} lags computed, {} non-empty",
            lag_distances.len(),
            non_empty_lags.len()
        );
    }
}
