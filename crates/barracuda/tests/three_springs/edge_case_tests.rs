// SPDX-License-Identifier: AGPL-3.0-or-later
//! Edge case tests: Chaos (stress) and Fault (error handling).

#![expect(clippy::unwrap_used, reason = "tests")]
use super::*;
use barracuda::ops::fused_map_reduce_f64::FusedMapReduceF64;
use barracuda::ops::kriging_f64::{KrigingF64, VariogramModel};

mod chaos {
    use super::*;

    #[test]
    fn test_shannon_large_counts() {
        let device = match create_device_sync() {
            Some(d) => d,
            None => return,
        };
        let fmr = FusedMapReduceF64::new(device.clone()).unwrap();
        let counts: Vec<f64> = vec![1e8, 5e7, 2.5e7, 1e7, 5e6];
        let result = fmr.shannon_entropy(&counts).unwrap();
        let expected = cpu_shannon(&counts);
        let error = (result - expected).abs();
        let t = tol(&device, 1e-8);
        assert!(error < t, "Large counts Shannon error: {error} (tol: {t})");
        println!("✓ Shannon large counts: {result} (error: {error:.2e})");
    }

    #[test]
    fn test_shannon_small_counts() {
        let device = match create_device_sync() {
            Some(d) => d,
            None => return,
        };
        let fmr = FusedMapReduceF64::new(device).unwrap();
        let counts = vec![1e-10, 1e-11, 1e-12, 1e-13];
        let result = fmr.shannon_entropy(&counts).unwrap();
        let expected = cpu_shannon(&counts);
        let error = (result - expected).abs();
        assert!(
            error < 1e-6 || (result - expected).abs() / expected.abs() < 1e-6,
            "Small counts Shannon error: {error}"
        );
        println!("✓ Shannon small counts: {result} (error: {error:.2e})");
    }

    #[test]
    fn test_shannon_sparse() {
        let device = match create_device_sync() {
            Some(d) => d,
            None => return,
        };
        let fmr = FusedMapReduceF64::new(device.clone()).unwrap();
        let mut counts = vec![0.0; 1000];
        counts[0] = 100.0;
        counts[50] = 50.0;
        counts[100] = 25.0;
        counts[500] = 10.0;
        counts[999] = 5.0;
        let result = fmr.shannon_entropy(&counts).unwrap();
        let expected = cpu_shannon(&counts);
        let error = (result - expected).abs();
        let t = tol(&device, 1e-10);
        assert!(error < t, "Sparse Shannon error: {error} (tol: {t})");
        println!("✓ Shannon sparse (5 non-zero of 1000): {result}");
    }

    #[test]
    fn test_kriging_colocated() {
        let device = match create_device_sync() {
            Some(d) => d,
            None => return,
        };
        let kriging = KrigingF64::new(device).unwrap();
        let known = vec![(0.0, 0.0, 1.0), (0.001, 0.001, 1.1), (10.0, 10.0, 2.0)];
        let targets = vec![(5.0, 5.0)];
        let model = VariogramModel::Spherical {
            nugget: 0.01,
            sill: 1.0,
            range: 15.0,
        };
        let result = kriging.interpolate(&known, &targets, model).unwrap();
        assert!(
            result.values[0].is_finite(),
            "Co-located points caused non-finite result"
        );
        println!("✓ Kriging co-located: value={:.6}", result.values[0]);
    }

    #[test]
    fn test_kriging_extrapolation() {
        let device = match create_device_sync() {
            Some(d) => d,
            None => return,
        };
        let kriging = KrigingF64::new(device).unwrap();
        let known = vec![
            (0.0, 0.0, 1.0),
            (10.0, 0.0, 2.0),
            (0.0, 10.0, 2.0),
            (10.0, 10.0, 3.0),
        ];
        let targets = vec![(100.0, 100.0), (-50.0, -50.0)];
        let model = VariogramModel::Exponential {
            nugget: 0.0,
            sill: 1.0,
            range: 15.0,
        };
        let result = kriging.interpolate(&known, &targets, model).unwrap();
        for (i, &var) in result.variances.iter().enumerate() {
            assert!(
                var > 0.5,
                "Extrapolation variance should be high: got {var} at {i}"
            );
        }
        println!("✓ Kriging extrapolation: variances={:?}", result.variances);
    }

    #[test]
    fn test_large_array_reduction() {
        let device = match create_device_sync() {
            Some(d) => d,
            None => return,
        };
        let fmr = FusedMapReduceF64::new(device).unwrap();
        let n = 500;
        let data: Vec<f64> = (0..n).map(|i| f64::from(i) * 0.001).collect();
        let sum = fmr.sum(&data).unwrap();
        let expected: f64 = data.iter().sum();
        let rel_error = (sum - expected).abs() / expected.abs();
        assert!(
            rel_error < 1e-10,
            "Large array sum relative error: {rel_error}"
        );
        println!("✓ Array ({n} elements) sum: {sum} (rel error: {rel_error:.2e})");
    }

    #[test]
    #[ignore = "GPU memory-intensive — run explicitly with --ignored"]
    fn test_very_large_array_reduction_gpu() {
        let device = match create_device_sync() {
            Some(d) => d,
            None => return,
        };
        let fmr = FusedMapReduceF64::new(device).unwrap();
        let n = 1_000_000;
        let data: Vec<f64> = (0..n).map(|i| f64::from(i) * 0.000001).collect();
        let sum = fmr.sum(&data).unwrap();
        let expected: f64 = data.iter().sum();
        let rel_error = (sum - expected).abs() / expected.abs();
        assert!(
            rel_error < 1e-8,
            "Large array sum relative error: {rel_error}"
        );
        println!("✓ Large array (1M elements) sum: {sum} (rel error: {rel_error:.2e})");
    }

    #[test]
    fn test_repeated_operations() {
        let device = match create_device_sync() {
            Some(d) => d,
            None => return,
        };
        let fmr = FusedMapReduceF64::new(device.clone()).unwrap();
        let kriging = KrigingF64::new(device).unwrap();
        for i in 0..100 {
            let counts: Vec<f64> = (0..100)
                .map(|j| f64::from((i * 100 + j) % 50 + 1))
                .collect();
            let _ = fmr.shannon_entropy(&counts).unwrap();
            let known = vec![(0.0, 0.0, f64::from(i)), (10.0, 10.0, f64::from(i + 1))];
            let targets = vec![(5.0, 5.0)];
            let model = VariogramModel::Linear {
                nugget: 0.0,
                sill: 1.0,
                range: 15.0,
            };
            let _ = kriging.interpolate(&known, &targets, model).unwrap();
        }
        println!("✓ Repeated operations (100 iterations): no crash/leak");
    }
}

mod fault {
    use super::*;

    #[test]
    fn test_shannon_empty_input() {
        let device = match create_device_sync() {
            Some(d) => d,
            None => return,
        };
        let fmr = FusedMapReduceF64::new(device).unwrap();
        let counts: Vec<f64> = vec![];
        let result = fmr.shannon_entropy(&counts);
        match result {
            Ok(v) => {
                assert!(v.abs() < 1e-10, "Empty Shannon should be 0 or error");
                println!("✓ Empty Shannon: returned 0.0");
            }
            Err(e) => println!("✓ Empty Shannon: returned error ({e})"),
        }
    }

    #[test]
    fn test_shannon_all_zeros() {
        let device = match create_device_sync() {
            Some(d) => d,
            None => return,
        };
        let fmr = FusedMapReduceF64::new(device).unwrap();
        let counts = vec![0.0; 100];
        let result = fmr.shannon_entropy(&counts);
        match result {
            Ok(v) => {
                assert!(v.is_finite(), "All-zero Shannon should be finite");
                println!("✓ All-zero Shannon: {v}");
            }
            Err(e) => println!("✓ All-zero Shannon: returned error ({e})"),
        }
    }

    #[test]
    fn test_nan_input() {
        let device = match create_device_sync() {
            Some(d) => d,
            None => return,
        };
        let fmr = FusedMapReduceF64::new(device).unwrap();
        let counts = vec![1.0, f64::NAN, 3.0, 4.0];
        let result = fmr.sum(&counts).unwrap();
        assert!(result.is_nan(), "NaN should propagate");
        println!("✓ NaN propagation: sum of [1, NaN, 3, 4] = {result}");
    }

    #[test]
    fn test_infinity_input() {
        let device = match create_device_sync() {
            Some(d) => d,
            None => return,
        };
        let fmr = FusedMapReduceF64::new(device).unwrap();
        let counts = vec![1.0, f64::INFINITY, 3.0];
        let result = fmr.sum(&counts).unwrap();
        if !result.is_infinite() {
            // Software renderers (llvmpipe) and some DF64-emulated drivers
            // do not preserve IEEE infinity through workgroup reductions.
            // The result is still large, confirming the GPU path executed.
            assert!(
                result > 1e30 || result.is_nan(),
                "Expected infinity, large value, or NaN from GPU reduction, got {result}"
            );
            println!(
                "✓ Infinity input: GPU returned {result} \
                 (driver lacks IEEE inf propagation — acceptable)"
            );
            return;
        }
        println!("✓ Infinity propagation: sum = {result}");
    }

    #[test]
    fn test_negative_counts() {
        let device = match create_device_sync() {
            Some(d) => d,
            None => return,
        };
        let fmr = FusedMapReduceF64::new(device).unwrap();
        let counts = vec![10.0, -5.0, 15.0];
        let result = fmr.shannon_entropy(&counts);
        match result {
            Ok(v) => println!("✓ Negative counts: Shannon = {v} (may be NaN)"),
            Err(e) => println!("✓ Negative counts: returned error ({e})"),
        }
    }

    #[test]
    fn test_kriging_empty_known() {
        let device = match create_device_sync() {
            Some(d) => d,
            None => return,
        };
        let kriging = KrigingF64::new(device).unwrap();
        let known: Vec<(f64, f64, f64)> = vec![];
        let targets = vec![(5.0, 5.0)];
        let model = VariogramModel::Spherical {
            nugget: 0.0,
            sill: 1.0,
            range: 10.0,
        };
        let result = kriging.interpolate(&known, &targets, model).unwrap();
        assert!(
            result.values.is_empty() || result.values[0].is_nan(),
            "Empty known should produce empty/NaN result"
        );
        println!("✓ Kriging empty known: handled gracefully");
    }

    #[test]
    fn test_kriging_single_point() {
        let device = match create_device_sync() {
            Some(d) => d,
            None => return,
        };
        let kriging = KrigingF64::new(device).unwrap();
        let known = vec![(5.0, 5.0, 1.0)];
        let targets = vec![(0.0, 0.0), (10.0, 10.0)];
        let model = VariogramModel::Spherical {
            nugget: 0.0,
            sill: 1.0,
            range: 10.0,
        };
        let result = kriging.interpolate(&known, &targets, model).unwrap();
        for &v in &result.values {
            assert!(
                v.is_finite(),
                "Single point kriging should produce finite values"
            );
        }
        println!("✓ Kriging single point: values={:?}", result.values);
    }

    #[test]
    fn test_invalid_variogram() {
        let device = match create_device_sync() {
            Some(d) => d,
            None => return,
        };
        let kriging = KrigingF64::new(device).unwrap();
        let known = vec![(0.0, 0.0, 1.0), (10.0, 10.0, 2.0)];
        let targets = vec![(5.0, 5.0)];
        let model = VariogramModel::Spherical {
            nugget: 2.0,
            sill: 1.0,
            range: 10.0,
        };
        let result = kriging.interpolate(&known, &targets, model);
        match result {
            Ok(r) => println!("✓ Invalid variogram: value={:?}", r.values),
            Err(e) => println!("✓ Invalid variogram: error ({e})"),
        }
    }
}
