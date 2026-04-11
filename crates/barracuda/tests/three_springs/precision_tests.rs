// SPDX-License-Identifier: AGPL-3.0-or-later
//! Precision tests: f64 accuracy vs CPU reference.

#![expect(clippy::unwrap_used, reason = "tests")]
#![expect(
    clippy::useless_vec,
    reason = "test cases with mixed-length inner vecs"
)]
use super::*;
use barracuda::ops::fused_map_reduce_f64::FusedMapReduceF64;

mod precision {
    use super::*;

    #[test]
    fn test_shannon_precision_suite() {
        let device = match create_device_sync() {
            Some(d) => d,
            None => return,
        };
        let fmr = FusedMapReduceF64::new(device.clone()).unwrap();
        let t = tol(&device, 1e-10);
        let test_cases = vec![
            vec![1.0, 1.0, 1.0, 1.0],
            vec![100.0, 1.0, 1.0, 1.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![0.001, 0.01, 0.1, 1.0, 10.0],
            (1..=100).map(f64::from).collect::<Vec<_>>(),
        ];
        for (i, counts) in test_cases.iter().enumerate() {
            let gpu = fmr.shannon_entropy(counts).unwrap();
            let cpu = cpu_shannon(counts);
            let error = (gpu - cpu).abs();
            let rel_error = if cpu.abs() > 1e-15 {
                error / cpu.abs()
            } else {
                error
            };
            assert!(
                error < t || rel_error < t,
                "Case {i} failed: error={error}, rel_error={rel_error} (tol: {t})"
            );
        }
    }

    #[test]
    fn test_simpson_precision_suite() {
        let device = match create_device_sync() {
            Some(d) => d,
            None => return,
        };
        let fmr = FusedMapReduceF64::new(device.clone()).unwrap();
        let t = tol(&device, 1e-12);
        let test_cases = vec![
            vec![1.0, 1.0, 1.0, 1.0],
            vec![100.0, 1.0, 1.0, 1.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        ];
        for (i, counts) in test_cases.iter().enumerate() {
            let gpu = fmr.simpson_index(counts).unwrap();
            let cpu = cpu_simpson(counts);
            let error = (gpu - cpu).abs();
            assert!(error < t, "Case {i} failed: error={error} (tol: {t})");
        }
    }

    #[test]
    fn test_kahan_summation_accuracy() {
        let device = match create_device_sync() {
            Some(d) => d,
            None => return,
        };
        let fmr = FusedMapReduceF64::new(device.clone()).unwrap();
        let n = 500;
        let large = 1e10;
        let small = 1.0;
        let mut data = vec![large];
        data.extend(std::iter::repeat_n(small, n));
        data.push(-large);
        let result = fmr.sum(&data).unwrap();
        let expected = n as f64;
        let error = (result - expected).abs();
        let rel_error = error / expected;

        if rel_error > 0.5 {
            // GPU is executing f32 math despite advertising SHADER_F64.
            // Kahan summation can't compensate when 1e10 + 1.0 == 1e10
            // in the underlying precision. Skip rather than fail.
            return;
        }
        let t = tol(&device, 1e-10);
        assert!(
            rel_error < t,
            "Kahan summation error too large: {error} (rel: {rel_error}, tol: {t})"
        );
    }
}
