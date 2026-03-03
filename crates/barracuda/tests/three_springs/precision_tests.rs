//! Precision tests: f64 accuracy vs CPU reference.

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
        let fmr = FusedMapReduceF64::new(device).unwrap();
        let test_cases = vec![
            vec![1.0, 1.0, 1.0, 1.0],
            vec![100.0, 1.0, 1.0, 1.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![0.001, 0.01, 0.1, 1.0, 10.0],
            (1..=100).map(|x| x as f64).collect::<Vec<_>>(),
        ];
        println!("Shannon precision suite:");
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
                error < 1e-10 || rel_error < 1e-10,
                "Case {} failed: error={}, rel_error={}",
                i,
                error,
                rel_error
            );
            println!("  Case {}: H={:.10}, error={:.2e}", i, gpu, error);
        }
    }

    #[test]
    fn test_simpson_precision_suite() {
        let device = match create_device_sync() {
            Some(d) => d,
            None => return,
        };
        let fmr = FusedMapReduceF64::new(device).unwrap();
        let test_cases = vec![
            vec![1.0, 1.0, 1.0, 1.0],
            vec![100.0, 1.0, 1.0, 1.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        ];
        println!("Simpson precision suite:");
        for (i, counts) in test_cases.iter().enumerate() {
            let gpu = fmr.simpson_index(counts).unwrap();
            let cpu = cpu_simpson(counts);
            let error = (gpu - cpu).abs();
            assert!(error < 1e-12, "Case {} failed: error={}", i, error);
            println!("  Case {}: D={:.12}, error={:.2e}", i, gpu, error);
        }
    }

    #[test]
    fn test_kahan_summation_accuracy() {
        let device = match create_device_sync() {
            Some(d) => d,
            None => return,
        };
        let fmr = FusedMapReduceF64::new(device).unwrap();
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
        assert!(
            rel_error < 1e-10,
            "Kahan summation error too large: {} (rel: {})",
            error,
            rel_error
        );
        println!(
            "✓ Kahan summation: sum={}, expected={}, rel_error={:.2e}",
            result, expected, rel_error
        );
    }
}
