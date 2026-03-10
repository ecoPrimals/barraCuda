// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

#[tokio::test]
async fn test_shannon_entropy_cpu() -> Result<()> {
    // Test CPU path (small array)
    let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
    else {
        return Ok(()); // Skip if no f64 GPU available
    };
    let fmr = FusedMapReduceF64::new(device)?;

    // Test case from wetSpring handoff: counts = [10, 20, 30, 40] → Shannon ≈ 1.27985422
    let counts = vec![10.0, 20.0, 30.0, 40.0];
    let shannon = fmr.shannon_entropy(&counts)?;

    // CPU reference
    let total: f64 = counts.iter().sum();
    let expected: f64 = counts
        .iter()
        .map(|&c| {
            let p = c / total;
            if p > 0.0 { -p * p.ln() } else { 0.0 }
        })
        .sum();

    let error = (shannon - expected).abs();
    assert!(
        error < 1e-10,
        "Shannon error {error} exceeds tolerance (got {shannon}, expected {expected})"
    );

    Ok(())
}

#[tokio::test]
async fn test_simpson_index_cpu() -> Result<()> {
    let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
    else {
        return Ok(()); // Skip if no f64 GPU available
    };
    let fmr = FusedMapReduceF64::new(device)?;

    let counts = vec![10.0, 20.0, 30.0, 40.0];
    let simpson = fmr.simpson_index(&counts)?;

    // CPU reference
    let total: f64 = counts.iter().sum();
    let expected: f64 = counts.iter().map(|&c| (c / total).powi(2)).sum();

    let error = (simpson - expected).abs();
    assert!(
        error < 1e-12,
        "Simpson error {error} exceeds tolerance (got {simpson}, expected {expected})"
    );

    Ok(())
}

#[tokio::test]
async fn test_large_array_sum_gpu() -> Result<()> {
    let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
    else {
        return Ok(()); // Skip if no f64 GPU available
    };
    let fmr = FusedMapReduceF64::new(device)?;

    // Large array to trigger GPU path
    let n = 100_000;
    let data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.001).collect();
    let sum = fmr.sum(&data)?;

    let expected: f64 = data.iter().sum();
    let error = (sum - expected).abs() / expected.abs();

    assert!(
        error < 1e-10,
        "Sum relative error {error} exceeds tolerance"
    );

    Ok(())
}
