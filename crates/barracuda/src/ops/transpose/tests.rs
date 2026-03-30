// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tests for Transpose operation

use crate::device::test_pool;
use crate::tensor::Tensor;

#[tokio::test]
async fn test_transpose_basic() {
    let device = test_pool::get_test_device().await;

    let input = Tensor::from_vec_on(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], device)
        .await
        .unwrap();

    let output = input.transpose().unwrap();
    let result = output.to_vec().unwrap();

    let expected = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    assert_eq!(output.shape(), &[3, 2]);
    for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!((r - e).abs() < 1e-5, "Mismatch at index {i}: {r} vs {e}");
    }
}

#[tokio::test]
async fn test_transpose_nd() {
    // ND transpose uses 7+ storage buffers — exceeds CPU downlevel limit of 4.
    let device = test_pool::get_test_device().await;

    let input = Tensor::from_vec_on(
        (0..24).map(|i| i as f32).collect(),
        vec![2, 3, 4],
        device.clone(),
    )
    .await
    .unwrap();

    let output = input.transpose_with_permutation(vec![0, 2, 1]).unwrap();
    assert_eq!(output.shape(), &[2, 4, 3]);
}
