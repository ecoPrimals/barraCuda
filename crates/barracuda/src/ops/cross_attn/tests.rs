// SPDX-License-Identifier: AGPL-3.0-only
//! Tests for Cross Attention
//!
//! Validates encoder-decoder attention with asymmetric sequence lengths.

use super::*;
use crate::device::test_pool::get_test_device_if_gpu_available;

#[tokio::test]
async fn test_cross_attention_basic() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };

    let batch = 1;
    let heads = 2;
    let dec_seq = 4; // Decoder sequence
    let enc_seq = 8; // Encoder sequence (longer)
    let dim = 16;

    // Decoder query
    let q = Tensor::from_vec_on(
        vec![0.5; batch * heads * dec_seq * dim],
        vec![batch, heads, dec_seq, dim],
        device.clone(),
    )
    .await
    .unwrap();

    // Encoder keys/values
    let k = Tensor::from_vec_on(
        vec![0.5; batch * heads * enc_seq * dim],
        vec![batch, heads, enc_seq, dim],
        device.clone(),
    )
    .await
    .unwrap();

    let v = Tensor::from_vec_on(
        vec![1.0; batch * heads * enc_seq * dim],
        vec![batch, heads, enc_seq, dim],
        device,
    )
    .await
    .unwrap();

    let output = q.cross_attention(&k, &v).unwrap();

    assert_eq!(output.shape(), &[batch, heads, dec_seq, dim]);
}

#[tokio::test]
async fn test_cross_attention_shape_validation() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };

    let batch = 2;
    let heads = 4;
    let dec_seq = 8;
    let enc_seq = 16;
    let dim = 32;

    let q = Tensor::from_vec_on(
        vec![0.5; batch * heads * dec_seq * dim],
        vec![batch, heads, dec_seq, dim],
        device.clone(),
    )
    .await
    .unwrap();

    let k = Tensor::from_vec_on(
        vec![0.5; batch * heads * enc_seq * dim],
        vec![batch, heads, enc_seq, dim],
        device.clone(),
    )
    .await
    .unwrap();

    let v = Tensor::from_vec_on(
        vec![1.0; batch * heads * enc_seq * dim],
        vec![batch, heads, enc_seq, dim],
        device.clone(),
    )
    .await
    .unwrap();

    // Valid: asymmetric seq_lens
    assert!(q.clone().cross_attention(&k, &v).is_ok());

    // Invalid: mismatched batch
    let k_bad = Tensor::from_vec_on(
        vec![0.5; heads * enc_seq * dim],
        vec![1, heads, enc_seq, dim],
        device,
    )
    .await
    .unwrap();

    assert!(q.cross_attention(&k_bad, &v).is_err());
}

#[tokio::test]
async fn test_cross_attention_whisper_style() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };

    // Whisper-style: short decoder, long encoder (audio)
    let batch = 1;
    let heads = 8;
    let dec_seq = 32; // Text tokens
    let enc_seq = 1500; // Audio frames
    let dim = 64;

    let q = Tensor::from_vec_on(
        vec![0.5; batch * heads * dec_seq * dim],
        vec![batch, heads, dec_seq, dim],
        device.clone(),
    )
    .await
    .unwrap();

    let k = Tensor::from_vec_on(
        vec![0.5; batch * heads * enc_seq * dim],
        vec![batch, heads, enc_seq, dim],
        device.clone(),
    )
    .await
    .unwrap();

    let v = Tensor::from_vec_on(
        vec![1.0; batch * heads * enc_seq * dim],
        vec![batch, heads, enc_seq, dim],
        device,
    )
    .await
    .unwrap();

    let output = q.cross_attention(&k, &v).unwrap();

    assert_eq!(output.shape(), &[batch, heads, dec_seq, dim]);
}
