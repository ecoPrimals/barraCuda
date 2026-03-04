// SPDX-License-Identifier: AGPL-3.0-or-later
#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
#[expect(clippy::module_inception, reason = "suppressed")]
mod tests {
    use crate::device::WgpuDevice;
    use crate::error::Result;
    use crate::ops::multi_head_attention::MultiHeadAttention;
    use crate::tensor::Tensor;
    use std::sync::Arc;

    async fn create_test_tensor(
        device: Arc<WgpuDevice>,
        shape: Vec<usize>,
        value: f32,
    ) -> Result<Tensor> {
        let size: usize = shape.iter().product();
        let data: Vec<f32> = vec![value; size];
        Tensor::from_vec_on(data, shape, device).await
    }

    #[tokio::test]
    async fn test_multi_head_attention_basic() {
        let Some(dev) = crate::device::test_pool::get_test_device_if_gpu_available().await else {
            return;
        };

        let batch = 1;
        let seq_len = 4;
        let d_model = 8;
        let num_heads = 2;

        let query = create_test_tensor(dev.clone(), vec![batch, seq_len, d_model], 0.5)
            .await
            .unwrap();
        let key = create_test_tensor(dev.clone(), vec![batch, seq_len, d_model], 0.5)
            .await
            .unwrap();
        let value = create_test_tensor(dev.clone(), vec![batch, seq_len, d_model], 0.5)
            .await
            .unwrap();

        let _weight_size = d_model * d_model;
        let w_q = create_test_tensor(dev.clone(), vec![d_model, d_model], 0.01)
            .await
            .unwrap();
        let w_k = create_test_tensor(dev.clone(), vec![d_model, d_model], 0.01)
            .await
            .unwrap();
        let w_v = create_test_tensor(dev.clone(), vec![d_model, d_model], 0.01)
            .await
            .unwrap();
        let w_o = create_test_tensor(dev.clone(), vec![d_model, d_model], 0.01)
            .await
            .unwrap();

        let output = MultiHeadAttention::new(query, key, value, w_q, w_k, w_v, w_o, num_heads)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(output.shape(), &[batch, seq_len, d_model]);
    }
}
