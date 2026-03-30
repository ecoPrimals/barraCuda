// SPDX-License-Identifier: AGPL-3.0-or-later
use super::{Fft1D, Ifft1D};
use crate::tensor::Tensor;

#[tokio::test]
async fn test_fft_ifft_inverse_property() {
    let device = crate::device::test_pool::get_test_device().await;

    // Test FFT(IFFT(x)) = x property
    // Input: Random complex signal
    let data = vec![
        1.0f32, 2.0, // 1+2i
        3.0, 4.0, // 3+4i
        5.0, 6.0, // 5+6i
        7.0, 8.0, // 7+8i
    ];

    let tensor = Tensor::from_data(&data, vec![4, 2], device).unwrap();

    // Forward FFT
    let fft = Fft1D::new(tensor, 4).unwrap();
    let spectrum = fft.execute().unwrap();

    // Inverse FFT
    let ifft = Ifft1D::new(spectrum, 4).unwrap();
    let reconstructed = ifft.execute().unwrap();

    let result_data = reconstructed.to_vec().unwrap();

    // Verify FFT(IFFT(x)) = x
    for (i, &expected) in data.iter().enumerate() {
        assert!(
            (result_data[i] - expected).abs() < 1e-4,
            "Mismatch at index {}: expected {}, got {}",
            i,
            expected,
            result_data[i]
        );
    }
}
