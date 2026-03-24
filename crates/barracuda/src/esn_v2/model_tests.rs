// SPDX-License-Identifier: AGPL-3.0-or-later
use super::*;
use crate::device::{Device, WorkloadHint};

#[tokio::test]
async fn test_esn_creation() {
    let config = ESNConfig::default();
    let esn = ESN::new(config).await.unwrap();
    assert!(!esn.is_trained());
    assert_eq!(esn.state().shape(), &[100, 1]);
}

#[tokio::test]
async fn test_esn_invalid_config() {
    let config = ESNConfig {
        input_size: 0,
        ..Default::default()
    };
    assert!(ESN::new(config).await.is_err());
}

#[tokio::test]
async fn test_esn_device_preference() {
    let config = ESNConfig::default();
    let esn = ESN::new(config).await.unwrap();
    let _esn_gpu = esn.prefer_device(Device::GPU);
}

#[tokio::test]
async fn test_esn_workload_hint() {
    let config = ESNConfig::default();
    let esn = ESN::new(config).await.unwrap();
    let _esn_large = esn.with_hint(WorkloadHint::LargeMatrices);
}

#[tokio::test]
async fn test_esn_device_query() {
    let config = ESNConfig::default();
    let esn = ESN::new(config).await.unwrap();
    let device = esn.query_device();
    assert!(matches!(device, Device::CPU | Device::GPU | Device::Auto));
}

#[tokio::test]
async fn test_esn_train_simple() {
    let config = ESNConfig {
        input_size: 1,
        reservoir_size: 20,
        output_size: 1,
        spectral_radius: 0.9,
        connectivity: 0.1,
        leak_rate: 0.3,
        regularization: 1e-6,
        seed: 42,
        ..Default::default()
    };

    let mut esn = ESN::new(config).await.unwrap();

    let inputs = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
    let targets = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];

    let error = esn.train(&inputs, &targets).await.unwrap();
    assert!(error >= 0.0);
    assert!(esn.is_trained());
    assert!(esn.w_out.is_some());
}

#[tokio::test]
async fn test_esn_predict_after_train() {
    let config = ESNConfig {
        input_size: 1,
        reservoir_size: 30,
        output_size: 1,
        spectral_radius: 0.95,
        connectivity: 0.15,
        leak_rate: 0.3,
        regularization: 1e-5,
        seed: 42,
        ..Default::default()
    };

    let mut esn = ESN::new(config).await.unwrap();

    let inputs = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
    let targets = vec![vec![2.0], vec![3.0], vec![4.0], vec![5.0]];

    esn.train(&inputs, &targets).await.unwrap();

    let prediction = esn.predict(&[10.0]).await.unwrap();
    assert_eq!(prediction.len(), 1);
    assert!(prediction[0] > 5.0 && prediction[0] < 20.0);
}

#[tokio::test]
async fn test_esn_train_mismatched_lengths() {
    let config = ESNConfig::default();
    let mut esn = ESN::new(config).await.unwrap();

    let inputs = vec![vec![0.0], vec![1.0]];
    let targets = vec![vec![1.0]];

    let result = esn.train(&inputs, &targets).await;
    assert!(result.is_err());
    assert!(!esn.is_trained());
}

#[tokio::test]
async fn test_esn_train_empty_data() {
    let config = ESNConfig::default();
    let mut esn = ESN::new(config).await.unwrap();

    let inputs: Vec<Vec<f32>> = vec![];
    let targets: Vec<Vec<f32>> = vec![];

    let result = esn.train(&inputs, &targets).await;
    assert!(result.is_err());
    assert!(!esn.is_trained());
}

#[tokio::test]
async fn test_esn_predict_untrained() {
    let config = ESNConfig::default();
    let mut esn = ESN::new(config).await.unwrap();

    let result = esn.predict(&[1.0]).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_esn_predict_wrong_input_size() {
    let config = ESNConfig {
        input_size: 2,
        reservoir_size: 20,
        output_size: 1,
        ..Default::default()
    };

    let mut esn = ESN::new(config).await.unwrap();

    let inputs = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
    let targets = vec![vec![1.0], vec![2.0]];
    esn.train(&inputs, &targets).await.unwrap();

    let result = esn.predict(&[1.0]).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_esn_multiple_outputs() {
    let config = ESNConfig {
        input_size: 2,
        reservoir_size: 40,
        output_size: 3,
        spectral_radius: 0.9,
        connectivity: 0.1,
        leak_rate: 0.3,
        regularization: 1e-5,
        seed: 42,
        ..Default::default()
    };

    let mut esn = ESN::new(config).await.unwrap();

    let inputs = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![
        vec![0.0, 0.0, 0.0],
        vec![1.0, 0.0, 1.0],
        vec![0.0, 1.0, 1.0],
        vec![1.0, 1.0, 2.0],
    ];

    let error = esn.train(&inputs, &targets).await.unwrap();
    assert!(error >= 0.0);
    assert!(esn.is_trained());

    let prediction = esn.predict(&[0.5, 0.5]).await.unwrap();
    assert_eq!(prediction.len(), 3);
}

#[tokio::test]
async fn test_esn_predict_return_state() {
    let config = ESNConfig {
        input_size: 1,
        reservoir_size: 20,
        output_size: 1,
        ..Default::default()
    };

    let mut esn = ESN::new(config).await.unwrap();

    let inputs = vec![vec![1.0], vec![2.0], vec![3.0]];
    let targets = vec![vec![2.0], vec![3.0], vec![4.0]];
    esn.train(&inputs, &targets).await.unwrap();

    let (output, state) = esn.predict_return_state(&[5.0]).await.unwrap();
    assert_eq!(output.len(), 1);
    assert_eq!(state.len(), 20);
    assert!(
        state.iter().any(|&v| v != 0.0),
        "State should be non-zero after update"
    );
}

#[tokio::test]
async fn test_esn_set_readout_weights() {
    let config = ESNConfig {
        input_size: 1,
        reservoir_size: 20,
        output_size: 1,
        ..Default::default()
    };

    let mut esn = ESN::new(config).await.unwrap();

    let inputs = vec![vec![1.0], vec![2.0]];
    let targets = vec![vec![2.0], vec![3.0]];
    esn.train(&inputs, &targets).await.unwrap();

    let original = esn.predict(&[5.0]).await.unwrap();

    let new_weights = Tensor::zeros_on(vec![20, 1], esn.device.clone())
        .await
        .unwrap();
    esn.set_readout_weights(new_weights).unwrap();

    let zeroed = esn.predict(&[5.0]).await.unwrap();
    assert!(
        (zeroed[0]).abs() < 1e-5,
        "Zero readout should produce near-zero output"
    );
    assert_ne!(
        original, zeroed,
        "Different readout weights should produce different output"
    );
}

#[tokio::test]
async fn test_esn_state_persistence() {
    let config = ESNConfig {
        input_size: 1,
        reservoir_size: 20,
        output_size: 1,
        ..Default::default()
    };

    let mut esn = ESN::new(config).await.unwrap();

    let inputs = vec![vec![1.0], vec![2.0], vec![3.0]];
    let targets = vec![vec![2.0], vec![3.0], vec![4.0]];
    esn.train(&inputs, &targets).await.unwrap();

    let pred1 = esn.predict(&[5.0]).await.unwrap();
    let pred2 = esn.predict(&[5.0]).await.unwrap();

    assert_ne!(pred1, pred2, "State should evolve between predictions");
}

#[tokio::test]
async fn test_esn_to_npu_weights() {
    let config = ESNConfig {
        input_size: 1,
        reservoir_size: 20,
        output_size: 1,
        ..Default::default()
    };

    let mut esn = ESN::new(config).await.unwrap();
    let inputs = vec![vec![1.0], vec![2.0], vec![3.0]];
    let targets = vec![vec![2.0], vec![3.0], vec![4.0]];
    esn.train(&inputs, &targets).await.unwrap();

    let npu = esn.to_npu_weights().unwrap();
    assert_eq!(npu.input_dim, 20);
    assert_eq!(npu.output_dim, 1);
    assert_eq!(npu.weights_i8.len(), 20);
    assert!(npu.scale > 0.0);
}

#[tokio::test]
async fn test_esn_to_npu_weights_untrained() {
    let config = ESNConfig::default();
    let esn = ESN::new(config).await.unwrap();
    assert!(esn.to_npu_weights().is_err());
}

#[tokio::test]
async fn test_esn_train_ridge_regression_linear() {
    let config = ESNConfig {
        input_size: 1,
        reservoir_size: 4,
        output_size: 1,
        ..Default::default()
    };

    let mut esn = ESN::new(config).await.unwrap();

    let n_samples = 10;
    let mut states = vec![0.0; 4 * n_samples];
    let mut targets = vec![0.0; n_samples];
    for k in 0..n_samples {
        let x = k as f64 * 0.5;
        states[k] = 1.0;
        states[n_samples + k] = x;
        states[2 * n_samples + k] = x * x;
        states[3 * n_samples + k] = x * x * x;
        targets[k] = 3.0f64.mul_add(x, 2.0);
    }

    esn.train_ridge_regression(&states, &targets, 1e-6).unwrap();
    assert!(esn.is_trained());

    let w = esn.w_out.as_ref().unwrap().to_vec().unwrap();
    assert_eq!(w.len(), 4);
    assert!((w[0] - 2.0).abs() < 0.1);
    assert!((w[1] - 3.0).abs() < 0.1);
}

#[tokio::test]
async fn test_esn_train_ridge_regression_regularization() {
    let config = ESNConfig {
        input_size: 1,
        reservoir_size: 5,
        output_size: 1,
        ..Default::default()
    };

    let mut esn_small = ESN::new(config.clone()).await.unwrap();
    let mut esn_large = ESN::new(config).await.unwrap();

    let n_samples = 8;
    let mut states = vec![0.0; 5 * n_samples];
    let mut targets = vec![0.0; n_samples];
    for k in 0..n_samples {
        for i in 0..5 {
            states[i * n_samples + k] = (k as f64 + i as f64) * 0.1;
        }
        targets[k] = (k as f64) * 0.2;
    }

    esn_small
        .train_ridge_regression(&states, &targets, 1e-6)
        .unwrap();
    esn_large
        .train_ridge_regression(&states, &targets, 10.0)
        .unwrap();

    let w_small = esn_small.w_out.as_ref().unwrap().to_vec().unwrap();
    let w_large = esn_large.w_out.as_ref().unwrap().to_vec().unwrap();

    let norm_small: f32 = w_small.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_large: f32 = w_large.iter().map(|x| x * x).sum::<f32>().sqrt();

    assert!(
        norm_large < norm_small,
        "Larger lambda should produce smaller weights: {norm_large} < {norm_small}"
    );
}

#[tokio::test]
async fn test_esn_export_import_weights() {
    let config = ESNConfig {
        input_size: 1,
        reservoir_size: 20,
        output_size: 1,
        ..Default::default()
    };

    let mut esn = ESN::new(config.clone()).await.unwrap();
    let inputs = vec![vec![1.0], vec![2.0], vec![3.0]];
    let targets = vec![vec![2.0], vec![3.0], vec![4.0]];
    esn.train(&inputs, &targets).await.unwrap();

    let exported = esn.export_weights().unwrap();
    assert_eq!(exported.w_in.len(), 20);
    assert_eq!(exported.w_res.len(), 400);
    assert!(exported.w_out.is_some());
    assert_eq!(exported.w_out.as_ref().unwrap().len(), 20);

    let mut esn2 = ESN::new(config.clone()).await.unwrap();
    esn2.import_weights(&exported.w_in, &exported.w_res, exported.w_out.as_deref())
        .unwrap();
    assert!(esn2.is_trained());
}

#[tokio::test]
async fn test_esn_exported_weights_migrate_to_multi_head() {
    let exported = super::ExportedWeights {
        w_in: vec![0.0; 20],
        w_res: vec![0.0; 400],
        w_out: Some(vec![1.0; 20]),
        input_size: 1,
        reservoir_size: 20,
        output_size: 1,
        leak_rate: 0.3,
        head_labels: Vec::new(),
    };

    let migrated = exported.migrate_to_multi_head(20, 11).unwrap();
    assert_eq!(migrated.w_out.as_ref().unwrap().len(), 11 * 20);
    assert_eq!(migrated.output_size, 11);
}

#[tokio::test]
async fn test_esn_reservoir_shape() {
    let config = ESNConfig {
        input_size: 3,
        reservoir_size: 16,
        output_size: 2,
        spectral_radius: 0.95,
        connectivity: 0.1,
        leak_rate: 0.2,
        regularization: 1e-4,
        seed: 42,
        ..Default::default()
    };

    let esn = ESN::new(config).await.unwrap();
    assert_eq!(esn.state().shape(), &[16, 1]);
}
