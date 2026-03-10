// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

#[tokio::test]
async fn test_analyzer_creation() {
    let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await else {
        return;
    };
    let analyzer = TimeSeriesAnalyzer::new(&device)
        .add_model(TimeSeriesModel::MovingAverage { window: 3 })
        .build()
        .await;

    assert!(analyzer.is_ok());
    let analyzer = analyzer.unwrap();
    assert_eq!(analyzer.models.len(), 1);
    assert!(analyzer.built);
}

#[tokio::test]
async fn test_moving_average_forecast() {
    let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await else {
        return;
    };
    let mut analyzer = TimeSeriesAnalyzer::new(&device)
        .add_model(TimeSeriesModel::MovingAverage { window: 3 })
        .build()
        .await
        .unwrap();

    let history = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let forecast = analyzer.forecast(&history, 5).await;

    assert!(forecast.is_ok());
    let forecast = forecast.unwrap();
    assert_eq!(forecast.values.len(), 5);
    assert_eq!(forecast.horizon, 5);

    // Should be close to average of last 3 values (3,4,5) = 4.0
    assert!((forecast.values[0] - 4.0).abs() < 0.1);
}

#[tokio::test]
async fn test_exponential_smoothing_forecast() {
    let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await else {
        return;
    };
    let mut analyzer = TimeSeriesAnalyzer::new(&device)
        .add_model(TimeSeriesModel::ExponentialSmoothing { alpha: 0.5 })
        .build()
        .await
        .unwrap();

    let history = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let forecast = analyzer.forecast(&history, 3).await;

    assert!(forecast.is_ok());
    let forecast = forecast.unwrap();
    assert_eq!(forecast.values.len(), 3);
}

#[tokio::test]
async fn test_anomaly_detection() {
    let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await else {
        return;
    };
    let mut analyzer = TimeSeriesAnalyzer::new(&device)
        .add_model(TimeSeriesModel::MovingAverage { window: 3 })
        .build()
        .await
        .unwrap();

    // Series with obvious anomaly
    let mut series = vec![1.0; 20];
    series[10] = 10.0; // Anomaly!

    let anomalies = analyzer.detect_anomalies(&series, 2.0).await;

    assert!(anomalies.is_ok());
    let anomalies = anomalies.unwrap();
    assert!(!anomalies.is_empty());

    // Should detect the anomaly at index 10
    assert!(anomalies.iter().any(|a| a.index == 10));
}

#[tokio::test]
async fn test_decomposition() {
    let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await else {
        return;
    };
    let analyzer = TimeSeriesAnalyzer::new(&device)
        .add_model(TimeSeriesModel::MovingAverage { window: 3 })
        .build()
        .await
        .unwrap();

    // Simple series with trend
    let series: Vec<f32> = (0..100).map(|i| i as f32).collect();

    let decomp = analyzer.decompose(&series, 10).await;

    assert!(decomp.is_ok());
    let decomp = decomp.unwrap();
    assert_eq!(decomp.trend.len(), 100);
    assert_eq!(decomp.seasonal.len(), 100);
    assert_eq!(decomp.residual.len(), 100);
}

#[tokio::test]
async fn test_esn_forecast() {
    let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await else {
        return;
    };
    let mut analyzer = TimeSeriesAnalyzer::new(&device)
        .add_model(TimeSeriesModel::ESN {
            reservoir_size: 50,
            spectral_radius: 0.9,
        })
        .build()
        .await
        .unwrap();

    // Simple increasing series
    let history: Vec<f32> = (0..20).map(|i| i as f32).collect();

    let forecast = analyzer.forecast(&history, 5).await;

    assert!(forecast.is_ok());
    let forecast = forecast.unwrap();
    assert_eq!(forecast.values.len(), 5);

    // ESN should learn the increasing pattern
    println!("ESN forecast: {:?}", forecast.values);
}

#[tokio::test]
async fn test_weighted_moving_average() {
    let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await else {
        return;
    };
    let mut analyzer = TimeSeriesAnalyzer::new(&device)
        .add_model(TimeSeriesModel::WeightedMovingAverage {
            weights: vec![0.5, 0.3, 0.2],
        })
        .build()
        .await
        .unwrap();

    let history = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let forecast = analyzer.forecast(&history, 3).await;

    assert!(forecast.is_ok());
    let forecast = forecast.unwrap();
    assert_eq!(forecast.values.len(), 3);
}
