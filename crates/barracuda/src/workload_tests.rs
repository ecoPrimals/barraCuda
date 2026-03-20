// SPDX-License-Identifier: AGPL-3.0-or-later
use super::*;

#[test]
fn test_sparsity_analysis() {
    // Test actual sparsity
    let sparse_data = vec![0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0];
    let profile = SparsityAnalyzer::analyze_data(&sparse_data);

    assert!((profile.actual_sparsity - 0.75).abs() < 0.01); // 6/8 = 75% sparse
    // Note: potential_sparsity may differ from actual if near-zeros differ

    // Test dense data
    let dense_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let profile = SparsityAnalyzer::analyze_data(&dense_data);

    assert!(profile.actual_sparsity < 0.01); // 0% sparse
    assert_eq!(profile.recommendation, DeviceRecommendation::PreferDense);
}

#[test]
fn test_workload_classification() {
    assert_eq!(
        WorkloadClassifier::classify_op("execute_mlp"),
        WorkloadType::ML
    );
    assert_eq!(WorkloadClassifier::classify_op("fhe_add"), WorkloadType::HE);
    assert_eq!(
        WorkloadClassifier::classify_op("kmer_count"),
        WorkloadType::Genomics
    );
    assert_eq!(
        WorkloadClassifier::classify_op("aes_encrypt"),
        WorkloadType::Crypto
    );
}

#[test]
fn test_device_selection() {
    let devices = vec![ComputeDevice::CPU, ComputeDevice::GPU, ComputeDevice::NPU];
    let selector = DeviceSelector::new(devices);

    // ML with energy priority → NPU
    let device = selector.select(
        WorkloadType::ML,
        0.5,
        100,
        Priority::Energy,
        DeviceHint::Auto,
    );
    assert_eq!(device, ComputeDevice::NPU);

    // ML with throughput priority, large batch → GPU
    let device = selector.select(
        WorkloadType::ML,
        0.5,
        128,
        Priority::Throughput,
        DeviceHint::Auto,
    );
    assert_eq!(device, ComputeDevice::GPU);

    // HE always → NPU
    let device = selector.select(
        WorkloadType::HE,
        0.8,
        100,
        Priority::Balanced,
        DeviceHint::Auto,
    );
    assert_eq!(device, ComputeDevice::NPU);
}

#[test]
fn test_sparsity_analysis_empty_data() {
    let empty_data: Vec<f32> = vec![];
    let profile = SparsityAnalyzer::analyze_data(&empty_data);
    assert_eq!(profile.actual_sparsity, 0.0);
    assert_eq!(profile.potential_sparsity, 0.0);
    assert_eq!(profile.recommendation, DeviceRecommendation::Neutral);
}

#[test]
fn test_sparsity_analysis_all_zeros() {
    let all_zeros = vec![0.0, 0.0, 0.0, 0.0];
    let profile = SparsityAnalyzer::analyze_data(&all_zeros);
    assert_eq!(profile.actual_sparsity, 1.0);
    assert_eq!(profile.potential_sparsity, 1.0);
    assert_eq!(profile.recommendation, DeviceRecommendation::ConsiderNPU);
}

#[test]
fn test_sparsity_analysis_near_zeros() {
    let near_zeros = vec![0.001, 0.005, 0.009, 1.0, 2.0];
    let profile = SparsityAnalyzer::analyze_data(&near_zeros);
    assert_eq!(profile.actual_sparsity, 0.0);
    assert_eq!(profile.potential_sparsity, 0.6);
}

#[test]
fn test_analyze_operation_relu() {
    let profile = SparsityAnalyzer::analyze_operation("relu_forward");
    assert_eq!(profile.potential_sparsity, 0.50);
    assert_eq!(profile.recommendation, DeviceRecommendation::PreferDense);
}

#[test]
fn test_analyze_operation_relu_threshold() {
    let profile = SparsityAnalyzer::analyze_operation("relu_threshold_layer");
    assert_eq!(profile.potential_sparsity, 0.75);
    assert_eq!(profile.recommendation, DeviceRecommendation::ConsiderNPU);
}

#[test]
fn test_analyze_operation_relu_mask() {
    let profile = SparsityAnalyzer::analyze_operation("relu_mask_op");
    assert_eq!(profile.potential_sparsity, 0.60);
    assert_eq!(profile.recommendation, DeviceRecommendation::ConsiderNPU);
}

#[test]
fn test_analyze_operation_dropout() {
    let profile = SparsityAnalyzer::analyze_operation("dropout");
    assert_eq!(profile.potential_sparsity, 0.30);
    assert_eq!(profile.recommendation, DeviceRecommendation::PreferDense);
}

#[test]
fn test_analyze_operation_threshold_only() {
    let profile = SparsityAnalyzer::analyze_operation("threshold_clamp");
    assert_eq!(profile.potential_sparsity, 0.40);
    assert_eq!(profile.recommendation, DeviceRecommendation::PreferDense);
}

#[test]
fn test_analyze_operation_unknown() {
    let profile = SparsityAnalyzer::analyze_operation("matmul_compute");
    assert_eq!(profile.potential_sparsity, 0.10);
    assert_eq!(profile.recommendation, DeviceRecommendation::PreferDense);
}

#[test]
fn test_workload_classification_ml_patterns() {
    assert_eq!(
        WorkloadClassifier::classify_op("conv2d_forward"),
        WorkloadType::ML
    );
    assert_eq!(
        WorkloadClassifier::classify_op("MATMUL_GEMM"),
        WorkloadType::ML
    );
    assert_eq!(
        WorkloadClassifier::classify_op("attention_scores"),
        WorkloadType::ML
    );
    assert_eq!(
        WorkloadClassifier::classify_op("layer_norm"),
        WorkloadType::ML
    );
}

#[test]
fn test_workload_classification_he_patterns() {
    assert_eq!(
        WorkloadClassifier::classify_op("tfhe_add"),
        WorkloadType::HE
    );
    assert_eq!(
        WorkloadClassifier::classify_op("homomorphic_mul"),
        WorkloadType::HE
    );
    assert_eq!(
        WorkloadClassifier::classify_op("bootstrap_key"),
        WorkloadType::HE
    );
}

#[test]
fn test_workload_classification_genomics_patterns() {
    assert_eq!(
        WorkloadClassifier::classify_op("dna_align"),
        WorkloadType::Genomics
    );
    assert_eq!(
        WorkloadClassifier::classify_op("sequence_match"),
        WorkloadType::Genomics
    );
    assert_eq!(
        WorkloadClassifier::classify_op("align_reads"),
        WorkloadType::Genomics
    );
}

#[test]
fn test_workload_classification_crypto_patterns() {
    assert_eq!(
        WorkloadClassifier::classify_op("chacha20_stream"),
        WorkloadType::Crypto
    );
    assert_eq!(
        WorkloadClassifier::classify_op("encrypt_block"),
        WorkloadType::Crypto
    );
    assert_eq!(
        WorkloadClassifier::classify_op("hash_sha256"),
        WorkloadType::Crypto
    );
}

#[test]
fn test_workload_classification_sparse_dense() {
    assert_eq!(
        WorkloadClassifier::classify_op("sparse_ops"),
        WorkloadType::Sparse
    );
    assert_eq!(
        WorkloadClassifier::classify_op("dense_layer"),
        WorkloadType::Dense
    );
    assert_eq!(
        WorkloadClassifier::classify_op("vector_add_op"),
        WorkloadType::Dense
    );
}

#[test]
fn test_workload_classification_unknown() {
    assert_eq!(
        WorkloadClassifier::classify_op("random_func"),
        WorkloadType::Unknown
    );
    assert_eq!(WorkloadClassifier::classify_op(""), WorkloadType::Unknown);
}

#[test]
fn test_device_selection_force_hint() {
    let selector = DeviceSelector::new(vec![ComputeDevice::CPU, ComputeDevice::GPU]);
    let device = selector.select(
        WorkloadType::ML,
        0.5,
        10_000,
        Priority::Throughput,
        DeviceHint::Force(ComputeDevice::CPU),
    );
    assert_eq!(device, ComputeDevice::CPU);
}

#[test]
fn test_device_selection_genomics_gpu() {
    let selector = DeviceSelector::new(vec![ComputeDevice::CPU, ComputeDevice::GPU]);
    let device = selector.select(
        WorkloadType::Genomics,
        0.1,
        10_000_000,
        Priority::Throughput,
        DeviceHint::Auto,
    );
    assert_eq!(device, ComputeDevice::GPU);
}

#[test]
fn test_device_selection_crypto_small_data() {
    let selector = DeviceSelector::new(vec![ComputeDevice::CPU, ComputeDevice::GPU]);
    let device = selector.select(
        WorkloadType::Crypto,
        0.0,
        100_000,
        Priority::Throughput,
        DeviceHint::Auto,
    );
    assert_eq!(device, ComputeDevice::CPU);
}

#[test]
fn test_device_selection_crypto_large_data() {
    let selector = DeviceSelector::new(vec![ComputeDevice::CPU, ComputeDevice::GPU]);
    let device = selector.select(
        WorkloadType::Crypto,
        0.0,
        10_000_000,
        Priority::Throughput,
        DeviceHint::Auto,
    );
    assert_eq!(device, ComputeDevice::GPU);
}

#[test]
fn test_device_selection_dense_small() {
    let selector = DeviceSelector::new(vec![ComputeDevice::CPU, ComputeDevice::GPU]);
    let device = selector.select(
        WorkloadType::Dense,
        0.0,
        512,
        Priority::Balanced,
        DeviceHint::Auto,
    );
    assert_eq!(device, ComputeDevice::CPU);
}

#[test]
fn test_device_selection_sparse_high_sparsity() {
    let selector = DeviceSelector::new(vec![ComputeDevice::CPU, ComputeDevice::NPU]);
    let device = selector.select(
        WorkloadType::Sparse,
        0.95,
        10_000,
        Priority::Energy,
        DeviceHint::Auto,
    );
    assert_eq!(device, ComputeDevice::NPU);
}

#[test]
fn test_device_selection_cpu_only() {
    let selector = DeviceSelector::new(vec![ComputeDevice::CPU]);
    let device = selector.select(
        WorkloadType::ML,
        0.5,
        10_000,
        Priority::Throughput,
        DeviceHint::Auto,
    );
    assert_eq!(device, ComputeDevice::CPU);
}

#[test]
fn test_decision_matrix_validation_data() {
    let matrix = DecisionMatrix::from_validation_data();

    assert!(
        matrix
            .get_energy(WorkloadType::ML, ComputeDevice::NPU)
            .is_some()
    );
    assert!(
        matrix
            .get_throughput(WorkloadType::ML, ComputeDevice::GPU)
            .is_some()
    );
    assert!(
        matrix
            .get_latency(WorkloadType::ML, ComputeDevice::CPU)
            .is_some()
    );

    let ml_npu_energy = matrix
        .get_energy(WorkloadType::ML, ComputeDevice::NPU)
        .unwrap();
    assert!((ml_npu_energy - 9.09).abs() < 0.01);

    let ml_gpu_throughput = matrix
        .get_throughput(WorkloadType::ML, ComputeDevice::GPU)
        .unwrap();
    assert!((ml_gpu_throughput - 1_330_679.0).abs() < 1.0);
}

#[test]
fn test_decision_matrix_missing_entries() {
    let matrix = DecisionMatrix::from_validation_data();
    assert!(
        matrix
            .get_energy(WorkloadType::Unknown, ComputeDevice::CPU)
            .is_none()
    );
    assert!(
        matrix
            .get_throughput(WorkloadType::Sparse, ComputeDevice::NPU)
            .is_none()
    );
}
