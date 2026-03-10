// SPDX-License-Identifier: AGPL-3.0-or-later
//! Operations module - Pure WGSL operations
//!
//! **Pure WGSL Architecture**:
//! - WGSL shaders ONLY (no CPU code!)
//! - wgpu handles execution on any device
//! - Single implementation per operation
//! - Zero duplication
//!
//! **Why this file is large (~720 lines)**: This is a barrel/re-export module.
//! It declares all op submodules and re-exports their public API. No logic lives
//! here—only `pub mod` and `pub use`. Splitting would fragment the ops namespace
//! without improving cohesion. The size reflects the breadth of operations, not
//! mixed concerns.

// Linear algebra operations (scientific computing)
pub mod linalg;

// Interpolation operations (RBF, splines, etc.)
pub mod interpolation;

// Full NCHW neural network operations (Conv2D GPU with stride/pad/dil/groups)
pub mod nn;

// Mixing operations for SCF solvers (Broyden, linear)
// Validated by hotSpring nuclear EOS study (169/169 acceptance checks)
pub mod mixing;

// Structured grid operations (gradients, Laplacian, FD stencils)
// Physics-agnostic primitives for fluid dynamics, nuclear physics, etc.
pub mod grid;

// NPU Bridge - Phase 3 unified API (Tensor ↔ NPU conversion)
pub mod npu_bridge;

// Neuromorphic operations (NPU/GPU/CPU universal)
pub mod sparse_matmul_quantized;

// Sparse GEMM: CSR × Dense matrix multiplication (f64 GPU)
pub mod sparse_gemm_f64;

// TransE knowledge graph triple scoring (f64 GPU)
pub mod transe_score_f64;

// 1D peak detection with prominence and width (f64 GPU)
pub mod peak_detect_f64;

// Attention mechanisms (Phase 4 - Universal Compute)
pub mod alibi;
pub mod attention;
pub mod causal_attn;
pub mod cross_attn;
pub mod dice;
pub mod flash_attention;
pub mod mha;
pub mod rope;
pub mod sparse_attn;

// Homomorphic encryption operations (FHE - GPU accelerated)
pub mod fhe_and;
pub mod fhe_extract; // Coefficient extraction (selective decryption!)
/// Fast polynomial multiply in NTT domain (NTT-based, 56x speedup).
pub mod fhe_fast_poly_mul;
pub mod fhe_intt; // Inverse NTT (completes NTT pipeline!)
pub mod fhe_key_switch; // Key switching (multi-key FHE!)
pub mod fhe_modulus_switch; // Modulus switching (noise reduction!)
pub mod fhe_ntt; // Number Theoretic Transform (56x speedup!)
pub mod fhe_or;
/// Point-wise multiply in NTT domain (O(N)).
pub mod fhe_pointwise_mul;
pub mod fhe_poly_add;
pub mod fhe_poly_mul; // Naive polynomial multiply (for comparison)
pub mod fhe_poly_sub;
pub mod fhe_rotate; // CKKS rotation (Galois automorphism!)
pub mod fhe_xor;

// Activation operations
pub mod elu_wgsl;
pub mod gelu_wgsl;
pub mod hardswish_wgsl;
pub mod leaky_relu_wgsl;
pub mod mish_wgsl;
pub mod relu;
pub mod selu_wgsl;
pub mod sigmoid;
pub mod softmax;
pub mod softplus_wgsl;
pub mod swish_wgsl;
// DF64 universal math shaders (f32-pair, ~48-bit mantissa, 9.9× f64 throughput)
pub mod df64_shaders;
// DF64 protein folding shaders (AlphaFold2-style, neuralSpring V60)
pub mod folding_df64;

// Element-wise operations
pub mod abs_wgsl;
pub mod add;
pub mod clamp_wgsl;
pub mod div;
pub mod exp_wgsl;
pub mod fma; // Fused multiply-add: d = a * b + c
pub mod log_wgsl;
pub mod mul;
pub mod neg_wgsl;
pub mod pow_wgsl;
pub mod reciprocal_wgsl;
pub mod sign_wgsl;
pub mod sqrt_wgsl;
pub mod sub;

// Comparison operations
pub mod eq;
pub mod gt;
pub mod lt;

// Trigonometric operations
pub mod acos_wgsl;
pub mod acosh_wgsl;
pub mod asin_wgsl;
pub mod asinh_wgsl;
pub mod atan_wgsl;
pub mod atanh_wgsl;
pub mod cos_wgsl;
pub mod cosh_wgsl;
pub mod sin_wgsl;
pub mod sinh_wgsl;
pub mod tan_wgsl;

// Rounding operations
pub mod ceil_wgsl;
pub mod floor_wgsl;
pub mod frac_wgsl;
pub mod round_wgsl;
pub mod trunc_wgsl;

// Special functions (mathematical physics, statistics)
pub mod bessel_i0_wgsl;
pub mod bessel_j0_wgsl;
pub mod bessel_j1_wgsl;
pub mod bessel_k0_wgsl;
pub mod beta_wgsl;
pub mod digamma_wgsl;
pub mod erf_wgsl;
pub mod erfc_wgsl;
pub mod hermite_f64_wgsl; // f64 Hermite polynomials (hotSpring nuclear physics)
pub mod hermite_wgsl;
pub mod laguerre_f64_wgsl; // f64 generalized Laguerre polynomials (hotSpring)
pub mod laguerre_wgsl;
pub mod legendre_f64_wgsl; // f64 Legendre polynomials (angular momentum, multipoles)
pub mod legendre_wgsl;

// Bessel functions (f64) - cylindrical/spherical problems
pub mod bessel_i0_f64_wgsl; // f64 Modified Bessel I0 (Kaiser windows)
pub mod bessel_j0_f64_wgsl; // f64 Bessel J0 (cylindrical waves)
pub mod bessel_j1_f64_wgsl; // f64 Bessel J1 (electromagnetic propagation)
pub mod bessel_k0_f64_wgsl; // f64 Modified Bessel K0 (Yukawa potential)

// Spherical harmonics (f64) - multipole expansion
pub mod spherical_harmonics_f64_wgsl;

// Statistical functions (f64) - statistics, ML
pub mod autocorrelation_f64_wgsl; // f64 general autocorrelation (time series, spectral)
pub mod beta_f64_wgsl; // f64 Beta function (Bayesian statistics)
pub mod correlation_f64_wgsl; // f64 Pearson correlation (portfolio analysis)
pub mod covariance_f64_wgsl; // f64 Covariance (PCA, Kalman)
pub mod digamma_f64_wgsl; // f64 Digamma function (Fisher information)
pub mod variance_f64_wgsl; // f64 Variance/StdDev (normalization)

// Spring handoff ops (neuralSpring V24, groundSpring V10/V54, wateringHole V69)
pub mod boltzmann_sampling_f64; // Boltzmann/softmax sampling (wateringHole V69)
pub mod fused_chi_squared_f64; // Fused chi-squared test (neuralSpring V24)
pub mod fused_kl_divergence_f64; // Fused KL divergence (neuralSpring V24)
pub mod rawr_weighted_mean_f64; // RAWR weighted mean + bootstrap CI (groundSpring V10/V54)

pub mod lgamma_wgsl;
pub mod norm_cdf_wgsl;
pub mod norm_ppf_wgsl;

// Reduction operations
pub mod loo_cv_wgsl;
pub mod max_wgsl;
pub mod mean;
pub mod min_wgsl;
pub mod norm;
pub mod prng_xoshiro_wgsl;
pub mod prod;
pub mod random_uniform_wgsl;
pub mod rsqrt_wgsl;
pub mod sobol_wgsl;
pub mod sparse_matvec_wgsl;
pub mod spherical_harmonics_wgsl;
pub mod std;
pub mod sum;
pub mod variance;

// Shape operations
pub mod concat;
pub mod pad_wgsl;
pub mod slice;
pub mod transpose;

// Selection and manipulation operations
pub mod argmax_wgsl;
pub mod argmin_wgsl;
pub mod squeeze;
pub mod tensor_axis_ops;
pub mod unsqueeze;
pub mod where_op;

// Neuromorphic operations
pub mod adaptive_avgpool2d;
pub mod adaptive_maxpool2d;
pub mod avg_pool1d_wgsl;
pub mod avgpool2d;
pub mod batch_norm;
pub mod cast;
pub mod conv2d;
pub mod dropout_wgsl;
pub mod embedding_wgsl;
pub mod gather_wgsl;
pub mod global_maxpool;
pub mod layer_norm_wgsl;
pub mod matmul;
pub mod max_pool1d_wgsl;
pub mod maxpool2d;
pub mod scatter_wgsl;
pub mod spatial_dropout;
pub mod topk;

// Reduction operations (f64) - Full f64 precision for scientific computing
pub mod cumprod_f64;
pub mod fused_map_reduce_f64; // Unified pattern: Shannon, Simpson, norms, etc.
pub mod kriging_f64; // Spatial interpolation (airSpring, wetSpring)

// Tridiagonal solvers - Critical for PDE/ODE (all springs)
// cyclic_reduction_wgsl.rs removed (API drift); use cyclic_reduction_f64 instead
pub mod cyclic_reduction_f64; // f64 cyclic reduction for PDEs (Feb 17, 2026)
pub mod weighted_dot_f64; // Weighted inner products (energy integrals, Galerkin)

// Statistical kernels
pub mod correlation_wgsl; // Pearson correlation
pub mod covariance_wgsl; // Sample covariance
pub mod moving_window_stats; // Sliding window mean/var/min/max (airSpring IoT)

// PDE/ODE infrastructure
pub mod batch_pair_reduce_f64; // O(N²) pairwise batch reduction (DADA2, BrayCurtis)
pub mod batch_tolerance_search_f64; // PFAS ion batch tolerance search
pub mod batched_ode_rk4; // BatchedOdeRK4F64 — full-GPU QS/c-di-GMP parameter sweep
pub mod batched_rk4_sweep; // BatchedRK4F64   — general-purpose N-trajectory orchestration (D-S21-001)
pub mod crank_nicolson; // Implicit PDE solver (Richards, heat, Schrödinger)
pub mod hill_f64; // Hill dose-response activation (wetSpring QS + healthSpring PK/PD)
pub mod kmd_grouping_f64;
pub mod population_pk_f64; // Population PK Monte Carlo (healthSpring absorption)
pub mod rk45_adaptive; // Adaptive Dormand-Prince RK45 for regulatory networks (neuralSpring)
pub mod rk_stage;
pub mod wright_fisher_f32; // Wright-Fisher drift+selection (neuralSpring popgen absorption) // RkIntegrator    — single-trajectory CPU-orchestrated RK4/RK45 // Kendrick Mass Defect homologue grouping
pub use batch_pair_reduce_f64::{BatchPairReduceF64, PairReduceOp};
pub use batch_tolerance_search_f64::BatchToleranceSearchF64;
pub use hill_f64::HillFunctionF64;
pub use kmd_grouping_f64::{KmdGroupingF64, KmdResult, repeat_units};
pub use population_pk_f64::{PopulationPkConfig, PopulationPkF64};
pub use rk_stage::{
    BatchedOdeRK4F64, BatchedRk4Config, OdeFunction, RkIntegrator, wgsl_rk4_parallel,
};
pub use wright_fisher_f32::{WrightFisherConfig, WrightFisherF32, seed_xoshiro_state};

// DF64 universal math shaders (compile via compile_shader_df64)
pub use df64_shaders::{
    WGSL_ELEMENTWISE_ADD_DF64, WGSL_ELEMENTWISE_FMA_DF64, WGSL_ELEMENTWISE_MUL_DF64,
    WGSL_ELEMENTWISE_SUB_DF64, WGSL_MEAN_REDUCE_DF64, WGSL_SUM_REDUCE_DF64,
};
pub use folding_df64::{FoldingOp, compile_folding_shader};

// Cosine similarity (f64)
pub mod cosine_similarity_f64; // Spectral matching, small-batch queries

// Batched element-wise operations (f64) - Unified pattern for all springs
pub mod batched_elementwise_f64; // FAO-56 ET₀, water balance, diversity metrics
pub mod max_abs_diff_f64;
pub mod norm_reduce_f64;
pub mod prod_reduce_f64;
pub mod sum_reduce_f64;
pub mod variance_reduce_f64;

// Utility operations
pub mod broadcast;
pub mod cumprod_wgsl;
pub mod cumsum_f64;
pub mod cumsum_wgsl;
pub mod fill;
pub mod flip_wgsl;
pub mod one_hot_wgsl;
pub mod repeat_wgsl;
pub mod roll_wgsl;

// Loss functions
pub mod binary_cross_entropy;
pub mod cross_entropy;
pub mod dice_loss;
pub mod focal_loss;
pub mod huber_loss;
pub mod l1_loss_wgsl;
pub mod mae_loss;
pub mod mse_loss;

// Advanced normalization
pub mod group_norm_wgsl;
pub mod instance_norm_wgsl;
pub mod rmsnorm;

// Convolution variants
pub mod conv1d;
pub mod conv3d;
pub mod depthwise_conv2d;
pub mod transposed_conv2d;

// Advanced operations
pub mod batch_matmul;
pub mod dotproduct;
pub mod filter;
pub mod global_avgpool;
pub mod map;
pub mod matmul_tiled;
pub mod reduce;
pub mod scan;
pub mod split;

// Optimizers
pub mod adadelta;
pub mod adagrad;
pub mod adam;
pub mod nadam;
pub mod rmsprop;
pub mod sgd;

// Attention mechanisms
pub mod alibi_position;
pub mod causal_attention;
pub mod cross_attention;
pub mod grouped_query_attention;
pub mod local_attention;
pub mod multi_head_attention;
pub mod rotary_embedding;
pub mod scaled_dot_product_attention;
pub mod sparse_attention;

// RNN/LSTM cells
pub mod bi_lstm;
pub mod gru_cell;
pub mod lstm_cell;
pub mod rnn_cell;
pub mod rnn_cell_wgsl;

// Advanced activations
pub mod celu_wgsl;
pub mod gelu_approximate_wgsl;
pub mod glu_wgsl;
pub mod hardshrink_wgsl;
pub mod hardsigmoid_wgsl;
pub mod hardtanh_wgsl;
pub mod log_softmax_wgsl;
pub mod logsigmoid_wgsl;
pub mod prelu_wgsl;
pub mod rrelu_wgsl;
pub mod silu_wgsl;
pub mod softshrink_wgsl;
pub mod softsign_wgsl;
pub mod tanhshrink_wgsl;
pub mod threshold_wgsl;

// Advanced convolutions
pub mod avgpool3d;
pub mod circular_pad2d;
pub mod circular_pad_wgsl;
pub mod deformable_conv2d;
/// Dilated (atrous) 2D convolution with configurable dilation rates.
pub mod dilated_conv2d;
pub mod gated_conv2d;
pub mod grouped_conv2d;
pub mod maxpool3d;
pub mod octave_conv2d;
pub mod reflection_pad_wgsl;
pub mod replication_pad_wgsl;
pub mod separable_conv2d;

// Advanced loss functions
pub mod bce_loss;
pub mod contrastive_loss;
pub mod cosine_embedding_loss;
pub mod hinge_loss;
pub mod kl_divergence;
pub mod kldiv_loss;
pub mod margin_ranking_loss;
pub mod multi_margin_loss;
pub mod multilabel_margin_loss;
pub mod nll_loss;
pub mod poisson_nll_loss;
pub mod triplet_loss;

// Advanced normalization
pub mod adaptive_instance_norm;
pub mod filter_response_norm;
pub mod local_response_norm;
pub mod spectral_norm;
pub mod spectral_normalization;
pub mod weight_norm;
pub mod weight_normalization;

// Advanced utilities
pub mod affine_grid;
pub mod bincount_wgsl;
pub mod bucketize_wgsl;
pub mod cdist_wgsl;
pub mod channel_shuffle_wgsl;
pub mod color_jitter_wgsl;
pub mod diag;
pub mod fold;
pub mod gather_nd;
pub mod grid_sample_wgsl;
pub mod histc;
pub mod index_add;
pub mod index_select;
pub mod index_select_wgsl;
pub mod interpolate_nearest_wgsl;
pub mod interpolate_wgsl;
pub mod inverse_wgsl;
pub mod logsumexp;
pub mod logsumexp_wgsl;
pub mod masked_fill_wgsl;
pub mod masked_select;
pub mod nonzero;
pub mod normalize;
pub mod pairwise_distance;
pub mod pdist;
pub mod renorm;
pub mod scatter_nd;
pub mod searchsorted;
pub mod sinkhorn_distance;
pub mod slice_assign;
pub mod trace_wgsl;
pub mod tril;
pub mod triu;
pub mod unfold;
pub mod unique;

// Tensor manipulation
pub mod chunk;
pub mod expand;
pub mod flatten;
pub mod movedim;
pub mod narrow_wgsl;
pub mod permute;
pub mod repeat_interleave;
pub mod stack;
pub mod tensor_split;
pub mod tile;

// Advanced matrix operations
pub mod bray_curtis_f64; // Bray-Curtis distance (absorbed from wetSpring)
pub mod cosine_similarity;
pub mod cross_product;
pub mod determinant;
pub mod matrix_inverse;
pub mod matrix_power;
pub mod matrix_rank;
pub mod outer_product;
pub mod tensor_dot;

// Gradient operations
pub mod clip_grad_norm;
pub mod clip_grad_value;

// Complex arithmetic (for FFT and wave physics)
pub mod complex;

// Fast Fourier Transform (evolved from NTT!)
pub mod fft;

// Molecular Dynamics primitives
pub mod md;

// Lattice QCD / gauge field theory (hotSpring v0.5.16 absorption, Feb 2026)
// complex_f64 + SU(3) WGSL libraries, Wilson plaquette, U(1) Higgs HMC, SU(3) HMC force
pub mod lattice;

// Nuclear structure GPU primitives (HFB, BCS, Skyrme)
// Absorbed from hotSpring v0.6.4 (Feb 2026)
pub mod physics;

// AlphaFold2 Evoformer primitives (neuralSpring S69 absorption)
// Triangle multiplication, MSA attention, IPA, triangle attention, backbone update, torsion angles
pub mod alphafold2;

// Anderson coupling (tight-binding Hamiltonian, groundSpring S69 absorption)
pub mod anderson_coupling;

// Lanczos eigensolver (symmetric eigenvalue problems, spectral methods)
pub mod lanczos;

// Statistical ops (f64) — matrix correlation, OLS regression (neuralSpring S69)
pub mod stats_f64;

// Life-science + analytical chemistry GPU primitives (wetSpring handoff v4, Feb 2026)
// Smith-Waterman banded alignment, Gillespie SSA, decision-tree inference, Felsenstein pruning
pub mod bio;

// Quantization
pub mod dequantize;
pub mod fake_quantize;
pub mod quantize;

// Object detection
pub mod anchor_generator;
pub mod bbox_transform;
pub mod box_iou;
pub mod giou_loss;
pub mod nms;
pub mod roi_align;
pub mod roi_pool;
pub mod soft_nms;

// Advanced pooling
pub mod adaptive_avg_pool1d;
pub mod adaptive_max_pool1d;
pub mod fractional_max_pool2d;
pub mod lp_pool2d;

// Enhanced losses
pub mod focal_loss_alpha;
pub mod focal_loss_v2;
pub mod smooth_l1_loss;

// Utility operations
pub mod layer_scale;
pub mod pixel_shuffle;
pub mod put;
pub mod reshape;
pub mod take;
pub mod upsample;
pub mod view;

// Graph Neural Networks (Category 13)
pub mod edge_conv;
pub mod gat_conv;
pub mod gcn_conv;
pub mod gin_conv;
pub mod global_pooling;
pub mod graph_batch_norm;
pub mod graph_conv;
pub mod graph_norm;
pub mod message_passing;
pub mod sage_conv;

// Advanced Optimizers & Learning (Category 14)
pub mod adabound;
pub mod adafactor;
pub mod adamw;
pub mod cyclical_lr;
pub mod lamb;
pub mod lookahead;
pub mod onecycle;
pub mod radam;
pub mod sgdw;

// Audio/Signal Processing (Category 15)
pub mod griffin_lim;
pub mod istft;
pub mod mel_scale;
pub mod mfcc;
pub mod pitch_shift;
pub mod spectral_norm_1d;
pub mod spectrogram;
pub mod stft;
pub mod time_stretch;
pub mod window_function;

// Advanced Sampling & Augmentation (Category 16)
pub mod cutmix;
pub mod elastic_transform;
pub mod grid_mask;
pub mod label_smoothing;
pub mod mixup;
pub mod mosaic;
pub mod random_affine;
pub mod random_crop;
pub mod random_erasing;
pub mod random_perspective;
pub mod random_rotation;

// Specialized Losses & Metrics (Category 17)
pub mod center_loss;
pub mod chamfer_distance;
pub mod earth_mover_distance;
pub mod iou_loss;
pub mod perceptual_loss;
pub mod psnr;
pub mod ssim;
pub mod tversky_loss;
pub mod wasserstein_loss;

// ── Re-exports: Operation traits and structs for use as `Op::run(tensor)` ──
// These are the primary API surface. Each op lives in its submodule; docs are on the source.
pub use abs_wgsl::Abs;
pub use add::Add;
pub use argmax_wgsl::Argmax;
pub use batch_norm::BatchNorm;
pub use ceil_wgsl::Ceil;
pub use clamp_wgsl::Clamp;
pub use concat::Concat;
pub use cos_wgsl::Cos;
pub use div::Div;
pub use dropout_wgsl::Dropout;
pub use elu_wgsl::ELU;
pub use eq::Eq;
pub use exp_wgsl::Exp;
pub use floor_wgsl::Floor;
pub use gather_nd::GatherNd;
pub use gather_wgsl::Gather;
pub use gelu_wgsl::GELU;
pub use gt::Gt;
pub use hardswish_wgsl::Hardswish;
pub use index_add::IndexAdd;
pub use index_select::IndexSelect;
pub use index_select_wgsl::IndexSelect as IndexSelectWgsl;
pub use layer_norm_wgsl::LayerNorm;
pub use leaky_relu_wgsl::LeakyRelu;
pub use log_wgsl::Log;
pub use lt::Lt;
pub use max_wgsl::Max;
pub use mean::Mean;
pub use min_wgsl::Min;
pub use mish_wgsl::Mish;
pub use mul::Mul;
pub use neg_wgsl::Neg;
pub use norm::Norm;
pub use pad_wgsl::Pad;
pub use pow_wgsl::Pow;
pub use prod::Prod;
pub use reciprocal_wgsl::Reciprocal;
pub use relu::ReLU;
pub use round_wgsl::Round;
pub use scatter_nd::ScatterNd;
pub use scatter_wgsl::Scatter;
pub use selu_wgsl::SELU;
pub use sigmoid::Sigmoid;
pub use sign_wgsl::Sign;
pub use sin_wgsl::Sin;
pub use slice::Slice;
pub use slice_assign::{SliceAssign, SliceRange};
pub use softmax::Softmax;
pub use sqrt_wgsl::Sqrt;
pub use squeeze::Squeeze;
pub use std::Std;
pub use sub::Sub;
pub use sum::Sum;
pub use swish_wgsl::Swish;
pub mod tanh; // WGSL implementation in tanh.rs (not tanh_wgsl.rs)
pub use transpose::Transpose;
pub use unsqueeze::Unsqueeze;
pub use variance::Variance;
pub use view::View;
pub use where_op::Where;
// topk exports functions, not struct
pub use adadelta::AdaDelta;
pub use adagrad::AdaGrad;
pub use adam::Adam;
pub use adaptive_avgpool2d::AdaptiveAvgPool2D;
pub use adaptive_maxpool2d::AdaptiveMaxPool2D;
pub use avgpool2d::AvgPool2D;
pub use batch_matmul::BatchMatMul;
pub use binary_cross_entropy::BinaryCrossEntropy;
pub use broadcast::Broadcast;
pub use cast::Cast;
pub use conv1d::Conv1D;
pub use conv2d::Conv2D;
pub use conv3d::Conv3D;
pub use cross_entropy::CrossEntropy;
pub use cumprod_f64::CumprodF64;
pub use cumsum_f64::CumsumF64;
pub use cumsum_wgsl::Cumsum;
pub use depthwise_conv2d::DepthwiseConv2D;
pub use dotproduct::DotProduct;
pub use embedding_wgsl::Embedding;
pub use fill::Fill;
pub use filter::{Filter, FilterOperation};
pub use flip_wgsl::Flip;
pub use focal_loss::FocalLoss;
pub use global_avgpool::GlobalAvgPool;
pub use global_maxpool::GlobalMaxPool;
pub use group_norm_wgsl::GroupNorm;
pub use huber_loss::HuberLoss;
pub use instance_norm_wgsl::InstanceNorm;
pub use l1_loss_wgsl::L1Loss;
pub use mae_loss::MAELoss;
pub use map::{Map, MapOperation};
pub use matmul::MatMul;
pub use matmul_tiled::MatmulTiled;
pub use max_abs_diff_f64::MaxAbsDiffF64;
pub use maxpool2d::MaxPool2D;
pub use mse_loss::MseLoss;
pub use norm_reduce_f64::NormReduceF64;
pub use one_hot_wgsl::OneHot;
pub use prod_reduce_f64::ProdReduceF64;
pub use reduce::{Reduce, ReduceOperation};
pub use repeat_wgsl::Repeat;
pub use rmsnorm::RMSNorm;
pub use rmsprop::RMSprop;
pub use scan::Scan;
pub use sgd::SGD;
pub use softplus_wgsl::Softplus;
pub use split::Split;
pub use sum_reduce_f64::SumReduceF64;
pub use transposed_conv2d::TransposedConv2D;
pub use variance_reduce_f64::VarianceReduceF64;
// Note: Reshape is already defined in tensor.rs
pub mod lovasz_loss;

// Week 7 operations - Universal compute via WGSL
pub use acos_wgsl::Acos;
pub use acosh_wgsl::Acosh;
pub use asin_wgsl::Asin;
pub use asinh_wgsl::Asinh;
pub use atan_wgsl::Atan;
pub use atanh_wgsl::Atanh;
pub use bessel_i0_wgsl::BesselI0;
pub use bessel_j0_wgsl::BesselJ0;
pub use bessel_j1_wgsl::BesselJ1;
pub use bessel_k0_wgsl::BesselK0;
pub use cosh_wgsl::Cosh;
pub use erf_wgsl::Erf;
pub use erfc_wgsl::Erfc;
pub use kl_divergence::KLDivergence; // Using kl_divergence.rs (has WGSL shader)
pub use lgamma_wgsl::Lgamma;
pub use logsumexp_wgsl::LogsumexpWgsl;
pub use sinh_wgsl::Sinh;
pub use smooth_l1_loss::SmoothL1Loss; // Using smooth_l1_loss.rs (has WGSL shader)
pub use tanh::Tanh; // Using tanh.rs (has WGSL shader) // Keep for now, need to analyze logsumexp.rs
