// SPDX-License-Identifier: AGPL-3.0-only
//! High-level Neural Network Training API
//!
//! Production-ready interface for building and training deep neural networks.
//! Wraps barraCuda operations into an ergonomic, PyTorch-like API with full
//! deep debt compliance.
//!
//! # Deep Debt Principles
//!
//! - **Zero unsafe code**: 100% safe Rust throughout
//! - **No hardcoding**: All parameters runtime-configurable
//! - **Capability-based**: Discovers hardware at runtime
//! - **No mocks**: All production implementations
//! - **Self-knowledge**: Runtime capability discovery
//! - **Modern idioms**: Async/await, builder patterns
//!
//! # Module Structure
//!
//! - `config` - Network configuration and hardware preferences
//! - `layer` - Layer types and definitions
//! - `optimizer` - Optimization algorithms
//! - `loss` - Loss function implementations
//! - `metrics` - Training and evaluation metrics
//!
//! # Example
//!
//! ```rust,ignore
//! use barracuda::nn::{NeuralNetwork, Layer, Optimizer, LossFunction};
//! use barracuda::prelude::WgpuDevice;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let device = WgpuDevice::new().await?;
//!
//! // Build network with capability detection
//! let mut model = NeuralNetwork::builder(&device)
//!     .add_layer(Layer::Linear { in_features: 784, out_features: 128 })
//!     .add_layer(Layer::ReLU)
//!     .add_layer(Layer::Linear { in_features: 128, out_features: 10 })
//!     .optimizer(Optimizer::Adam { lr: 0.001, betas: (0.9, 0.999), eps: 1e-8 })
//!     .loss(LossFunction::CrossEntropy)
//!     .build()
//!     .await?;
//!
//! // Train (discovers optimal hardware at runtime)
//! let train_history = model.train(&train_data, epochs).await?;
//! # Ok(())
//! # }
//! ```

// S66: Blanket #![allow(dead_code)] removed. Individual items annotated where
// genuinely pending implementation.

// Re-export core types
pub use config::{HardwarePreference, NetworkConfig};
pub use layer::Layer;
pub use loss::LossFunction;
pub use metrics::{EvalMetrics, TrainHistory, TrainingMetrics};
pub use optimizer::Optimizer;

pub mod esn_classifier;
pub mod fused_mlp;
pub mod lstm_reservoir;
pub mod simple_mlp;

mod config;
mod layer;
mod loss;
mod metrics;
mod optimizer;

pub use esn_classifier::{EsnClassifier, EsnConfig};
pub use fused_mlp::{Activation, fused_mlp};
pub use lstm_reservoir::{LstmReservoir, LstmReservoirConfig, LstmState};
pub use simple_mlp::SimpleMlp;
