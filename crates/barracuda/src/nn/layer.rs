// SPDX-License-Identifier: AGPL-3.0-or-later
//! Neural Network Layer Types
//!
//! Layer definitions for building neural networks.
//! Capability-based: Hardware requirements discovered at runtime.

/// Neural network layer types
#[derive(Debug, Clone)]
pub enum Layer {
    /// Linear (fully connected) layer
    Linear {
        in_features: usize,
        out_features: usize,
    },
    /// 2D Convolution
    Conv2D {
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
    },
    /// Max pooling 2D
    MaxPool2D { kernel_size: usize, stride: usize },
    /// Batch normalization
    BatchNorm { num_features: usize },
    /// Layer normalization
    LayerNorm { normalized_shape: Vec<usize> },
    /// Dropout
    Dropout { rate: f32 },
    /// ReLU activation
    ReLU,
    /// GELU activation
    GELU,
    /// Tanh activation
    Tanh,
    /// Sigmoid activation
    Sigmoid,
    /// Softmax activation
    Softmax,
}
