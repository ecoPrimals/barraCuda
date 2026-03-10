// SPDX-License-Identifier: AGPL-3.0-only
//! Neural Network Layer Types
//!
//! Layer definitions for building neural networks.
//! Capability-based: Hardware requirements discovered at runtime.

/// Neural network layer types.
#[derive(Debug, Clone)]
pub enum Layer {
    /// Linear (fully connected) layer
    Linear {
        /// Input feature dimension
        in_features: usize,
        /// Output feature dimension
        out_features: usize,
    },
    /// 2D Convolution
    Conv2D {
        /// Number of input channels
        in_channels: usize,
        /// Number of output channels
        out_channels: usize,
        /// Kernel size (square)
        kernel_size: usize,
    },
    /// Max pooling 2D
    MaxPool2D {
        /// Pooling kernel size
        kernel_size: usize,
        /// Stride between pooling windows
        stride: usize,
    },
    /// Batch normalization
    BatchNorm {
        /// Number of features to normalize
        num_features: usize,
    },
    /// Layer normalization
    LayerNorm {
        /// Shape of dimensions to normalize (e.g. [C] or [H, W, C])
        normalized_shape: Vec<usize>,
    },
    /// Dropout
    Dropout {
        /// Dropout probability (0.0–1.0)
        rate: f32,
    },
    /// `ReLU` activation
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
