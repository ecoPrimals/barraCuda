// SPDX-License-Identifier: AGPL-3.0-or-later
//! Unified Math Base - Hardware-Agnostic Mathematical Primitives
//!
//! **Philosophy**: Define WHAT to compute, not HOW
//!
//! This module provides the mathematical foundation for `BarraCuda`:
//! - Pure mathematical definitions (no hardware assumptions)
//! - Trait-based abstraction (works with any backend)
//! - Composable primitives (build complex ops from simple ones)
//! - Type-safe tensors (compile-time shape checking where possible)
//!
//! **Deep Debt Principles**:
//! - ✅ Hardware agnostic (no GPU/CPU/TPU assumptions)
//! - ✅ Math-first (correct semantics before optimization)
//! - ✅ Composable (primitives combine naturally)
//! - ✅ Type-safe (catch errors at compile time)

use std::fmt;

/// Mathematical operation primitive
///
/// Represents a fundamental mathematical operation that can be
/// executed on any hardware backend (GPU, CPU, TPU, NPU).
#[derive(Debug, Clone, PartialEq)]
pub enum MathOp {
    // ═══════════════════════════════════════════════════════════
    // UNARY OPERATIONS (one input, one output)
    // ═══════════════════════════════════════════════════════════
    /// Negate: y = -x
    Negate,

    /// Absolute value: y = |x|
    Abs,

    /// Square: y = x²
    Square,

    /// Square root: y = √x
    Sqrt,

    /// Reciprocal: y = 1/x
    Reciprocal,

    /// Exponential: y = eˣ
    Exp,

    /// Natural log: y = ln(x)
    Log,

    /// Sine: y = sin(x)
    Sin,

    /// Cosine: y = cos(x)
    Cos,

    /// Tangent: y = tan(x)
    Tan,

    // ═══════════════════════════════════════════════════════════
    // BINARY OPERATIONS (two inputs, one output)
    // ═══════════════════════════════════════════════════════════
    /// Addition: z = x + y
    Add,

    /// Subtraction: z = x - y
    Sub,

    /// Multiplication: z = x * y
    Mul,

    /// Division: z = x / y
    Div,

    /// Power: z = xʸ
    Pow,

    /// Maximum: z = max(x, y)
    Max,

    /// Minimum: z = min(x, y)
    Min,

    // ═══════════════════════════════════════════════════════════
    // REDUCTION OPERATIONS (reduce along dimension)
    // ═══════════════════════════════════════════════════════════
    /// Sum reduction: ∑x
    ReduceSum {
        /// Dimension along which to reduce; `None` reduces all.
        dim: Option<usize>,
        /// Whether to keep the reduced dimension with size 1.
        keepdim: bool,
    },

    /// Mean reduction: mean(x)
    ReduceMean {
        /// Dimension along which to reduce; `None` reduces all.
        dim: Option<usize>,
        /// Whether to keep the reduced dimension with size 1.
        keepdim: bool,
    },

    /// Max reduction: max(x)
    ReduceMax {
        /// Dimension along which to reduce; `None` reduces all.
        dim: Option<usize>,
        /// Whether to keep the reduced dimension with size 1.
        keepdim: bool,
    },

    /// Min reduction: min(x)
    ReduceMin {
        /// Dimension along which to reduce; `None` reduces all.
        dim: Option<usize>,
        /// Whether to keep the reduced dimension with size 1.
        keepdim: bool,
    },

    /// Product reduction: ∏x
    ReduceProd {
        /// Dimension along which to reduce; `None` reduces all.
        dim: Option<usize>,
        /// Whether to keep the reduced dimension with size 1.
        keepdim: bool,
    },

    // ═══════════════════════════════════════════════════════════
    // MATRIX OPERATIONS
    // ═══════════════════════════════════════════════════════════
    /// Matrix multiply: C = A @ B
    MatMul {
        /// Whether to transpose the first matrix.
        transpose_a: bool,
        /// Whether to transpose the second matrix.
        transpose_b: bool,
    },

    /// Matrix transpose: Aᵀ
    Transpose {
        /// Permutation of dimension indices.
        perm: Vec<usize>,
    },

    /// Batch matrix multiply: [C₁, C₂, ...] = [A₁, A₂, ...] @ [B₁, B₂, ...]
    BatchMatMul {
        /// Whether to transpose the first matrix in each batch.
        transpose_a: bool,
        /// Whether to transpose the second matrix in each batch.
        transpose_b: bool,
    },

    // ═══════════════════════════════════════════════════════════
    // SHAPE OPERATIONS
    // ═══════════════════════════════════════════════════════════
    /// Reshape: change shape without copying data
    Reshape {
        /// Target shape dimensions (-1 for inferred).
        new_shape: Vec<i64>,
    },

    /// Broadcast: expand shape by repeating values
    Broadcast {
        /// Target shape to broadcast to.
        target_shape: Vec<usize>,
    },

    /// Squeeze: remove dimensions of size 1
    Squeeze {
        /// Dimensions to squeeze; `None` squeezes all size-1 dims.
        dims: Option<Vec<usize>>,
    },

    /// Unsqueeze: add dimensions of size 1
    Unsqueeze {
        /// Dimensions at which to insert size-1.
        dims: Vec<usize>,
    },

    /// Concat: join tensors along dimension
    Concat {
        /// Dimension along which to concatenate.
        dim: usize,
    },

    /// Split: split tensor along dimension
    Split {
        /// Dimension along which to split.
        dim: usize,
        /// Size of each split chunk.
        sizes: Vec<usize>,
    },

    // ═══════════════════════════════════════════════════════════
    // ACTIVATION FUNCTIONS
    // ═══════════════════════════════════════════════════════════
    /// `ReLU`: max(0, x)
    ReLU,

    /// Sigmoid: 1 / (1 + e⁻ˣ)
    Sigmoid,

    /// Tanh: (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)
    Tanh,

    /// Softmax: eˣⁱ / ∑eˣʲ
    Softmax {
        /// Dimension along which to apply softmax.
        dim: i64,
    },

    /// GELU: x * Φ(x) where Φ is standard normal CDF
    GELU,

    // ═══════════════════════════════════════════════════════════
    // CONVOLUTION OPERATIONS
    // ═══════════════════════════════════════════════════════════
    /// 2D Convolution
    Conv2D {
        /// Stride (height, width).
        stride: (usize, usize),
        /// Padding (height, width).
        padding: (usize, usize),
        /// Dilation (height, width).
        dilation: (usize, usize),
        /// Number of groups for grouped convolution.
        groups: usize,
    },

    /// 2D Max Pooling
    MaxPool2D {
        /// Kernel size (height, width).
        kernel_size: (usize, usize),
        /// Stride (height, width).
        stride: (usize, usize),
        /// Padding (height, width).
        padding: (usize, usize),
    },

    /// 2D Average Pooling
    AvgPool2D {
        /// Kernel size (height, width).
        kernel_size: (usize, usize),
        /// Stride (height, width).
        stride: (usize, usize),
        /// Padding (height, width).
        padding: (usize, usize),
    },
}

/// Unified math primitive trait
///
/// **Hardware-agnostic**: Define math operations without knowing the backend
pub trait MathPrimitive {
    /// Data type (f32, f64, i32, etc.)
    type Scalar: Copy + fmt::Debug;

    /// Execute unary operation: y = op(x)
    fn unary(&self, op: UnaryOp, x: &[Self::Scalar]) -> Vec<Self::Scalar>;

    /// Execute binary operation: z = op(x, y)
    fn binary(&self, op: BinaryOp, x: &[Self::Scalar], y: &[Self::Scalar]) -> Vec<Self::Scalar>;

    /// Execute reduction: y = reduce(x, dim)
    fn reduce(
        &self,
        op: ReduceOp,
        x: &[Self::Scalar],
        shape: &[usize],
        dim: Option<usize>,
    ) -> Vec<Self::Scalar>;

    /// Execute matrix multiply: C = A @ B
    fn matmul(
        &self,
        a: &[Self::Scalar],
        b: &[Self::Scalar],
        m: usize,
        n: usize,
        k: usize,
    ) -> Vec<Self::Scalar>;
}

/// Unary operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// Negate: y = -x
    Negate,
    /// Absolute value: y = |x|
    Abs,
    /// Square: y = x²
    Square,
    /// Square root: y = √x
    Sqrt,
    /// Reciprocal: y = 1/x
    Reciprocal,
    /// Exponential: y = eˣ
    Exp,
    /// Natural log: y = ln(x)
    Log,
    /// Sine: y = sin(x)
    Sin,
    /// Cosine: y = cos(x)
    Cos,
    /// Tangent: y = tan(x)
    Tan,
    /// `ReLU`: max(0, x)
    ReLU,
    /// Sigmoid: 1 / (1 + e⁻ˣ)
    Sigmoid,
    /// Tanh: (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)
    Tanh,
    /// GELU: x * Φ(x) where Φ is standard normal CDF
    GELU,
}

/// Binary operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    /// Addition: z = x + y
    Add,
    /// Subtraction: z = x - y
    Sub,
    /// Multiplication: z = x * y
    Mul,
    /// Division: z = x / y
    Div,
    /// Power: z = xʸ
    Pow,
    /// Maximum: z = max(x, y)
    Max,
    /// Minimum: z = min(x, y)
    Min,
}

/// Reduction operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    /// Sum reduction: ∑x
    Sum,
    /// Mean reduction: mean(x)
    Mean,
    /// Max reduction: max(x)
    Max,
    /// Min reduction: min(x)
    Min,
    /// Product reduction: ∏x
    Prod,
}

/// Hardware-agnostic tensor descriptor
///
/// Describes tensor metadata without storing data
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDescriptor {
    /// Shape (dimensions)
    pub shape: Vec<usize>,

    /// Data type
    pub dtype: DType,

    /// Strides (for non-contiguous tensors)
    pub strides: Vec<usize>,

    /// Total number of elements
    pub numel: usize,
}

impl TensorDescriptor {
    /// Create a new tensor descriptor from shape and data type.
    #[must_use]
    pub fn new(shape: Vec<usize>, dtype: DType) -> Self {
        let numel = shape.iter().product();
        let strides = Self::compute_strides(&shape);
        Self {
            shape,
            dtype,
            strides,
            numel,
        }
    }

    /// Compute strides for a contiguous tensor of the given shape.
    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Returns the rank (number of dimensions).
    #[must_use]
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Returns `true` if the tensor is a scalar (single element).
    #[must_use]
    pub fn is_scalar(&self) -> bool {
        self.numel == 1
    }

    /// Returns `true` if the tensor is a vector (rank 1).
    #[must_use]
    pub fn is_vector(&self) -> bool {
        self.rank() == 1
    }

    /// Returns `true` if the tensor is a matrix (rank 2).
    #[must_use]
    pub fn is_matrix(&self) -> bool {
        self.rank() == 2
    }
}

/// Data type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    /// 32-bit floating point
    F32,
    /// 64-bit floating point
    F64,
    /// 32-bit signed integer
    I32,
    /// 64-bit signed integer
    I64,
    /// 32-bit unsigned integer
    U32,
    /// 64-bit unsigned integer
    U64,
    /// Boolean
    Bool,
}

impl DType {
    /// Returns the size in bytes of a single element of this type.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::F64 | DType::I64 | DType::U64 => 8,
            DType::Bool => 1,
        }
    }
}

/// Math operation graph node
///
/// Represents a node in the computation graph
#[derive(Debug, Clone)]
pub struct OpNode {
    /// Operation
    pub op: MathOp,

    /// Input tensor descriptors
    pub inputs: Vec<TensorDescriptor>,

    /// Output tensor descriptor
    pub output: TensorDescriptor,

    /// Operation name (for debugging)
    pub name: Option<String>,
}

impl OpNode {
    /// Create a new operation node.
    #[must_use]
    pub fn new(op: MathOp, inputs: Vec<TensorDescriptor>, output: TensorDescriptor) -> Self {
        Self {
            op,
            inputs,
            output,
            name: None,
        }
    }

    /// Set the operation name for debugging.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_descriptor_creation() {
        let desc = TensorDescriptor::new(vec![2, 3, 4], DType::F32);
        assert_eq!(desc.shape, vec![2, 3, 4]);
        assert_eq!(desc.numel, 24);
        assert_eq!(desc.rank(), 3);
        assert!(!desc.is_scalar());
        assert!(!desc.is_vector());
        assert!(!desc.is_matrix());
    }

    #[test]
    fn test_dtype_size() {
        assert_eq!(DType::F32.size_bytes(), 4);
        assert_eq!(DType::F64.size_bytes(), 8);
        assert_eq!(DType::Bool.size_bytes(), 1);
    }

    #[test]
    fn test_tensor_shapes() {
        let scalar = TensorDescriptor::new(vec![1], DType::F32);
        assert!(scalar.is_scalar());

        let vector = TensorDescriptor::new(vec![10], DType::F32);
        assert!(vector.is_vector());

        let matrix = TensorDescriptor::new(vec![3, 4], DType::F32);
        assert!(matrix.is_matrix());
    }

    #[test]
    fn test_math_op_clone() {
        let op = MathOp::Add;
        let cloned = op.clone();
        assert_eq!(op, cloned);
    }
}
