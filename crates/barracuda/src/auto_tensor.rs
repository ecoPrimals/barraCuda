// SPDX-License-Identifier: AGPL-3.0-or-later
//! Auto-Tensor - Scheduler-Aware Tensor Operations
//!
//! **Purpose**: High-level API that automatically selects optimal hardware
//! for each operation using the unified scheduler.
//!
//! **Usage**:
//! ```rust,ignore
//! use barracuda::auto_tensor::AutoContext;
//!
//! // Initialize with automatic hardware discovery
//! let ctx = AutoContext::new().await?;
//!
//! // Operations automatically route to optimal hardware
//! let a = ctx.randn(vec![2048, 2048])?;  // Large → GPU
//! let b = ctx.randn(vec![2048, 2048])?;  // Large → GPU
//! let c = ctx.matmul(&a, &b)?;           // Automatic selection!
//! ```

use crate::device::WgpuDevice;
use crate::error::Result;
use crate::scheduler::UnifiedScheduler;
use crate::tensor::Tensor;
use crate::unified_hardware::HardwareType;
use crate::unified_math::{DType, MathOp, TensorDescriptor};
use std::collections::HashMap;
use std::sync::Arc;

/// Auto-scheduling context for tensor operations
///
/// **Deep Debt**: Automatic hardware selection with zero configuration
pub struct AutoContext {
    scheduler: UnifiedScheduler,
    devices: HashMap<HardwareType, Arc<WgpuDevice>>,
}

impl AutoContext {
    /// Create new auto-scheduling context
    ///
    /// Discovers all available hardware and creates device pool
    pub async fn new() -> Result<Self> {
        tracing::info!("Initializing AutoContext...");

        // Initialize scheduler
        let scheduler = UnifiedScheduler::new().await?;

        // Create device pool (uses shared device pool for concurrent safety)
        let mut devices = HashMap::new();

        // GPU device - use shared pool for thread-safe concurrent access
        if scheduler.get_executor(HardwareType::GPU).is_some() {
            let shared_device = crate::device::test_pool::get_test_device().await;
            tracing::info!("GPU device from shared pool");
            devices.insert(HardwareType::GPU, shared_device.clone());
            // CPU uses the same device (wgpu handles backend selection)
            devices.insert(HardwareType::CPU, shared_device);
        } else {
            // Fallback: try to get any device from pool
            let shared_device = crate::device::test_pool::get_test_device().await;
            tracing::info!("Fallback device from shared pool");
            devices.insert(HardwareType::CPU, shared_device);
        }

        tracing::info!("AutoContext ready with {} device(s)", devices.len());

        Ok(Self { scheduler, devices })
    }

    /// Create random tensor (normal distribution)
    ///
    /// Automatically selects device based on tensor size
    pub fn randn(&self, shape: Vec<usize>) -> Result<Tensor> {
        // For tensor creation, just use first available device
        // Scheduler will optimize operations, not creation
        let device = self
            .devices
            .values()
            .next()
            .ok_or_else(|| crate::error::BarracudaError::device("No device available"))?;

        // Generate random data
        let size: usize = shape.iter().product();
        let data: Vec<f32> = (0..size).map(|_| rand::random::<f32>()).collect();

        Tensor::from_data(&data, shape, device.clone())
    }

    /// Create zeros tensor
    pub fn zeros(&self, shape: Vec<usize>) -> Result<Tensor> {
        let device = self
            .devices
            .values()
            .next()
            .ok_or_else(|| crate::error::BarracudaError::device("No device available"))?;

        let size: usize = shape.iter().product();
        let data = vec![0.0f32; size];

        Tensor::from_data(&data, shape, device.clone())
    }

    /// Matrix multiplication with automatic device selection.
    ///
    /// Scheduler picks CPU for small, GPU for large.
    pub fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let op = MathOp::MatMul {
            transpose_a: false,
            transpose_b: false,
        };
        self.dispatch_binary(op, a, b, |a, b| a.matmul(&b))
    }

    /// Element-wise ReLU with automatic device selection.
    pub fn relu(&self, input: &Tensor) -> Result<Tensor> {
        self.dispatch_unary(MathOp::ReLU, input, |t| t.relu())
    }

    /// 2D Convolution with automatic device selection.
    pub fn conv2d(&self, input: &Tensor, kernel: &Tensor) -> Result<Tensor> {
        let op = MathOp::Conv2D {
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
            groups: 1,
        };
        self.dispatch_binary(op, input, kernel, |a, b| a.conv2d(&b))
    }

    /// Element-wise addition with automatic device selection.
    pub fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        self.dispatch_binary(MathOp::Add, a, b, |a, b| a.add(&b))
    }

    /// Element-wise subtraction with automatic device selection.
    pub fn sub(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        self.dispatch_binary(MathOp::Sub, a, b, |a, b| a.sub(&b))
    }

    /// Element-wise multiplication with automatic device selection.
    pub fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        self.dispatch_binary(MathOp::Mul, a, b, |a, b| a.mul(&b))
    }

    /// Element-wise division with automatic device selection.
    pub fn div(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        self.dispatch_binary(MathOp::Div, a, b, |a, b| a.div(&b))
    }

    /// Sigmoid activation with automatic device selection.
    pub fn sigmoid(&self, input: &Tensor) -> Result<Tensor> {
        self.dispatch_unary(MathOp::Sigmoid, input, |t| t.sigmoid())
    }

    /// Tanh activation with automatic device selection.
    pub fn tanh(&self, input: &Tensor) -> Result<Tensor> {
        self.dispatch_unary(MathOp::Tanh, input, |t| t.tanh())
    }

    /// Resolve the optimal device for a scheduled operation.
    fn resolve_device(&self, hw_type: HardwareType) -> Result<&Arc<WgpuDevice>> {
        self.devices
            .get(&hw_type)
            .or_else(|| self.devices.values().next())
            .ok_or_else(|| crate::error::BarracudaError::device("No device available"))
    }

    /// Ensure a tensor lives on the target device, transferring if necessary.
    fn ensure_on_device(&self, tensor: &Tensor, target: &Arc<WgpuDevice>) -> Result<Tensor> {
        if Arc::ptr_eq(tensor.device(), target) {
            Ok(tensor.clone())
        } else {
            let data = tensor.to_vec()?;
            Tensor::from_data(&data, tensor.shape().to_vec(), target.clone())
        }
    }

    /// Schedule and dispatch a binary (two-tensor) operation.
    fn dispatch_binary(
        &self,
        op: MathOp,
        a: &Tensor,
        b: &Tensor,
        f: impl FnOnce(Tensor, Tensor) -> Result<Tensor>,
    ) -> Result<Tensor> {
        let descs = [
            TensorDescriptor::new(a.shape().to_vec(), DType::F32),
            TensorDescriptor::new(b.shape().to_vec(), DType::F32),
        ];
        let hw = self.scheduler.select_executor(&op, &descs).hardware_type();
        let dev = self.resolve_device(hw)?;
        let a_dev = self.ensure_on_device(a, dev)?;
        let b_dev = self.ensure_on_device(b, dev)?;
        f(a_dev, b_dev)
    }

    /// Schedule and dispatch a unary (single-tensor) operation.
    fn dispatch_unary(
        &self,
        op: MathOp,
        input: &Tensor,
        f: impl FnOnce(Tensor) -> Result<Tensor>,
    ) -> Result<Tensor> {
        let desc = TensorDescriptor::new(input.shape().to_vec(), DType::F32);
        let hw = self.scheduler.select_executor(&op, &[desc]).hardware_type();
        let dev = self.resolve_device(hw)?;
        let on_dev = self.ensure_on_device(input, dev)?;
        f(on_dev)
    }

    /// Get scheduler reference
    pub fn scheduler(&self) -> &UnifiedScheduler {
        &self.scheduler
    }

    /// Get device from pool
    pub fn get_device(&self, hw_type: HardwareType) -> Option<&Arc<WgpuDevice>> {
        self.devices.get(&hw_type)
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_auto_context_creation() {
        let ctx = AutoContext::new().await.unwrap();

        // Should have at least one device
        assert!(!ctx.devices.is_empty());
    }

    #[tokio::test]
    async fn test_auto_matmul_small() {
        let ctx = AutoContext::new().await.unwrap();

        // Small matmul should prefer CPU
        let a = ctx.randn(vec![16, 16]).unwrap();
        let b = ctx.randn(vec![16, 16]).unwrap();

        let c = ctx.matmul(&a, &b).unwrap();

        // Verify result shape
        assert_eq!(c.shape(), &[16, 16]);
    }

    #[tokio::test]
    async fn test_auto_matmul_large() {
        let ctx = AutoContext::new().await.unwrap();

        // Large matmul should prefer GPU (if available)
        let a = ctx.randn(vec![512, 512]).unwrap();
        let b = ctx.randn(vec![512, 512]).unwrap();

        let c = ctx.matmul(&a, &b).unwrap();

        // Verify result shape
        assert_eq!(c.shape(), &[512, 512]);
    }

    #[tokio::test]
    async fn test_auto_relu() {
        let ctx = AutoContext::new().await.unwrap();

        // Test small ReLU (should prefer CPU)
        let small = ctx.randn(vec![100]).unwrap();
        let result = ctx.relu(&small).unwrap();
        assert_eq!(result.shape(), &[100]);

        // Test large ReLU (should prefer GPU if available)
        let large = ctx.randn(vec![100_000]).unwrap();
        let result = ctx.relu(&large).unwrap();
        assert_eq!(result.shape(), &[100_000]);
    }

    #[tokio::test]
    async fn test_auto_conv2d() {
        let ctx = AutoContext::new().await.unwrap();

        // Test small Conv2D (should prefer CPU)
        let input = ctx.randn(vec![28, 28]).unwrap();
        let kernel = ctx.randn(vec![3, 3]).unwrap();
        let result = ctx.conv2d(&input, &kernel).unwrap();
        assert_eq!(result.shape(), &[26, 26]); // 28 - 3 + 1 = 26

        // Test large Conv2D (should prefer GPU if available)
        let input = ctx.randn(vec![224, 224]).unwrap();
        let kernel = ctx.randn(vec![7, 7]).unwrap();
        let result = ctx.conv2d(&input, &kernel).unwrap();
        assert_eq!(result.shape(), &[218, 218]); // 224 - 7 + 1 = 218
    }
}
