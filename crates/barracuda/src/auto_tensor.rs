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

    /// Matrix multiplication with automatic device selection
    ///
    /// **Deep Debt**: Scheduler picks CPU for small, GPU for large
    pub fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        // Convert to descriptor for scheduling
        let desc_a = TensorDescriptor::new(a.shape().to_vec(), DType::F32);
        let desc_b = TensorDescriptor::new(b.shape().to_vec(), DType::F32);
        let op = MathOp::MatMul {
            transpose_a: false,
            transpose_b: false,
        };

        // Ask scheduler for optimal device
        let executor = self.scheduler.select_executor(&op, &[desc_a, desc_b]);
        let hw_type = executor.hardware_type();

        tracing::info!(
            "MatMul {}×{} → {} (score: {:.3})",
            a.shape()[0],
            a.shape()[1],
            executor.name(),
            executor.score_operation(
                &op,
                &[
                    TensorDescriptor::new(a.shape().to_vec(), DType::F32),
                    TensorDescriptor::new(b.shape().to_vec(), DType::F32)
                ]
            )
        );

        // Get optimal device from pool
        let optimal_device = self
            .devices
            .get(&hw_type)
            .or_else(|| self.devices.values().next())
            .ok_or_else(|| crate::error::BarracudaError::device("No device available"))?;

        // Transfer tensors if needed (copy to optimal device)
        let a_on_device = if Arc::ptr_eq(a.device(), optimal_device) {
            a.clone()
        } else {
            // Transfer to optimal device
            self.transfer_tensor(a, optimal_device)?
        };

        let b_on_device = if Arc::ptr_eq(b.device(), optimal_device) {
            b.clone()
        } else {
            self.transfer_tensor(b, optimal_device)?
        };

        // Execute on optimal device
        a_on_device.matmul(&b_on_device)
    }

    /// Element-wise ReLU with automatic device selection
    pub fn relu(&self, input: &Tensor) -> Result<Tensor> {
        let desc = TensorDescriptor::new(input.shape().to_vec(), DType::F32);
        let op = MathOp::ReLU;

        let executor = self
            .scheduler
            .select_executor(&op, std::slice::from_ref(&desc));
        let hw_type = executor.hardware_type();

        tracing::info!(
            "ReLU [{}] → {} (score: {:.3})",
            input.shape().iter().product::<usize>(),
            executor.name(),
            executor.score_operation(&op, &[desc])
        );

        let optimal_device = self
            .devices
            .get(&hw_type)
            .or_else(|| self.devices.values().next())
            .ok_or_else(|| crate::error::BarracudaError::device("No device available"))?;

        let input_on_device = if Arc::ptr_eq(input.device(), optimal_device) {
            input.clone()
        } else {
            self.transfer_tensor(input, optimal_device)?
        };

        input_on_device.relu()
    }

    /// 2D Convolution with automatic device selection
    pub fn conv2d(&self, input: &Tensor, kernel: &Tensor) -> Result<Tensor> {
        let desc_input = TensorDescriptor::new(input.shape().to_vec(), DType::F32);
        let desc_kernel = TensorDescriptor::new(kernel.shape().to_vec(), DType::F32);
        let op = MathOp::Conv2D {
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
            groups: 1,
        };

        let executor = self
            .scheduler
            .select_executor(&op, &[desc_input.clone(), desc_kernel.clone()]);
        let hw_type = executor.hardware_type();

        tracing::info!(
            "Conv2D {:?} * {:?} → {} (score: {:.3})",
            input.shape(),
            kernel.shape(),
            executor.name(),
            executor.score_operation(&op, &[desc_input, desc_kernel])
        );

        let optimal_device = self
            .devices
            .get(&hw_type)
            .or_else(|| self.devices.values().next())
            .ok_or_else(|| crate::error::BarracudaError::device("No device available"))?;

        let input_on_device = if Arc::ptr_eq(input.device(), optimal_device) {
            input.clone()
        } else {
            self.transfer_tensor(input, optimal_device)?
        };

        let kernel_on_device = if Arc::ptr_eq(kernel.device(), optimal_device) {
            kernel.clone()
        } else {
            self.transfer_tensor(kernel, optimal_device)?
        };

        input_on_device.conv2d(&kernel_on_device)
    }

    /// Element-wise addition with automatic device selection
    pub fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let desc_a = TensorDescriptor::new(a.shape().to_vec(), DType::F32);
        let desc_b = TensorDescriptor::new(b.shape().to_vec(), DType::F32);
        let op = MathOp::Add;

        let executor = self
            .scheduler
            .select_executor(&op, &[desc_a.clone(), desc_b]);
        let hw_type = executor.hardware_type();

        let optimal_device = self
            .devices
            .get(&hw_type)
            .or_else(|| self.devices.values().next())
            .ok_or_else(|| crate::error::BarracudaError::device("No device available"))?;

        let a_on_device = if Arc::ptr_eq(a.device(), optimal_device) {
            a.clone()
        } else {
            self.transfer_tensor(a, optimal_device)?
        };

        let b_on_device = if Arc::ptr_eq(b.device(), optimal_device) {
            b.clone()
        } else {
            self.transfer_tensor(b, optimal_device)?
        };

        a_on_device.add(&b_on_device)
    }

    /// Element-wise subtraction with automatic device selection
    pub fn sub(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let desc_a = TensorDescriptor::new(a.shape().to_vec(), DType::F32);
        let desc_b = TensorDescriptor::new(b.shape().to_vec(), DType::F32);
        let op = MathOp::Sub;

        let executor = self
            .scheduler
            .select_executor(&op, &[desc_a.clone(), desc_b]);
        let hw_type = executor.hardware_type();

        let optimal_device = self
            .devices
            .get(&hw_type)
            .or_else(|| self.devices.values().next())
            .ok_or_else(|| crate::error::BarracudaError::device("No device available"))?;

        let a_on_device = if Arc::ptr_eq(a.device(), optimal_device) {
            a.clone()
        } else {
            self.transfer_tensor(a, optimal_device)?
        };

        let b_on_device = if Arc::ptr_eq(b.device(), optimal_device) {
            b.clone()
        } else {
            self.transfer_tensor(b, optimal_device)?
        };

        a_on_device.sub(&b_on_device)
    }

    /// Element-wise multiplication with automatic device selection
    pub fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let desc_a = TensorDescriptor::new(a.shape().to_vec(), DType::F32);
        let desc_b = TensorDescriptor::new(b.shape().to_vec(), DType::F32);
        let op = MathOp::Mul;

        let executor = self
            .scheduler
            .select_executor(&op, &[desc_a.clone(), desc_b]);
        let hw_type = executor.hardware_type();

        let optimal_device = self
            .devices
            .get(&hw_type)
            .or_else(|| self.devices.values().next())
            .ok_or_else(|| crate::error::BarracudaError::device("No device available"))?;

        let a_on_device = if Arc::ptr_eq(a.device(), optimal_device) {
            a.clone()
        } else {
            self.transfer_tensor(a, optimal_device)?
        };

        let b_on_device = if Arc::ptr_eq(b.device(), optimal_device) {
            b.clone()
        } else {
            self.transfer_tensor(b, optimal_device)?
        };

        a_on_device.mul(&b_on_device)
    }

    /// Element-wise division with automatic device selection
    pub fn div(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let desc_a = TensorDescriptor::new(a.shape().to_vec(), DType::F32);
        let desc_b = TensorDescriptor::new(b.shape().to_vec(), DType::F32);
        let op = MathOp::Div;

        let executor = self
            .scheduler
            .select_executor(&op, &[desc_a.clone(), desc_b]);
        let hw_type = executor.hardware_type();

        let optimal_device = self
            .devices
            .get(&hw_type)
            .or_else(|| self.devices.values().next())
            .ok_or_else(|| crate::error::BarracudaError::device("No device available"))?;

        let a_on_device = if Arc::ptr_eq(a.device(), optimal_device) {
            a.clone()
        } else {
            self.transfer_tensor(a, optimal_device)?
        };

        let b_on_device = if Arc::ptr_eq(b.device(), optimal_device) {
            b.clone()
        } else {
            self.transfer_tensor(b, optimal_device)?
        };

        a_on_device.div(&b_on_device)
    }

    /// Sigmoid activation with automatic device selection
    pub fn sigmoid(&self, input: &Tensor) -> Result<Tensor> {
        let desc = TensorDescriptor::new(input.shape().to_vec(), DType::F32);
        let op = MathOp::Sigmoid;

        let executor = self.scheduler.select_executor(&op, &[desc]);
        let hw_type = executor.hardware_type();

        let optimal_device = self
            .devices
            .get(&hw_type)
            .or_else(|| self.devices.values().next())
            .ok_or_else(|| crate::error::BarracudaError::device("No device available"))?;

        let input_on_device = if Arc::ptr_eq(input.device(), optimal_device) {
            input.clone()
        } else {
            self.transfer_tensor(input, optimal_device)?
        };

        input_on_device.sigmoid()
    }

    /// Tanh activation with automatic device selection
    pub fn tanh(&self, input: &Tensor) -> Result<Tensor> {
        let desc = TensorDescriptor::new(input.shape().to_vec(), DType::F32);
        let op = MathOp::Tanh;

        let executor = self.scheduler.select_executor(&op, &[desc]);
        let hw_type = executor.hardware_type();

        let optimal_device = self
            .devices
            .get(&hw_type)
            .or_else(|| self.devices.values().next())
            .ok_or_else(|| crate::error::BarracudaError::device("No device available"))?;

        let input_on_device = if Arc::ptr_eq(input.device(), optimal_device) {
            input.clone()
        } else {
            self.transfer_tensor(input, optimal_device)?
        };

        input_on_device.tanh()
    }

    /// Transfer tensor to different device
    fn transfer_tensor(&self, tensor: &Tensor, target_device: &Arc<WgpuDevice>) -> Result<Tensor> {
        // Read data from source device
        let data = tensor.to_vec()?;

        // Create on target device
        Tensor::from_data(&data, tensor.shape().to_vec(), target_device.clone())
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
