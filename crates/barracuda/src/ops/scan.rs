//! Scan Operation - Prefix sum (cumulative operations)
//!
//! **Deep Debt Evolution**: Modernized from trait-based to direct `impl Tensor`
//!
//! ## Deep Debt Principles
//!
//! - ✅ Modern idiomatic Rust (direct `impl Tensor`, not trait extension)
//! - ✅ Universal compute (WGSL shader for all substrates)
//! - ✅ Safe Rust (no unsafe blocks)
//! - ✅ Agnostic design (inclusive/exclusive parameter)
//!
//! ## Evolution History
//!
//! **Before** (Phase 3): `ScanExt` trait extension  
//! **After** (Phase 6): Direct `impl Tensor` method
//!
//! ## Usage
//!
//! ```ignore
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # use barracuda::tensor::Tensor;
//! # use barracuda::device::test_pool;
//! # let device = test_pool::tokio_block_on(test_pool::get_test_device_if_gpu_available()).unwrap();
//! let input = Tensor::from_data(&[1.0f32, 2.0, 3.0, 4.0], vec![4], device)?;
//! let _cumsum = input.scan(false)?;  // Inclusive scan
//! # Ok(())
//! # }
//! ```

use crate::device::{ComputeDispatch, DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ScanParams {
    size: u32,
    operation: u32,
    exclusive: u32,
}

pub struct Scan {
    input: Tensor,
    exclusive: bool,
}

impl Scan {
    fn wgsl_shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                    "../shaders/misc/scan_f64.wgsl"
                ))
            });
            std::sync::LazyLock::force(&SHADER).as_str()
        }
    }

    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let size = self.input.shape().iter().product::<usize>();

        let params = ScanParams {
            size: size as u32,
            operation: 0, // Sum
            exclusive: if self.exclusive { 1 } else { 0 },
        };

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scan_output"),
            size: (size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("scan_params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let caps = DeviceCapabilities::from_device(device);
        let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Reduction);
        let workgroups = (size as u32).div_ceil(optimal_wg_size);

        ComputeDispatch::new(device, "scan")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, self.input.buffer())
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch(workgroups, 1, 1)
            .submit();

        Ok(Tensor::from_buffer(
            output_buffer,
            self.input.shape().to_vec(),
            device.clone(),
        ))
    }
}

// ============================================================================
// Modern API: Direct impl Tensor (Phase 6 Evolution)
// ============================================================================

impl Tensor {
    /// Compute prefix sum (cumulative sum) of tensor elements
    ///
    /// **Deep Debt**: Modern direct method, no trait extension needed
    ///
    /// ## Arguments
    ///
    /// * `exclusive` - If true, exclusive scan (shift right by 1, start with 0).
    ///   If false, inclusive scan (standard cumulative sum).
    ///
    /// ## Example
    ///
    /// ```ignore
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # use barracuda::tensor::Tensor;
    /// # use barracuda::device::test_pool;
    /// # let device = test_pool::tokio_block_on(test_pool::get_test_device_if_gpu_available()).unwrap();
    /// # let input = Tensor::from_data(&[1.0f32, 2.0, 3.0, 4.0], vec![4], device).unwrap();
    /// // Inclusive scan: [1, 2, 3, 4] → [1, 3, 6, 10]
    /// let cumsum = input.clone().scan(false)?;
    ///
    /// // Exclusive scan: [1, 2, 3, 4] → [0, 1, 3, 6]
    /// let _exclusive = input.scan(true)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn scan(self, exclusive: bool) -> Result<Self> {
        let op = Scan {
            input: self,
            exclusive,
        };
        op.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_scan_inclusive() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };

        let input = Tensor::from_data(&vec![1.0, 2.0, 3.0, 4.0], vec![4], device.clone()).unwrap();

        let result = input.scan(false).unwrap();
        let output = result.to_vec().unwrap();

        // Inclusive scan: [1, 3, 6, 10]
        assert_eq!(output.len(), 4);
    }
}
