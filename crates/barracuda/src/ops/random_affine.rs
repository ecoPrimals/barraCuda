// SPDX-License-Identifier: AGPL-3.0-or-later
//! RandomAffine - Random affine transformations
//!
//! Applies random rotation, translation, scale, and shear.
//! Comprehensive geometric augmentation.
//!
//! Deep Debt Principles:
//! - Pure GPU/WGSL execution
//! - Safe Rust wrappers
//! - Hardware-agnostic via WebGPU
//! - Runtime device discovery
//! - Zero CPU fallbacks in execution

use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;

const SHADER_F64: &str = include_str!("../shaders/augmentation/random_affine_f64.wgsl");
static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| crate::shaders::precision::downcast_f64_to_f32(SHADER_F64));

/// RandomAffine operation
pub struct RandomAffine {
    input: Tensor,
    degrees: f32,
    translate: (f32, f32),
    scale: (f32, f32),
    shear: f32,
    seed: u64,
}

impl RandomAffine {
    /// Create a new random affine operation
    pub fn new(
        input: Tensor,
        degrees: f32,
        translate: (f32, f32),
        scale: (f32, f32),
        shear: f32,
        seed: u64,
    ) -> Result<Self> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(crate::error::BarracudaError::invalid_op(
                "RandomAffine",
                format!("Expected 3D tensor (C, H, W), got {}D", shape.len()),
            ));
        }

        Ok(Self {
            input,
            degrees,
            translate,
            scale,
            shear,
            seed,
        })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

    /// Execute the random affine operation
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();

        let channels = shape[0];
        let height = shape[1];
        let width = shape[2];

        // Generate random parameters from seed (CPU-side, deterministic)
        let angle = self.degrees * (((self.seed * 1_103_515_245) % 2000) as f32 / 1000.0 - 1.0);
        let tx = self.translate.0
            * width as f32
            * (((self.seed * 22_695_477) % 2000) as f32 / 1000.0 - 1.0);
        let ty = self.translate.1
            * height as f32
            * (((self.seed * 1_664_525) % 2000) as f32 / 1000.0 - 1.0);
        let sc = self.scale.0
            + (self.scale.1 - self.scale.0) * ((self.seed * 48_271) % 1000) as f32 / 1000.0;
        let sh = self.shear * (((self.seed * 69_621) % 2000) as f32 / 1000.0 - 1.0);

        // Build affine matrix
        let angle_rad = angle * std::f32::consts::PI / 180.0;
        let shear_rad = sh * std::f32::consts::PI / 180.0;

        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();
        let tan_sh = shear_rad.tan();

        let a = sc * cos_a;
        let b = sc * (-sin_a + cos_a * tan_sh);
        let c = tx;
        let d = sc * sin_a;
        let e = sc * (cos_a + sin_a * tan_sh);
        let f = ty;

        let cx = width as f32 / 2.0;
        let cy = height as f32 / 2.0;

        let output_size = channels * height * width;

        // Create buffers
        let input_buffer = self.input.buffer();

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RandomAffine Output"),
            size: (output_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            channels: u32,
            height: u32,
            width: u32,
            a: f32,
            b: f32,
            c: f32,
            d: f32,
            e: f32,
            f: f32,
            cx: f32,
            cy: f32,
        }

        let params = Params {
            channels: channels as u32,
            height: height as u32,
            width: width as u32,
            a,
            b,
            c,
            d,
            e,
            f,
            cx,
            cy,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("RandomAffine Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let caps = DeviceCapabilities::from_device(device);
        let (wg_x, wg_y) = caps.optimal_workgroup_size_2d(WorkloadType::Convolution);
        let workgroups_x = (width as u32).div_ceil(wg_x);
        let workgroups_y = (height as u32).div_ceil(wg_y);

        ComputeDispatch::new(device, "random_affine")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, input_buffer)
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch(workgroups_x, workgroups_y, 1)
            .submit();

        // Read back results
        let output_data = crate::utils::read_buffer(device, &output_buffer, output_size)?;

        Ok(Tensor::new(
            output_data,
            vec![channels, height, width],
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Apply random affine transformation
    ///
    /// # Arguments
    ///
    /// * `degrees` - Max rotation in degrees
    /// * `translate` - Max translation fraction (x, y)
    /// * `scale` - Scale range (min, max)
    /// * `shear` - Max shear in degrees
    /// * `seed` - Random seed for deterministic transformation
    pub fn random_affine(
        self,
        degrees: f32,
        translate: (f32, f32),
        scale: (f32, f32),
        shear: f32,
        seed: u64,
    ) -> Result<Self> {
        RandomAffine::new(self, degrees, translate, scale, shear, seed)?.execute()
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_random_affine() {
        let image_data = vec![1.0; 3 * 64 * 64];
        let tensor = Tensor::from_vec(image_data.clone(), vec![3, 64, 64])
            .await
            .unwrap();
        let transformed_tensor = tensor
            .random_affine(15.0, (0.1, 0.1), (0.9, 1.1), 5.0, 42_424)
            .unwrap();
        let transformed = transformed_tensor.to_vec().unwrap();
        assert_eq!(transformed.len(), image_data.len());
    }
}
