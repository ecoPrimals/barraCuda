//! 3D Fast Fourier Transform Operation
//!
//! **Purpose**: 3D frequency analysis for PPPM molecular dynamics
//! **Algorithm**: Dimension-wise decomposition using 1D FFT
//!
//! **CRITICAL FOR PPPM**: This operation unblocks molecular dynamics!

use super::Fft1D;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// 3D Complex FFT operation
pub struct Fft3D {
    input: Tensor,
    nx: u32,
    ny: u32,
    nz: u32,
}

impl Fft3D {
    pub fn new(input: Tensor, nx: u32, ny: u32, nz: u32) -> Result<Self> {
        let shape = input.shape();

        if shape.len() != 4 {
            return Err(BarracudaError::Device(
                "FFT 3D input must have 4 dimensions [nx, ny, nz, 2]".to_string(),
            ));
        }

        if shape[0] != nx as usize
            || shape[1] != ny as usize
            || shape[2] != nz as usize
            || shape[3] != 2
        {
            return Err(BarracudaError::Device(format!(
                "FFT 3D shape mismatch: expected [{nx}, {ny}, {nz}, 2], got {shape:?}"
            )));
        }

        if nx & (nx - 1) != 0 || ny & (ny - 1) != 0 || nz & (nz - 1) != 0 {
            return Err(BarracudaError::Device(
                "FFT 3D dimensions must be powers of 2".to_string(),
            ));
        }

        Ok(Self { input, nx, ny, nz })
    }

    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();

        // FFT along X
        let mut x_results = Vec::new();
        for y in 0..self.ny {
            for z in 0..self.nz {
                let pencil = self.extract_x_pencil(y, z)?;
                let tensor = Tensor::from_data(&pencil, vec![self.nx as usize, 2], device.clone())?;
                let fft = Fft1D::new(tensor, self.nx)?;
                x_results.extend(fft.execute()?.to_vec()?);
            }
        }

        let x_transformed = Tensor::from_data(
            &x_results,
            vec![self.nx as usize, self.ny as usize, self.nz as usize, 2],
            device.clone(),
        )?;

        // FFT along Y
        let mut y_results = Vec::new();
        for x in 0..self.nx {
            for z in 0..self.nz {
                let pencil = self.extract_y_pencil(&x_transformed, x, z)?;
                let tensor = Tensor::from_data(&pencil, vec![self.ny as usize, 2], device.clone())?;
                let fft = Fft1D::new(tensor, self.ny)?;
                y_results.extend(fft.execute()?.to_vec()?);
            }
        }

        let y_transformed = Tensor::from_data(
            &y_results,
            vec![self.nx as usize, self.ny as usize, self.nz as usize, 2],
            device.clone(),
        )?;

        // FFT along Z
        let mut z_results = Vec::new();
        for x in 0..self.nx {
            for y in 0..self.ny {
                let pencil = self.extract_z_pencil(&y_transformed, x, y)?;
                let tensor = Tensor::from_data(&pencil, vec![self.nz as usize, 2], device.clone())?;
                let fft = Fft1D::new(tensor, self.nz)?;
                z_results.extend(fft.execute()?.to_vec()?);
            }
        }

        Tensor::from_data(
            &z_results,
            vec![self.nx as usize, self.ny as usize, self.nz as usize, 2],
            device.clone(),
        )
    }

    fn extract_x_pencil(&self, y: u32, z: u32) -> Result<Vec<f32>> {
        let data = self.input.to_vec()?;
        let mut pencil = Vec::with_capacity((self.nx * 2) as usize);
        for x in 0..self.nx {
            let idx = ((x * self.ny * self.nz + y * self.nz + z) * 2) as usize;
            pencil.push(data[idx]);
            pencil.push(data[idx + 1]);
        }
        Ok(pencil)
    }

    fn extract_y_pencil(&self, tensor: &Tensor, x: u32, z: u32) -> Result<Vec<f32>> {
        let data = tensor.to_vec()?;
        let mut pencil = Vec::with_capacity((self.ny * 2) as usize);
        for y in 0..self.ny {
            let idx = ((x * self.ny * self.nz + y * self.nz + z) * 2) as usize;
            pencil.push(data[idx]);
            pencil.push(data[idx + 1]);
        }
        Ok(pencil)
    }

    fn extract_z_pencil(&self, tensor: &Tensor, x: u32, y: u32) -> Result<Vec<f32>> {
        let data = tensor.to_vec()?;
        let mut pencil = Vec::with_capacity((self.nz * 2) as usize);
        for z in 0..self.nz {
            let idx = ((x * self.ny * self.nz + y * self.nz + z) * 2) as usize;
            pencil.push(data[idx]);
            pencil.push(data[idx + 1]);
        }
        Ok(pencil)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_fft_3d_simple() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };

        // 2×2×2 FFT test
        let data = vec![
            1.0f32, 0.0, 2.0, 0.0, // (0,0,:)
            3.0, 0.0, 4.0, 0.0, // (0,1,:)
            5.0, 0.0, 6.0, 0.0, // (1,0,:)
            7.0, 0.0, 8.0, 0.0, // (1,1,:)
        ];

        let tensor = Tensor::from_data(&data, vec![2, 2, 2, 2], device.clone()).unwrap();
        let fft = Fft3D::new(tensor, 2, 2, 2).unwrap();
        let result = fft.execute().unwrap();
        let result_data = result.to_vec().unwrap();
        assert_eq!(result_data.len(), 16); // 2×2×2×2 = 16 floats
    }
}
