// SPDX-License-Identifier: AGPL-3.0-or-later
//! Langevin Thermostat
//!
//! **Physics**: Stochastic dynamics with friction and random noise
//! **Properties**: Samples canonical ensemble, ergodic, handles non-equilibrium
//! **Use Case**: Brownian dynamics, implicit solvent, driven systems
//!
//! **Algorithm** (BAOAB splitting):
//! The Langevin equation: m*dv/dt = F - γ*v + σ*ξ(t)
//! where γ is friction, σ = sqrt(2*γ*`k_B`*T/m), and ξ is white noise.
//!
//! BAOAB integration (best for configurational sampling):
//! 1. B: v += (dt/2) * a         (half-kick from forces)
//! 2. A: x += (dt/2) * v         (half-drift)
//! 3. O: v = c*v + `σ_eff`*R       (friction + noise)
//! 4. A: x += (dt/2) * v         (half-drift)
//! 5. B: v += (dt/2) * a         (half-kick with new forces)
//!
//! where c = exp(-γ*dt) and `σ_eff` = `sqrt(k_B`*T/m * (1 - c²))
//!
//! **Deep Debt Compliance**:
//! - ✅ Pure WGSL shader
//! - ✅ Zero unsafe code
//! - ✅ f64 precision

use crate::device::capabilities::WORKGROUP_SIZE_COMPACT;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use rand::Rng;

/// Langevin thermostat parameters
///
/// Maintains the friction and noise parameters across timesteps.
pub struct LangevinParams {
    /// Friction coefficient γ (units of 1/time)
    pub gamma: f64,
    /// Target temperature (reduced units)
    pub t_target: f64,
    /// Particle mass (3.0 in OCP reduced units)
    pub mass: f64,
    /// Timestep
    pub dt: f64,
    /// Pre-computed: exp(-γ*dt)
    exp_factor: f64,
    /// Pre-computed: `sqrt(k_B`*T/m * (1 - exp(-2*γ*dt)))
    noise_factor: f64,
}

impl LangevinParams {
    /// Create Langevin thermostat parameters
    /// # Arguments
    /// * `gamma` - Friction coefficient (typical: 1/τ where τ is relaxation time)
    /// * `t_target` - Target temperature (reduced units)
    /// * `mass` - Particle mass (3.0 in OCP reduced units)
    /// * `dt` - Integration timestep
    /// # Notes
    /// - Higher γ → faster equilibration but slower dynamics
    /// - Lower γ → more Hamiltonian-like behavior
    /// - Typical γ: 0.1/dt to 10/dt
    #[must_use]
    pub fn new(gamma: f64, t_target: f64, mass: f64, dt: f64) -> Self {
        let exp_factor = (-gamma * dt).exp();
        // σ = sqrt(k_B * T / m) in reduced units (k_B = 1)
        // noise_factor = σ * sqrt(1 - exp(-2*γ*dt))
        let sigma = (t_target / mass).sqrt();
        let noise_factor = sigma * (1.0 - (-2.0 * gamma * dt).exp()).sqrt();

        Self {
            gamma,
            t_target,
            mass,
            dt,
            exp_factor,
            noise_factor,
        }
    }

    /// Get the friction decay factor exp(-γ*dt)
    #[must_use]
    pub fn exp_factor(&self) -> f64 {
        self.exp_factor
    }

    /// Get the noise amplitude factor
    #[must_use]
    pub fn noise_factor(&self) -> f64 {
        self.noise_factor
    }
}

/// Langevin friction + noise step
///
/// Applies the "O" step of BAOAB: `v_new` = c*v + `σ_eff`*R
/// where c = exp(-γ*dt), `σ_eff` = `sqrt(k_B`*T/m * (1 - c²)), and R is Gaussian noise.
pub struct LangevinStep {
    velocities: Tensor,
    noise: Tensor,
    n_particles: usize,
    params: LangevinParams,
}

impl LangevinStep {
    /// Create a new Langevin step
    /// # Arguments
    /// * `velocities` - Velocity tensor [N, 3] (f64)
    /// * `noise` - Gaussian noise tensor [N, 3] (f64), mean=0, std=1
    /// * `params` - Pre-computed Langevin parameters
    /// # Errors
    /// Returns error if tensor shapes don't match.
    pub fn new(velocities: Tensor, noise: Tensor, params: LangevinParams) -> Result<Self> {
        let vel_shape = velocities.shape();
        if vel_shape.len() != 2 || vel_shape[1] != 3 {
            return Err(BarracudaError::InvalidShape {
                expected: vec![0, 3],
                actual: vel_shape.to_vec(),
            });
        }

        let n_particles = vel_shape[0];

        if noise.shape() != vel_shape {
            return Err(BarracudaError::InvalidShape {
                expected: vel_shape.to_vec(),
                actual: noise.shape().to_vec(),
            });
        }

        Ok(Self {
            velocities,
            noise,
            n_particles,
            params,
        })
    }

    /// Generate Gaussian noise tensor for Langevin dynamics
    /// # Arguments
    /// * `n_particles` - Number of particles
    /// * `device` - GPU device
    /// * `rng` - Random number generator
    /// # Returns
    /// Tensor [N, 3] of Gaussian random numbers (mean=0, std=1)
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn generate_noise<R: Rng>(
        n_particles: usize,
        device: &std::sync::Arc<crate::device::WgpuDevice>,
        rng: &mut R,
    ) -> Result<Tensor> {
        // Box-Muller transform for Gaussian samples
        let mut noise_data = Vec::with_capacity(n_particles * 3);
        for _ in 0..=(n_particles * 3 / 2) {
            let u1: f64 = rng.random::<f64>().max(1e-30);
            let u2: f64 = rng.random();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            noise_data.push(r * theta.cos());
            noise_data.push(r * theta.sin());
        }
        noise_data.truncate(n_particles * 3);

        let noise_bytes: &[u8] = bytemuck::cast_slice(&noise_data);
        let noise_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Langevin Noise"),
                contents: noise_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });

        Ok(Tensor::from_buffer(
            noise_buffer,
            vec![n_particles, 3],
            device.clone(),
        ))
    }

    /// Execute the friction + noise step (in-place update)
    /// # Returns
    /// Updated velocities tensor
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.velocities.device();

        // Params: [n, gamma, sigma, dt, exp_factor, noise_factor, _, _]
        let params_data: Vec<f64> = vec![
            self.n_particles as f64,
            self.params.gamma,
            0.0, // sigma (computed differently)
            self.params.dt,
            self.params.exp_factor,
            self.params.noise_factor,
            0.0,
            0.0,
        ];
        let params_bytes: &[u8] = bytemuck::cast_slice(&params_data);
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Langevin Params"),
                contents: params_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });

        let shader = device.compile_shader(
            include_str!("langevin.wgsl"),
            Some("Langevin Thermostat Shader"),
        );

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Langevin BGL"),
                    entries: &[
                        // Velocities (read-write)
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Noise (read)
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Params (read)
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Langevin PL"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Langevin Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Langevin BG"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.velocities.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.noise.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Langevin Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Langevin Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);

            let workgroups = (self.n_particles as u32).div_ceil(WORKGROUP_SIZE_COMPACT);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_commands(Some(encoder.finish()));

        Ok(self.velocities)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_langevin_params_creation() {
        // OCP at T* = 0.1, γ = 1.0
        let params = LangevinParams::new(1.0, 0.1, 3.0, 0.01);

        // exp(-γ*dt) = exp(-0.01) ≈ 0.99
        let expected_exp = (-0.01_f64).exp();
        assert!(
            (params.exp_factor - expected_exp).abs() < 1e-10,
            "exp_factor: {} vs {}",
            params.exp_factor,
            expected_exp
        );

        // σ = sqrt(T/m) = sqrt(0.1/3) ≈ 0.1826
        // noise_factor = σ * sqrt(1 - exp(-0.02)) ≈ σ * 0.1414
        assert!(params.noise_factor > 0.0, "noise_factor should be positive");
    }

    #[tokio::test]
    async fn test_langevin_noise_generation() {
        let device = crate::device::test_pool::get_test_device().await;

        let mut rng = rand::rng();
        let noise = LangevinStep::generate_noise(100, &device, &mut rng).unwrap();

        assert_eq!(noise.shape(), &[100, 3]);
    }

    #[tokio::test]
    async fn test_langevin_step() {
        let device = crate::device::test_pool::get_test_device().await;

        // Check for f64 support
        if !device
            .device
            .features()
            .contains(wgpu::Features::SHADER_F64)
        {
            return;
        }

        let velocities: Vec<f64> = vec![1.0, 0.0, 0.0];
        let noise_data: Vec<f64> = vec![0.5, 0.0, 0.0]; // Small positive noise

        let vel_bytes: &[u8] = bytemuck::cast_slice(&velocities);
        let vel_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Test Velocities"),
                contents: vel_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let vel_tensor = Tensor::from_buffer(vel_buffer, vec![1, 3], device.clone());

        let noise_bytes: &[u8] = bytemuck::cast_slice(&noise_data);
        let noise_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Test Noise"),
                contents: noise_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });
        let noise_tensor = Tensor::from_buffer(noise_buffer, vec![1, 3], device);

        let params = LangevinParams::new(1.0, 0.1, 3.0, 0.01);
        let step = LangevinStep::new(vel_tensor, noise_tensor, params).unwrap();
        let _vel_after = step.execute().unwrap();
    }
}
