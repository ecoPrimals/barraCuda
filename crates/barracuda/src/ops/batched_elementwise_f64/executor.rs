// SPDX-License-Identifier: AGPL-3.0-or-later
//! Batched element-wise GPU executor for f64 operations.

#[cfg(test)]
use super::cpu_ref;
use super::op::{Op, StationDayInput, WaterBalanceInput};
use crate::device::WgpuDevice;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::driver_profile::{Fp64Strategy, GpuDriverProfile};
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const SHADER_BATCHED_ELEMENTWISE_F64: &str =
    include_str!("../../shaders/science/batched_elementwise_f64.wgsl");

/// Try to build a DF64-rewritten variant of the batched elementwise shader.
///
/// Pre-injects `math_f64` polyfills so naga can validate, then rewrites
/// f64 infix ops to DF64 bridge calls. Returns `None` if the naga
/// rewriter can't handle this shader's complexity.
fn df64_shader_source() -> Option<&'static str> {
    static DF64_SOURCE: std::sync::LazyLock<Option<String>> = std::sync::LazyLock::new(|| {
        use crate::shaders::precision::ShaderTemplate;
        let with_polyfills = ShaderTemplate::with_math_f64_auto(SHADER_BATCHED_ELEMENTWISE_F64);
        crate::shaders::sovereign::df64_rewrite::rewrite_f64_infix_full(&with_polyfills).ok()
    });
    DF64_SOURCE.as_deref()
}

/// Parameters for batched elementwise shader
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    batch_size: u32,
    stride: u32,
    operation: u32,
    _pad: u32,
    aux_param: f64,
}

/// Batched element-wise executor for f64 data
///
/// Processes multiple independent computations in parallel, one per batch element.
/// Useful for station-days (ET₀), field-cells (water balance), or samples (diversity).
pub struct BatchedElementwiseF64 {
    device: Arc<WgpuDevice>,
}

impl BatchedElementwiseF64 {
    /// Create a new batched elementwise executor
    /// # Errors
    /// Returns [`Err`] if device initialization fails.
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// Execute batched computation
    /// # Arguments
    /// * `data` - Flattened input array [`batch_size` * stride]
    /// * `batch_size` - Number of batch elements
    /// * `op` - Operation to perform
    /// # Returns
    /// Output array [`batch_size`]
    /// # Errors
    /// Returns [`Err`] if input data length is too short for the given batch size
    /// and stride, buffer allocation fails, the device is lost, or buffer readback
    /// fails.
    pub fn execute(&self, data: &[f64], batch_size: usize, op: Op) -> Result<Vec<f64>> {
        self.execute_with_aux(data, batch_size, op, 0.0)
    }

    /// Execute batched computation with auxiliary parameter
    /// # Panics
    /// Panics if DF64 shader source is unavailable when hybrid FP64 strategy is selected (internal consistency).
    /// # Arguments
    /// * `data` - Flattened input array [`batch_size` * stride]
    /// * `batch_size` - Number of batch elements
    /// * `op` - Operation to perform
    /// * `aux_param` - Auxiliary parameter (e.g., total for normalization)
    /// # Returns
    /// Output array [`batch_size`]
    /// # Errors
    /// Returns [`Err`] if input data length is too short for the given batch size
    /// and stride, buffer allocation fails, the device is lost, or buffer readback
    /// fails.
    pub fn execute_with_aux(
        &self,
        data: &[f64],
        batch_size: usize,
        op: Op,
        aux_param: f64,
    ) -> Result<Vec<f64>> {
        if batch_size == 0 {
            return Ok(Vec::new());
        }

        let stride = op.stride();
        let expected_len = batch_size * stride;

        if data.len() < expected_len {
            return Err(crate::error::BarracudaError::InvalidInput {
                message: format!(
                    "Input data length {} too short for {} batches with stride {} (expected {})",
                    data.len(),
                    batch_size,
                    stride,
                    expected_len
                ),
            });
        }

        // Create input buffer
        let input_buffer =
            self.device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("BatchedEW Input"),
                    contents: bytemuck::cast_slice(data),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        // Create output buffer
        let output_buffer = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BatchedEW Output"),
            size: (batch_size * 8) as u64, // f64 = 8 bytes
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create params buffer
        let params = Params {
            batch_size: batch_size as u32,
            stride: stride as u32,
            operation: op as u32,
            _pad: 0,
            aux_param,
        };
        let params_buffer =
            self.device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("BatchedEW Params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let profile = GpuDriverProfile::from_device(&self.device);
        let use_df64 = matches!(profile.fp64_strategy(), Fp64Strategy::Hybrid)
            && df64_shader_source().is_some();

        if use_df64 {
            let Some(src) = df64_shader_source() else {
                return Err(crate::error::BarracudaError::Internal(
                    "DF64 shader source unavailable despite Hybrid strategy".into(),
                ));
            };
            ComputeDispatch::new(&self.device, "batched_elementwise_df64")
                .shader(src, "batched_compute")
                .df64()
                .storage_read(0, &input_buffer)
                .storage_rw(1, &output_buffer)
                .uniform(2, &params_buffer)
                .dispatch(batch_size as u32, 1, 1)
                .submit()?;
        } else {
            ComputeDispatch::new(&self.device, "batched_elementwise_f64")
                .shader(SHADER_BATCHED_ELEMENTWISE_F64, "batched_compute")
                .f64()
                .storage_read(0, &input_buffer)
                .storage_rw(1, &output_buffer)
                .uniform(2, &params_buffer)
                .dispatch(batch_size as u32, 1, 1)
                .submit()?;
        }

        // Read results
        self.read_results(&output_buffer, batch_size)
    }

    /// Read results from GPU buffer
    fn read_results(&self, buffer: &wgpu::Buffer, count: usize) -> Result<Vec<f64>> {
        self.device.read_buffer::<f64>(buffer, count)
    }

    /// CPU fallback for small batches
    #[cfg(test)]
    #[expect(dead_code, reason = "CPU reference for GPU validation")]
    fn execute_cpu(
        &self,
        data: &[f64],
        batch_size: usize,
        op: Op,
        aux_param: f64,
    ) -> Result<Vec<f64>> {
        let stride = op.stride();
        let mut results = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let base = i * stride;
            let result = match op {
                Op::Fao56Et0 => {
                    let tmax = data[base];
                    let tmin = data[base + 1];
                    let rh_max = data[base + 2];
                    let rh_min = data[base + 3];
                    let wind_2m = data[base + 4];
                    let rs = data[base + 5];
                    let elevation = data[base + 6];
                    let lat = data[base + 7];
                    let doy = data[base + 8] as u32;

                    cpu_ref::fao56_et0_cpu(
                        tmax, tmin, rh_max, rh_min, wind_2m, rs, elevation, lat, doy,
                    )
                }
                Op::WaterBalance => {
                    let dr_prev = data[base];
                    let precip = data[base + 1];
                    let irrig = data[base + 2];
                    let etc = data[base + 3];
                    let taw = data[base + 4];
                    let raw = data[base + 5];

                    cpu_ref::water_balance_cpu(dr_prev, precip, irrig, etc, taw, raw)
                }
                Op::Custom => data[base],
                Op::SensorCalibration => {
                    let raw = data[base];
                    2e-13 * raw.powi(3) - 4e-9 * raw.powi(2) + 4e-5 * raw - 0.0677
                }
                Op::HargreavesEt0 => {
                    let tmax = data[base];
                    let tmin = data[base + 1];
                    let lat_rad = data[base + 2];
                    let doy = data[base + 3];
                    cpu_ref::hargreaves_et0_cpu(tmax, tmin, lat_rad, doy)
                }
                Op::KcClimateAdjust => {
                    let kc_table = data[base];
                    let u2 = data[base + 1];
                    let rh_min = data[base + 2];
                    let h = data[base + 3];
                    let adj =
                        (0.04 * (u2 - 2.0) - 0.004 * (rh_min - 45.0)) * (h / 3.0_f64).powf(0.3);
                    (kc_table + adj).max(0.0)
                }
                Op::DualKcKe => {
                    let kcb: f64 = data[base];
                    let kc_max: f64 = data[base + 1];
                    let few: f64 = data[base + 2];
                    let mulch: f64 = data[base + 3];
                    let de_prev: f64 = data[base + 4];
                    let rew: f64 = data[base + 5];
                    let tew: f64 = data[base + 6];
                    let p_eff: f64 = data[base + 7];

                    let de = (de_prev - p_eff).clamp(0.0, tew);
                    let kr: f64 = if de > rew {
                        ((tew - de) / (tew - rew).max(0.001)).max(0.0)
                    } else {
                        1.0
                    };
                    let ke_full = kr * (kc_max - kcb);
                    let ke_limit = few * kc_max;
                    (ke_full.min(ke_limit) * mulch).max(0.0)
                }
                Op::VanGenuchtenTheta => cpu_ref::van_genuchten_theta_cpu(
                    data[base],
                    data[base + 1],
                    data[base + 2],
                    data[base + 3],
                    data[base + 4],
                ),
                Op::VanGenuchtenK => cpu_ref::van_genuchten_k_cpu(
                    data[base],
                    data[base + 1],
                    data[base + 2],
                    data[base + 3],
                    data[base + 4],
                    data[base + 5],
                    data[base + 6],
                ),
                Op::ThornthwaiteEt0 => cpu_ref::thornthwaite_et0_cpu(
                    data[base],
                    data[base + 1],
                    data[base + 2],
                    data[base + 3],
                    data[base + 4],
                ),
                Op::Gdd => {
                    let t_mean = data[base];
                    (t_mean - aux_param).max(0.0)
                }
                Op::PedotransferPolynomial => cpu_ref::pedotransfer_polynomial_cpu(
                    data[base],
                    data[base + 1],
                    data[base + 2],
                    data[base + 3],
                    data[base + 4],
                    data[base + 5],
                    data[base + 6],
                ),
                Op::MakkinkEt0 => {
                    cpu_ref::makkink_et0_cpu(data[base], data[base + 1], data[base + 2])
                }
                Op::TurcEt0 => cpu_ref::turc_et0_cpu(data[base], data[base + 1], data[base + 2]),
                Op::HamonEt0 => cpu_ref::hamon_et0_cpu(data[base], data[base + 1]),
                Op::ScsCnRunoff => {
                    cpu_ref::scs_cn_runoff_cpu(data[base], data[base + 1], data[base + 2])
                }
                Op::StewartYieldWater => {
                    cpu_ref::stewart_yield_water_cpu(data[base], data[base + 1])
                }
                Op::BlaneyCriddleEt0 => cpu_ref::blaney_criddle_et0_cpu(data[base], data[base + 1]),
            };
            results.push(result);
        }

        Ok(results)
    }

    /// Compute FAO-56 ET₀ for multiple station-days
    /// # Arguments
    /// * `station_days` - Slice of `StationDayInput` tuples
    /// # Returns
    /// ET₀ values in mm/day for each station-day
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, the device is lost, or buffer
    /// readback fails.
    pub fn fao56_et0_batch(&self, station_days: &[StationDayInput]) -> Result<Vec<f64>> {
        let batch_size = station_days.len();
        let mut data = Vec::with_capacity(batch_size * 9);

        for &(tmax, tmin, rh_max, rh_min, wind, rs, elev, lat, doy) in station_days {
            data.push(tmax);
            data.push(tmin);
            data.push(rh_max);
            data.push(rh_min);
            data.push(wind);
            data.push(rs);
            data.push(elev);
            data.push(lat);
            data.push(doy as f64);
        }

        self.execute(&data, batch_size, Op::Fao56Et0)
    }

    /// Compute water balance update for multiple fields
    /// # Arguments
    /// * `fields` - Slice of `WaterBalanceInput` tuples
    /// # Returns
    /// New depletion values for each field
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, the device is lost, or buffer
    /// readback fails.
    pub fn water_balance_batch(&self, fields: &[WaterBalanceInput]) -> Result<Vec<f64>> {
        let batch_size = fields.len();
        let mut data = Vec::with_capacity(batch_size * 7);

        for &(dr_prev, precip, irrig, etc, taw, raw, p_frac) in fields {
            data.push(dr_prev);
            data.push(precip);
            data.push(irrig);
            data.push(etc);
            data.push(taw);
            data.push(raw);
            data.push(p_frac);
        }

        self.execute(&data, batch_size, Op::WaterBalance)
    }
}
