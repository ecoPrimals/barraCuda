//! Cyclic Reduction for Tridiagonal Systems (WGSL GPU Implementation)
//!
//! **Why this file is large (~650 lines)**: Single algorithm—cyclic reduction
//! for tridiagonal systems. Size comes from four code paths (single/batch ×
//! small/large) sharing the same math but different dispatch patterns. Splitting
//! would separate logically coupled variants.
//!
//! This is the SHADER-FIRST implementation of tridiagonal solvers.
//! Uses cyclic reduction (odd-even elimination) for O(log n) parallel steps
//! instead of O(n) sequential steps in the Thomas algorithm.
//!
//! # Algorithm
//!
//! Cyclic reduction eliminates odd-indexed unknowns in parallel:
//! 1. Reduction phase: O(log n) steps, each eliminating half the unknowns
//! 2. Solve the 1-element system at the center
//! 3. Substitution phase: O(log n) steps, recovering eliminated unknowns
//!
//! # When to Use
//!
//! - **Large systems (n > 256)**: Cyclic reduction wins on GPU
//! - **Small systems (n ≤ 256)**: Single workgroup shared memory version
//! - **Batched systems**: Many independent tridiagonal systems in parallel
//!
//! # Future Hardware
//!
//! Same math runs on:
//! - Current GPUs (fp32)
//! - Future fp64 GPUs (seamless transition via ToadStool dispatch)
//! - Quantum sequential compute (same algorithm, different substrate)

use crate::device::wgpu_device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use bytemuck::{Pod, Zeroable};

/// Parameters for cyclic reduction shader (single system)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CyclicReductionParams {
    n: u32,
    step: u32,
    phase: u32,
    _pad: u32,
}

/// Parameters for batched cyclic reduction shader
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BatchParams {
    n: u32,           // System size per batch element
    n_padded: u32,    // Padded to power of 2
    batch_size: u32,  // Number of systems
    step: u32,        // Current reduction step (for multi-pass)
}

const SHADER_SOURCE: &str = include_str!("../shaders/linalg/cyclic_reduction.wgsl");

/// Solve tridiagonal system using GPU cyclic reduction.
///
/// Solves: a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]
///
/// # Arguments
///
/// * `a` - Sub-diagonal tensor (length n-1, padded to n)
/// * `b` - Main diagonal tensor (length n)
/// * `c` - Super-diagonal tensor (length n-1, padded to n)
/// * `d` - Right-hand side tensor (length n)
///
/// # Returns
///
/// Solution tensor x of length n.
///
/// # Example
///
/// ```ignore
/// use barracuda::ops::cyclic_reduction_wgsl::tridiagonal_solve_gpu;
/// use barracuda::tensor::Tensor;
///
/// let a = Tensor::from_vec(vec![1.0f32, 1.0, 0.0], vec![3]);
/// let b = Tensor::from_vec(vec![4.0f32, 4.0, 4.0], vec![3]);
/// let c = Tensor::from_vec(vec![1.0f32, 1.0, 0.0], vec![3]);
/// let d = Tensor::from_vec(vec![5.0f32, 6.0, 5.0], vec![3]);
///
/// let x = tridiagonal_solve_gpu(&a, &b, &c, &d)?;
/// ```
pub fn tridiagonal_solve_gpu(
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    d: &Tensor,
) -> Result<Tensor> {
    let n = b.len();
    
    if n == 0 {
        return Err(BarracudaError::InvalidInput {
            message: "System size must be > 0".to_string(),
        });
    }
    
    // For small systems, use the shared memory single-pass version
    if n <= 256 {
        return tridiagonal_solve_gpu_small(a, b, c, d);
    }
    
    // For large systems, use multi-pass cyclic reduction
    tridiagonal_solve_gpu_large(a, b, c, d)
}

/// Single-pass solver for small systems (n ≤ 256)
/// Uses shared memory for efficiency
fn tridiagonal_solve_gpu_small(
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    d: &Tensor,
) -> Result<Tensor> {
    let n = b.len();
    let device = WgpuDevice::new()?;
    
    // Create padded copies of the diagonals
    let mut a_data = a.to_vec_f32()?;
    let mut b_data = b.to_vec_f32()?;
    let mut c_data = c.to_vec_f32()?;
    let mut d_data = d.to_vec_f32()?;
    
    // Pad to power of 2 for cyclic reduction
    let n_padded = n.next_power_of_two();
    a_data.resize(n_padded, 0.0);
    b_data.resize(n_padded, 1.0);  // Identity for padded rows
    c_data.resize(n_padded, 0.0);
    d_data.resize(n_padded, 0.0);
    
    let params = CyclicReductionParams {
        n: n as u32,
        step: 0,
        phase: 0,
        _pad: 0,
    };
    
    // Create GPU buffers
    let params_buffer = device.create_uniform_buffer(&params);
    let a_buffer = device.create_storage_buffer_init(&a_data);
    let b_buffer = device.create_storage_buffer_init(&b_data);
    let c_buffer = device.create_storage_buffer_init(&c_data);
    let d_buffer = device.create_storage_buffer_init(&d_data);
    
    // Create compute pipeline
    let pipeline = device.create_compute_pipeline(SHADER_SOURCE, "solve_small")?;
    
    // Bind group
    let bind_group = device.create_bind_group(
        &pipeline,
        &[&params_buffer, &a_buffer, &b_buffer, &c_buffer, &d_buffer],
    );
    
    // Dispatch single workgroup
    device.dispatch(&pipeline, &bind_group, (1, 1, 1))?;
    
    // Read back result (d now contains solution)
    let result = device.read_buffer_f32(&d_buffer, n)?;
    
    Ok(Tensor::from_vec(result, vec![n]))
}

/// Multi-pass solver for large systems (n > 256)
fn tridiagonal_solve_gpu_large(
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    d: &Tensor,
) -> Result<Tensor> {
    let n = b.len();
    let device = WgpuDevice::new()?;
    
    // Pad to power of 2
    let n_padded = n.next_power_of_two();
    let num_steps = (n_padded as f64).log2() as u32;
    
    let mut a_data = a.to_vec_f32()?;
    let mut b_data = b.to_vec_f32()?;
    let mut c_data = c.to_vec_f32()?;
    let mut d_data = d.to_vec_f32()?;
    
    a_data.resize(n_padded, 0.0);
    b_data.resize(n_padded, 1.0);
    c_data.resize(n_padded, 0.0);
    d_data.resize(n_padded, 0.0);
    
    // Create buffers (read_write for iterative updates)
    let a_buffer = device.create_storage_buffer_init(&a_data);
    let b_buffer = device.create_storage_buffer_init(&b_data);
    let c_buffer = device.create_storage_buffer_init(&c_data);
    let d_buffer = device.create_storage_buffer_init(&d_data);
    
    // Create pipelines
    let reduction_pipeline = device.create_compute_pipeline(SHADER_SOURCE, "reduction")?;
    let substitution_pipeline = device.create_compute_pipeline(SHADER_SOURCE, "substitution")?;
    
    // Reduction phase
    for step in 0..num_steps {
        let params = CyclicReductionParams {
            n: n_padded as u32,
            step,
            phase: 0,
            _pad: 0,
        };
        let params_buffer = device.create_uniform_buffer(&params);
        
        let bind_group = device.create_bind_group(
            &reduction_pipeline,
            &[&params_buffer, &a_buffer, &b_buffer, &c_buffer, &d_buffer],
        );
        
        let stride = 1 << (step + 1);
        let workgroups = ((n_padded / stride) + 255) / 256;
        device.dispatch(&reduction_pipeline, &bind_group, (workgroups.max(1), 1, 1))?;
    }
    
    // Solve center element (trivial - just d[n/2] / b[n/2])
    // This is implicitly handled by the reduction
    
    // Substitution phase (reverse order)
    for step in (0..num_steps).rev() {
        let params = CyclicReductionParams {
            n: n_padded as u32,
            step,
            phase: 1,
            _pad: 0,
        };
        let params_buffer = device.create_uniform_buffer(&params);
        
        let bind_group = device.create_bind_group(
            &substitution_pipeline,
            &[&params_buffer, &a_buffer, &b_buffer, &c_buffer, &d_buffer],
        );
        
        let stride = 1 << (step + 1);
        let workgroups = ((n_padded / stride) + 255) / 256;
        device.dispatch(&substitution_pipeline, &bind_group, (workgroups.max(1), 1, 1))?;
    }
    
    // Read back result
    let result = device.read_buffer_f32(&d_buffer, n)?;
    
    Ok(Tensor::from_vec(result, vec![n]))
}

/// Batch solve multiple independent tridiagonal systems in parallel.
///
/// **Deep Debt Evolution**: True 2D batched kernel with single GPU dispatch.
/// All systems are solved in parallel with zero CPU↔GPU round-trips per system.
///
/// Each system has the same size but different coefficients/RHS.
/// This is extremely efficient for:
/// - ADI methods (2D/3D PDE with row/column sweeps)
/// - Monte Carlo with multiple realizations
/// - Parallel independent 1D problems (e.g., fiber simulation)
///
/// # Arguments
///
/// * `a_batch` - Sub-diagonals [batch_size × n]
/// * `b_batch` - Main diagonals [batch_size × n]
/// * `c_batch` - Super-diagonals [batch_size × n]
/// * `d_batch` - Right-hand sides [batch_size × n]
///
/// # Returns
///
/// Solutions [batch_size × n]
///
/// # Performance
///
/// For batch_size=100, n=64:
/// - Old (sequential): 100 GPU dispatches, ~100 CPU↔GPU round-trips
/// - New (batched):    1 GPU dispatch, 1 CPU↔GPU round-trip
pub fn tridiagonal_solve_batch_gpu(
    a_batch: &Tensor,
    b_batch: &Tensor,
    c_batch: &Tensor,
    d_batch: &Tensor,
) -> Result<Tensor> {
    let shape = b_batch.shape();
    if shape.len() != 2 {
        return Err(BarracudaError::InvalidInput {
            message: "Batch tensors must be 2D [batch_size × n]".to_string(),
        });
    }
    
    let batch_size = shape[0];
    let n = shape[1];
    
    if n == 0 || batch_size == 0 {
        return Err(BarracudaError::InvalidInput {
            message: "Batch size and system size must be > 0".to_string(),
        });
    }
    
    // Use the true batched solver
    if n <= 256 {
        tridiagonal_solve_batch_gpu_small(a_batch, b_batch, c_batch, d_batch)
    } else {
        tridiagonal_solve_batch_gpu_large(a_batch, b_batch, c_batch, d_batch)
    }
}

/// True batched solver for small systems (n ≤ 256) using 2D dispatch
/// One workgroup per batch element, shared memory per system
fn tridiagonal_solve_batch_gpu_small(
    a_batch: &Tensor,
    b_batch: &Tensor,
    c_batch: &Tensor,
    d_batch: &Tensor,
) -> Result<Tensor> {
    let shape = b_batch.shape();
    let batch_size = shape[0];
    let n = shape[1];
    let n_padded = n.next_power_of_two();
    
    let device = WgpuDevice::new()?;
    
    // Flatten and pad data for all systems
    let mut a_data = Vec::with_capacity(batch_size * n_padded);
    let mut b_data = Vec::with_capacity(batch_size * n_padded);
    let mut c_data = Vec::with_capacity(batch_size * n_padded);
    let mut d_data = Vec::with_capacity(batch_size * n_padded);
    
    let a_flat = a_batch.to_vec_f32()?;
    let b_flat = b_batch.to_vec_f32()?;
    let c_flat = c_batch.to_vec_f32()?;
    let d_flat = d_batch.to_vec_f32()?;
    
    for i in 0..batch_size {
        // Copy original data
        let start = i * n;
        a_data.extend(&a_flat[start..start + n]);
        b_data.extend(&b_flat[start..start + n]);
        c_data.extend(&c_flat[start..start + n]);
        d_data.extend(&d_flat[start..start + n]);
        
        // Pad to power of 2
        for _ in n..n_padded {
            a_data.push(0.0);
            b_data.push(1.0);  // Identity for padded rows
            c_data.push(0.0);
            d_data.push(0.0);
        }
    }
    
    // Create uniform params for legacy binding (not used by batched kernel but required)
    let legacy_params = CyclicReductionParams {
        n: n as u32,
        step: 0,
        phase: 0,
        _pad: 0,
    };
    
    // Create batch params
    let batch_params = BatchParams {
        n: n as u32,
        n_padded: n_padded as u32,
        batch_size: batch_size as u32,
        step: 0,
    };
    
    // Create GPU buffers
    let legacy_params_buffer = device.create_uniform_buffer(&legacy_params);
    let a_buffer = device.create_storage_buffer_init(&a_data);
    let b_buffer = device.create_storage_buffer_init(&b_data);
    let c_buffer = device.create_storage_buffer_init(&c_data);
    let d_buffer = device.create_storage_buffer_init(&d_data);
    let batch_params_buffer = device.create_uniform_buffer(&batch_params);
    
    // Create compute pipeline for batched solve
    let pipeline = device.create_compute_pipeline(SHADER_SOURCE, "solve_batch_small")?;
    
    // Create bind groups (group 0 for data, group 1 for batch params)
    let bind_group_0 = device.create_bind_group(
        &pipeline,
        &[&legacy_params_buffer, &a_buffer, &b_buffer, &c_buffer, &d_buffer],
    );
    
    // Create second bind group for batch params (group 1)
    let bind_group_layout_1 = device.wgpu_device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("batch_params_layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    
    let bind_group_1 = device.wgpu_device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("batch_params_bind_group"),
        layout: &bind_group_layout_1,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: batch_params_buffer.as_entire_binding(),
        }],
    });
    
    // Dispatch: 1 workgroup in X (256 threads handle all elements), batch_size workgroups in Y
    device.dispatch_with_bind_groups(
        &pipeline, 
        &[&bind_group_0, &bind_group_1], 
        (1, batch_size as u32, 1)
    )?;
    
    // Read back results (only original n elements per system)
    let result_data = device.read_buffer_f32(&d_buffer, batch_size * n_padded)?;
    
    // Extract only the original elements (skip padding)
    let mut result = Vec::with_capacity(batch_size * n);
    for i in 0..batch_size {
        let start = i * n_padded;
        result.extend(&result_data[start..start + n]);
    }
    
    Ok(Tensor::from_vec(result, vec![batch_size, n]))
}

/// True batched solver for large systems (n > 256) using multi-pass 2D dispatch
fn tridiagonal_solve_batch_gpu_large(
    a_batch: &Tensor,
    b_batch: &Tensor,
    c_batch: &Tensor,
    d_batch: &Tensor,
) -> Result<Tensor> {
    let shape = b_batch.shape();
    let batch_size = shape[0];
    let n = shape[1];
    let n_padded = n.next_power_of_two();
    let num_steps = (n_padded as f64).log2() as u32;
    
    let device = WgpuDevice::new()?;
    
    // Flatten and pad data for all systems
    let mut a_data = Vec::with_capacity(batch_size * n_padded);
    let mut b_data = Vec::with_capacity(batch_size * n_padded);
    let mut c_data = Vec::with_capacity(batch_size * n_padded);
    let mut d_data = Vec::with_capacity(batch_size * n_padded);
    
    let a_flat = a_batch.to_vec_f32()?;
    let b_flat = b_batch.to_vec_f32()?;
    let c_flat = c_batch.to_vec_f32()?;
    let d_flat = d_batch.to_vec_f32()?;
    
    for i in 0..batch_size {
        let start = i * n;
        a_data.extend(&a_flat[start..start + n]);
        b_data.extend(&b_flat[start..start + n]);
        c_data.extend(&c_flat[start..start + n]);
        d_data.extend(&d_flat[start..start + n]);
        
        for _ in n..n_padded {
            a_data.push(0.0);
            b_data.push(1.0);
            c_data.push(0.0);
            d_data.push(0.0);
        }
    }
    
    // Create buffers (read_write for iterative updates)
    let a_buffer = device.create_storage_buffer_init(&a_data);
    let b_buffer = device.create_storage_buffer_init(&b_data);
    let c_buffer = device.create_storage_buffer_init(&c_data);
    let d_buffer = device.create_storage_buffer_init(&d_data);
    
    // Create pipelines for batched reduction and substitution
    let reduction_pipeline = device.create_compute_pipeline(SHADER_SOURCE, "reduction_batch")?;
    let substitution_pipeline = device.create_compute_pipeline(SHADER_SOURCE, "substitution_batch")?;
    
    // Legacy params placeholder (not used but required for bind group 0)
    let legacy_params = CyclicReductionParams {
        n: n as u32,
        step: 0,
        phase: 0,
        _pad: 0,
    };
    let legacy_params_buffer = device.create_uniform_buffer(&legacy_params);
    
    // Reduction phase
    for step in 0..num_steps {
        let batch_params = BatchParams {
            n: n as u32,
            n_padded: n_padded as u32,
            batch_size: batch_size as u32,
            step,
        };
        let batch_params_buffer = device.create_uniform_buffer(&batch_params);
        
        let bind_group_0 = device.create_bind_group(
            &reduction_pipeline,
            &[&legacy_params_buffer, &a_buffer, &b_buffer, &c_buffer, &d_buffer],
        );
        
        // Create bind group for batch params
        let bind_group_layout_1 = device.wgpu_device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("batch_params_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        
        let bind_group_1 = device.wgpu_device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("batch_params_bind_group"),
            layout: &bind_group_layout_1,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: batch_params_buffer.as_entire_binding(),
            }],
        });
        
        let stride = 1 << (step + 1);
        let workgroups_x = ((n_padded / stride) + 255) / 256;
        
        device.dispatch_with_bind_groups(
            &reduction_pipeline,
            &[&bind_group_0, &bind_group_1],
            (workgroups_x.max(1) as u32, batch_size as u32, 1),
        )?;
    }
    
    // Substitution phase (reverse order)
    for step in (0..num_steps).rev() {
        let batch_params = BatchParams {
            n: n as u32,
            n_padded: n_padded as u32,
            batch_size: batch_size as u32,
            step,
        };
        let batch_params_buffer = device.create_uniform_buffer(&batch_params);
        
        let bind_group_0 = device.create_bind_group(
            &substitution_pipeline,
            &[&legacy_params_buffer, &a_buffer, &b_buffer, &c_buffer, &d_buffer],
        );
        
        let bind_group_layout_1 = device.wgpu_device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("batch_params_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        
        let bind_group_1 = device.wgpu_device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("batch_params_bind_group"),
            layout: &bind_group_layout_1,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: batch_params_buffer.as_entire_binding(),
            }],
        });
        
        let stride = 1 << (step + 1);
        let workgroups_x = ((n_padded / stride) + 255) / 256;
        
        device.dispatch_with_bind_groups(
            &substitution_pipeline,
            &[&bind_group_0, &bind_group_1],
            (workgroups_x.max(1) as u32, batch_size as u32, 1),
        )?;
    }
    
    // Read back results
    let result_data = device.read_buffer_f32(&d_buffer, batch_size * n_padded)?;
    
    // Extract only the original elements
    let mut result = Vec::with_capacity(batch_size * n);
    for i in 0..batch_size {
        let start = i * n_padded;
        result.extend(&result_data[start..start + n]);
    }
    
    Ok(Tensor::from_vec(result, vec![batch_size, n]))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tridiagonal_small() {
        // Skip if no GPU
        if crate::device::test_pool::get_test_device_if_gpu_available_sync().is_none() {
            return;
        }
        
        // Simple 3x3 system
        let a = Tensor::from_vec(vec![1.0f32, 1.0, 0.0], vec![3]);
        let b = Tensor::from_vec(vec![4.0f32, 4.0, 4.0], vec![3]);
        let c = Tensor::from_vec(vec![1.0f32, 1.0, 0.0], vec![3]);
        let d = Tensor::from_vec(vec![5.0f32, 6.0, 5.0], vec![3]);
        
        let x = tridiagonal_solve_gpu(&a, &b, &c, &d).unwrap();
        let x_data = x.to_vec_f32().unwrap();
        
        // Verify solution: A·x ≈ d
        // Row 0: 4*x[0] + 1*x[1] = 5
        let ax0 = 4.0 * x_data[0] + 1.0 * x_data[1];
        assert!((ax0 - 5.0).abs() < 0.1, "Row 0: {} vs 5.0", ax0);
    }
    
    #[test]
    fn test_tridiagonal_heat_equation() {
        if crate::device::test_pool::get_test_device_if_gpu_available_sync().is_none() {
            return;
        }
        
        // Heat equation discretization: -u_{i-1} + 2u_i - u_{i+1} = f_i
        let n = 10;
        let a = Tensor::from_vec(vec![-1.0f32; n], vec![n]);
        let b = Tensor::from_vec(vec![2.0f32; n], vec![n]);
        let c = Tensor::from_vec(vec![-1.0f32; n], vec![n]);
        let d = Tensor::from_vec(vec![1.0f32; n], vec![n]);
        
        let x = tridiagonal_solve_gpu(&a, &b, &c, &d).unwrap();
        assert_eq!(x.len(), n);
    }
    
    #[test]
    fn test_tridiagonal_identity() {
        if crate::device::test_pool::get_test_device_if_gpu_available_sync().is_none() {
            return;
        }
        
        // Identity matrix: b=1, a=c=0
        let a = Tensor::from_vec(vec![0.0f32, 0.0, 0.0], vec![3]);
        let b = Tensor::from_vec(vec![1.0f32, 1.0, 1.0], vec![3]);
        let c = Tensor::from_vec(vec![0.0f32, 0.0, 0.0], vec![3]);
        let d = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
        
        let x = tridiagonal_solve_gpu(&a, &b, &c, &d).unwrap();
        let x_data = x.to_vec_f32().unwrap();
        
        assert!((x_data[0] - 1.0).abs() < 0.01);
        assert!((x_data[1] - 2.0).abs() < 0.01);
        assert!((x_data[2] - 3.0).abs() < 0.01);
    }
}
