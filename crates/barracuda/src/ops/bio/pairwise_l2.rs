// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pairwise L2 Distance — GPU kernel.
//!
//! Computes the upper-triangle pairwise Euclidean (L2) distance matrix for N
//! feature vectors of dimension D. Each thread handles one pair. Output is
//! N*(N-1)/2 L2 distances.
//!
//! Provenance: neuralSpring metalForge → toadStool absorption

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::WgpuDevice;

static WGSL_PAIRWISE_L2: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
        "../../shaders/math/pairwise_l2_f64.wgsl"
    ))
});

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PairwiseL2Params {
    n: u32,
    dim: u32,
}

pub struct PairwiseL2Gpu {
    device: Arc<WgpuDevice>,
}

impl PairwiseL2Gpu {
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        Self { device }
    }

    /// Compute pairwise L2 distances for `n` vectors of dimension `dim`.
    ///
    /// `input_buf`: `[n × dim]` f32 (row-major feature vectors)
    /// `output_buf`: `[n*(n-1)/2]` f32 (L2 distances)
    pub fn dispatch(&self, input_buf: &wgpu::Buffer, output_buf: &wgpu::Buffer, n: u32, dim: u32) {
        let d = self.device.device();

        let params = PairwiseL2Params { n, dim };
        let params_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PairwiseL2 Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let n_pairs = n * (n - 1) / 2;
        let wg_count = n_pairs.div_ceil(WORKGROUP_SIZE_1D);

        ComputeDispatch::new(&self.device, "PairwiseL2")
            .shader(&WGSL_PAIRWISE_L2, "main")
            .storage_read(0, input_buf)
            .storage_rw(1, output_buf)
            .uniform(2, &params_buf)
            .dispatch(wg_count, 1, 1)
            .submit();
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::{PairwiseL2Gpu, WGSL_PAIRWISE_L2};

    #[test]
    fn sanity_constants_exported() {
        assert!(!&WGSL_PAIRWISE_L2.is_empty());
        assert!(&WGSL_PAIRWISE_L2.contains("fn main"));
        assert!(&WGSL_PAIRWISE_L2.contains("PairwiseParams"));
        assert!(std::any::type_name::<PairwiseL2Gpu>().contains("PairwiseL2Gpu"));
    }
}
