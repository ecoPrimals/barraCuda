// SPDX-License-Identifier: AGPL-3.0-or-later

//! Free energy surface (FES) reconstruction via metadynamics Gaussian summation.

/// GPU FES reconstruction from HILLS Gaussian bias data.
pub const WGSL_FES_GAUSSIAN_SUM: &str =
    include_str!("../../shaders/physics/fes_gaussian_sum_f64.wgsl");
