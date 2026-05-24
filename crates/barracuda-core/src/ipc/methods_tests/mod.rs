// SPDX-License-Identifier: AGPL-3.0-or-later
#![expect(
    clippy::unwrap_used,
    reason = "test assertions: unwrap is idiomatic for test code"
)]

use super::super::jsonrpc::{INTERNAL_ERROR, INVALID_PARAMS, METHOD_NOT_FOUND};
use super::batch::tensor_batch_submit;
use super::compute::{compute_dispatch, parse_shape};
use super::device::{list as device_list, probe as device_probe};
use super::dispatch;
use super::fhe::{fhe_ntt, fhe_pointwise_mul};
use super::health::{
    btsp_capabilities, health_check, health_liveness, health_readiness, health_version,
    tolerances_get, validate_gpu_stack,
};
use super::primal::{announce, capabilities, identity, info};
use super::tensor::{
    tensor_add, tensor_clamp, tensor_create, tensor_matmul, tensor_reduce, tensor_scale,
    tensor_sigmoid,
};
use super::signal::{signal_bandpass, signal_derivative, signal_detect_peaks};
use super::stats::{
    stats_bray_curtis, stats_fit_exponential, stats_fit_logarithmic, stats_fit_quadratic,
    stats_gamma_cdf, stats_gamma_fit, stats_hill, stats_rarefaction_curve, stats_simpson,
};
use super::{REGISTERED_METHODS, normalize_method};
use crate::BarraCudaPrimal;

fn test_primal() -> BarraCudaPrimal {
    BarraCudaPrimal::new()
}

mod batch_tests;
mod comprehensive_tests;
mod device_health_tests;
mod dispatch_compute_tests;
mod primal_wire_tests;
mod registry_tests;
mod signal_tests;
mod stats_diversity_tests;
mod stats_regression_tests;
mod tensor_fhe_tests;
