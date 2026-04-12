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
    health_check, health_liveness, health_readiness, tolerances_get, validate_gpu_stack,
};
use super::primal::{capabilities, identity, info};
use super::tensor::{tensor_create, tensor_matmul};
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
mod tensor_fhe_tests;
