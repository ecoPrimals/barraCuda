// SPDX-License-Identifier: AGPL-3.0-or-later
//! ML inference handlers for JSON-RPC IPC.
//!
//! Inline-data CPU paths for lightweight ML operations suitable for
//! composition graph nodes. GPU tensor ops live in `tensor.rs`.
//!
//! Sub-modules:
//! - `forward` — stateless inference (MLP forward, attention, ESN predict)
//! - `train` — training pipelines (MLP train, perceptron pipeline)
//! - `infer` — batch inference on telemetry vectors
//! - `persistence` — model save/load to disk

mod forward;
mod infer;
mod persistence;
mod train;

use barracuda::nn::simple_mlp::Activation;

pub(super) use forward::{ml_attention, ml_esn_predict, ml_mlp_forward};
pub(super) use infer::ml_mlp_infer;
pub(super) use persistence::{ml_mlp_load, ml_mlp_save};
pub(super) use train::{ml_mlp_train, ml_perceptron_train};

fn parse_activation(s: &str) -> Option<Activation> {
    match s {
        "relu" => Some(Activation::Relu),
        "tanh" => Some(Activation::Tanh),
        "sigmoid" => Some(Activation::Sigmoid),
        "gelu" => Some(Activation::Gelu),
        "identity" | "none" | "linear" => Some(Activation::Identity),
        _ => None,
    }
}
