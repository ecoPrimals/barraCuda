// SPDX-License-Identifier: AGPL-3.0-or-later
#![expect(
    clippy::unwrap_used,
    reason = "test assertions: unwrap is idiomatic for test code"
)]

use super::*;

mod batch;
mod btsp_encryption;
mod connection;
mod dispatch;
mod genetics;
mod resolve;
mod socket;
