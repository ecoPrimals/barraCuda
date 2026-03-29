// SPDX-License-Identifier: AGPL-3.0-or-later

//! Capability-based primal discovery and response parsing.
//!
//! Primals in the ecoPrimals ecosystem report their capabilities in at
//! least 6 different JSON shapes. This module normalises all of them
//! into `Vec<String>` so higher-level code can discover by capability
//! (e.g. `shader.compile`, `crypto.hash`) without knowing the exact
//! wire format each primal uses.
//!
//! Absorbed from ludoSpring V34 (March 2026) and generalised for
//! barraCuda's cross-primal discovery needs.
//!
//! # Supported Formats
//!
//! | Format | Shape | Example |
//! |--------|-------|---------|
//! | A | Flat string array | `["cap1", "cap2"]` |
//! | B | Object array with `name` | `[{"name": "cap1"}]` |
//! | C | Nested wrapper | `{"capabilities": [...]}` |
//! | D | Double-nested | `{"capabilities": {"capabilities": [...]}}` |
//! | E | BearDog `provided_capabilities` | `[{"type": "crypto", "methods": [...]}]` |
//! | F | Top-level flat array (Songbird) | (root is array) |

pub mod capabilities;
pub mod tolerances;
