// SPDX-License-Identifier: AGPL-3.0-or-later
//! Inter-Primal Communication (IPC) for barraCuda.
//!
//! Implements JSON-RPC 2.0 per wateringHole `PRIMAL_IPC_PROTOCOL.md` and
//! `UNIVERSAL_IPC_STANDARD_V3.md`. Each primal implements the protocol
//! itself — no shared IPC crate.
//!
//! ## Transport
//!
//! Primary: Unix domain socket at `$BIOMEOS_SOCKET_DIR/math.sock`
//! (or `math-{family_id}.sock` when `FAMILY_ID` is set; per PRIMAL_SELF_KNOWLEDGE_STANDARD §3)
//! Legacy symlink: `barracuda.sock → math.sock` for backward compatibility
//! Fallback: TCP, resolved via [`transport::resolve_bind_address()`](crate::ipc::transport::resolve_bind_address) —
//! `BARRACUDA_IPC_BIND` or `BARRACUDA_IPC_HOST`:`BARRACUDA_IPC_PORT`,
//! default `127.0.0.1:0` (ephemeral)
//!
//! ## Wire Format
//!
//! Newline-delimited JSON-RPC 2.0 per `PRIMAL_IPC_PROTOCOL.md` v3.1.
//! One JSON object per line, terminated by `\n`. Batch requests (JSON arrays)
//! are supported per JSON-RPC 2.0 §6.
//!
//! ## Method Schemas (31 methods)
//!
//! All methods follow `{domain}.{operation}` per `SEMANTIC_METHOD_NAMING_STANDARD.md`.
//! Legacy `barracuda.{domain}.{operation}` prefix accepted for backward compatibility.
//!
//! ### Ecosystem Probes (non-negotiable)
//!
//! | Method | Aliases | Params | Result | GPU |
//! |--------|---------|--------|--------|-----|
//! | `health.liveness` | `ping`, `health` | none | `{"status": "alive"}` | no |
//! | `health.readiness` | — | none | `{"status": "ready"\|"not_ready", "gpu_available": bool}` | no |
//! | `health.check` | `status`, `check` | none | `{"name": str, "version": str, "status": str}` | no |
//! | `capabilities.list` | `capability.list`, `primal.capabilities` | none | see below | no |
//!
//! ### Primal Identity
//!
//! | Method | Params | Result | GPU |
//! |--------|--------|--------|-----|
//! | `primal.info` | none | `{"primal": str, "version": str, "protocol": "json-rpc-2.0", "namespace": str, "license": "AGPL-3.0-or-later"}` | no |
//! | `primal.capabilities` | none | `{"provides": [{"id": str, "version": str}], "requires": [...], "domains": [...], "methods": [str], "hardware": {"gpu_available": bool, "f64_shaders": bool, "spirv_passthrough": bool}}` | no |
//!
//! ### Device
//!
//! | Method | Params | Result | GPU |
//! |--------|--------|--------|-----|
//! | `device.list` | none | `{"devices": [{"name", "vendor", "device_type", "backend", "driver", "driver_info"}]}` | no |
//! | `device.probe` | none | `{"available": bool, "max_buffer_size": u64, ...}` or `{"available": false, "reason": str}` | no |
//! | `tolerances.get` | `{"name": str}` (optional, default `"default"`) | `{"name": str, "abs_tol": f64, "rel_tol": f64}` | no |
//! | `validate.gpu_stack` | none | `{"gpu_available": true, "status": str, "tests": [{"test": str, "pass": bool}]}` | yes |
//!
//! ### Math & Activation (CPU)
//!
//! | Method | Params | Result | GPU |
//! |--------|--------|--------|-----|
//! | `math.sigmoid` | `{"data": [f64]}` | `{"result": [f64]}` | no |
//! | `math.log2` | `{"data": [f64]}` | `{"result": [f64]}` | no |
//! | `activation.fitts` | `{"distance": f64, "width": f64, "a"?: f64, "b"?: f64, "variant"?: "shannon"\|"fitts"}` | `{"result": f64, "movement_time": f64, "index_of_difficulty": f64, "variant": str}` | no |
//! | `activation.hick` | `{"n_choices": u64, "a"?: f64, "b"?: f64, "include_no_choice"?: bool}` | `{"result": f64, "reaction_time": rt, "information_bits": f64, "include_no_choice": bool}` | no |
//!
//! ### Statistics (CPU)
//!
//! All stats methods use **sample** statistics (Bessel's correction, N-1 denominator).
//! `stats.std_dev` and `stats.variance` include `"convention": "sample"` in the response.
//!
//! | Method | Params | Result | GPU |
//! |--------|--------|--------|-----|
//! | `stats.mean` | `{"data": [f64]}` | `{"result": f64}` | no |
//! | `stats.std_dev` | `{"data": [f64]}` | `{"result": f64, "convention": "sample", "denominator": "N-1"}` | no |
//! | `stats.variance` | `{"data": [f64]}` | `{"result": f64, "convention": "sample", "denominator": "N-1"}` | no |
//! | `stats.correlation` | `{"x": [f64], "y": [f64]}` | `{"result": f64}` (Pearson r ∈ [-1,1]) | no |
//! | `stats.weighted_mean` | `{"values": [f64], "weights": [f64]}` | `{"result": f64}` | no |
//!
//! ### Linear Algebra (CPU inline-data)
//!
//! | Method | Params | Result | GPU |
//! |--------|--------|--------|-----|
//! | `linalg.solve` | `{"matrix": [[f64]], "b": [f64]}` | `{"result": [f64]}` (solution x where Ax=b) | no |
//! | `linalg.eigenvalues` | `{"matrix": [[f64]]}` | `{"result": [f64]}` (eigenvalues of symmetric matrix) | no |
//!
//! ### Spectral (CPU inline-data)
//!
//! | Method | Params | Result | GPU |
//! |--------|--------|--------|-----|
//! | `spectral.fft` | `{"data": [f64]}` | `{"result": [f64], "real": [f64], "imag": [f64], "n": u64}` | no |
//! | `spectral.power_spectrum` | `{"data": [f64]}` | `{"result": [f64], "n": u64}` | no |
//!
//! ### Noise & RNG (CPU)
//!
//! | Method | Params | Result | GPU |
//! |--------|--------|--------|-----|
//! | `noise.perlin2d` | `{"x": f64, "y": f64}` | `{"result": f64}` | no |
//! | `noise.perlin3d` | `{"x": f64, "y": f64, "z": f64}` | `{"result": f64}` | no |
//! | `rng.uniform` | `{"n"?: u64, "min"?: f64, "max"?: f64, "seed"?: u64}` | `{"result": [f64]}` | no |
//!
//! ### Compute (GPU)
//!
//! | Method | Params | Result | GPU |
//! |--------|--------|--------|-----|
//! | `compute.dispatch` | `{"op": "zeros"\|"ones", "shape": [u64]}` | `{"status": "completed", "op": str, "tensor_id": str, "shape": [u64]}` | yes |
//! | `compute.dispatch` | `{"op": "read", "tensor_id": str}` | `{"status": "completed", "tensor_id": str, "shape": [u64], "data": [f32]}` | yes |
//!
//! ### Tensor (GPU)
//!
//! | Method | Params | Result | GPU |
//! |--------|--------|--------|-----|
//! | `tensor.create` | `{"shape": [u64], "data"?: [f32]}` | `{"status": "completed", "tensor_id": str, "result_id": str, "shape": [u64], "elements": u64, "dtype": "f32"}` | yes |
//! | `tensor.matmul` | `{"lhs_id": str, "rhs_id": str}` | `{"status": "completed", "result_id": str, "shape": [u64], "elements": u64}` | yes |
//! | `tensor.matmul_inline` | `{"lhs": [[f64]], "rhs": [[f64]]}` | `{"result": [[f64]], "shape": [u64, u64]}` | no |
//! | `tensor.add` | `{"tensor_id": str, "other_id"?: str, "scalar"?: f64}` | `{"status": "completed", "result_id": str, "shape": [u64], "elements": u64}` | yes |
//! | `tensor.scale` | `{"tensor_id": str, "scalar": f64}` | `{"status": "completed", "result_id": str, "shape": [u64], "elements": u64}` | yes |
//! | `tensor.clamp` | `{"tensor_id": str, "min": f64, "max": f64}` | `{"status": "completed", "result_id": str, "shape": [u64], "elements": u64}` | yes |
//! | `tensor.reduce` | `{"tensor_id": str, "op"?: "sum"\|"mean"\|"max"\|"min"}` | `{"result": f64, "status": "completed", "value": f64, "op": str}` | yes |
//! | `tensor.sigmoid` | `{"tensor_id": str}` | `{"status": "completed", "result_id": str, "shape": [u64], "elements": u64}` | yes |
//!
//! ### Batch Pipeline (GPU)
//!
//! | Method | Params | Result | GPU |
//! |--------|--------|--------|-----|
//! | `tensor.batch.submit` | `{"ops": [BatchOp]}` | `{"status": "completed", "outputs": {alias: TensorResult\|ReadbackResult}, "ops_executed": u64}` | yes |
//!
//! `BatchOp` types: `create`, `add`, `mul`, `fma`, `scale`, `matmul`, `relu`,
//! `gelu`, `softmax`, `layer_norm`, `reshape`, `readback`. Each op references
//! previous results by `alias`. See `methods/batch.rs` for full wire contract.
//!
//! ### FHE (GPU)
//!
//! | Method | Params | Result | GPU |
//! |--------|--------|--------|-----|
//! | `fhe.ntt` | `{"modulus": u64, "degree": u64, "root_of_unity": u64, "coefficients": [u64]}` | `{"status": "completed", "modulus": u64, "degree": u64, "result": [u64]}` | yes |
//! | `fhe.pointwise_mul` | `{"modulus": u64, "degree": u64, "a": [u64], "b": [u64]}` | `{"status": "completed", "modulus": u64, "degree": u64, "result": [u64]}` | yes |

pub mod btsp;
pub mod btsp_frame;
pub mod jsonrpc;
pub mod methods;
pub mod transport;

pub use btsp::{BtspCipher, BtspSession};
pub use btsp_frame::{BtspFrameError, BtspFrameReader, BtspFrameWriter};
pub use jsonrpc::{JsonRpcError, JsonRpcRequest, JsonRpcResponse};
pub use transport::IpcServer;
