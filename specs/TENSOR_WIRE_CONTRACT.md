<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tensor IPC Wire Contract

**Status**: Sprint 73 — 90 methods, stadial gate compliant, multi-model regression + signal processing + ecology + gamma absorbed from springs
**Version**: 1.6.0
**Authority**: barraCuda primal (self-knowledge)
**Implements**: wateringHole `PRIMAL_IPC_PROTOCOL.md` v3.1, `SEMANTIC_METHOD_NAMING_STANDARD.md`

---

## Purpose

This document defines the **exact** JSON-RPC response schemas for all
`tensor.*` methods exposed by barraCuda. Springs composing with barraCuda
(primalSpring, hotSpring, neuralSpring, etc.) use typed extractors
(`call_extract_f64`, `call_extract_vec_f64`) that rely on **consistent keys**
across all tensor operations.

Prior to this contract, different tensor methods returned results under
different keys (`tensor_id`, `result_id`, `result`, `data`), forcing springs
to guess the shape. This contract eliminates that ambiguity.

---

## Response Categories

All responses are JSON-RPC 2.0 success envelopes:
```json
{"jsonrpc": "2.0", "result": {<payload>}, "id": N}
```

The `<payload>` shape depends on the operation category:

### Category 1: Tensor-Producing Operations

Methods that create or transform a tensor, storing the result in barraCuda's
tensor store and returning a handle for chaining.

**Schema**:
```json
{
  "status": "completed",
  "result_id": "t_abc123...",
  "shape": [M, N],
  "elements": E
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | `"completed"` | Always present, always `"completed"` on success |
| `result_id` | `string` | Opaque tensor handle — use in subsequent calls |
| `shape` | `[u64]` | Tensor dimensions |
| `elements` | `u64` | Product of shape dimensions |

**Methods in this category**:
- `tensor.matmul` — matrix multiplication
- `tensor.add` — element-wise addition (tensor + tensor or tensor + scalar)
- `tensor.scale` — scalar multiplication
- `tensor.clamp` — element-wise clamping
- `tensor.sigmoid` — element-wise sigmoid activation

### Category 1a: Tensor Creation

`tensor.create` is a tensor-producing op with additional metadata:

```json
{
  "status": "completed",
  "tensor_id": "t_abc123...",
  "result_id": "t_abc123...",
  "shape": [M, N],
  "elements": E,
  "dtype": "f32"
}
```

`tensor_id` and `result_id` are identical — `tensor_id` exists for backward
compatibility; `result_id` exists for uniformity with other tensor-producing ops.
Springs SHOULD use `result_id` for new code.

#### CPU Fallback (Sprint 44c)

On headless hosts without GPU, `tensor.create` automatically falls back to
CPU-resident storage. The response includes an additional field:

```json
{
  "status": "completed",
  "tensor_id": "abc123...",
  "result_id": "abc123...",
  "shape": [M, N],
  "elements": E,
  "dtype": "f32",
  "backend": "cpu"
}
```

When `"backend": "cpu"` is present, the tensor lives in CPU memory. All
subsequent handle-based ops (`tensor.matmul`, `tensor.add`, `tensor.scale`,
`tensor.clamp`, `tensor.reduce`, `tensor.sigmoid`) work transparently on
CPU tensors — callers do not need to change their code.

The `"backend"` field is absent when tensors are GPU-resident (default).
Springs SHOULD NOT branch on `backend` — the API is identical for both paths.

### Category 2: Scalar-Producing Operations

Methods that reduce a tensor to a scalar value.

**Schema**:
```json
{
  "status": "completed",
  "value": 42.0,
  "op": "sum"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | `"completed"` | Always present |
| `value` | `f64` | The scalar result |
| `op` | `string` | The reduction operation performed |

**Methods in this category**:
- `tensor.reduce` — reduce to scalar (`sum`, `mean`, `max`, `min`)

### Category 3: Batch Pipeline

`tensor.batch.submit` executes a fused multi-op pipeline in a single GPU
command submission. This is the composition primitive for springs that need
"matmul → softmax → readback" as one IPC round-trip.

**Request**:
```json
{
  "ops": [
    {"op": "create", "alias": "x", "data": [1,2,3,4], "shape": [1,4]},
    {"op": "create", "alias": "w", "data": [1,0,0,1], "shape": [4,1]},
    {"op": "matmul", "alias": "h", "a": "x", "b": "w"},
    {"op": "relu", "alias": "out", "input": "h"},
    {"op": "readback", "alias": "final", "input": "out"}
  ]
}
```

**Response**:
```json
{
  "status": "completed",
  "outputs": {
    "x":     {"result_id": "t_...", "shape": [1,4], "elements": 4},
    "h":     {"result_id": "t_...", "shape": [1,1], "elements": 1},
    "out":   {"result_id": "t_...", "shape": [1,1], "elements": 1},
    "final": {"data": [1.0], "shape": [1,1]}
  },
  "ops_executed": 5
}
```

**Supported batch ops**:

| Op | Params | Description |
|----|--------|-------------|
| `create` | `alias`, `shape`, `data?` | Upload data to GPU |
| `add` | `alias`, `a`, `b` | Element-wise add |
| `sub` | `alias`, `a`, `b` | Element-wise subtract (Sprint 66 — leapfrog HMC) |
| `mul` | `alias`, `a`, `b` | Element-wise multiply |
| `negate` | `alias`, `input` | Element-wise sign flip (Sprint 66 — force conventions) |
| `fma` | `alias`, `a`, `b`, `c` | Fused multiply-add: a*b+c |
| `scale` | `alias`, `input`, `scalar` | Scalar multiplication |
| `matmul` | `alias`, `a`, `b` | Matrix multiplication |
| `relu` | `alias`, `input` | ReLU activation |
| `gelu` | `alias`, `input` | GELU activation |
| `softmax` | `alias`, `input` | Row-wise softmax |
| `layer_norm` | `alias`, `input`, `feature_size` | Layer normalization |
| `reshape` | `alias`, `input`, `shape` | Metadata-only reshape |
| `readback` | `alias`, `input` | Read GPU data to host |

Each op references previous results by `alias`. The `alias` field is used as
the key in the `outputs` map. Readback results include `data` (host f32 array)
instead of `result_id`.

---

## Extractor Patterns for Springs

### Extract scalar (e.g. after `tensor.reduce`)
```rust
let value = response["result"]["value"].as_f64().unwrap();
```

### Extract tensor handle (e.g. after `tensor.matmul`)
```rust
let result_id = response["result"]["result_id"].as_str().unwrap();
```

### Extract batch readback data
```rust
let data = response["result"]["outputs"]["final"]["data"]
    .as_array().unwrap();
```

---

## Error Responses

All tensor methods use standard JSON-RPC error codes:

| Code | Meaning | When |
|------|---------|------|
| -32602 | Invalid params | Missing field, bad shape, unknown alias |
| -32603 | Internal error | No GPU, device lost, execution failure |

---

## Backward Compatibility

- `tensor.create` still includes `tensor_id` alongside `result_id`
- All ops now include `status: "completed"` (previously only `tensor.matmul` did)
- `tensor.reduce` returns `value` instead of the previous `result` key

Springs using the old `result` key for `tensor.reduce` must update to `value`.
All other changes are additive (new fields, not renamed fields).

---

## IPC Namespace Guide

barraCuda exposes operations across multiple semantic namespaces. Each namespace
groups related operations per `SEMANTIC_METHOD_NAMING_STANDARD.md`:

| Namespace | Domain | Methods |
|-----------|--------|---------|
| `tensor.*` | Handle-based GPU/CPU tensor ops | `create`, `matmul`, `matmul_inline`, `add`, `scale`, `clamp`, `reduce`, `sigmoid`, `batch.submit` |
| `stats.*` | Descriptive statistics | `mean`, `std_dev`, `variance`, `weighted_mean`, `correlation`, `shannon`, `entropy`, `covariance`, `spearman`, `fit_linear`, `fit_quadratic`, `fit_exponential`, `fit_logarithmic`, `empirical_spectral_density`, `chi_squared`, `anova_oneway`, `simpson`, `bray_curtis`, `hill`, `rarefaction_curve`, `gamma_fit`, `gamma_cdf` |
| `signal.*` | Signal processing | `detect_peaks`, `bandpass`, `derivative` |
| `activation.*` | Psychophysical / activation functions | `fitts`, `hick`, `softmax`, `gelu` |
| `linalg.*` | Linear algebra (CPU, inline-data) | `solve`, `eigenvalues`, `svd`, `qr`, `graph_laplacian` |
| `spectral.*` | Spectral analysis | `fft`, `power_spectrum`, `stft` |
| `noise.*` | Procedural noise | `perlin2d`, `perlin3d` |
| `rng.*` | Random number generation | `uniform` |
| `fhe.*` | Fully homomorphic encryption | `ntt`, `pointwise_mul` |
| `math.*` | Scalar math functions | `sigmoid`, `log2` |
| `ml.*` | Machine learning | `mlp_forward`, `mlp_train`, `attention`, `esn_predict` |
| `ode.*` | Differential equations | `step` |
| `nautilus.*` | Anomaly detection sessions | `create`, `observe`, `train`, `predict`, `export`, `import` |
| `precision.*` | Precision routing advisory | `route` |
| `compute.*` | Low-level GPU dispatch | `dispatch`, `health` |

**When to use `tensor.matmul_inline` vs `tensor.matmul`**: Use `matmul_inline`
for one-shot matrix multiplications with inline data (no handle round-trip).
Use `tensor.create` + `tensor.matmul` for multi-step pipelines where tensors
are reused across operations.

**The `math.*` namespace is intentionally sparse** — it provides scalar-valued
functions not covered by other namespaces. Statistics go in `stats.*`, matrix
ops in `linalg.*`, spectral ops in `spectral.*`.

---

## Method Stability Tiers

Per `INTERSTADIAL_EXIT_CRITERIA.md`, every registered method has a stability
tier annotation. Tiers define the contract strength for downstream consumers:

| Tier | Meaning |
|------|---------|
| **stable** | Wire format frozen. Breaking changes require semver major bump. |
| **evolving** | Wire format may change between minor versions. Additive-only changes preferred. |
| **internal** | Not part of the public contract. May be removed or restructured. |

| Namespace | Stability | Rationale |
|-----------|-----------|-----------|
| `health.*` | stable | Non-negotiable ecosystem probes per wateringHole standard |
| `capabilities.list` | stable | Wire Standard L2 |
| `identity.get` | stable | Wire Standard L2 |
| `primal.info` | stable | Runtime discovery contract |
| `primal.capabilities` | stable | Alias for `capabilities.list` |
| `primal.announce` | stable | biomeOS Neural API self-registration (Wave 43 schema) |
| `auth.*` | stable | MethodGate JH-0 introspection |
| `device.*` | stable | Hardware discovery surface |
| `precision.route` | stable | Cross-primal routing advisory |
| `tensor.*` | stable | Core GPU/CPU tensor ops — frozen schemas |
| `stats.*` | stable | CPU statistics — frozen schemas |
| `linalg.*` | stable | CPU linear algebra — frozen schemas |
| `spectral.*` | stable | Spectral analysis — frozen schemas |
| `math.*` | stable | Scalar math — frozen schemas |
| `activation.*` | stable | Activation functions — frozen schemas |
| `compute.dispatch` | stable | Low-level GPU dispatch |
| `fhe.*` | stable | Fully homomorphic encryption — frozen schemas |
| `ml.*` | evolving | ML ops — additive expansion expected |
| `ode.*` | evolving | Differential equation solvers — may grow |
| `nautilus.*` | evolving | Anomaly detection sessions — schema stabilising |
| `noise.*` | stable | Procedural noise — frozen |
| `rng.*` | stable | RNG — frozen |
| `btsp.negotiate` | stable | BTSP Phase 3 cipher upgrade |
| `btsp.capabilities` | stable | BTSP cipher suite advertisement |
| `tolerances.get` | stable | Numerical tolerance query |
| `validate.gpu_stack` | stable | GPU validation suite |

---

## Socket Naming

barraCuda uses **domain-based socket naming** per `PRIMAL_SELF_KNOWLEDGE_STANDARD.md` §3:

- **Authoritative socket**: `$BIOMEOS_SOCKET_DIR/math.sock`
  (or `math-{family_id}.sock` when `FAMILY_ID` is set)
- **Legacy symlink**: `barracuda.sock → math.sock` (backward compatibility)

The socket named `math.sock` is the JSON-RPC IPC endpoint. The tarpc binary
protocol socket is `barracuda-{family_id}.sock`. Springs discovering barraCuda
should use `discover_by_capability("tensor")` or `discover_by_capability("math")`
— capability-based discovery, never hardcoded socket names.

---

## Sovereign Dispatch Contract (CG-3: GPU API Alignment)

barraCuda's sovereign dispatch wire (`sovereign_dispatch_wire.rs`) defines the
contract for the HMMA tensor-core execution path: barraCuda → coralReef → toadStool.

### barraCuda → coralReef (compile request)

When `kernel_router` routes a `DenseMatmul` with tensor-core-eligible precision
(F16/BF16/TF32) to `KernelTarget::Sovereign`, barraCuda calls coralReef's
`shader.compile.wgsl` (or `shader.compile.gemm`) with:

```json
{
  "method": "shader.compile.wgsl",
  "params": {
    "wgsl_source": "<matmul shader>",
    "target_sm": 70,
    "precision_advice": {
      "tier": "f16",
      "needs_transcendental_lowering": false,
      "hardware_hint": "tensor_core"
    }
  }
}
```

### coralReef → barraCuda (compile response)

coralReef returns a `CompileResponse` / `CompilationInfoResponse`. The fields
barraCuda consumes to build `ShaderDispatchInfo`:

| coralReef field | barraCuda field | Type | Description |
|-----------------|-----------------|------|-------------|
| `binary_b64` | `binary` (base64-decoded) | `[u8]` | Native PTX/SASS binary |
| `gprs` | `gpr_count` | `u32` | General-purpose registers used |
| `workgroup` | `workgroup` | `[u32; 3]` | Workgroup dimensions |
| `shared_memory` | `shared_mem_bytes` | `u32` | Shared memory bytes |
| `barriers` | `barrier_count` | `u32` | Barrier count |

### barraCuda → toadStool (dispatch submission)

barraCuda's `submit_dispatch()` sends to toadStool `compute.dispatch.submit`:

```json
{
  "method": "compute.dispatch.submit",
  "params": {
    "binary": "<base64 native binary>",
    "workgroups": [Wx, Wy, Wz],
    "gpr_count": N,
    "shared_mem_bytes": M,
    "barrier_count": B,
    "hardware_hint": "tensor_core",
    "buffers": [
      {"index": 0, "data": "<base64>", "size": S, "read_only": true},
      {"index": 1, "data": "<base64>", "size": S, "read_only": false}
    ]
  }
}
```

### Status

- barraCuda routing: **DONE** (Sprint 64, `KernelTarget::Sovereign`)
- barraCuda wire format: **DONE** (`sovereign_dispatch_wire.rs`)
- coralReef HMMA codegen: **PENDING** (blocks end-to-end execution)
- toadStool QMD consumption: **PENDING** (QMD builders exist, IPC handshake pending)

The `submit_and_map` internal API is not part of this cross-primal contract —
it is barraCuda's local GPU readback mechanism for wgpu-resident results. The
sovereign path uses IPC-based readback via `compute.dispatch.submit` response.

---

**License**: AGPL-3.0-or-later
