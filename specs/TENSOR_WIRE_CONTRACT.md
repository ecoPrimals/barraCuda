<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tensor IPC Wire Contract

**Status**: Sprint 46b — NUCLEUS env wiring + role-based naming + 12-axis clean
**Version**: 1.1.0
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
| `mul` | `alias`, `a`, `b` | Element-wise multiply |
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
| `stats.*` | Descriptive statistics | `mean`, `std_dev`, `variance`, `weighted_mean`, `correlation` |
| `activation.*` | Psychophysical / activation functions | `fitts`, `hick` |
| `linalg.*` | Linear algebra (CPU, inline-data) | `solve`, `eigenvalues` |
| `spectral.*` | Spectral analysis | `fft`, `power_spectrum` |
| `noise.*` | Procedural noise | `perlin2d`, `perlin3d` |
| `fhe.*` | Fully homomorphic encryption | `ntt`, `pointwise_mul` |
| `math.*` | Scalar math functions | `sigmoid`, `log2` |
| `compute.*` | Low-level GPU dispatch | `dispatch`, `health` |

**When to use `tensor.matmul_inline` vs `tensor.matmul`**: Use `matmul_inline`
for one-shot matrix multiplications with inline data (no handle round-trip).
Use `tensor.create` + `tensor.matmul` for multi-step pipelines where tensors
are reused across operations.

**The `math.*` namespace is intentionally sparse** — it provides scalar-valued
functions not covered by other namespaces. Statistics go in `stats.*`, matrix
ops in `linalg.*`, spectral ops in `spectral.*`.

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

**License**: AGPL-3.0-or-later
