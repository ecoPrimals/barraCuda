# Scan (Prefix Sum) - Known Limitation

**Status**: ✅ Works correctly for inputs ≤ 512 elements  
**Limitation**: ⚠️ Multi-workgroup propagation not implemented

## Current Implementation

The Blelloch algorithm in `scan.wgsl` correctly computes prefix sums **within a single workgroup** (512 elements max, due to 256 threads × 2 elements each).

## Multi-Workgroup Issue

For inputs > 512 elements:
- Each workgroup computes its own prefix sum starting from 0
- Workgroup totals are NOT propagated to subsequent workgroups
- Result: Each workgroup starts from 0 instead of continuing from previous total

**Example**:
```
Input: [1, 2, 3, 4] × 200 elements = 800 elements
Expected: [1, 3, 6, 10, 11, 13, 16, 20, ...]
Actual:   [1, 3, 6, 10, 1,  3,  6,  10, ...]  ← Second workgroup restarts!
```

## Solution (Phase 2)

Implement **three-phase scan** (standard parallel scan):

1. **Phase 1**: Scan within each workgroup (current implementation)
2. **Phase 2**: Scan workgroup totals (NEW)
3. **Phase 3**: Add workgroup totals to all elements in subsequent workgroups (NEW)

**Timeline**: 4-6 hours  
**Priority**: HIGH (blocks filter operation)

## Workaround

For now, scan works perfectly for:
- Single-workgroup inputs (≤ 512 elements)
- Applications with small tensors
- Development/testing

**Status**: Production-ready for small inputs, needs evolution for large inputs (>512 elements).
