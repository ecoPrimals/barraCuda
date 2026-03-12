// SPDX-License-Identifier: AGPL-3.0-only
// quality_filter.wgsl — Per-read parallel quality trimming
//
// One GPU thread per FASTQ read. Replicates the exact CPU logic:
//   leading trim → trailing trim → sliding window → min length check
//
// Input: packed quality bytes (4 per u32), per-read offsets + lengths
// Output: per-read (start << 16 | end), or 0 for failed reads
//
// Provenance: wetSpring handoff v5 (Feb 20, 2026) → ToadStool absorption

struct Params {
    n_reads: u32,
    leading_min_quality: u32,
    trailing_min_quality: u32,
    window_min_quality: u32,
    window_size: u32,
    min_length: u32,
    phred_offset: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> qual_data: array<u32>;
@group(0) @binding(2) var<storage, read> read_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> read_lengths: array<u32>;
@group(0) @binding(4) var<storage, read_write> results: array<u32>;

fn get_phred(byte_offset: u32) -> u32 {
    let word_idx = byte_offset / 4u;
    let byte_pos = byte_offset % 4u;
    let raw = (qual_data[word_idx] >> (byte_pos * 8u)) & 0xFFu;
    if (raw >= params.phred_offset) {
        return raw - params.phred_offset;
    }
    return 0u;
}

@compute @workgroup_size(256)
fn quality_filter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let read_idx = gid.x;
    if (read_idx >= params.n_reads) {
        return;
    }

    let offset = read_offsets[read_idx];
    let len = read_lengths[read_idx];

    if (len == 0u) {
        results[read_idx] = 0u;
        return;
    }

    // Leading trim: first base with quality >= threshold
    var lead: u32 = len;
    for (var i: u32 = 0u; i < len; i++) {
        if (get_phred(offset + i) >= params.leading_min_quality) {
            lead = i;
            break;
        }
    }
    if (lead >= len) {
        results[read_idx] = 0u;
        return;
    }

    // Trailing trim: last base with quality >= threshold
    var trail: u32 = 0u;
    for (var i: i32 = i32(len) - 1; i >= 0; i--) {
        if (get_phred(offset + u32(i)) >= params.trailing_min_quality) {
            trail = u32(i) + 1u;
            break;
        }
    }
    if (trail <= lead) {
        results[read_idx] = 0u;
        return;
    }

    // Sliding window on [lead..trail)
    let sub_len = trail - lead;
    var win_end: u32 = sub_len;

    if (sub_len < params.window_size) {
        var short_sum: u32 = 0u;
        for (var i: u32 = 0u; i < sub_len; i++) {
            short_sum += get_phred(offset + lead + i);
        }
        if (short_sum < params.window_min_quality * sub_len) {
            win_end = 0u;
        }
    } else {
        var window_sum: u32 = 0u;
        for (var i: u32 = 0u; i < params.window_size; i++) {
            window_sum += get_phred(offset + lead + i);
        }

        let threshold = params.window_min_quality * params.window_size;

        if (window_sum < threshold) {
            win_end = 0u;
        } else {
            for (var i: u32 = 1u; i <= sub_len - params.window_size; i++) {
                window_sum -= get_phred(offset + lead + i - 1u);
                window_sum += get_phred(offset + lead + i + params.window_size - 1u);
                if (window_sum < threshold) {
                    win_end = i;
                    break;
                }
            }
        }
    }

    let final_start = lead;
    let final_end = lead + win_end;
    let trimmed_len = final_end - final_start;

    if (trimmed_len < params.min_length) {
        results[read_idx] = 0u;
        return;
    }

    results[read_idx] = (final_start << 16u) | final_end;
}
