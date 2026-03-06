// SPDX-License-Identifier: AGPL-3.0-or-later
#![expect(clippy::unwrap_used, reason = "tests")]
//! Unit tests for `BatchedEncoder`

use barracuda::device::{BatchedEncoder, test_pool};

const SHADER_FILL: &str = r"
    @group(0) @binding(0) var<storage, read_write> out: array<f32>;
    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        out[gid.x] = f32(gid.x);
    }
";

const SHADER_DOUBLE: &str = r"
    @group(0) @binding(0) var<storage, read> inp: array<f32>;
    @group(0) @binding(1) var<storage, read_write> out: array<f32>;
    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        out[gid.x] = inp[gid.x] * 2.0;
    }
";

#[tokio::test]
async fn batched_encoder_empty_submits_cleanly() {
    let device = test_pool::get_test_device().await;
    let batch = BatchedEncoder::new(&device);
    batch.submit();
}

#[tokio::test]
async fn batched_encoder_two_passes_execute() {
    let device = test_pool::get_test_device().await;
    let n = 256usize;
    let buf_a = device.create_buffer_f32(n).unwrap();
    let buf_b = device.create_buffer_f32(n).unwrap();

    let mut batch = BatchedEncoder::new(&device);
    let _ = batch
        .dispatch("fill", SHADER_FILL, "main")
        .storage_rw(0, &buf_a)
        .workgroups(1, 1, 1);
    let _ = batch
        .dispatch("double", SHADER_DOUBLE, "main")
        .storage_read(0, &buf_a)
        .storage_rw(1, &buf_b)
        .workgroups(1, 1, 1);
    batch.submit();

    let out = device.read_buffer_f32(&buf_b, n).unwrap();
    for (i, &v) in out.iter().enumerate() {
        assert!(
            (v - (i as f32 * 2.0)).abs() < 1e-5,
            "buf_b[{}] = {}, expected {}",
            i,
            v,
            i as f32 * 2.0
        );
    }
}
