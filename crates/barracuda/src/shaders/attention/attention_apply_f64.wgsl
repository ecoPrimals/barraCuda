// Attention Apply: Apply attention weights to values
// Pass 3 of multi-pass attention implementation
//
// Computes: output[i,d] = sum_j (weights[i,j] * V[j,d])
// weights: [batch, heads, q_seq_len, kv_seq_len]
// V: [batch, heads, kv_seq_len, head_dim]
// output: [batch, heads, q_seq_len, head_dim]

struct AttentionParams {
    batch_size: u32,
    num_heads: u32,
    q_seq_len: u32,
    kv_seq_len: u32,
    head_dim: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<f64>;
@group(0) @binding(1) var<storage, read> value: array<f64>;
@group(0) @binding(2) var<storage, read_write> output: array<f64>;
@group(0) @binding(3) var<uniform> params: AttentionParams;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_head = global_id.z;
    let i = global_id.y; // query position (output row)
    let d = global_id.x; // head dimension
    
    if (i >= params.q_seq_len || d >= params.head_dim) {
        return;
    }
    
    let batch = batch_head / params.num_heads;
    let head = batch_head % params.num_heads;
    
    if (batch >= params.batch_size) {
        return;
    }
    
    var weighted_sum = 0.0;
    
    for (var j = 0u; j < params.kv_seq_len; j = j + 1u) {
        let weight_idx = batch * params.num_heads * params.q_seq_len * params.kv_seq_len
                       + head * params.q_seq_len * params.kv_seq_len
                       + i * params.kv_seq_len
                       + j;
        
        let value_idx = batch * params.num_heads * params.kv_seq_len * params.head_dim
                      + head * params.kv_seq_len * params.head_dim
                      + j * params.head_dim
                      + d;
        
        weighted_sum += weights[weight_idx] * value[value_idx];
    }
    
    let out_idx = batch * params.num_heads * params.q_seq_len * params.head_dim
                + head * params.q_seq_len * params.head_dim
                + i * params.head_dim
                + d;
    
    output[out_idx] = weighted_sum;
}
