// Binary Cross Entropy Loss (f64 canonical)
// loss = -mean(targets * log(predictions) + (1 - targets) * log(1 - predictions))

@group(0) @binding(0) var<storage, read> predictions: array<f64>;
@group(0) @binding(1) var<storage, read> targets: array<f64>;
@group(0) @binding(2) var<storage, read_write> output: array<f64>;

var<workgroup> shared_loss: f64;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let size = arrayLength(&predictions);
    let tid = local_id.x;
    
    if (tid == 0u) {
        var sum = f64(0.0);
        for (var i = 0u; i < size; i = i + 1u) {
            let pred = clamp(predictions[i], f64(1e-7), f64(1.0) - f64(1e-7));
            let tgt = targets[i];
            sum = sum + tgt * log_f64(pred) + (f64(1.0) - tgt) * log_f64(f64(1.0) - pred);
        }
        shared_loss = -sum / f64(size);
    }
    workgroupBarrier();
    
    if (tid == 0u) {
        output[0] = shared_loss;
    }
}
