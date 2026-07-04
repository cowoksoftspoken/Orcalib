@group(0) @binding(0) var<storage, read> grad_out: array<f32>;
@group(0) @binding(1) var<storage, read> primal: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn relu_backward_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    if (id < arrayLength(&out)) {
        if (primal[id] > 0.0) {
            out[id] = grad_out[id];
        } else {
            out[id] = 0.0;
        }
    }
}

@compute @workgroup_size(64)
fn sigmoid_backward_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    if (id < arrayLength(&out)) {
        let y = primal[id];
        out[id] = grad_out[id] * y * (1.0 - y);
    }
}

@compute @workgroup_size(64)
fn exp_backward_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    if (id < arrayLength(&out)) {
        out[id] = grad_out[id] * primal[id];
    }
}

@compute @workgroup_size(64)
fn log_backward_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    if (id < arrayLength(&out)) {
        out[id] = grad_out[id] / max(primal[id], 1e-7);
    }
}
