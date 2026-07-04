@group(0) @binding(0) var<storage, read> in_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn relu_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    if (id < arrayLength(&out)) {
        out[id] = max(in_data[id], 0.0);
    }
}

@compute @workgroup_size(64)
fn sigmoid_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    if (id < arrayLength(&out)) {
        let val = in_data[id];
        out[id] = 1.0 / (1.0 + exp(-val));
    }
}

@compute @workgroup_size(64)
fn exp_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    if (id < arrayLength(&out)) {
        out[id] = exp(in_data[id]);
    }
}

@compute @workgroup_size(64)
fn log_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    if (id < arrayLength(&out)) {
        out[id] = log(max(in_data[id], 1e-7));
    }
}

@compute @workgroup_size(64)
fn sqrt_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    if (id < arrayLength(&out)) {
        out[id] = sqrt(in_data[id]);
    }
}

