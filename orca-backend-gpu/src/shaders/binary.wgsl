@group(0) @binding(0) var<storage, read> lhs: array<f32>;
@group(0) @binding(1) var<storage, read> rhs: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn add_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    if (id < arrayLength(&out)) {
        out[id] = lhs[id] + rhs[id];
    }
}

@compute @workgroup_size(64)
fn sub_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    if (id < arrayLength(&out)) {
        out[id] = lhs[id] - rhs[id];
    }
}

@compute @workgroup_size(64)
fn mul_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    if (id < arrayLength(&out)) {
        out[id] = lhs[id] * rhs[id];
    }
}

@compute @workgroup_size(64)
fn div_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    if (id < arrayLength(&out)) {
        out[id] = lhs[id] / rhs[id];
    }
}

