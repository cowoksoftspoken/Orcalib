@group(0) @binding(0) var<storage, read> in_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_data: array<f32>;

struct Uniforms {
    m: u32,
    n: u32,
}
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(16, 16)
fn transpose_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x < uniforms.n && y < uniforms.m) {
        let in_idx = y * uniforms.n + x;
        let out_idx = x * uniforms.m + y;
        out_data[out_idx] = in_data[in_idx];
    }
}
