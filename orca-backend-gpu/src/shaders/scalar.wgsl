@group(0) @binding(0) var<storage, read> in_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

struct Uniforms {
    scalar: f32,
}
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(64)
fn mul_scalar_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    if (id < arrayLength(&out)) {
        out[id] = in_data[id] * uniforms.scalar;
    }
}
