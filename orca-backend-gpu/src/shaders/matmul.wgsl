@group(0) @binding(0) var<storage, read> lhs: array<f32>;
@group(0) @binding(1) var<storage, read> rhs: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Uniforms {
    M: u32,
    K: u32,
    N: u32,
    _padding: u32,
}
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(16, 16)
fn matmul_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;

    if (row < uniforms.M && col < uniforms.N) {
        var sum = 0.0;
        for (var i = 0u; i < uniforms.K; i = i + 1u) {
            sum = sum + lhs[row * uniforms.K + i] * rhs[i * uniforms.N + col];
        }
        out[row * uniforms.N + col] = sum;
    }
}
