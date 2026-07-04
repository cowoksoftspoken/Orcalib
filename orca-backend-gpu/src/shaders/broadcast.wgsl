@group(0) @binding(0) var<storage, read> in_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_data: array<f32>;

struct Uniforms {
    in_strides: vec4<u32>,
    out_strides: vec4<u32>,
    padded_in_shape: vec4<u32>,
    out_shape: vec4<u32>,
    rank: u32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
}
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(64)
fn expand_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    if (id >= arrayLength(&out_data)) {
        return;
    }

    // Convert output linear index to multi-dimensional index
    var multi_out: vec4<u32> = vec4<u32>(0u, 0u, 0u, 0u);
    var rem = id;
    
    if (uniforms.rank > 0u) { multi_out[0] = rem / uniforms.out_strides[0]; rem = rem % uniforms.out_strides[0]; }
    if (uniforms.rank > 1u) { multi_out[1] = rem / uniforms.out_strides[1]; rem = rem % uniforms.out_strides[1]; }
    if (uniforms.rank > 2u) { multi_out[2] = rem / uniforms.out_strides[2]; rem = rem % uniforms.out_strides[2]; }
    if (uniforms.rank > 3u) { multi_out[3] = rem / uniforms.out_strides[3]; rem = rem % uniforms.out_strides[3]; }

    // Map to input multi-dimensional index (0 if broadcasted)
    var multi_in: vec4<u32> = vec4<u32>(0u, 0u, 0u, 0u);
    if (uniforms.rank > 0u) { multi_in[0] = select(multi_out[0], 0u, uniforms.padded_in_shape[0] == 1u); }
    if (uniforms.rank > 1u) { multi_in[1] = select(multi_out[1], 0u, uniforms.padded_in_shape[1] == 1u); }
    if (uniforms.rank > 2u) { multi_in[2] = select(multi_out[2], 0u, uniforms.padded_in_shape[2] == 1u); }
    if (uniforms.rank > 3u) { multi_in[3] = select(multi_out[3], 0u, uniforms.padded_in_shape[3] == 1u); }

    // Convert input multi-dimensional index to linear index
    var in_idx = 0u;
    if (uniforms.rank > 0u) { in_idx = in_idx + multi_in[0] * uniforms.in_strides[0]; }
    if (uniforms.rank > 1u) { in_idx = in_idx + multi_in[1] * uniforms.in_strides[1]; }
    if (uniforms.rank > 2u) { in_idx = in_idx + multi_in[2] * uniforms.in_strides[2]; }
    if (uniforms.rank > 3u) { in_idx = in_idx + multi_in[3] * uniforms.in_strides[3]; }

    out_data[id] = in_data[in_idx];
}
