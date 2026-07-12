@group(0) @binding(0) var<storage, read> in_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_data: array<f32>;

struct Uniforms {
    in_strides: vec4<u32>,
    out_strides: vec4<u32>,
    out_shape: vec4<u32>,
    dim0: u32,
    dim1: u32,
    rank: u32,
    num_elements: u32,
}
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(256)
fn transpose_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.num_elements) {
        return;
    }

    // Compute N-D coordinate in the output tensor
    var out_coord = vec4<u32>(0u, 0u, 0u, 0u);
    var rem = idx;
    
    // Unroll up to 4 dimensions (we only support up to rank 4 here for simplicity)
    // The dimensions are padded left, so the valid dimensions are (4 - rank) to 3.
    let start_dim = 4u - uniforms.rank;
    
    for (var i = 0u; i < 4u; i = i + 1u) {
        if (i >= start_dim) {
            let stride = uniforms.out_strides[i];
            out_coord[i] = rem / stride;
            rem = rem % stride;
        }
    }

    // The coordinate in the input tensor is the same, but with dim0 and dim1 swapped!
    // Note: dim0 and dim1 are passed as their padded indices (i.e. original_dim + (4 - rank))
    var in_coord = out_coord;
    let temp = in_coord[uniforms.dim0];
    in_coord[uniforms.dim0] = in_coord[uniforms.dim1];
    in_coord[uniforms.dim1] = temp;

    // Compute the 1D index in the input tensor
    var in_idx = 0u;
    for (var i = 0u; i < 4u; i = i + 1u) {
        if (i >= start_dim) {
            in_idx = in_idx + in_coord[i] * uniforms.in_strides[i];
        }
    }

    out_data[idx] = in_data[in_idx];
}
