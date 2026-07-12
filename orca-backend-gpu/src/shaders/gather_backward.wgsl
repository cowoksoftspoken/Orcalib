@group(0) @binding(0) var<storage, read> grad_out: array<f32>;
@group(0) @binding(1) var<storage, read> index: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_in: array<f32>;

struct Uniforms {
    dim: u32,
    rank: u32,
    in_elements: u32,
    idx_dim_size: u32,
    idx_strides: vec4<u32>,
    in_strides: vec4<u32>,
};

@group(0) @binding(3) var<uniform> info: Uniforms;

@compute @workgroup_size(64)
fn gather_bw_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let in_idx = global_id.x; 
    if (in_idx >= info.in_elements) {
        return;
    }

    var in_multi = vec4<u32>(0u, 0u, 0u, 0u);
    var temp = in_idx;
    for (var i = 0u; i < info.rank; i = i + 1u) {
        in_multi[i] = temp / info.in_strides[i];
        temp = temp % info.in_strides[i];
    }

    var sum = 0.0;
    for (var k = 0u; k < info.idx_dim_size; k = k + 1u) {
        var idx_multi = in_multi;
        idx_multi[info.dim] = k;

        var idx_1d = 0u;
        for (var i = 0u; i < info.rank; i = i + 1u) {
            idx_1d = idx_1d + idx_multi[i] * info.idx_strides[i];
        }

        let gather_idx = u32(index[idx_1d]);
        if (gather_idx == in_multi[info.dim]) {
            sum = sum + grad_out[idx_1d];
        }
    }

    grad_in[in_idx] = sum;
}
