@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read> index: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_data: array<f32>;

struct Uniforms {
    dim: u32,
    rank: u32,
    num_elements: u32,
    pad: u32,
    in_strides: vec4<u32>,
    index_strides: vec4<u32>,
};

@group(0) @binding(3) var<uniform> info: Uniforms;

@compute @workgroup_size(64)
fn gather_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    if (id >= info.num_elements) {
        return;
    }

    var multi = vec4<u32>(0u, 0u, 0u, 0u);
    var temp = id;
    for (var i = 0u; i < info.rank; i++) {
        multi[i] = temp / info.index_strides[i];
        temp = temp % info.index_strides[i];
    }

    multi[info.dim] = u32(index[id]);

    var src_idx = 0u;
    for (var i = 0u; i < info.rank; i++) {
        src_idx = src_idx + multi[i] * info.in_strides[i];
    }

    out_data[id] = src[src_idx];
}
