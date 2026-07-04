struct ShapeInfo {
    in_strides: vec4<u32>,
    out_strides: vec4<u32>,
    in_shape: vec4<u32>,
    out_shape: vec4<u32>, // padded_out
    rank: u32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
}

@group(0) @binding(0) var<storage, read> in_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_data: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> info: ShapeInfo;



@compute @workgroup_size(64)
fn sum_to_shape_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    if (id >= arrayLength(&in_data)) {
        return;
    }

    // Calculate multi-dimensional index for input
    var multi_in: vec4<u32> = vec4<u32>(0u, 0u, 0u, 0u);
    var rem = id;
    
    // Unroll up to 4 dimensions (max supported rank for now)
    if (info.rank > 0u) { multi_in[0] = rem / info.in_strides[0]; rem = rem % info.in_strides[0]; }
    if (info.rank > 1u) { multi_in[1] = rem / info.in_strides[1]; rem = rem % info.in_strides[1]; }
    if (info.rank > 2u) { multi_in[2] = rem / info.in_strides[2]; rem = rem % info.in_strides[2]; }
    if (info.rank > 3u) { multi_in[3] = rem / info.in_strides[3]; rem = rem % info.in_strides[3]; }

    // Map to multi-dimensional index for output (force to 0 if out_shape[d] == 1)
    var multi_out: vec4<u32> = vec4<u32>(0u, 0u, 0u, 0u);
    if (info.rank > 0u) { multi_out[0] = select(multi_in[0], 0u, info.out_shape[0] == 1u); }
    if (info.rank > 1u) { multi_out[1] = select(multi_in[1], 0u, info.out_shape[1] == 1u); }
    if (info.rank > 2u) { multi_out[2] = select(multi_in[2], 0u, info.out_shape[2] == 1u); }
    if (info.rank > 3u) { multi_out[3] = select(multi_in[3], 0u, info.out_shape[3] == 1u); }

    // Calculate linear index for output
    var out_idx = 0u;
    if (info.rank > 0u) { out_idx = out_idx + multi_out[0] * info.out_strides[0]; }
    if (info.rank > 1u) { out_idx = out_idx + multi_out[1] * info.out_strides[1]; }
    if (info.rank > 2u) { out_idx = out_idx + multi_out[2] * info.out_strides[2]; }
    if (info.rank > 3u) { out_idx = out_idx + multi_out[3] * info.out_strides[3]; }

    // Atomic add to output
    let val = in_data[id];
    var old_u32 = atomicLoad(&out_data[out_idx]);
    loop {
        let old_f32 = bitcast<f32>(old_u32);
        let new_f32 = old_f32 + val;
        let new_u32 = bitcast<u32>(new_f32);
        let res = atomicCompareExchangeWeak(&out_data[out_idx], old_u32, new_u32);
        if (res.exchanged) {
            break;
        }
        old_u32 = res.old_value;
    }
}
