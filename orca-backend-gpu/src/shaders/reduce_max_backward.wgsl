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

@group(0) @binding(0) var<storage, read> grad_out: array<f32>;
@group(0) @binding(1) var<storage, read> in_primal: array<f32>;
@group(0) @binding(2) var<storage, read> out_primal: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_in: array<f32>;
@group(0) @binding(4) var<uniform> info: ShapeInfo;

@compute @workgroup_size(64)
fn max_to_shape_bw_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_id = global_id.x; // out_id is actually the index in `grad_out` (which has out_shape)
    
    var total_out = 1u;
    if (info.rank > 0u) { total_out = total_out * info.out_shape[0]; }
    if (info.rank > 1u) { total_out = total_out * info.out_shape[1]; }
    if (info.rank > 2u) { total_out = total_out * info.out_shape[2]; }
    if (info.rank > 3u) { total_out = total_out * info.out_shape[3]; }
    
    if (out_id >= total_out) {
        return;
    }

    var out_multi: vec4<u32> = vec4<u32>(0u, 0u, 0u, 0u);
    var rem = out_id;
    if (info.rank > 0u) { out_multi[0] = rem / info.out_strides[0]; rem = rem % info.out_strides[0]; }
    if (info.rank > 1u) { out_multi[1] = rem / info.out_strides[1]; rem = rem % info.out_strides[1]; }
    if (info.rank > 2u) { out_multi[2] = rem / info.out_strides[2]; rem = rem % info.out_strides[2]; }
    if (info.rank > 3u) { out_multi[3] = rem / info.out_strides[3]; rem = rem % info.out_strides[3]; }

    var start0 = out_multi[0]; var end0 = out_multi[0] + 1u;
    var start1 = out_multi[1]; var end1 = out_multi[1] + 1u;
    var start2 = out_multi[2]; var end2 = out_multi[2] + 1u;
    var start3 = out_multi[3]; var end3 = out_multi[3] + 1u;

    if (info.rank > 0u && info.out_shape[0] == 1u && info.in_shape[0] > 1u) { start0 = 0u; end0 = info.in_shape[0]; }
    if (info.rank > 1u && info.out_shape[1] == 1u && info.in_shape[1] > 1u) { start1 = 0u; end1 = info.in_shape[1]; }
    if (info.rank > 2u && info.out_shape[2] == 1u && info.in_shape[2] > 1u) { start2 = 0u; end2 = info.in_shape[2]; }
    if (info.rank > 3u && info.out_shape[3] == 1u && info.in_shape[3] > 1u) { start3 = 0u; end3 = info.in_shape[3]; }

    let g_out = grad_out[out_id];
    let max_val = out_primal[out_id];
    var count = 0.0;

    // First pass to count how many inputs match the max value
    if (info.rank == 0u) {
        if (in_primal[0] == max_val) { count = count + 1.0; }
    } else {
        for (var i0 = start0; i0 < end0; i0 = i0 + 1u) {
            var idx0 = 0u; if (info.rank > 0u) { idx0 = i0 * info.in_strides[0]; }
            for (var i1 = start1; i1 < end1; i1 = i1 + 1u) {
                var idx1 = idx0; if (info.rank > 1u) { idx1 = idx1 + i1 * info.in_strides[1]; }
                for (var i2 = start2; i2 < end2; i2 = i2 + 1u) {
                    var idx2 = idx1; if (info.rank > 2u) { idx2 = idx2 + i2 * info.in_strides[2]; }
                    for (var i3 = start3; i3 < end3; i3 = i3 + 1u) {
                        var in_idx = idx2; if (info.rank > 3u) { in_idx = in_idx + i3 * info.in_strides[3]; }
                        if (in_primal[in_idx] == max_val) { count = count + 1.0; }
                    }
                }
            }
        }
    }

    let grad_share = g_out / count;

    // Second pass to distribute gradients
    if (info.rank == 0u) {
        if (in_primal[0] == max_val) { grad_in[0] = grad_share; } else { grad_in[0] = 0.0; }
    } else {
        for (var i0 = start0; i0 < end0; i0 = i0 + 1u) {
            var idx0 = 0u; if (info.rank > 0u) { idx0 = i0 * info.in_strides[0]; }
            for (var i1 = start1; i1 < end1; i1 = i1 + 1u) {
                var idx1 = idx0; if (info.rank > 1u) { idx1 = idx1 + i1 * info.in_strides[1]; }
                for (var i2 = start2; i2 < end2; i2 = i2 + 1u) {
                    var idx2 = idx1; if (info.rank > 2u) { idx2 = idx2 + i2 * info.in_strides[2]; }
                    for (var i3 = start3; i3 < end3; i3 = i3 + 1u) {
                        var in_idx = idx2; if (info.rank > 3u) { in_idx = in_idx + i3 * info.in_strides[3]; }
                        if (in_primal[in_idx] == max_val) {
                            grad_in[in_idx] = grad_share;
                        } else {
                            grad_in[in_idx] = 0.0;
                        }
                    }
                }
            }
        }
    }
}
