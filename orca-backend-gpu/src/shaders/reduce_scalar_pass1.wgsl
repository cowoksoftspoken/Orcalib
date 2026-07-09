const WORKGROUP_SIZE: u32 = 256u;

@group(0) @binding(0) var<storage, read> in_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_data: array<f32>;

var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn reduce_scalar_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let id = global_id.x;
    let lid = local_id.x;

    // Load global data into shared memory
    if (id < arrayLength(&in_data)) {
        sdata[lid] = in_data[id];
    } else {
        sdata[lid] = 0.0;
    }

    workgroupBarrier();

    // Parallel tree reduction within the workgroup
    for (var s = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            sdata[lid] = sdata[lid] + sdata[lid + s];
        }
        workgroupBarrier();
    }

    // Only thread 0 of the workgroup writes to global memory intermediate block
    if (lid == 0u) {
        out_data[group_id.x] = sdata[0];
    }
}
