@group(0) @binding(0) var<storage, read> in_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_data: array<f32>;

var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn reduce_scalar_pass2(
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let lid = local_id.x;
    var sum = 0.0;
    
    // Each thread processes multiple elements if num_elements > 256
    let num_elements = arrayLength(&in_data);
    for (var i = lid; i < num_elements; i = i + 256u) {
        sum = sum + in_data[i];
    }
    sdata[lid] = sum;
    workgroupBarrier();
    
    // Reduce sdata
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            sdata[lid] = sdata[lid] + sdata[lid + s];
        }
        workgroupBarrier();
    }
    
    if (lid == 0u) {
        out_data[0] = sdata[0];
    }
}
