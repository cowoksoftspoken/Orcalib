@group(0) @binding(0) var<storage, read> in_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_data: array<atomic<u32>>;

@compute @workgroup_size(256)
fn check_main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let id = global_id.x;
    if (id >= arrayLength(&in_data)) {
        return;
    }
    
    let val = in_data[id];
    let bits = bitcast<u32>(val);
    let exponent = (bits >> 23u) & 0xFFu;
    
    if (exponent == 0xFFu) {
        atomicStore(&out_data[0], 1u);
    }
}
