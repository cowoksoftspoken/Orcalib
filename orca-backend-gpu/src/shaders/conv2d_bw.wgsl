struct ConvUniforms {
    n: u32,
    c_in: u32,
    h_in: u32,
    w_in: u32,
    c_out: u32,
    k_h: u32,
    k_w: u32,
    h_out: u32,
    w_out: u32,
    stride: u32,
    padding: u32,
    has_bias: u32,
    dilation: u32,
    groups: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> grad_out: array<f32>;
@group(0) @binding(1) var<storage, read> input_or_weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_in: array<f32>; // NO ATOMIC!
@group(0) @binding(3) var<uniform> uniforms: ConvUniforms;

var<workgroup> sdata: array<f32, 256>;

// BACKWARD INPUT
// input_or_weight is weight.
// grad_in is size of input.
@compute @workgroup_size(8, 8, 1)
fn backward_input_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let iw = global_id.x;
    let ih = global_id.y;
    let b_ic = global_id.z; // (b * c_in) + ic
    
    if (iw >= uniforms.w_in || ih >= uniforms.h_in || b_ic >= uniforms.n * uniforms.c_in) {
        return;
    }
    
    let b = b_ic / uniforms.c_in;
    let ic = b_ic % uniforms.c_in;
    
    let in_c_per_g = uniforms.c_in / uniforms.groups;
    let out_c_per_g = uniforms.c_out / uniforms.groups;
    let g = ic / in_c_per_g;
    let ic_w = ic % in_c_per_g;
    
    var val: f32 = 0.0;
    
    // Loop over output channels for this group
    for (var oc = g * out_c_per_g; oc < (g + 1u) * out_c_per_g; oc = oc + 1u) {
        for (var kh = 0u; kh < uniforms.k_h; kh = kh + 1u) {
            for (var kw = 0u; kw < uniforms.k_w; kw = kw + 1u) {
                // h_in = h_out * stride - padding + kh * dilation
                // => h_out = (h_in + padding - kh * dilation) / stride
                // Must be exact division!
                let oh_num = i32(ih) + i32(uniforms.padding) - i32(kh * uniforms.dilation);
                let ow_num = i32(iw) + i32(uniforms.padding) - i32(kw * uniforms.dilation);
                
                if (oh_num >= 0 && ow_num >= 0 && 
                    oh_num % i32(uniforms.stride) == 0 && ow_num % i32(uniforms.stride) == 0) {
                    
                    let oh = u32(oh_num) / uniforms.stride;
                    let ow = u32(ow_num) / uniforms.stride;
                    
                    if (oh < uniforms.h_out && ow < uniforms.w_out) {
                        let go_idx = b * (uniforms.c_out * uniforms.h_out * uniforms.w_out) + 
                                     oc * (uniforms.h_out * uniforms.w_out) + 
                                     oh * uniforms.w_out + 
                                     ow;
                                     
                        let w_idx = oc * (in_c_per_g * uniforms.k_h * uniforms.k_w) + 
                                    ic_w * (uniforms.k_h * uniforms.k_w) + 
                                    kh * uniforms.k_w + 
                                    kw;
                                    
                        val = val + grad_out[go_idx] * input_or_weight[w_idx];
                    }
                }
            }
        }
    }
    
    let in_idx = b * (uniforms.c_in * uniforms.h_in * uniforms.w_in) + 
                 ic * (uniforms.h_in * uniforms.w_in) + 
                 ih * uniforms.w_in + 
                 iw;
    grad_in[in_idx] = val;
}

// BACKWARD WEIGHT
// input_or_weight is input.
// grad_in is size of weight.
@compute @workgroup_size(256, 1, 1)
fn backward_weight_main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let oc = group_id.x;
    let ic_w = group_id.y;
    let k_linear = group_id.z;
    
    let in_c_per_g = uniforms.c_in / uniforms.groups;
    let out_c_per_g = uniforms.c_out / uniforms.groups;
    
    if (oc >= uniforms.c_out || ic_w >= in_c_per_g || k_linear >= uniforms.k_h * uniforms.k_w) {
        return;
    }
    
    let kw = k_linear % uniforms.k_w;
    let kh = k_linear / uniforms.k_w;
    let tid = local_id.x;
    
    let g = oc / out_c_per_g;
    let ic = g * in_c_per_g + ic_w;
    
    var val: f32 = 0.0;
    let total_iters = uniforms.n * uniforms.h_out * uniforms.w_out;
    
    for (var i = tid; i < total_iters; i = i + 256u) {
        let ow = i % uniforms.w_out;
        var temp_i = i / uniforms.w_out;
        let oh = temp_i % uniforms.h_out;
        let b = temp_i / uniforms.h_out;
        
        let ih_signed = i32(oh * uniforms.stride) - i32(uniforms.padding) + i32(kh * uniforms.dilation);
        let iw_signed = i32(ow * uniforms.stride) - i32(uniforms.padding) + i32(kw * uniforms.dilation);
        
        if (ih_signed >= 0 && ih_signed < i32(uniforms.h_in) && iw_signed >= 0 && iw_signed < i32(uniforms.w_in)) {
            let ih = u32(ih_signed);
            let iw = u32(iw_signed);
            
            let go_idx = b * (uniforms.c_out * uniforms.h_out * uniforms.w_out) + 
                         oc * (uniforms.h_out * uniforms.w_out) + 
                         oh * uniforms.w_out + 
                         ow;
                         
            let in_idx = b * (uniforms.c_in * uniforms.h_in * uniforms.w_in) + 
                         ic * (uniforms.h_in * uniforms.w_in) + 
                         ih * uniforms.w_in + 
                         iw;
                         
            val = val + grad_out[go_idx] * input_or_weight[in_idx];
        }
    }
    
    sdata[tid] = val;
    workgroupBarrier();
    
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            sdata[tid] = sdata[tid] + sdata[tid + s];
        }
        workgroupBarrier();
    }
    
    if (tid == 0u) {
        let w_idx = oc * (in_c_per_g * uniforms.k_h * uniforms.k_w) + 
                    ic_w * (uniforms.k_h * uniforms.k_w) + 
                    kh * uniforms.k_w + 
                    kw;
        grad_in[w_idx] = sdata[0];
    }
}

// BACKWARD BIAS
// input_or_weight is NOT bound (we can just bind a dummy buffer).
// grad_in is size of c_out.
@compute @workgroup_size(256, 1, 1)
fn backward_bias_main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let oc = group_id.x;
    let tid = local_id.x;
    
    if (oc >= uniforms.c_out) {
        return;
    }
    
    var val: f32 = 0.0;
    let total_iters = uniforms.n * uniforms.h_out * uniforms.w_out;
    
    for (var i = tid; i < total_iters; i = i + 256u) {
        let ow = i % uniforms.w_out;
        var temp_i = i / uniforms.w_out;
        let oh = temp_i % uniforms.h_out;
        let b = temp_i / uniforms.h_out;
        
        let go_idx = b * (uniforms.c_out * uniforms.h_out * uniforms.w_out) + 
                     oc * (uniforms.h_out * uniforms.w_out) + 
                     oh * uniforms.w_out + 
                     ow;
        val = val + grad_out[go_idx];
    }
    
    sdata[tid] = val;
    workgroupBarrier();
    
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            sdata[tid] = sdata[tid] + sdata[tid + s];
        }
        workgroupBarrier();
    }
    
    if (tid == 0u) {
        // touch unused binding to prevent it from being optimized out
        let dummy = input_or_weight[0];
        grad_in[oc] = sdata[0];
    }
}
