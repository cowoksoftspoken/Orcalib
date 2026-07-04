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
@group(0) @binding(2) var<storage, read_write> grad_in: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> uniforms: ConvUniforms;



// BACKWARD INPUT
// input_or_weight is weight.
// grad_in is size of input.
@compute @workgroup_size(8, 8, 1)
fn backward_input_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ow = global_id.x;
    let oh = global_id.y;
    let b_oc = global_id.z; // (b * c_out) + oc
    
    if (ow >= uniforms.w_out || oh >= uniforms.h_out || b_oc >= uniforms.n * uniforms.c_out) {
        return;
    }
    
    let b = b_oc / uniforms.c_out;
    let oc = b_oc % uniforms.c_out;
    
    let go_idx = b * (uniforms.c_out * uniforms.h_out * uniforms.w_out) + 
                 oc * (uniforms.h_out * uniforms.w_out) + 
                 oh * uniforms.w_out + 
                 ow;
                 
    let go_val = grad_out[go_idx];
    
    let in_c_per_g = uniforms.c_in / uniforms.groups;
    let out_c_per_g = uniforms.c_out / uniforms.groups;
    let g = oc / out_c_per_g;
    
    for (var ic = g * in_c_per_g; ic < (g + 1u) * in_c_per_g; ic = ic + 1u) {
        let ic_w = ic % in_c_per_g;
        for (var kh = 0u; kh < uniforms.k_h; kh = kh + 1u) {
            for (var kw = 0u; kw < uniforms.k_w; kw = kw + 1u) {
                let ih_signed = i32(oh * uniforms.stride) - i32(uniforms.padding) + i32(kh * uniforms.dilation);
                let iw_signed = i32(ow * uniforms.stride) - i32(uniforms.padding) + i32(kw * uniforms.dilation);
                
                if (ih_signed >= 0 && ih_signed < i32(uniforms.h_in) && iw_signed >= 0 && iw_signed < i32(uniforms.w_in)) {
                    let ih = u32(ih_signed);
                    let iw = u32(iw_signed);
                    
                    let in_idx = b * (uniforms.c_in * uniforms.h_in * uniforms.w_in) + 
                                 ic * (uniforms.h_in * uniforms.w_in) + 
                                 ih * uniforms.w_in + 
                                 iw;
                                 
                    let w_idx = oc * (in_c_per_g * uniforms.k_h * uniforms.k_w) + 
                                ic_w * (uniforms.k_h * uniforms.k_w) + 
                                kh * uniforms.k_w + 
                                kw;
                                
                    let val = go_val * input_or_weight[w_idx];
                    var old_u32 = atomicLoad(&grad_in[in_idx]);
                    loop {
                        let old_f32 = bitcast<f32>(old_u32);
                        let new_f32 = old_f32 + val;
                        let new_u32 = bitcast<u32>(new_f32);
                        let res = atomicCompareExchangeWeak(&grad_in[in_idx], old_u32, new_u32);
                        if (res.exchanged) {
                            break;
                        }
                        old_u32 = res.old_value;
                    }
                }
            }
        }
    }
}

// BACKWARD WEIGHT
// input_or_weight is input.
// grad_in is size of weight.
@compute @workgroup_size(8, 8, 1)
fn backward_weight_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ow = global_id.x;
    let oh = global_id.y;
    let b_oc = global_id.z; // (b * c_out) + oc
    
    if (ow >= uniforms.w_out || oh >= uniforms.h_out || b_oc >= uniforms.n * uniforms.c_out) {
        return;
    }
    
    let b = b_oc / uniforms.c_out;
    let oc = b_oc % uniforms.c_out;
    
    let go_idx = b * (uniforms.c_out * uniforms.h_out * uniforms.w_out) + 
                 oc * (uniforms.h_out * uniforms.w_out) + 
                 oh * uniforms.w_out + 
                 ow;
                 
    let go_val = grad_out[go_idx];
    
    for (var ic = 0u; ic < uniforms.c_in; ic = ic + 1u) {
        let in_c_per_g = uniforms.c_in / uniforms.groups;
        let out_c_per_g = uniforms.c_out / uniforms.groups;
        let g = oc / out_c_per_g;
        
        if (ic >= g * in_c_per_g && ic < (g + 1u) * in_c_per_g) {
            let ic_w = ic % in_c_per_g;
            for (var kh = 0u; kh < uniforms.k_h; kh = kh + 1u) {
                for (var kw = 0u; kw < uniforms.k_w; kw = kw + 1u) {
                    let ih_signed = i32(oh * uniforms.stride) - i32(uniforms.padding) + i32(kh * uniforms.dilation);
                    let iw_signed = i32(ow * uniforms.stride) - i32(uniforms.padding) + i32(kw * uniforms.dilation);
                    
                    if (ih_signed >= 0 && ih_signed < i32(uniforms.h_in) && iw_signed >= 0 && iw_signed < i32(uniforms.w_in)) {
                        let ih = u32(ih_signed);
                        let iw = u32(iw_signed);
                        
                        let in_idx = b * (uniforms.c_in * uniforms.h_in * uniforms.w_in) + 
                                     ic * (uniforms.h_in * uniforms.w_in) + 
                                     ih * uniforms.w_in + 
                                     iw;
                                     
                        let w_idx = oc * (in_c_per_g * uniforms.k_h * uniforms.k_w) + 
                                    ic_w * (uniforms.k_h * uniforms.k_w) + 
                                    kh * uniforms.k_w + 
                                    kw;
                                
                    let val = go_val * input_or_weight[in_idx];
                    var old_u32 = atomicLoad(&grad_in[w_idx]);
                    loop {
                        let old_f32 = bitcast<f32>(old_u32);
                        let new_f32 = old_f32 + val;
                        let new_u32 = bitcast<u32>(new_f32);
                        let res = atomicCompareExchangeWeak(&grad_in[w_idx], old_u32, new_u32);
                        if (res.exchanged) {
                            break;
                        }
                        old_u32 = res.old_value;
                    }
                }
            }
        }
    }
}
}
// BACKWARD BIAS
// input_or_weight is NOT bound (we can just bind a dummy buffer).
// grad_in is size of c_out.
@compute @workgroup_size(8, 8, 1)
fn backward_bias_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ow = global_id.x;
    let oh = global_id.y;
    let b_oc = global_id.z; // (b * c_out) + oc
    
    if (ow >= uniforms.w_out || oh >= uniforms.h_out || b_oc >= uniforms.n * uniforms.c_out) {
        return;
    }
    
    let b = b_oc / uniforms.c_out;
    let oc = b_oc % uniforms.c_out;
    
    let go_idx = b * (uniforms.c_out * uniforms.h_out * uniforms.w_out) + 
                 oc * (uniforms.h_out * uniforms.w_out) + 
                 oh * uniforms.w_out + 
                 ow;
                 
    let val = grad_out[go_idx];
    var old_u32 = atomicLoad(&grad_in[oc]);
    loop {
        let old_f32 = bitcast<f32>(old_u32);
        let new_f32 = old_f32 + val;
        let new_u32 = bitcast<u32>(new_f32);
        let res = atomicCompareExchangeWeak(&grad_in[oc], old_u32, new_u32);
        if (res.exchanged) {
            break;
        }
        old_u32 = res.old_value;
    }
}
