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

@group(0) @binding(0) var<storage, read> in_data: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_data: array<f32>;
@group(0) @binding(4) var<uniform> uniforms: ConvUniforms;

@compute @workgroup_size(8, 8, 1)
fn forward_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ow = global_id.x;
    let oh = global_id.y;
    let b_oc = global_id.z; // z is (b * c_out) + oc
    
    if (ow >= uniforms.w_out || oh >= uniforms.h_out || b_oc >= uniforms.n * uniforms.c_out) {
        return;
    }
    
    let b = b_oc / uniforms.c_out;
    let oc = b_oc % uniforms.c_out;
    
    let in_c_per_g = uniforms.c_in / uniforms.groups;
    let out_c_per_g = uniforms.c_out / uniforms.groups;
    let g = oc / out_c_per_g;
    
    var sum = 0.0;
    
    for (var ic = g * in_c_per_g; ic < (g + 1u) * in_c_per_g; ic = ic + 1u) {
        let ic_w = ic % in_c_per_g;
        for (var kh = 0u; kh < uniforms.k_h; kh = kh + 1u) {
            for (var kw = 0u; kw < uniforms.k_w; kw = kw + 1u) {
                // Calculate input coordinates
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
                                
                    sum = sum + in_data[in_idx] * weight[w_idx];
                }
            }
        }
    }
    
    if (uniforms.has_bias == 1u) {
        sum = sum + bias[oc];
    }
    
    let out_idx = b * (uniforms.c_out * uniforms.h_out * uniforms.w_out) + 
                  oc * (uniforms.h_out * uniforms.w_out) + 
                  oh * uniforms.w_out + 
                  ow;
                  
    out_data[out_idx] = sum;
}
