#![allow(clippy::needless_range_loop)]
use orca_core::{Device, DType, Shape, Result, OrcaError};
use orca_tensor::{Backend, Storage};
use crate::storage::CpuByteStorage;
use crate::math::CpuNumeric;
use crate::{dispatch_dtype, dispatch_float};

/// The CPU execution backend.
#[derive(Debug, Clone, Default)]
pub struct CpuBackend;

impl CpuBackend {
    pub fn pad_shape_left(shape: &[usize], target_rank: usize) -> Vec<usize> {
        let mut padded = vec![1; target_rank];
        if shape.len() <= target_rank {
            let offset = target_rank - shape.len();
            for (i, &dim) in shape.iter().enumerate() {
                padded[offset + i] = dim;
            }
        }
        padded
    }

    pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![0; shape.len()];
        let mut current = 1;
        for i in (0..shape.len()).rev() {
            strides[i] = current;
            current *= shape[i];
        }
        strides
    }

    fn linear_to_multi(mut index: usize, shape: &[usize]) -> Vec<usize> {
        let mut multi = vec![0; shape.len()];
        for i in (0..shape.len()).rev() {
            multi[i] = index % shape[i];
            index /= shape[i];
        }
        multi
    }

    fn multi_to_linear(multi: &[usize], strides: &[usize]) -> usize {
        multi.iter().zip(strides.iter()).map(|(m, s)| m * s).sum()
    }

    pub fn from_slice<T: Copy>(&self, shape: &Shape, data: &[T], _dtype: DType) -> Result<CpuByteStorage> {
        let element_size = std::mem::size_of::<T>();
        let num_elements = shape.num_elements();
        let total_bytes = num_elements * element_size;
        let mut storage = CpuByteStorage::new(total_bytes, num_elements, element_size);
        storage.as_mut_slice::<T>().copy_from_slice(data);
        Ok(storage)
    }
}

impl Backend for CpuBackend {
    type Storage = CpuByteStorage;

    fn device(&self) -> Device {
        Device::Cpu
    }

    fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let element_size = match dtype {
            DType::F32 | DType::I32 => 4,
            DType::F64 | DType::I64 => 8,
            DType::F16 | DType::BF16 => 2,
            DType::U8 | DType::Bool => 1,
        };

        let num_elements = shape.num_elements();
        let total_bytes = num_elements * element_size;
        
        Ok(CpuByteStorage::new(total_bytes, num_elements, element_size))
    }

    fn from_f32_slice(&self, shape: &Shape, data: &[f32]) -> Result<Self::Storage> {
        self.from_slice::<f32>(shape, data, DType::F32)
    }

    fn to_f32_vec(&self, storage: &Self::Storage) -> Result<Vec<f32>> {
        // Technically this should be aware of dtype and convert!
        // But since to_f32_vec doesn't take dtype, we assume the storage contains f32.
        let bytes = storage.as_bytes();
        let mut vec = vec![0.0f32; storage.len()];
        
        let dest_bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(vec.as_mut_ptr() as *mut u8, bytes.len())
        };
        dest_bytes.copy_from_slice(bytes);
        
        Ok(vec)
    }

    fn add(&self, lhs: &Self::Storage, rhs: &Self::Storage, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        dispatch_dtype!(dtype, T, {
            let lhs_slice = lhs.as_slice::<T>();
            let rhs_slice = rhs.as_slice::<T>();
            let mut result = Vec::with_capacity(lhs_slice.len());
            for i in 0..lhs_slice.len() {
                result.push(lhs_slice[i].add(rhs_slice[i]));
            }
            self.from_slice::<T>(shape, &result, dtype)
        })
    }

    fn matmul(&self, lhs: &Self::Storage, rhs: &Self::Storage, lhs_shape: &Shape, rhs_shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        dispatch_dtype!(dtype, T, {
            let m = lhs_shape[0];
            let k = lhs_shape[1];
            let n = rhs_shape[1];
            
            let lhs_slice = lhs.as_slice::<T>();
            let rhs_slice = rhs.as_slice::<T>();
            
            let mut result = vec![T::zero(); m * n];
            
            for i in 0..m {
                for j in 0..n {
                    let mut sum = T::zero();
                    for p in 0..k {
                        sum = sum.add(lhs_slice[i * k + p].mul(rhs_slice[p * n + j]));
                    }
                    result[i * n + j] = sum;
                }
            }
            
            let out_shape = Shape::new(vec![m, n]);
            self.from_slice::<T>(&out_shape, &result, dtype)
        })
    }

    fn transpose(&self, storage: &Self::Storage, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        dispatch_dtype!(dtype, T, {
            let rows = shape[0];
            let cols = shape[1];
            
            let in_slice = storage.as_slice::<T>();
            let mut out_slice = vec![T::zero(); rows * cols];
            
            for i in 0..rows {
                for j in 0..cols {
                    out_slice[j * rows + i] = in_slice[i * cols + j];
                }
            }
            
            let out_shape = Shape::new(vec![cols, rows]);
            self.from_slice::<T>(&out_shape, &out_slice, dtype)
        })
    }

    fn sub(&self, lhs: &Self::Storage, rhs: &Self::Storage, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        dispatch_dtype!(dtype, T, {
            let lhs_slice = lhs.as_slice::<T>();
            let rhs_slice = rhs.as_slice::<T>();
            
            let mut result = Vec::with_capacity(lhs_slice.len());
            for i in 0..lhs_slice.len() {
                result.push(lhs_slice[i].sub(rhs_slice[i]));
            }
            
            self.from_slice::<T>(shape, &result, dtype)
        })
    }

    fn mul_scalar(&self, storage: &Self::Storage, scalar: f32, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        dispatch_dtype!(dtype, T, {
            let in_slice = storage.as_slice::<T>();
            let scalar_t = T::from_f32(scalar);
            
            let mut result = Vec::with_capacity(in_slice.len());
            for i in 0..in_slice.len() {
                result.push(in_slice[i].mul(scalar_t));
            }
            
            self.from_slice::<T>(shape, &result, dtype)
        })
    }

    fn mul(&self, lhs: &Self::Storage, rhs: &Self::Storage, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        if dtype != DType::F32 {
            return Err(OrcaError::UnsupportedDType { op: "mul (cpu)", dtype });
        }
        
        let lhs_f32 = unsafe { std::slice::from_raw_parts(lhs.as_bytes().as_ptr() as *const f32, lhs.len()) };
        let rhs_f32 = unsafe { std::slice::from_raw_parts(rhs.as_bytes().as_ptr() as *const f32, rhs.len()) };
        
        let mut result = Vec::with_capacity(lhs.len());
        for i in 0..lhs.len() {
            result.push(lhs_f32[i] * rhs_f32[i]);
        }
        
        self.from_f32_slice(shape, &result)
    }

    fn relu(&self, storage: &Self::Storage, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        dispatch_float!(dtype, T, {
            let in_slice = storage.as_slice::<T>();
            let mut result = Vec::with_capacity(in_slice.len());
            for i in 0..in_slice.len() {
                let val = in_slice[i];
                result.push(if val.to_f32() > 0.0 { val } else { T::zero() });
            }
            self.from_slice::<T>(shape, &result, dtype)
        })
    }

    fn sigmoid(&self, storage: &Self::Storage, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        dispatch_float!(dtype, T, {
            let in_slice = storage.as_slice::<T>();
            let mut result = Vec::with_capacity(in_slice.len());
            for i in 0..in_slice.len() {
                let val = in_slice[i];
                let sig_f32 = 1.0 / (1.0 + (-val.to_f32()).exp());
                result.push(T::from_f32(sig_f32));
            }
            self.from_slice::<T>(shape, &result, dtype)
        })
    }

    fn relu_backward(&self, grad_out: &Self::Storage, in_primal: &Self::Storage, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        dispatch_float!(dtype, T, {
            let grad_slice = grad_out.as_slice::<T>();
            let in_slice = in_primal.as_slice::<T>();
            let mut result = Vec::with_capacity(grad_slice.len());
            for i in 0..grad_slice.len() {
                result.push(if in_slice[i].to_f32() > 0.0 { grad_slice[i] } else { T::zero() });
            }
            self.from_slice::<T>(shape, &result, dtype)
        })
    }

    fn sigmoid_backward(&self, grad_out: &Self::Storage, out_primal: &Self::Storage, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        dispatch_float!(dtype, T, {
            let grad_slice = grad_out.as_slice::<T>();
            let out_slice = out_primal.as_slice::<T>();
            let mut result = Vec::with_capacity(grad_slice.len());
            for i in 0..grad_slice.len() {
                let y = out_slice[i];
                let val = grad_slice[i].mul(y).mul(T::one().sub(y));
                result.push(val);
            }
            self.from_slice::<T>(shape, &result, dtype)
        })
    }

    fn expand(&self, storage: &Self::Storage, in_shape: &Shape, out_shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        dispatch_dtype!(dtype, T, {
            let in_slice = storage.as_slice::<T>();
            let mut result = vec![T::zero(); out_shape.num_elements()];
            
            let padded_in = Self::pad_shape_left(&in_shape.0, out_shape.rank());
            let in_strides = Self::compute_strides(&padded_in);
            
            for i in 0..result.len() {
                let multi_out = Self::linear_to_multi(i, &out_shape.0);
                let mut multi_in = vec![0; out_shape.rank()];
                for d in 0..out_shape.rank() {
                    multi_in[d] = if padded_in[d] == 1 { 0 } else { multi_out[d] };
                }
                let in_idx = Self::multi_to_linear(&multi_in, &in_strides);
                result[i] = in_slice[in_idx];
            }
            self.from_slice::<T>(out_shape, &result, dtype)
        })
    }

    fn sum_to_shape(&self, storage: &Self::Storage, in_shape: &Shape, out_shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        dispatch_dtype!(dtype, T, {
            let in_slice = storage.as_slice::<T>();
            let mut result = vec![T::zero(); out_shape.num_elements()];
            
            let padded_out = Self::pad_shape_left(&out_shape.0, in_shape.rank());
            let out_strides = Self::compute_strides(&padded_out);
            
            for i in 0..in_slice.len() {
                let multi_in = Self::linear_to_multi(i, &in_shape.0);
                let mut multi_out = vec![0; in_shape.rank()];
                for d in 0..in_shape.rank() {
                    multi_out[d] = if padded_out[d] == 1 { 0 } else { multi_in[d] };
                }
                let out_idx = Self::multi_to_linear(&multi_out, &out_strides);
                result[out_idx] = result[out_idx].add(in_slice[i]);
            }
            self.from_slice::<T>(out_shape, &result, dtype)
        })
    }

    fn reshape(&self, storage: &Self::Storage, in_shape: &Shape, out_shape: &Shape, _dtype: DType) -> Result<Self::Storage> {
        if in_shape.num_elements() != out_shape.num_elements() {
            return Err(OrcaError::ShapeMismatch {
                op: "reshape",
                expected: format!("{} elements", in_shape.num_elements()),
                got: format!("{} elements", out_shape.num_elements()),
            });
        }
        Ok(storage.clone()) // Since CPU storage is a flat contiguous Vec<u8> right now
    }

    fn exp(&self, storage: &Self::Storage, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        dispatch_float!(dtype, T, {
            let in_slice = storage.as_slice::<T>();
            let mut result = Vec::with_capacity(in_slice.len());
            for i in 0..in_slice.len() {
                result.push(T::from_f32(in_slice[i].to_f32().exp()));
            }
            self.from_slice::<T>(shape, &result, dtype)
        })
    }

    fn log(&self, storage: &Self::Storage, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        dispatch_float!(dtype, T, {
            let in_slice = storage.as_slice::<T>();
            let mut result = Vec::with_capacity(in_slice.len());
            for i in 0..in_slice.len() {
                // Add a small epsilon to prevent log(0) which becomes -inf and creates NaNs.
                result.push(T::from_f32((in_slice[i].to_f32().max(1e-7)).ln()));
            }
            self.from_slice::<T>(shape, &result, dtype)
        })
    }

    fn exp_backward(&self, grad_out: &Self::Storage, out_primal: &Self::Storage, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        dispatch_float!(dtype, T, {
            let grad_slice = grad_out.as_slice::<T>();
            let out_slice = out_primal.as_slice::<T>();
            let mut result = Vec::with_capacity(grad_slice.len());
            for i in 0..grad_slice.len() {
                result.push(grad_slice[i].mul(out_slice[i]));
            }
            self.from_slice::<T>(shape, &result, dtype)
        })
    }

    fn log_backward(&self, grad_out: &Self::Storage, in_primal: &Self::Storage, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        dispatch_float!(dtype, T, {
            let grad_slice = grad_out.as_slice::<T>();
            let in_slice = in_primal.as_slice::<T>();
            let mut result = Vec::with_capacity(grad_slice.len());
            for i in 0..grad_slice.len() {
                result.push(grad_slice[i].div(in_slice[i]));
            }
            self.from_slice::<T>(shape, &result, dtype)
        })
    }

    fn div(&self, lhs: &Self::Storage, rhs: &Self::Storage, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        dispatch_float!(dtype, T, {
            let lhs_slice = lhs.as_slice::<T>();
            let rhs_slice = rhs.as_slice::<T>();
            let mut result = Vec::with_capacity(lhs_slice.len());
            for i in 0..lhs_slice.len() {
                result.push(lhs_slice[i].div(rhs_slice[i]));
            }
            self.from_slice::<T>(shape, &result, dtype)
        })
    }

    fn sqrt(&self, storage: &Self::Storage, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        dispatch_float!(dtype, T, {
            let in_slice = storage.as_slice::<T>();
            let mut result = Vec::with_capacity(in_slice.len());
            for i in 0..in_slice.len() {
                result.push(T::from_f32(in_slice[i].to_f32().sqrt()));
            }
            self.from_slice::<T>(shape, &result, dtype)
        })
    }

    fn div_backward_lhs(&self, grad_out: &Self::Storage, rhs_primal: &Self::Storage, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        dispatch_float!(dtype, T, {
            let grad_slice = grad_out.as_slice::<T>();
            let rhs_slice = rhs_primal.as_slice::<T>();
            let mut result = Vec::with_capacity(grad_slice.len());
            for i in 0..grad_slice.len() {
                result.push(grad_slice[i].div(rhs_slice[i]));
            }
            self.from_slice::<T>(shape, &result, dtype)
        })
    }

    fn div_backward_rhs(&self, grad_out: &Self::Storage, lhs_primal: &Self::Storage, rhs_primal: &Self::Storage, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        dispatch_float!(dtype, T, {
            let grad_slice = grad_out.as_slice::<T>();
            let lhs_slice = lhs_primal.as_slice::<T>();
            let rhs_slice = rhs_primal.as_slice::<T>();
            let mut result = Vec::with_capacity(grad_slice.len());
            for i in 0..grad_slice.len() {
                let grad_f32 = grad_slice[i].to_f32();
                let l_f32 = lhs_slice[i].to_f32();
                let r_f32 = rhs_slice[i].to_f32();
                result.push(T::from_f32(grad_f32 * (-l_f32 / (r_f32 * r_f32))));
            }
            self.from_slice::<T>(shape, &result, dtype)
        })
    }

    fn sqrt_backward(&self, grad_out: &Self::Storage, out_primal: &Self::Storage, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        dispatch_float!(dtype, T, {
            let grad_slice = grad_out.as_slice::<T>();
            let out_slice = out_primal.as_slice::<T>();
            let mut result = Vec::with_capacity(grad_slice.len());
            for i in 0..grad_slice.len() {
                let grad_f32 = grad_slice[i].to_f32();
                let out_f32 = out_slice[i].to_f32();
                result.push(T::from_f32(grad_f32 / (2.0 * out_f32)));
            }
            self.from_slice::<T>(shape, &result, dtype)
        })
    }

    fn accumulate_grad(&self, lhs: &Self::Storage, rhs: &Self::Storage) -> Result<Self::Storage> {
        let lhs_f32 = unsafe { std::slice::from_raw_parts(lhs.as_bytes().as_ptr() as *const f32, lhs.len()) };
        let rhs_f32 = unsafe { std::slice::from_raw_parts(rhs.as_bytes().as_ptr() as *const f32, rhs.len()) };
        
        let mut result = Vec::with_capacity(lhs.len());
        for i in 0..lhs.len() {
            result.push(lhs_f32[i] + rhs_f32[i]);
        }
        
        let dummy_shape = Shape::new(vec![lhs.len()]);
        self.from_slice::<f32>(&dummy_shape, &result, DType::F32)
    }

    fn conv2d(&self, input: &Self::Storage, weight: &Self::Storage, bias: Option<&Self::Storage>,
              in_shape: &Shape, weight_shape: &Shape,
              padding: usize, stride: usize, dilation: usize, groups: usize, dtype: DType) -> Result<Self::Storage> {
        if dtype != DType::F32 { return Err(OrcaError::UnsupportedDType { op: "conv2d", dtype }); }
        
        let n = in_shape[0];
        let c_in = in_shape[1];
        let h_in = in_shape[2];
        let w_in = in_shape[3];

        let c_out = weight_shape[0];
        let k_h = weight_shape[2];
        let k_w = weight_shape[3];

        let h_out = (h_in + 2 * padding - dilation * (k_h - 1) - 1) / stride + 1;
        let w_out = (w_in + 2 * padding - dilation * (k_w - 1) - 1) / stride + 1;

        let in_c_per_g = c_in / groups;
        let out_c_per_g = c_out / groups;

        let in_f32 = unsafe { std::slice::from_raw_parts(input.as_bytes().as_ptr() as *const f32, input.len()) };
        let w_f32 = unsafe { std::slice::from_raw_parts(weight.as_bytes().as_ptr() as *const f32, weight.len()) };
        let b_f32 = bias.map(|b| unsafe { std::slice::from_raw_parts(b.as_bytes().as_ptr() as *const f32, b.len()) });

        let mut result = vec![0.0; n * c_out * h_out * w_out];
        
        for b in 0..n {
            for oc in 0..c_out {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut sum = 0.0;
                        let g = oc / out_c_per_g;
                        for ic in (g * in_c_per_g)..((g + 1) * in_c_per_g) {
                            let ic_w = ic % in_c_per_g;
                            for kh in 0..k_h {
                                for kw in 0..k_w {
                                    let ih = (oh * stride) as isize - padding as isize + (kh * dilation) as isize;
                                    let iw = (ow * stride) as isize - padding as isize + (kw * dilation) as isize;
                                    
                                    if ih >= 0 && ih < h_in as isize && iw >= 0 && iw < w_in as isize {
                                        let in_idx = b * (c_in * h_in * w_in) + ic * (h_in * w_in) + (ih as usize) * w_in + (iw as usize);
                                        let w_idx = oc * (in_c_per_g * k_h * k_w) + ic_w * (k_h * k_w) + kh * k_w + kw;
                                        sum += in_f32[in_idx] * w_f32[w_idx];
                                    }
                                }
                            }
                        }
                        if let Some(bias_slice) = b_f32 {
                            sum += bias_slice[oc];
                        }
                        let out_idx = b * (c_out * h_out * w_out) + oc * (h_out * w_out) + oh * w_out + ow;
                        result[out_idx] = sum;
                    }
                }
            }
        }
        
        let out_shape = Shape::new(vec![n, c_out, h_out, w_out]);
        self.from_f32_slice(&out_shape, &result)
    }

    fn conv2d_backward_input(&self, grad_out: &Self::Storage, weight: &Self::Storage,
                             in_shape: &Shape, weight_shape: &Shape,
                             padding: usize, stride: usize, dilation: usize, groups: usize, _dtype: DType) -> Result<Self::Storage> {
        let n = in_shape[0];
        let c_in = in_shape[1];
        let h_in = in_shape[2];
        let w_in = in_shape[3];

        let c_out = weight_shape[0];
        let k_h = weight_shape[2];
        let k_w = weight_shape[3];

        let h_out = (h_in + 2 * padding - dilation * (k_h - 1) - 1) / stride + 1;
        let w_out = (w_in + 2 * padding - dilation * (k_w - 1) - 1) / stride + 1;

        let in_c_per_g = c_in / groups;
        let out_c_per_g = c_out / groups;

        let go_f32 = unsafe { std::slice::from_raw_parts(grad_out.as_bytes().as_ptr() as *const f32, grad_out.len()) };
        let w_f32 = unsafe { std::slice::from_raw_parts(weight.as_bytes().as_ptr() as *const f32, weight.len()) };

        let mut grad_in = vec![0.0; n * c_in * h_in * w_in];

        for b in 0..n {
            for oc in 0..c_out {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let go_idx = b * (c_out * h_out * w_out) + oc * (h_out * w_out) + oh * w_out + ow;
                        let g_val = go_f32[go_idx];
                        let g = oc / out_c_per_g;
                        for ic in (g * in_c_per_g)..((g + 1) * in_c_per_g) {
                            let ic_w = ic % in_c_per_g;
                            for kh in 0..k_h {
                                for kw in 0..k_w {
                                    let ih = (oh * stride) as isize - padding as isize + (kh * dilation) as isize;
                                    let iw = (ow * stride) as isize - padding as isize + (kw * dilation) as isize;
                                    
                                    if ih >= 0 && ih < h_in as isize && iw >= 0 && iw < w_in as isize {
                                        let in_idx = b * (c_in * h_in * w_in) + ic * (h_in * w_in) + (ih as usize) * w_in + (iw as usize);
                                        let w_idx = oc * (in_c_per_g * k_h * k_w) + ic_w * (k_h * k_w) + kh * k_w + kw;
                                        grad_in[in_idx] += g_val * w_f32[w_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        self.from_f32_slice(in_shape, &grad_in)
    }

    fn conv2d_backward_weight(&self, grad_out: &Self::Storage, input: &Self::Storage,
                              in_shape: &Shape, weight_shape: &Shape,
                              padding: usize, stride: usize, dilation: usize, groups: usize, _dtype: DType) -> Result<Self::Storage> {
        let n = in_shape[0];
        let c_in = in_shape[1];
        let h_in = in_shape[2];
        let w_in = in_shape[3];

        let c_out = weight_shape[0];
        let k_h = weight_shape[2];
        let k_w = weight_shape[3];

        let h_out = (h_in + 2 * padding - dilation * (k_h - 1) - 1) / stride + 1;
        let w_out = (w_in + 2 * padding - dilation * (k_w - 1) - 1) / stride + 1;

        let in_c_per_g = c_in / groups;
        let out_c_per_g = c_out / groups;

        let go_f32 = unsafe { std::slice::from_raw_parts(grad_out.as_bytes().as_ptr() as *const f32, grad_out.len()) };
        let in_f32 = unsafe { std::slice::from_raw_parts(input.as_bytes().as_ptr() as *const f32, input.len()) };

        let mut grad_w = vec![0.0; c_out * c_in * k_h * k_w];

        for b in 0..n {
            for oc in 0..c_out {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let go_idx = b * (c_out * h_out * w_out) + oc * (h_out * w_out) + oh * w_out + ow;
                        let g_val = go_f32[go_idx];
                        
                        let g = oc / out_c_per_g;
                        for ic in (g * in_c_per_g)..((g + 1) * in_c_per_g) {
                            let ic_w = ic % in_c_per_g;
                            for kh in 0..k_h {
                                for kw in 0..k_w {
                                    let ih = (oh * stride) as isize - padding as isize + (kh * dilation) as isize;
                                    let iw = (ow * stride) as isize - padding as isize + (kw * dilation) as isize;
                                    
                                    if ih >= 0 && ih < h_in as isize && iw >= 0 && iw < w_in as isize {
                                        let in_idx = b * (c_in * h_in * w_in) + ic * (h_in * w_in) + (ih as usize) * w_in + (iw as usize);
                                        let w_idx = oc * (in_c_per_g * k_h * k_w) + ic_w * (k_h * k_w) + kh * k_w + kw;
                                        grad_w[w_idx] += g_val * in_f32[in_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        self.from_f32_slice(weight_shape, &grad_w)
    }

    fn conv2d_backward_bias(&self, grad_out: &Self::Storage, out_shape: &Shape, _dtype: DType) -> Result<Self::Storage> {
        let n = out_shape[0];
        let c_out = out_shape[1];
        let h_out = out_shape[2];
        let w_out = out_shape[3];

        let go_f32 = unsafe { std::slice::from_raw_parts(grad_out.as_bytes().as_ptr() as *const f32, grad_out.len()) };
        let mut grad_b = vec![0.0; c_out];

        for b in 0..n {
            for oc in 0..c_out {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let go_idx = b * (c_out * h_out * w_out) + oc * (h_out * w_out) + oh * w_out + ow;
                        grad_b[oc] += go_f32[go_idx];
                    }
                }
            }
        }
        self.from_f32_slice(&Shape::new(vec![c_out]), &grad_b)
    }
}
