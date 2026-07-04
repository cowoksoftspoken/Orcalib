use std::ops::Add;
use crate::backend::Backend;
use orca_core::{Device, DType, Shape, OrcaError, Result};

/// The core multidimensional array representation in Orca.
/// 
/// A Tensor is parameterized over a generic `Backend` trait, enabling
/// zero-cost dispatch to different execution environments (CPU, CUDA, etc.)
#[derive(Debug, Clone)]
pub struct Tensor<B: Backend> {
    /// The underlying raw data storage.
    storage: B::Storage,
    /// The dimensions of the tensor.
    shape: Shape,
    /// Memory strides (used for view semantics).
    strides: Vec<usize>,
    /// The data type of the tensor elements.
    dtype: DType,
    /// The backend instance associated with this tensor.
    backend: B,
}

impl<B: Backend> Tensor<B> {
    pub fn zeros(backend: B, shape: impl Into<Shape>, dtype: DType) -> Result<Self> {
        let shape = shape.into();
        // Default contiguous strides
        let strides = Self::compute_contiguous_strides(&shape);
        let storage = backend.zeros(&shape, dtype)?;

        Ok(Self {
            storage,
            shape,
            strides,
            dtype,
            backend,
        })
    }

    /// Creates a new tensor filled with random uniform values between low and high.
    pub fn rand_uniform(backend: B, shape: impl Into<Shape>, low: f32, high: f32, _dtype: DType) -> Result<Self> {
        use rand::Rng;
        let shape = shape.into();
        let num_elements = shape.num_elements();
        
        let mut rng = rand::thread_rng();
        let mut data = Vec::with_capacity(num_elements);
        for _ in 0..num_elements {
            data.push(rng.gen_range(low..high));
        }
        
        Self::from_f32_slice(backend, &data, shape.clone())
    }

    /// Creates a dropout mask tensor scaled by 1/(1-p).
    pub fn rand_dropout_mask(backend: B, shape: impl Into<Shape>, p: f32, _dtype: DType) -> Result<Self> {
        use rand::Rng;
        let shape = shape.into();
        let num_elements = shape.num_elements();
        
        let mut rng = rand::thread_rng();
        let mut data = Vec::with_capacity(num_elements);
        let scale = 1.0 / (1.0 - p);
        
        for _ in 0..num_elements {
            let val = if rng.gen::<f32>() > p { scale } else { 0.0 };
            data.push(val);
        }
        
        Self::from_f32_slice(backend, &data, shape.clone())
    }

    /// Returns the shape of this tensor.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns the strides of this tensor.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Returns the data type of the tensor.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the device where this tensor is stored.
    pub fn device(&self) -> Device {
        self.backend.device()
    }
    
    /// Gets a reference to the backend.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Gets a reference to the underlying storage.
    pub fn storage(&self) -> &B::Storage {
        &self.storage
    }

    /// Gets a mutable reference to the underlying storage.
    pub fn storage_mut(&mut self) -> &mut B::Storage {
        &mut self.storage
    }

    /// Returns the number of dimensions.
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Creates a tensor from raw components.
    pub fn from_raw_parts(
        backend: B,
        storage: B::Storage,
        shape: Shape,
        strides: Vec<usize>,
        dtype: DType,
    ) -> Self {
        Self {
            backend,
            storage,
            shape,
            strides,
            dtype,
        }
    }

    /// Creates a tensor from an f32 slice.
    pub fn from_f32_slice(backend: B, data: &[f32], shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        let strides = Self::compute_contiguous_strides(&shape);
        if data.len() != shape.num_elements() {
            return Err(OrcaError::ShapeMismatch {
                op: "from_f32_slice",
                expected: format!("{} elements", shape.num_elements()),
                got: format!("{} elements", data.len()),
            });
        }
        let storage = backend.from_f32_slice(&shape, data)?;
        
        Ok(Self {
            storage,
            shape,
            strides,
            dtype: DType::F32,
            backend,
        })
    }

    /// Extracts the tensor data as an f32 vec.
    pub fn to_f32_vec(&self) -> Result<Vec<f32>> {
        if self.dtype != DType::F32 {
            return Err(OrcaError::UnsupportedDType {
                op: "to_f32_vec",
                dtype: self.dtype,
            });
        }
        self.backend.to_f32_vec(&self.storage)
    }

    /// Helper to compute contiguous strides for a given shape.
    fn compute_contiguous_strides(shape: &Shape) -> Vec<usize> {
        if shape.is_empty() {
            return vec![];
        }
        let mut strides = vec![1; shape.rank()];
        for i in (0..shape.rank() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Matrix multiplication.
    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        if self.shape.rank() != 2 || rhs.shape.rank() != 2 {
            return Err(OrcaError::ShapeMismatch {
                op: "matmul",
                expected: "2D tensors".into(),
                got: format!("{}D and {}D", self.shape.rank(), rhs.shape.rank()),
            });
        }
        if self.shape[1] != rhs.shape[0] {
            return Err(OrcaError::ShapeMismatch {
                op: "matmul",
                expected: format!("Inner dimensions must match: {} == {}", self.shape[1], rhs.shape[0]),
                got: format!("{} != {}", self.shape[1], rhs.shape[0]),
            });
        }
        if self.backend.device() != rhs.backend.device() {
            return Err(OrcaError::DeviceMismatch(self.backend.device(), rhs.backend.device()));
        }
        if self.dtype != rhs.dtype {
            return Err(OrcaError::InternalError("DTypes must match for matmul".into()));
        }

        let out_shape = Shape::new(vec![self.shape[0], rhs.shape[1]]);
        let storage = self.backend.matmul(&self.storage, &rhs.storage, &self.shape, &rhs.shape, self.dtype)?;
        let strides = Self::compute_contiguous_strides(&out_shape);
        
        Ok(Tensor {
            storage,
            shape: out_shape,
            strides,
            dtype: self.dtype,
            backend: self.backend.clone(),
        })
    }

    /// Transposes a 2D tensor.
    pub fn transpose(&self) -> Result<Self> {
        if self.shape.rank() != 2 {
            return Err(OrcaError::ShapeMismatch {
                op: "transpose",
                expected: "2D tensor".into(),
                got: format!("{}D", self.shape.rank()),
            });
        }
        
        let out_shape = Shape::new(vec![self.shape[1], self.shape[0]]);
        let storage = self.backend.transpose(&self.storage, &self.shape, self.dtype)?;
        let strides = Self::compute_contiguous_strides(&out_shape);

        Ok(Tensor {
            storage,
            shape: out_shape,
            strides,
            dtype: self.dtype,
            backend: self.backend.clone(),
        })
    }

    /// Multiply by a scalar.
    pub fn mul_scalar(&self, scalar: f32) -> Result<Self> {
        let storage = self.backend.mul_scalar(&self.storage, scalar, &self.shape, self.dtype)?;
        
        Ok(Tensor {
            storage,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            dtype: self.dtype,
            backend: self.backend.clone(),
        })
    }

    /// Applies ReLU activation element-wise.
    pub fn relu(&self) -> Result<Self> {
        let storage = self.backend.relu(&self.storage, &self.shape, self.dtype)?;
        Ok(Self {
            storage,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            dtype: self.dtype,
            backend: self.backend.clone(),
        })
    }

    pub fn sigmoid(&self) -> Result<Self> {
        let storage = self.backend.sigmoid(&self.storage, &self.shape, self.dtype)?;
        Ok(Self {
            storage,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            dtype: self.dtype,
            backend: self.backend.clone(),
        })
    }

    pub fn expand(&self, out_shape: &Shape) -> Result<Self> {
        let storage = self.backend.expand(&self.storage, &self.shape, out_shape, self.dtype)?;
        // Strides for the expanded tensor should be contiguous for now since we physically copied the data
        let mut strides = vec![0; out_shape.rank()];
        let mut current = 1;
        for i in (0..out_shape.rank()).rev() {
            strides[i] = current;
            current *= out_shape.0[i];
        }
        Ok(Self {
            storage,
            shape: out_shape.clone(),
            strides,
            dtype: self.dtype,
            backend: self.backend.clone(),
        })
    }

    pub fn sum_to_shape(&self, out_shape: &Shape) -> Result<Self> {
        let storage = self.backend.sum_to_shape(&self.storage, &self.shape, out_shape, self.dtype)?;
        let mut strides = vec![0; out_shape.rank()];
        let mut current = 1;
        for i in (0..out_shape.rank()).rev() {
            strides[i] = current;
            current *= out_shape.0[i];
        }
        Ok(Self {
            storage,
            shape: out_shape.clone(),
            strides,
            dtype: self.dtype,
            backend: self.backend.clone(),
        })
    }

    pub fn sum(&self) -> Result<Self> {
        let out_shape = Shape::new(vec![1]);
        self.sum_to_shape(&out_shape)
    }

    pub fn reshape(&self, out_shape: &Shape) -> Result<Self> {
        let storage = self.backend.reshape(&self.storage, &self.shape, out_shape, self.dtype)?;
        let mut strides = vec![0; out_shape.rank()];
        let mut current = 1;
        for i in (0..out_shape.rank()).rev() {
            strides[i] = current;
            current *= out_shape.0[i];
        }
        Ok(Self {
            storage,
            shape: out_shape.clone(),
            strides,
            dtype: self.dtype,
            backend: self.backend.clone(),
        })
    }

    pub fn exp(&self) -> Result<Self> {
        let storage = self.backend.exp(&self.storage, &self.shape, self.dtype)?;
        Ok(Self {
            storage,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            dtype: self.dtype,
            backend: self.backend.clone(),
        })
    }

    pub fn log(&self) -> Result<Self> {
        let storage = self.backend.log(&self.storage, &self.shape, self.dtype)?;
        Ok(Self { storage, shape: self.shape.clone(), strides: self.strides.clone(), dtype: self.dtype, backend: self.backend.clone() })
    }

    pub fn conv2d(&self, weight: &Self, bias: Option<&Self>, padding: usize, stride: usize, dilation: usize, groups: usize) -> Result<Self> {
        let storage = self.backend.conv2d(&self.storage, &weight.storage, bias.map(|b| &b.storage), &self.shape, &weight.shape, padding, stride, dilation, groups, self.dtype)?;
        
        let in_h = self.shape[2];
        let in_w = self.shape[3];
        let k_h = weight.shape[2];
        let k_w = weight.shape[3];
        let out_h = (in_h + 2 * padding - dilation * (k_h - 1) - 1) / stride + 1;
        let out_w = (in_w + 2 * padding - dilation * (k_w - 1) - 1) / stride + 1;
        let out_shape = Shape::new(vec![self.shape[0], weight.shape[0], out_h, out_w]);
        let strides = Self::compute_contiguous_strides(&out_shape);
        
        Ok(Self { storage, shape: out_shape, strides, dtype: self.dtype, backend: self.backend.clone() })
    }

    pub fn sqrt(&self) -> Result<Self> {
        let storage = self.backend.sqrt(&self.storage, &self.shape, self.dtype)?;
        Ok(Self {
            storage,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            dtype: self.dtype,
            backend: self.backend.clone(),
        })
    }
}

use std::ops::{Sub, Mul, Div};

impl<B: Backend> Div for &Tensor<B> {
    type Output = Result<Tensor<B>>;

    fn div(self, rhs: Self) -> Self::Output {
        if self.shape != rhs.shape {
            return Err(OrcaError::ShapeMismatch {
                op: "div",
                expected: self.shape.to_string(),
                got: rhs.shape.to_string(),
            });
        }
        if self.backend.device() != rhs.backend.device() {
            return Err(OrcaError::DeviceMismatch(self.backend.device(), rhs.backend.device()));
        }
        if self.dtype != rhs.dtype {
            return Err(OrcaError::InternalError("DTypes must match for element-wise division".into()));
        }

        let storage = self.backend.div(&self.storage, &rhs.storage, &self.shape, self.dtype)?;
        
        Ok(Tensor {
            storage,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            dtype: self.dtype,
            backend: self.backend.clone(),
        })
    }
}

impl<B: Backend> Add for &Tensor<B> {
    type Output = Result<Tensor<B>>;

    fn add(self, rhs: Self) -> Self::Output {
        if self.shape != rhs.shape {
            return Err(OrcaError::ShapeMismatch {
                op: "add",
                expected: self.shape.to_string(),
                got: rhs.shape.to_string(),
            });
        }
        if self.backend.device() != rhs.backend.device() {
            return Err(OrcaError::DeviceMismatch(self.backend.device(), rhs.backend.device()));
        }
        if self.dtype != rhs.dtype {
            // In the future we should promote dtypes, but for now we require them to match
            return Err(OrcaError::InternalError("DTypes must match for addition".into()));
        }

        let storage = self.backend.add(&self.storage, &rhs.storage, &self.shape, self.dtype)?;
        
        Ok(Tensor {
            storage,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            dtype: self.dtype,
            backend: self.backend.clone(),
        })
    }
}

impl<B: Backend> Sub for &Tensor<B> {
    type Output = Result<Tensor<B>>;

    fn sub(self, rhs: Self) -> Self::Output {
        if self.shape != rhs.shape {
            return Err(OrcaError::ShapeMismatch {
                op: "sub",
                expected: self.shape.to_string(),
                got: rhs.shape.to_string(),
            });
        }
        if self.backend.device() != rhs.backend.device() {
            return Err(OrcaError::DeviceMismatch(self.backend.device(), rhs.backend.device()));
        }
        if self.dtype != rhs.dtype {
            return Err(OrcaError::InternalError("DTypes must match for subtraction".into()));
        }

        let storage = self.backend.sub(&self.storage, &rhs.storage, &self.shape, self.dtype)?;
        
        Ok(Tensor {
            storage,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            dtype: self.dtype,
            backend: self.backend.clone(),
        })
    }
}

impl<B: Backend> Mul for &Tensor<B> {
    type Output = Result<Tensor<B>>;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.shape != rhs.shape {
            return Err(OrcaError::ShapeMismatch {
                op: "mul",
                expected: self.shape.to_string(),
                got: rhs.shape.to_string(),
            });
        }
        if self.backend.device() != rhs.backend.device() {
            return Err(OrcaError::DeviceMismatch(self.backend.device(), rhs.backend.device()));
        }
        if self.dtype != rhs.dtype {
            return Err(OrcaError::InternalError("DTypes must match for element-wise multiplication".into()));
        }

        let storage = self.backend.mul(&self.storage, &rhs.storage, &self.shape, self.dtype)?;
        
        Ok(Tensor {
            storage,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            dtype: self.dtype,
            backend: self.backend.clone(),
        })
    }
}
