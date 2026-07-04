use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use orca_core::{DType, Device, Shape};
use orca_tensor::Tensor;
use orca_backend_cpu::CpuBackend;
use orca_backend_gpu::GpuBackend;
use orca_autograd::{Autodiff, tensor::AutogradTensorExt};
use std::sync::OnceLock;

#[derive(FromPyObject)]
pub enum MulOperand<'a> {
    Scalar(f32),
    Tensor(PyRef<'a, PyTensor>),
}

fn global_backend_cpu() -> Autodiff<CpuBackend> {
    static BACKEND: OnceLock<Autodiff<CpuBackend>> = OnceLock::new();
    BACKEND.get_or_init(|| Autodiff::new(CpuBackend)).clone()
}

fn global_backend_gpu() -> Autodiff<GpuBackend> {
    static BACKEND: OnceLock<Autodiff<GpuBackend>> = OnceLock::new();
    BACKEND.get_or_init(|| Autodiff::new(GpuBackend::default())).clone()
}

#[derive(Clone)]
pub enum PyTensorInner {
    Cpu(Tensor<Autodiff<CpuBackend>>),
    Gpu(Tensor<Autodiff<GpuBackend>>),
}

/// Python wrapper for Orca's DType
#[pyclass(name = "DType", module = "orca.core")]
#[derive(Clone)]
pub struct PyDType(DType);

#[pymethods]
impl PyDType {
    #[classattr]
    const FLOAT32: PyDType = PyDType(DType::F32);
    #[classattr]
    const FLOAT64: PyDType = PyDType(DType::F64);
    #[classattr]
    const FLOAT16: PyDType = PyDType(DType::F16);
    #[classattr]
    const BFLOAT16: PyDType = PyDType(DType::BF16);
    #[classattr]
    const INT32: PyDType = PyDType(DType::I32);
    #[classattr]
    const INT64: PyDType = PyDType(DType::I64);
    #[classattr]
    const UINT8: PyDType = PyDType(DType::U8);
    #[classattr]
    const BOOL: PyDType = PyDType(DType::Bool);

    fn __repr__(&self) -> String {
        format!("orca.{}", self.0)
    }
}

/// Python wrapper for Orca's Device
#[pyclass(name = "Device", module = "orca.core")]
#[derive(Clone)]
pub struct PyDevice(Device);

#[pymethods]
impl PyDevice {
    #[new]
    fn new(device_str: &str) -> PyResult<Self> {
        match device_str {
            "cpu" => Ok(Self(Device::Cpu)),
            s if s.starts_with("cuda") || s.starts_with("gpu") => {
                let idx: usize = s.split(':').nth(1).unwrap_or("0").parse().unwrap_or(0);
                Ok(Self(Device::Gpu(idx)))
            }
            _ => Err(PyValueError::new_err(format!("Unknown device: {}", device_str))),
        }
    }

    fn __repr__(&self) -> String {
        format!("device(type='{}')", self.0)
    }
}

/// A multi-dimensional array with optional automatic differentiation.
#[pyclass(name = "Tensor", module = "orca.core", subclass)]
pub struct PyTensor(pub PyTensorInner);

macro_rules! dispatch_val {
    ($self:expr, $inner:ident, $expr:expr) => {
        match &$self.0 {
            PyTensorInner::Cpu($inner) => $expr,
            PyTensorInner::Gpu($inner) => $expr,
        }
    }
}

macro_rules! dispatch_tensor {
    ($self:expr, $inner:ident, $expr:expr) => {
        match &$self.0 {
            PyTensorInner::Cpu($inner) => {
                let res = $expr.map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(PyTensor(PyTensorInner::Cpu(res)))
            },
            PyTensorInner::Gpu($inner) => {
                let res = $expr.map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(PyTensor(PyTensorInner::Gpu(res)))
            },
        }
    }
}

macro_rules! dispatch_binary {
    ($self:expr, $other:expr, $inner:ident, $other_inner:ident, $expr:expr) => {
        match (&$self.0, &$other.0) {
            (PyTensorInner::Cpu($inner), PyTensorInner::Cpu($other_inner)) => {
                let res = $expr.map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(PyTensor(PyTensorInner::Cpu(res)))
            },
            (PyTensorInner::Gpu($inner), PyTensorInner::Gpu($other_inner)) => {
                let res = $expr.map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(PyTensor(PyTensorInner::Gpu(res)))
            },
            _ => Err(PyValueError::new_err("Cannot operate on tensors from different devices")),
        }
    }
}

#[pymethods]
impl PyTensor {
    #[staticmethod]
    #[pyo3(signature = (shape, dtype=None, device=None))]
    fn zeros(shape: Vec<usize>, dtype: Option<PyDType>, device: Option<PyDevice>) -> PyResult<Self> {
        let dtype = dtype.map(|d| d.0).unwrap_or(DType::F32);
        let is_gpu = match device {
            Some(d) => matches!(d.0, Device::Gpu(_)),
            None => false,
        };

        if is_gpu {
            let tensor = Tensor::zeros(global_backend_gpu(), shape, dtype)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(Self(PyTensorInner::Gpu(tensor)))
        } else {
            let tensor = Tensor::zeros(global_backend_cpu(), shape, dtype)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(Self(PyTensorInner::Cpu(tensor)))
        }
    }

    #[staticmethod]
    #[pyo3(signature = (shape, low, high, dtype=None, device=None, requires_grad=false))]
    fn rand_uniform(shape: Vec<usize>, low: f32, high: f32, dtype: Option<PyDType>, device: Option<PyDevice>, requires_grad: bool) -> PyResult<Self> {
        let dtype = dtype.map(|d| d.0).unwrap_or(DType::F32);
        let is_gpu = match device {
            Some(d) => matches!(d.0, Device::Gpu(_)),
            None => false,
        };

        if is_gpu {
            let mut tensor = Tensor::rand_uniform(global_backend_gpu(), shape, low, high, dtype)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            if requires_grad { tensor.require_grad(); }
            Ok(Self(PyTensorInner::Gpu(tensor)))
        } else {
            let mut tensor = Tensor::rand_uniform(global_backend_cpu(), shape, low, high, dtype)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            if requires_grad { tensor.require_grad(); }
            Ok(Self(PyTensorInner::Cpu(tensor)))
        }
    }

    #[staticmethod]
    #[pyo3(signature = (shape, p, dtype=None, device=None))]
    fn rand_dropout_mask(shape: Vec<usize>, p: f32, dtype: Option<PyDType>, device: Option<PyDevice>) -> PyResult<Self> {
        let dtype = dtype.map(|d| d.0).unwrap_or(DType::F32);
        let is_gpu = match device {
            Some(d) => matches!(d.0, Device::Gpu(_)),
            None => false,
        };

        if is_gpu {
            let tensor = Tensor::rand_dropout_mask(global_backend_gpu(), shape, p, dtype)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(Self(PyTensorInner::Gpu(tensor)))
        } else {
            let tensor = Tensor::rand_dropout_mask(global_backend_cpu(), shape, p, dtype)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(Self(PyTensorInner::Cpu(tensor)))
        }
    }

    #[staticmethod]
    #[pyo3(signature = (data, shape=None, requires_grad=false, device=None))]
    fn from_list(data: Vec<f32>, shape: Option<Vec<usize>>, requires_grad: bool, device: Option<PyDevice>) -> PyResult<Self> {
        let shape = shape.unwrap_or_else(|| vec![data.len()]);
        let is_gpu = match device {
            Some(d) => matches!(d.0, Device::Gpu(_) | Device::Cuda(_)),
            None => false,
        };

        if is_gpu {
            let mut tensor = Tensor::from_f32_slice(global_backend_gpu(), &data, shape)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            if requires_grad { tensor.require_grad(); }
            Ok(Self(PyTensorInner::Gpu(tensor)))
        } else {
            let mut tensor = Tensor::from_f32_slice(global_backend_cpu(), &data, shape)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            if requires_grad { tensor.require_grad(); }
            Ok(Self(PyTensorInner::Cpu(tensor)))
        }
    }

    fn to(&self, device_str: &str) -> PyResult<Self> {
        let is_gpu = device_str.starts_with("cuda") || device_str.starts_with("gpu");
        let current_is_gpu = matches!(self.0, PyTensorInner::Gpu(_));
        
        if is_gpu == current_is_gpu {
            return Ok(PyTensor(self.0.clone()));
        }

        let data = self.to_list()?;
        let shape = self.shape();
        let requires_grad = self.requires_grad();
        
        Self::from_list(data, Some(shape), requires_grad, Some(PyDevice::new(device_str)?))
    }

    #[getter]
    fn shape(&self) -> Vec<usize> { dispatch_val!(self, t, t.shape().to_vec()) }

    #[getter]
    fn dtype(&self) -> PyDType { dispatch_val!(self, t, PyDType(t.dtype())) }

    #[getter]
    fn device(&self) -> PyDevice { dispatch_val!(self, t, PyDevice(t.device())) }

    fn __repr__(&self) -> String {
        format!("Tensor(shape={:?}, dtype={}, device={})", self.shape(), self.dtype().0, self.device().0)
    }

    fn to_list(&self) -> PyResult<Vec<f32>> {
        dispatch_val!(self, t, t.to_f32_vec().map_err(|e| PyValueError::new_err(e.to_string())))
    }

    fn __add__(&self, other: &Self) -> PyResult<Self> {
        dispatch_binary!(self, other, s, o, {
            let out_shape = s.shape().broadcast(o.shape()).ok_or_else(|| PyValueError::new_err("Shapes are not broadcastable"))?;
            let s_b = s.expand(&out_shape).map_err(|e| PyValueError::new_err(e.to_string()))?;
            let o_b = o.expand(&out_shape).map_err(|e| PyValueError::new_err(e.to_string()))?;
            &s_b + &o_b
        })
    }

    fn __sub__(&self, other: &Self) -> PyResult<Self> {
        dispatch_binary!(self, other, s, o, {
            let out_shape = s.shape().broadcast(o.shape()).ok_or_else(|| PyValueError::new_err("Shapes are not broadcastable"))?;
            let s_b = s.expand(&out_shape).map_err(|e| PyValueError::new_err(e.to_string()))?;
            let o_b = o.expand(&out_shape).map_err(|e| PyValueError::new_err(e.to_string()))?;
            &s_b - &o_b
        })
    }

    fn __mul__(&self, operand: MulOperand) -> PyResult<Self> {
        match operand {
            MulOperand::Scalar(s_val) => dispatch_tensor!(self, t, t.mul_scalar(s_val)),
            MulOperand::Tensor(other) => dispatch_binary!(self, other, s, o, {
                let out_shape = s.shape().broadcast(o.shape()).ok_or_else(|| PyValueError::new_err("Shapes are not broadcastable"))?;
                let s_b = s.expand(&out_shape).map_err(|e| PyValueError::new_err(e.to_string()))?;
                let o_b = o.expand(&out_shape).map_err(|e| PyValueError::new_err(e.to_string()))?;
                &s_b * &o_b
            }),
        }
    }

    fn __matmul__(&self, other: &Self) -> PyResult<Self> {
        dispatch_binary!(self, other, s, o, s.matmul(o))
    }

    fn __truediv__(&self, other: &Self) -> PyResult<Self> {
        dispatch_binary!(self, other, s, o, {
            let out_shape = s.shape().broadcast(o.shape()).ok_or_else(|| PyValueError::new_err("Shapes are not broadcastable"))?;
            let s_b = s.expand(&out_shape).map_err(|e| PyValueError::new_err(e.to_string()))?;
            let o_b = o.expand(&out_shape).map_err(|e| PyValueError::new_err(e.to_string()))?;
            &s_b / &o_b
        })
    }

    fn transpose(&self) -> PyResult<Self> { dispatch_tensor!(self, t, t.transpose()) }
    fn relu(&self) -> PyResult<Self> { dispatch_tensor!(self, t, t.relu()) }
    fn sqrt(&self) -> PyResult<Self> { dispatch_tensor!(self, t, t.sqrt()) }
    fn sigmoid(&self) -> PyResult<Self> { dispatch_tensor!(self, t, t.sigmoid()) }
    fn exp(&self) -> PyResult<Self> {
        dispatch_tensor!(self, inner, inner.exp())
    }

    fn log(&self) -> PyResult<Self> {
        dispatch_tensor!(self, inner, inner.log())
    }

    fn __neg__(&self) -> PyResult<Self> {
        // -x is exactly x * -1.0
        match &self.0 {
            PyTensorInner::Cpu(inner) => {
                let res = inner.mul_scalar(-1.0).map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(PyTensor(PyTensorInner::Cpu(res)))
            },
            PyTensorInner::Gpu(inner) => {
                let res = inner.mul_scalar(-1.0).map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(PyTensor(PyTensorInner::Gpu(res)))
            }
        }
    }

    fn sum(&self) -> PyResult<Self> { dispatch_tensor!(self, t, t.sum()) }
    
    fn mean(&self) -> PyResult<Self> {
        dispatch_tensor!(self, t, {
            t.sum().and_then(|sum| sum.mul_scalar(1.0 / t.shape().num_elements() as f32))
        })
    }

    #[pyo3(signature = (weight, bias=None, padding=0, stride=1, dilation=1, groups=1))]
    fn conv2d(&self, weight: &Self, bias: Option<&Self>, padding: usize, stride: usize, dilation: usize, groups: usize) -> PyResult<Self> {
        match (&self.0, &weight.0) {
            (PyTensorInner::Cpu(o), PyTensorInner::Cpu(w)) => {
                let res = match bias {
                    Some(b) => {
                        if let PyTensorInner::Cpu(b_in) = &b.0 {
                            o.conv2d(w, Some(b_in), padding, stride, dilation, groups).map_err(|e| PyValueError::new_err(e.to_string()))?
                        } else {
                            return Err(PyValueError::new_err("Cannot operate on tensors from different devices"));
                        }
                    },
                    None => o.conv2d(w, None, padding, stride, dilation, groups).map_err(|e| PyValueError::new_err(e.to_string()))?
                };
                Ok(PyTensor(PyTensorInner::Cpu(res)))
            },
            (PyTensorInner::Gpu(o), PyTensorInner::Gpu(w)) => {
                let res = match bias {
                    Some(b) => {
                        if let PyTensorInner::Gpu(b_in) = &b.0 {
                            o.conv2d(w, Some(b_in), padding, stride, dilation, groups).map_err(|e| PyValueError::new_err(e.to_string()))?
                        } else {
                            return Err(PyValueError::new_err("Cannot operate on tensors from different devices"));
                        }
                    },
                    None => o.conv2d(w, None, padding, stride, dilation, groups).map_err(|e| PyValueError::new_err(e.to_string()))?
                };
                Ok(PyTensor(PyTensorInner::Gpu(res)))
            },
            _ => Err(PyValueError::new_err("Cannot operate on tensors from different devices")),
        }
    }

    fn expand(&self, shape: Vec<usize>) -> PyResult<Self> { dispatch_tensor!(self, t, t.expand(&Shape::new(shape))) }
    fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> { dispatch_tensor!(self, t, t.reshape(&Shape::new(shape))) }
    fn sum_to_shape(&self, shape: Vec<usize>) -> PyResult<Self> { dispatch_tensor!(self, t, t.sum_to_shape(&Shape::new(shape))) }

    #[getter]
    fn requires_grad(&self) -> bool { dispatch_val!(self, t, t.storage().node_id.is_some()) }

    fn zero_grad(&self) -> PyResult<()> {
        dispatch_val!(self, t, t.zero_grad().map_err(|e| PyValueError::new_err(e.to_string())))
    }

    fn backward(&self) -> PyResult<()> {
        dispatch_val!(self, t, t.backward().map_err(|e| PyValueError::new_err(e.to_string())))
    }

    fn grad(&self) -> PyResult<Option<Self>> {
        match &self.0 {
            PyTensorInner::Cpu(t) => {
                if let Some(grad_tensor) = t.grad() {
                    let vec = grad_tensor.to_f32_vec().map_err(|e| PyValueError::new_err(e.to_string()))?;
                    let shape = grad_tensor.shape().to_vec();
                    let new_tensor = Tensor::from_f32_slice(global_backend_cpu(), &vec, shape)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    Ok(Some(PyTensor(PyTensorInner::Cpu(new_tensor))))
                } else {
                    Ok(None)
                }
            },
            PyTensorInner::Gpu(t) => {
                if let Some(grad_tensor) = t.grad() {
                    let vec = grad_tensor.to_f32_vec().map_err(|e| PyValueError::new_err(e.to_string()))?;
                    let shape = grad_tensor.shape().to_vec();
                    let new_tensor = Tensor::from_f32_slice(global_backend_gpu(), &vec, shape)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    Ok(Some(PyTensor(PyTensorInner::Gpu(new_tensor))))
                } else {
                    Ok(None)
                }
            }
        }
    }
}

#[pymodule]
fn orca_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDType>()?;
    m.add_class::<PyDevice>()?;
    m.add_class::<PyTensor>()?;
    Ok(())
}
