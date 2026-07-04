use std::fmt;

/// The physical device where a tensor's memory resides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    /// The host CPU
    Cpu,
    /// An NVIDIA GPU identified by its index (e.g., Cuda(0))
    Cuda(usize),
    /// A generic GPU backend powered by WGPU
    Gpu(usize),
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda(idx) => write!(f, "cuda:{}", idx),
            Device::Gpu(idx) => write!(f, "gpu:{}", idx),
        }
    }
}
