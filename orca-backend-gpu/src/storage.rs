use std::sync::Arc;
use orca_tensor::Storage;

/// GPU storage backed by a reference-counted wgpu Buffer.
#[derive(Clone, Debug)]
pub struct GpuStorage {
    pub buffer: Arc<wgpu::Buffer>,
    pub num_elements: usize,
    pub element_size: usize,
}

impl GpuStorage {
    pub fn new(buffer: wgpu::Buffer, num_elements: usize, element_size: usize) -> Self {
        Self {
            buffer: Arc::new(buffer),
            num_elements,
            element_size,
        }
    }
}

impl Storage for GpuStorage {
    fn len(&self) -> usize {
        self.num_elements
    }
}
