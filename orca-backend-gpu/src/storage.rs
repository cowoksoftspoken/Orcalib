use std::sync::Arc;
use orca_tensor::Storage;

#[derive(Debug)]
pub struct PooledBuffer {
    pub buffer: Option<wgpu::Buffer>,
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        // Pool is removed, buffer drops naturally
    }
}

/// GPU storage backed by a reference-counted wgpu Buffer.
#[derive(Clone, Debug)]
pub struct GpuStorage {
    pub inner: Arc<PooledBuffer>,
    pub num_elements: usize,
    pub element_size: usize,
}

impl GpuStorage {
    pub fn new(
        buffer: wgpu::Buffer,
        num_elements: usize,
        element_size: usize
    ) -> Self {
        let inner = PooledBuffer {
            buffer: Some(buffer),
        };
        Self {
            inner: Arc::new(inner),
            num_elements,
            element_size,
        }
    }
    /// Returns the underlying wgpu buffer.
    ///
    /// # Safety Invariant  
    /// Buffer is always `Some` after construction — `PooledBuffer::drop` is the
    /// only consumer, and it runs after all references are gone.
    pub fn buffer(&self) -> &wgpu::Buffer {
        self.inner.buffer.as_ref()
            .expect("BUG: GpuStorage buffer was None — this should never happen")
    }
}

impl Storage for GpuStorage {
    fn len(&self) -> usize {
        self.num_elements
    }
}
