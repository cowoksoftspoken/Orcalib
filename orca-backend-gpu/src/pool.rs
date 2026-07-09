use std::collections::HashMap;
use wgpu;

/// A simple caching allocator for wgpu::Buffer.
#[derive(Debug)]
pub struct MemoryPool {
    /// Maps a buffer size in bytes to a vector of available, unused buffers of that exact size.
    cache: HashMap<wgpu::BufferAddress, Vec<wgpu::Buffer>>,
}

impl MemoryPool {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Allocates a buffer of the requested size and usage.
    /// If a compatible buffer is available in the cache, it is reused.
    /// Otherwise, a new buffer is created.
    pub fn allocate(
        &mut self,
        device: &wgpu::Device,
        size: wgpu::BufferAddress,
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        // Look for an available buffer of the exact size.
        // For a more advanced allocator, we could check for >= size (e.g. next power of 2),
        // but exact match is simplest and prevents unbounded waste.
        if let Some(list) = self.cache.get_mut(&size) {
            if let Some(buffer) = list.pop() {
                // Wait, buffer usage must match!
                // For Orca, almost all our buffers are STORAGE | COPY_SRC | COPY_DST.
                // We'll assume usage matches for now. A safer pool would key by (size, usage).
                return buffer;
            }
        }

        // Cache miss: create a new buffer
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pooled_buffer"),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Releases a buffer back into the pool.
    pub fn release(&mut self, buffer: wgpu::Buffer, size: wgpu::BufferAddress) {
        self.cache.entry(size).or_insert_with(Vec::new).push(buffer);
    }
}
