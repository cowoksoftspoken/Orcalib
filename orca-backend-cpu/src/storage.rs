use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::sync::Arc;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use orca_tensor::Storage;

/// A robust, SIMD-aligned (64-byte) raw memory buffer for CPU tensors.
/// Provides safe slice casting for various numeric types without alignment issues (Undefined Behavior).
struct AlignedBuffer {
    ptr: *mut u8,
    layout: Layout,
    capacity_bytes: usize,
}

unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

impl AlignedBuffer {
    fn new(size_in_bytes: usize) -> Self {
        // Use 64-byte alignment to be ready for AVX-512 and cache line optimizations
        let align = 64; 
        let layout = Layout::from_size_align(size_in_bytes.max(1), align)
            .expect("Invalid layout for CpuStorage");
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Self { ptr, layout, capacity_bytes: size_in_bytes }
    }
    
    fn as_bytes(&self) -> &[u8] {
        if self.capacity_bytes == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.ptr, self.capacity_bytes) }
    }
    
    fn as_mut_bytes(&mut self) -> &mut [u8] {
        if self.capacity_bytes == 0 {
            return &mut [];
        }
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.capacity_bytes) }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        if self.capacity_bytes > 0 && !self.ptr.is_null() {
            unsafe { dealloc(self.ptr, self.layout) };
        }
    }
}

impl Clone for AlignedBuffer {
    fn clone(&self) -> Self {
        let mut new_buf = AlignedBuffer::new(self.capacity_bytes);
        new_buf.as_mut_bytes().copy_from_slice(self.as_bytes());
        new_buf
    }
}

impl Debug for AlignedBuffer {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("AlignedBuffer")
            .field("capacity_bytes", &self.capacity_bytes)
            .finish()
    }
}

/// CPU storage backed by a reference-counted, aligned byte array.
#[derive(Clone, Debug)]
pub struct CpuByteStorage {
    data: Arc<AlignedBuffer>,
    num_elements: usize,
}

impl CpuByteStorage {
    pub fn new(size_in_bytes: usize, num_elements: usize, _element_size: usize) -> Self {
        Self {
            data: Arc::new(AlignedBuffer::new(size_in_bytes)),
            num_elements,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.data.as_bytes()
    }

    pub fn as_mut_bytes(&mut self) -> &mut [u8] {
        Arc::make_mut(&mut self.data).as_mut_bytes()
    }

    pub fn as_slice<T>(&self) -> &[T] {
        if self.num_elements == 0 {
            return &[];
        }
        assert!((self.data.ptr as usize).is_multiple_of(std::mem::align_of::<T>()), "Alignment mismatch for tensor type");
        unsafe { std::slice::from_raw_parts(self.data.ptr as *const T, self.num_elements) }
    }

    pub fn as_mut_slice<T>(&mut self) -> &mut [T] {
        if self.num_elements == 0 {
            return &mut [];
        }
        let buf = Arc::make_mut(&mut self.data);
        assert!((buf.ptr as usize).is_multiple_of(std::mem::align_of::<T>()), "Alignment mismatch for tensor type");
        unsafe { std::slice::from_raw_parts_mut(buf.ptr as *mut T, self.num_elements) }
    }
}

impl Storage for CpuByteStorage {
    fn len(&self) -> usize {
        self.num_elements
    }
}
