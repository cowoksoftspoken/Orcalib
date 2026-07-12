//! CPU backend implementation for the Orca framework.

pub mod backend;
pub mod math;
pub mod storage;

pub use backend::CpuBackend;
pub use storage::CpuByteStorage;
