//! CPU backend implementation for the Orca framework.

pub mod backend;
pub mod storage;
pub mod math;

pub use backend::CpuBackend;
pub use storage::CpuByteStorage;
