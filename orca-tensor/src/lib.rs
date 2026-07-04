//! Tensor representation and traits for the Orca framework.

pub mod backend;
pub mod tensor;

pub use backend::{Backend, Storage};
pub use tensor::Tensor;
