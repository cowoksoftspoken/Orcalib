//! Core types and vocabulary for the Orca deep learning framework.
//!
//! This crate contains fundamental structures that are shared across all other
//! Orca crates, such as `DType`, `Device`, `Shape`, and the unified `OrcaError`.

pub mod device;
pub mod dtype;
pub mod error;
pub mod shape;

pub use device::Device;
pub use dtype::DType;
pub use error::{OrcaError, Result};
pub use shape::Shape;
