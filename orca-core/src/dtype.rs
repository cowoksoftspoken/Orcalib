use std::fmt;

/// The data type of a tensor's elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// 32-bit floating point
    F32,
    /// 64-bit floating point
    F64,
    /// 16-bit floating point (IEEE 754)
    F16,
    /// 16-bit floating point (Brain Float)
    BF16,
    /// 32-bit signed integer
    I32,
    /// 64-bit signed integer
    I64,
    /// 8-bit unsigned integer
    U8,
    /// Boolean
    Bool,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::F32 => "float32",
            Self::F64 => "float64",
            Self::F16 => "float16",
            Self::BF16 => "bfloat16",
            Self::I32 => "int32",
            Self::I64 => "int64",
            Self::U8 => "uint8",
            Self::Bool => "bool",
        };
        write!(f, "{}", s)
    }
}
