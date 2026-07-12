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

impl DType {
    /// Promotes this type with another type, returning the type that can represent both without loss of precision.
    /// Following general ML framework type promotion rules.
    pub fn promote(self, other: Self) -> Self {
        if self == other {
            return self;
        }

        match (self, other) {
            (DType::F64, _) | (_, DType::F64) => DType::F64,
            (DType::F32, _) | (_, DType::F32) => DType::F32,
            (DType::BF16, _) | (_, DType::BF16) => DType::BF16,
            (DType::F16, _) | (_, DType::F16) => DType::F16,
            (DType::I64, _) | (_, DType::I64) => DType::I64,
            (DType::I32, _) | (_, DType::I32) => DType::I32,
            (DType::U8, _) | (_, DType::U8) => DType::U8,
            _ => DType::F32, // Fallback
        }
    }

    /// Get the size of this type in bytes.
    pub fn element_size(&self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F64 | Self::I64 => 8,
            Self::F16 | Self::BF16 => 2,
            Self::U8 | Self::Bool => 1,
        }
    }
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
