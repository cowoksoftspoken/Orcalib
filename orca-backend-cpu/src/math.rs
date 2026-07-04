use std::fmt::Debug;

pub trait CpuNumeric: Copy + Debug + Send + Sync + Default + 'static {
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn mul(self, other: Self) -> Self;
    fn div(self, other: Self) -> Self;
    
    // For scalar operations
    fn from_f32(val: f32) -> Self;
    fn to_f32(self) -> f32;
    fn max(self, other: Self) -> Self;
    fn zero() -> Self;
    fn one() -> Self;
}

pub trait CpuFloat: CpuNumeric {
    fn exp(self) -> Self;
    fn log(self) -> Self;
    fn sqrt(self) -> Self;
}

macro_rules! impl_numeric_float {
    ($t:ty) => {
        impl CpuNumeric for $t {
            #[inline(always)] fn add(self, other: Self) -> Self { self + other }
            #[inline(always)] fn sub(self, other: Self) -> Self { self - other }
            #[inline(always)] fn mul(self, other: Self) -> Self { self * other }
            #[inline(always)] fn div(self, other: Self) -> Self { self / other }
            #[inline(always)] fn from_f32(val: f32) -> Self { val as $t }
            #[inline(always)] fn to_f32(self) -> f32 { self as f32 }
            #[inline(always)] fn max(self, other: Self) -> Self { if self > other { self } else { other } }
            #[inline(always)] fn zero() -> Self { 0.0 }
            #[inline(always)] fn one() -> Self { 1.0 }
        }

        impl CpuFloat for $t {
            #[inline(always)] fn exp(self) -> Self { self.exp() }
            #[inline(always)] fn log(self) -> Self { self.ln() }
            #[inline(always)] fn sqrt(self) -> Self { self.sqrt() }
        }
    };
}

impl_numeric_float!(f32);
impl_numeric_float!(f64);

macro_rules! impl_numeric_int {
    ($t:ty) => {
        impl CpuNumeric for $t {
            #[inline(always)] fn add(self, other: Self) -> Self { self + other }
            #[inline(always)] fn sub(self, other: Self) -> Self { self - other }
            #[inline(always)] fn mul(self, other: Self) -> Self { self * other }
            #[inline(always)] fn div(self, other: Self) -> Self { self / other }
            #[inline(always)] fn from_f32(val: f32) -> Self { val as $t }
            #[inline(always)] fn to_f32(self) -> f32 { self as f32 }
            #[inline(always)] fn max(self, other: Self) -> Self { if self > other { self } else { other } }
            #[inline(always)] fn zero() -> Self { 0 }
            #[inline(always)] fn one() -> Self { 1 }
        }
    };
}

impl_numeric_int!(i32);
impl_numeric_int!(i64);
impl_numeric_int!(u8);

impl CpuNumeric for bool {
    #[inline(always)] fn add(self, other: Self) -> Self { self | other }
    #[inline(always)] fn sub(self, other: Self) -> Self { self ^ other }
    #[inline(always)] fn mul(self, other: Self) -> Self { self & other }
    #[inline(always)] fn div(self, _other: Self) -> Self { self }
    #[inline(always)] fn from_f32(val: f32) -> Self { val != 0.0 }
    #[inline(always)] fn to_f32(self) -> f32 { if self { 1.0 } else { 0.0 } }
    #[inline(always)] fn max(self, other: Self) -> Self { self | other }
    #[inline(always)] fn zero() -> Self { false }
    #[inline(always)] fn one() -> Self { true }
}

#[macro_export]
macro_rules! dispatch_dtype {
    ($dtype:expr, $type_var:ident, $body:expr) => {
        match $dtype {
            orca_core::DType::F32 => { type $type_var = f32; $body },
            orca_core::DType::F64 => { type $type_var = f64; $body },
            orca_core::DType::I32 => { type $type_var = i32; $body },
            orca_core::DType::I64 => { type $type_var = i64; $body },
            orca_core::DType::U8  => { type $type_var = u8; $body },
            orca_core::DType::Bool=> { type $type_var = bool; $body },
            _ => Err(orca_core::OrcaError::UnsupportedDType { op: "generic_dispatch", dtype: $dtype.clone() }),
        }
    }
}

#[macro_export]
macro_rules! dispatch_float {
    ($dtype:expr, $type_var:ident, $body:expr) => {
        match $dtype {
            orca_core::DType::F32 => { type $type_var = f32; $body },
            orca_core::DType::F64 => { type $type_var = f64; $body },
            _ => Err(orca_core::OrcaError::UnsupportedDType { op: "float_dispatch", dtype: $dtype.clone() }),
        }
    }
}
