use std::fmt;
use std::ops::Deref;

/// Represents the dimensions of a tensor.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape(pub Vec<usize>);

impl Shape {
    /// Create a new shape from a vector of dimensions.
    pub fn new(dims: Vec<usize>) -> Self {
        Self(dims)
    }

    /// Calculate the total number of elements in this shape.
    pub fn num_elements(&self) -> usize {
        self.0.iter().product()
    }

    /// Get the rank (number of dimensions) of this shape.
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// Computes the broadcasted shape between `self` and `other` according to standard numpy broadcasting rules.
    /// Returns `None` if the shapes are not compatible.
    pub fn broadcast(&self, other: &Shape) -> Option<Shape> {
        let mut result = Vec::new();
        let mut i = self.rank() as isize - 1;
        let mut j = other.rank() as isize - 1;

        while i >= 0 || j >= 0 {
            let dim1 = if i >= 0 { self.0[i as usize] } else { 1 };
            let dim2 = if j >= 0 { other.0[j as usize] } else { 1 };

            if dim1 == dim2 {
                result.push(dim1);
            } else if dim1 == 1 {
                result.push(dim2);
            } else if dim2 == 1 {
                result.push(dim1);
            } else {
                return None;
            }

            i -= 1;
            j -= 1;
        }

        result.reverse();
        Some(Shape::new(result))
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self(dims)
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self(dims.to_vec())
    }
}

impl Deref for Shape {
    type Target = [usize];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, dim) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", dim)?;
        }
        write!(f, "]")
    }
}
