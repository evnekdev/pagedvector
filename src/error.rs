use std::error::Error;
use std::fmt;

/// An attempted logical index was outside a [`crate::PagedVec`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexOutOfBounds {
    /// The attempted index.
    pub index: usize,
    /// The logical length of the vector.
    pub len: usize,
}

impl fmt::Display for IndexOutOfBounds {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "index {} is out of bounds for PagedVec of length {}",
            self.index, self.len
        )
    }
}

impl Error for IndexOutOfBounds {}

/// Construction errors for [`crate::PagedVec`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PagedVecError {
    /// Pages must contain at least one logical element.
    ZeroPageSize,
}

impl fmt::Display for PagedVecError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroPageSize => formatter.write_str("page size must be greater than zero"),
        }
    }
}

impl Error for PagedVecError {}
