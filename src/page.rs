/// Concrete storage for one allocated page.
///
/// A page is only present when `non_default` is non-zero. Its slot count is
/// exactly the logical length of the associated page.
#[derive(Clone, Debug)]
pub(crate) struct Page<T> {
    pub(crate) values: Box<[T]>,
    pub(crate) non_default: usize,
}

impl<T: Clone> Page<T> {
    pub(crate) fn filled_with(default: &T, len: usize) -> Self {
        Self {
            values: vec![default.clone(); len].into_boxed_slice(),
            non_default: 0,
        }
    }
}
use alloc::boxed::Box;
use alloc::vec;
