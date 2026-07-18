use std::iter::FusedIterator;
use std::slice;

use crate::page::Page;
use crate::paged_vec::PagedVec;

/// An iterator over every logical value in a [`PagedVec`].
///
/// Values from unallocated pages borrow the vector's configured default value.
/// No dense temporary allocation is performed.
#[derive(Clone, Debug)]
pub struct Iter<'a, T> {
    vector: &'a PagedVec<T>,
    front: usize,
    back: usize,
}

impl<'a, T> Iter<'a, T> {
    pub(crate) fn new(vector: &'a PagedVec<T>) -> Self {
        Self {
            vector,
            front: 0,
            back: vector.len,
        }
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.front == self.back {
            return None;
        }

        let index = self.front;
        self.front += 1;
        self.vector
            .get(index)
            .or_else(|| unreachable!("iterator index must be in bounds"))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.back - self.front;
        (remaining, Some(remaining))
    }
}

impl<T> DoubleEndedIterator for Iter<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front == self.back {
            return None;
        }

        self.back -= 1;
        self.vector
            .get(self.back)
            .or_else(|| unreachable!("iterator index must be in bounds"))
    }
}

impl<T> ExactSizeIterator for Iter<'_, T> {}
impl<T> FusedIterator for Iter<'_, T> {}

/// An iterator over logical indices whose values differ from a [`PagedVec`]'s
/// configured default.
///
/// It skips unallocated pages and yields `(index, value)` pairs in ascending
/// logical-index order without allocating.
#[derive(Clone, Debug)]
pub struct NonDefaultIter<'a, T> {
    pages: &'a [Option<Page<T>>],
    default: &'a T,
    page_size: usize,
    page_index: usize,
    value_index: usize,
    remaining: usize,
}

impl<'a, T> NonDefaultIter<'a, T> {
    pub(crate) fn new(vector: &'a PagedVec<T>) -> Self {
        Self {
            pages: &vector.pages,
            default: &vector.default,
            page_size: vector.page_size(),
            page_index: 0,
            value_index: 0,
            remaining: vector.non_default,
        }
    }
}

impl<'a, T: PartialEq> Iterator for NonDefaultIter<'a, T> {
    type Item = (usize, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(page) = self.pages.get(self.page_index) {
            let Some(page) = page.as_ref() else {
                self.page_index += 1;
                self.value_index = 0;
                continue;
            };

            while self.value_index < page.values.len() {
                let value_index = self.value_index;
                self.value_index += 1;
                let value = &page.values[value_index];
                if value != self.default {
                    self.remaining -= 1;
                    return Some((self.page_index * self.page_size + value_index, value));
                }
            }

            self.page_index += 1;
            self.value_index = 0;
        }

        debug_assert_eq!(self.remaining, 0);
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T: PartialEq> ExactSizeIterator for NonDefaultIter<'_, T> {}
impl<T: PartialEq> FusedIterator for NonDefaultIter<'_, T> {}

/// An iterator over physically allocated pages in ascending page-index order.
///
/// Each yielded slice is complete physical page storage. It can contain values
/// equal to the configured default because physical allocation is per page,
/// whereas non-defaultness is per logical value.
#[derive(Clone, Debug)]
pub struct AllocatedPages<'a, T> {
    pages: std::iter::Enumerate<slice::Iter<'a, Option<Page<T>>>>,
}

impl<'a, T> AllocatedPages<'a, T> {
    pub(crate) fn new(vector: &'a PagedVec<T>) -> Self {
        Self {
            pages: vector.pages.iter().enumerate(),
        }
    }
}

impl<'a, T> Iterator for AllocatedPages<'a, T> {
    type Item = (usize, &'a [T]);

    fn next(&mut self) -> Option<Self::Item> {
        for (page_index, page) in self.pages.by_ref() {
            if let Some(page) = page {
                return Some((page_index, page.values.as_ref()));
            }
        }
        None
    }
}

impl<T> DoubleEndedIterator for AllocatedPages<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        for (page_index, page) in self.pages.by_ref().rev() {
            if let Some(page) = page {
                return Some((page_index, page.values.as_ref()));
            }
        }
        None
    }
}

impl<T> FusedIterator for AllocatedPages<'_, T> {}

/// An iterator over the indices of physically allocated pages.
#[derive(Clone, Debug)]
pub struct AllocatedPageIndices<'a, T> {
    pages: AllocatedPages<'a, T>,
}

impl<'a, T> AllocatedPageIndices<'a, T> {
    pub(crate) fn new(vector: &'a PagedVec<T>) -> Self {
        Self {
            pages: AllocatedPages::new(vector),
        }
    }
}

impl<'a, T> Iterator for AllocatedPageIndices<'a, T> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.pages.next().map(|(page_index, _)| page_index)
    }
}

impl<T> DoubleEndedIterator for AllocatedPageIndices<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.pages.next_back().map(|(page_index, _)| page_index)
    }
}

impl<T> FusedIterator for AllocatedPageIndices<'_, T> {}
