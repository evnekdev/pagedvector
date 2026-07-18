use std::num::NonZeroUsize;
use std::ops::Index;

use crate::error::{IndexOutOfBounds, PagedVecError};
use crate::iter::{AllocatedPageIndices, AllocatedPages, Iter, NonDefaultIter};
use crate::page::Page;

/// A fixed-length vector whose physical storage is allocated one page at a
/// time for non-default values.
///
/// Each logical index below [`Self::len`] has a value. When its page is not
/// allocated, that value is [`Self::default_value`]. Allocated pages are
/// canonical: each has at least one non-default value, and their bookkeeping
/// exactly matches their contents.
///
/// # Logical values, non-default values, and allocated pages
///
/// A logical value exists at every index below [`Self::len`]. A non-default
/// value is a logical value unequal to [`Self::default_value`]. An allocated
/// page is physical storage containing at least one non-default value, but it
/// may also contain default-valued slots. These concepts are intentionally
/// distinct: [`Self::iter`] visits logical values, [`Self::non_default_iter`]
/// visits non-default values, and [`Self::allocated_pages`] visits physical
/// storage.
///
/// Common operations are O(1), apart from page allocation and cloning the
/// default value into a newly allocated page. [`Self::allocated_page_count`]
/// is O(number of logical pages) in the current dense page-table backend.
#[derive(Clone, Debug)]
pub struct PagedVec<T> {
    pub(crate) len: usize,
    pub(crate) page_size: NonZeroUsize,
    pub(crate) default: T,
    pub(crate) non_default: usize,
    pub(crate) pages: Vec<Option<Page<T>>>,
}

impl<T> PagedVec<T> {
    /// Creates a vector with `len` logical values, all initially equal to
    /// `default`.
    ///
    /// # Panics
    ///
    /// Panics if `page_size` is zero. Use [`Self::try_new`] when the page size
    /// comes from fallible input.
    pub fn new(len: usize, default: T, page_size: usize) -> Self {
        Self::try_new(len, default, page_size)
            .expect("PagedVec::new requires a page size greater than zero")
    }

    /// Creates a vector, returning an error if `page_size` is zero.
    pub fn try_new(len: usize, default: T, page_size: usize) -> Result<Self, PagedVecError> {
        let page_size = NonZeroUsize::new(page_size).ok_or(PagedVecError::ZeroPageSize)?;
        let page_count = page_count_for(len, page_size);
        let pages = std::iter::repeat_with(|| None).take(page_count).collect();

        Ok(Self {
            len,
            page_size,
            default,
            non_default: 0,
            pages,
        })
    }

    /// Returns the number of logical elements.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the vector has no logical elements.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the fixed number of slots in a full page.
    #[must_use]
    pub const fn page_size(&self) -> usize {
        self.page_size.get()
    }

    /// Returns the total number of logical pages in the page table.
    #[must_use]
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Returns the configured logical value of every unallocated slot.
    #[must_use]
    pub const fn default_value(&self) -> &T {
        &self.default
    }

    /// Returns the number of logical elements that differ from the default.
    #[must_use]
    pub const fn non_default_len(&self) -> usize {
        self.non_default
    }

    /// Returns the number of physically allocated pages.
    ///
    /// This scans the current dense page table and is O([`Self::page_count`]).
    #[must_use]
    pub fn allocated_page_count(&self) -> usize {
        self.pages.iter().filter(|page| page.is_some()).count()
    }

    /// Returns whether `page_index` has physical storage.
    ///
    /// Returns `None` when `page_index` is outside the logical page table.
    #[must_use]
    pub fn is_page_allocated(&self, page_index: usize) -> Option<bool> {
        self.pages.get(page_index).map(Option::is_some)
    }

    /// Returns whether the page containing `index` has physical storage.
    ///
    /// Returns `None` when `index` is outside the logical vector. `Some(true)`
    /// does not mean the value at this exact index is non-default: an allocated
    /// page can contain default-valued slots.
    #[must_use]
    pub fn is_allocated(&self, index: usize) -> Option<bool> {
        self.page_index(index)
            .and_then(|page_index| self.is_page_allocated(page_index))
    }

    /// Returns the page index containing `index`, or `None` when `index` is
    /// outside the logical vector.
    #[must_use]
    pub fn page_index(&self, index: usize) -> Option<usize> {
        (index < self.len).then(|| index / self.page_size())
    }

    /// Returns the offset within its page for `index`, or `None` when `index`
    /// is outside the logical vector.
    #[must_use]
    pub fn page_offset(&self, index: usize) -> Option<usize> {
        (index < self.len).then(|| index % self.page_size())
    }

    /// Returns the logical value at `index`.
    ///
    /// In-bounds unallocated pages return a shared reference to the configured
    /// default value. Out-of-bounds indices return `None`.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&T> {
        let (page_index, page_offset) = self.split_index(index)?;
        Some(match &self.pages[page_index] {
            Some(page) => &page.values[page_offset],
            None => &self.default,
        })
    }

    /// Returns an iterator over all logical values in index order.
    ///
    /// Values in unallocated pages borrow the configured default value. The
    /// iterator does not materialize a dense temporary vector.
    #[must_use]
    pub fn iter(&self) -> Iter<'_, T> {
        Iter::new(self)
    }

    /// Returns concrete physical storage for an allocated page.
    ///
    /// `None` means that `page_index` is either outside the page table or that
    /// the page has no physical allocation. This method never fabricates a
    /// default-filled slice for an unallocated page.
    #[must_use]
    pub fn allocated_page(&self, page_index: usize) -> Option<&[T]> {
        self.pages
            .get(page_index)
            .and_then(Option::as_ref)
            .map(|page| page.values.as_ref())
    }

    /// Returns an iterator over physically allocated pages in ascending order.
    ///
    /// Each slice contains complete page storage and can include values equal
    /// to the configured default. Use [`Self::non_default_iter`] when logical
    /// non-default values are required instead.
    #[must_use]
    pub fn allocated_pages(&self) -> AllocatedPages<'_, T> {
        AllocatedPages::new(self)
    }

    /// Returns an iterator over indices of physically allocated pages in
    /// ascending order.
    #[must_use]
    pub fn allocated_page_indices(&self) -> AllocatedPageIndices<'_, T> {
        AllocatedPageIndices::new(self)
    }

    /// Materializes all logical values in index order.
    #[must_use]
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.iter().cloned().collect()
    }

    /// Consumes the vector and materializes all logical values in index order.
    ///
    /// Unallocated slots are represented by cloned default values.
    #[must_use]
    pub fn into_vec(self) -> Vec<T>
    where
        T: Clone,
    {
        self.iter().cloned().collect()
    }

    fn split_index(&self, index: usize) -> Option<(usize, usize)> {
        (index < self.len).then(|| (index / self.page_size(), index % self.page_size()))
    }

    fn checked_index(&self, index: usize) -> Result<(usize, usize), IndexOutOfBounds> {
        self.split_index(index).ok_or(IndexOutOfBounds {
            index,
            len: self.len,
        })
    }

    pub(crate) fn logical_page_len(&self, page_index: usize) -> Option<usize> {
        logical_page_len_for(self.len, self.page_size, page_index)
    }
}

impl<T: PartialEq> PagedVec<T> {
    /// Returns whether the logical value at `index` equals the configured
    /// default, or `None` when `index` is out of bounds.
    #[must_use]
    pub fn is_default(&self, index: usize) -> Option<bool> {
        self.get(index).map(|value| value == &self.default)
    }

    /// Returns an iterator over `(index, value)` pairs whose values differ from
    /// the configured default.
    ///
    /// Unallocated pages are skipped without inspecting synthetic default
    /// values. The iterator's exact length is [`Self::non_default_len`].
    #[must_use]
    pub fn non_default_iter(&self) -> NonDefaultIter<'_, T> {
        NonDefaultIter::new(self)
    }

    /// Returns whether any logical value equals `value`.
    #[must_use]
    pub fn contains(&self, value: &T) -> bool {
        self.iter().any(|candidate| candidate == value)
    }

    /// Builds a paged vector from dense logical values.
    ///
    /// Pages containing only values equal to `default` are not allocated. The
    /// resulting vector is canonical, including a shorter final page when the
    /// input length is not divisible by `page_size`.
    pub fn from_vec(values: Vec<T>, default: T, page_size: usize) -> Result<Self, PagedVecError> {
        let mut vector = Self::try_new(values.len(), default, page_size)?;
        let mut values = values.into_iter();

        for page_index in 0..vector.page_count() {
            let page_len = vector
                .logical_page_len(page_index)
                .expect("page index from page table must be valid");
            let page_values = values.by_ref().take(page_len).collect::<Vec<_>>();
            let page_non_default = page_values
                .iter()
                .filter(|value| *value != &vector.default)
                .count();

            if page_non_default != 0 {
                vector.pages[page_index] = Some(Page {
                    values: page_values.into_boxed_slice(),
                    non_default: page_non_default,
                });
                vector.non_default += page_non_default;
            }
        }

        vector.debug_assert_invariants();
        Ok(vector)
    }

    pub(crate) fn validate_invariants(&self) -> Result<(), &'static str> {
        if self.pages.len() != page_count_for(self.len, self.page_size) {
            return Err("page table length does not match logical length");
        }

        let mut total_non_default = 0usize;
        for (page_index, page) in self.pages.iter().enumerate() {
            let Some(page) = page else {
                continue;
            };

            let expected_len = self
                .logical_page_len(page_index)
                .ok_or("allocated page is outside the logical page table")?;
            if page.values.len() != expected_len {
                return Err("allocated page length does not match its logical length");
            }

            let actual_non_default = page
                .values
                .iter()
                .filter(|value| *value != &self.default)
                .count();
            if actual_non_default != page.non_default {
                return Err("allocated page non-default counter is incorrect");
            }
            if page.non_default == 0 {
                return Err("default-only page remains allocated");
            }
            total_non_default += page.non_default;
        }

        if total_non_default != self.non_default {
            return Err("global non-default counter is incorrect");
        }

        Ok(())
    }

    #[cfg(debug_assertions)]
    pub(crate) fn debug_assert_invariants(&self) {
        if let Err(error) = self.validate_invariants() {
            panic!("PagedVec invariant violation: {error}");
        }
    }

    #[cfg(not(debug_assertions))]
    pub(crate) fn debug_assert_invariants(&self) {}
}

impl<T: Clone + PartialEq> PagedVec<T> {
    /// Stores `value` at `index`.
    ///
    /// Assigning the default to an unallocated page is a no-op. Assigning the
    /// last non-default value in a page deallocates that page.
    pub fn set(&mut self, index: usize, value: T) -> Result<(), IndexOutOfBounds> {
        let (page_index, page_offset) = self.checked_index(index)?;
        self.set_at(page_index, page_offset, value);
        self.debug_assert_invariants();
        Ok(())
    }

    /// Restores the logical value at `index` to [`Self::default_value`].
    pub fn reset(&mut self, index: usize) -> Result<(), IndexOutOfBounds> {
        let (page_index, page_offset) = self.checked_index(index)?;
        self.set_at(page_index, page_offset, self.default.clone());
        self.debug_assert_invariants();
        Ok(())
    }

    /// Applies `f` to a detached copy of the logical value at `index` and
    /// commits the resulting value if the closure returns normally.
    ///
    /// This provides controlled mutation without exposing a mutable reference
    /// into page storage. If `f` panics, the vector is unchanged and therefore
    /// remains canonical.
    pub fn update<R>(
        &mut self,
        index: usize,
        f: impl FnOnce(&mut T) -> R,
    ) -> Result<R, IndexOutOfBounds> {
        self.checked_index(index)?;
        let mut value = self
            .get(index)
            .expect("checked index must be available")
            .clone();
        let result = f(&mut value);
        self.set(index, value)
            .expect("checked index must remain available");
        Ok(result)
    }

    fn set_at(&mut self, page_index: usize, page_offset: usize, value: T) {
        if value == self.default {
            let mut deallocate = false;
            if let Some(page) = self.pages[page_index].as_mut() {
                let was_non_default = page.values[page_offset] != self.default;
                if was_non_default {
                    page.values[page_offset] = value;
                    page.non_default -= 1;
                    self.non_default -= 1;
                }
                deallocate = page.non_default == 0;
            }
            if deallocate {
                self.pages[page_index] = None;
            }
            return;
        }

        if self.pages[page_index].is_none() {
            let page_len = self
                .logical_page_len(page_index)
                .expect("checked index must have a logical page");
            self.pages[page_index] = Some(Page::filled_with(&self.default, page_len));
        }

        let page = self.pages[page_index]
            .as_mut()
            .expect("allocated page must be present");
        let was_default = page.values[page_offset] == self.default;
        page.values[page_offset] = value;
        if was_default {
            page.non_default += 1;
            self.non_default += 1;
        }
    }
}

impl<T> Index<usize> for PagedVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).unwrap_or_else(|| {
            panic!(
                "index {index} out of bounds for PagedVec of length {}",
                self.len
            )
        })
    }
}

/// Equality compares logical length, configured default value, and every
/// logical value. It deliberately ignores page size and physical allocation
/// layout, so equivalent logical contents can compare equal across different
/// page layouts. Different configured defaults compare unequal even when all
/// current logical values match.
impl<T: PartialEq> PartialEq for PagedVec<T> {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len
            && self.default == other.default
            && (0..self.len).all(|index| self.get(index) == other.get(index))
    }
}

impl<T: Eq> Eq for PagedVec<T> {}

impl<'a, T> IntoIterator for &'a PagedVec<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub(crate) fn page_count_for(len: usize, page_size: NonZeroUsize) -> usize {
    let full_pages = len / page_size.get();
    if len.is_multiple_of(page_size.get()) {
        full_pages
    } else {
        full_pages + 1
    }
}

pub(crate) fn logical_page_len_for(
    len: usize,
    page_size: NonZeroUsize,
    page_index: usize,
) -> Option<usize> {
    let page_count = page_count_for(len, page_size);
    if page_index >= page_count {
        return None;
    }

    if page_index + 1 == page_count {
        let final_page_len = len % page_size.get();
        Some(if final_page_len == 0 {
            page_size.get()
        } else {
            final_page_len
        })
    } else {
        Some(page_size.get())
    }
}

#[cfg(test)]
mod tests {
    use std::iter::FusedIterator;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use proptest::prelude::*;

    use super::*;

    fn assert_matches(paged: &PagedVec<i32>, model: &[i32], default: i32) {
        assert_eq!(paged.len(), model.len());
        assert_eq!(paged.to_vec(), model);
        assert_eq!(paged.iter().copied().collect::<Vec<_>>(), model);
        assert_eq!(
            paged.non_default_len(),
            model.iter().filter(|value| **value != default).count()
        );

        let expected_non_default = model
            .iter()
            .enumerate()
            .filter(|(_, value)| **value != default)
            .map(|(index, value)| (index, *value))
            .collect::<Vec<_>>();
        assert_eq!(
            paged
                .non_default_iter()
                .map(|(index, value)| (index, *value))
                .collect::<Vec<_>>(),
            expected_non_default
        );

        let expected_allocated_page_indices = (0..paged.page_count())
            .filter(|page_index| {
                let start = page_index * paged.page_size();
                let end = (start + paged.page_size()).min(model.len());
                model[start..end].iter().any(|value| *value != default)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            paged.allocated_page_count(),
            expected_allocated_page_indices.len()
        );
        assert_eq!(
            paged.allocated_page_indices().collect::<Vec<_>>(),
            expected_allocated_page_indices
        );
        assert_eq!(
            paged
                .allocated_pages()
                .map(|(page_index, _)| page_index)
                .collect::<Vec<_>>(),
            expected_allocated_page_indices
        );
        assert_eq!(paged.validate_invariants(), Ok(()));
    }

    #[test]
    fn zero_length_vector_has_no_pages() {
        let paged = PagedVec::new(0, 0_i32, 4);
        assert!(paged.is_empty());
        assert_eq!(paged.page_count(), 0);
        assert_eq!(paged.get(0), None);
        assert_eq!(paged.allocated_page_count(), 0);
        assert_matches(&paged, &[], 0);
    }

    #[test]
    fn checked_constructor_rejects_zero_page_size() {
        assert!(matches!(
            PagedVec::try_new(1, 0_i32, 0),
            Err(PagedVecError::ZeroPageSize)
        ));
    }

    #[test]
    #[should_panic(expected = "page size greater than zero")]
    fn panicking_constructor_rejects_zero_page_size() {
        let _ = PagedVec::new(1, 0_i32, 0);
    }

    #[test]
    fn page_size_one_allocates_one_slot_pages() {
        let mut paged = PagedVec::new(3, 0_i32, 1);
        paged.set(1, 9).unwrap();
        assert_eq!(paged.page_count(), 3);
        assert_eq!(paged.allocated_page(1), Some(&[9][..]));
        assert_eq!(paged.allocated_page_count(), 1);
        assert_matches(&paged, &[0, 9, 0], 0);
    }

    #[test]
    fn final_page_has_only_its_logical_length() {
        let mut paged = PagedVec::new(1_025, 0_i32, 1_024);
        paged.set(1_024, 7).unwrap();
        assert_eq!(paged.page_count(), 2);
        assert_eq!(paged.allocated_page(1).unwrap().len(), 1);
        assert_eq!(paged.logical_page_len(1), Some(1));
        assert_matches(
            &paged,
            &vec![0; 1_024].into_iter().chain([7]).collect::<Vec<_>>(),
            0,
        );
    }

    #[test]
    fn exact_page_multiple_uses_full_final_page() {
        let mut paged = PagedVec::new(8, 0_i32, 4);
        paged.set(7, 3).unwrap();
        assert_eq!(paged.page_count(), 2);
        assert_eq!(paged.allocated_page(1).unwrap().len(), 4);
        assert_matches(&paged, &[0, 0, 0, 0, 0, 0, 0, 3], 0);
    }

    #[test]
    fn set_tracks_default_transitions_and_deallocates() {
        let mut paged = PagedVec::new(8, 0_i32, 4);
        paged.set(1, 0).unwrap();
        assert_eq!(paged.allocated_page_count(), 0);

        paged.set(1, 10).unwrap();
        assert_eq!(paged.non_default_len(), 1);
        assert_eq!(paged.allocated_page_count(), 1);

        paged.set(1, 11).unwrap();
        assert_eq!(paged.non_default_len(), 1);
        assert_eq!(paged[1], 11);

        paged.set(1, 0).unwrap();
        assert_eq!(paged.non_default_len(), 0);
        assert_eq!(paged.allocated_page_count(), 0);
        assert_matches(&paged, &[0; 8], 0);
    }

    #[test]
    fn pages_remain_allocated_until_their_last_non_default_value_is_reset() {
        let mut paged = PagedVec::new(8, 0_i32, 4);
        paged.set(0, 1).unwrap();
        paged.set(3, 2).unwrap();
        paged.set(4, 3).unwrap();
        assert_eq!(paged.allocated_page_count(), 2);
        assert_eq!(paged.non_default_len(), 3);

        paged.reset(0).unwrap();
        assert_eq!(paged.allocated_page_count(), 2);
        paged.reset(3).unwrap();
        assert_eq!(paged.allocated_page_count(), 1);
        paged.reset(4).unwrap();
        assert_eq!(paged.allocated_page_count(), 0);
        assert_matches(&paged, &[0; 8], 0);
    }

    #[test]
    fn get_and_default_queries_follow_collection_conventions() {
        let mut paged = PagedVec::new(2, 5_i32, 2);
        assert_eq!(paged.get(0), Some(&5));
        assert_eq!(paged.get(2), None);
        assert_eq!(paged.is_default(0), Some(true));
        assert_eq!(paged.is_default(2), None);

        paged.set(0, 7).unwrap();
        assert_eq!(paged.get(0), Some(&7));
        assert_eq!(paged.is_default(0), Some(false));
        assert_eq!(paged.default_value(), &5);
        assert_matches(&paged, &[7, 5], 5);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn indexing_panics_out_of_bounds() {
        let paged = PagedVec::new(1, 0_i32, 1);
        let _ = paged[1];
    }

    #[test]
    fn mutation_reports_out_of_bounds() {
        let mut paged = PagedVec::new(2, 0_i32, 1);
        let error = IndexOutOfBounds { index: 2, len: 2 };
        assert_eq!(paged.set(2, 1), Err(error));
        assert_eq!(paged.reset(2), Err(error));
        assert_eq!(paged.update(2, |_| ()), Err(error));
        assert_matches(&paged, &[0, 0], 0);
    }

    #[test]
    fn page_access_is_physical_only() {
        let mut paged = PagedVec::new(5, 0_i32, 4);
        assert_eq!(paged.allocated_page(0), None);
        assert_eq!(paged.allocated_page(2), None);
        paged.set(4, 1).unwrap();
        assert_eq!(paged.allocated_page(1), Some(&[1][..]));
        assert_eq!(paged.page_index(4), Some(1));
        assert_eq!(paged.page_offset(4), Some(0));
        assert_eq!(paged.page_index(5), None);
        assert_eq!(paged.page_offset(5), None);
        assert_matches(&paged, &[0, 0, 0, 0, 1], 0);
    }

    #[test]
    fn update_commits_changes_and_is_panic_safe() {
        let mut paged = PagedVec::new(3, 0_i32, 2);
        let result = paged.update(1, |value| {
            *value = 4;
            "updated"
        });
        assert_eq!(result, Ok("updated"));
        assert_matches(&paged, &[0, 4, 0], 0);

        let panic_result = catch_unwind(AssertUnwindSafe(|| {
            let _ = paged.update(2, |value| {
                *value = 9;
                panic!("closure failure");
            });
        }));
        assert!(panic_result.is_err());
        assert_matches(&paged, &[0, 4, 0], 0);

        paged.update(1, |value| *value = 0).unwrap();
        assert_matches(&paged, &[0, 0, 0], 0);
    }

    #[test]
    fn equality_compares_logical_values_and_default_but_not_page_size() {
        let mut small_pages = PagedVec::new(5, 0_i32, 2);
        let mut large_pages = PagedVec::new(5, 0_i32, 4);
        small_pages.set(4, 9).unwrap();
        large_pages.set(4, 9).unwrap();
        assert_eq!(small_pages, large_pages);
        assert_eq!(small_pages.clone(), small_pages);

        let mut different_default = PagedVec::new(5, 1_i32, 4);
        for index in 0..5 {
            different_default
                .set(index, if index == 4 { 9 } else { 0 })
                .unwrap();
        }
        assert_eq!(small_pages.to_vec(), different_default.to_vec());
        assert_ne!(small_pages, different_default);

        let mut changed_history = PagedVec::new(5, 0_i32, 2);
        changed_history.set(1, 8).unwrap();
        changed_history.reset(1).unwrap();
        let untouched = PagedVec::new(5, 0_i32, 2);
        assert_eq!(changed_history, untouched);
        assert_eq!(changed_history.allocated_page_count(), 0);
    }

    #[test]
    fn iterators_cover_logical_sparse_and_allocated_views() {
        fn assert_logical_traits<
            I: Iterator + DoubleEndedIterator + ExactSizeIterator + FusedIterator,
        >() {
        }
        fn assert_sparse_traits<I: Iterator + ExactSizeIterator + FusedIterator>() {}
        fn assert_allocated_traits<I: Iterator + DoubleEndedIterator + FusedIterator>() {}

        assert_logical_traits::<Iter<'_, i32>>();
        assert_sparse_traits::<NonDefaultIter<'_, i32>>();
        assert_allocated_traits::<AllocatedPages<'_, i32>>();
        assert_allocated_traits::<AllocatedPageIndices<'_, i32>>();

        let mut paged = PagedVec::new(6, 0_i32, 4);
        paged.set(1, 2).unwrap();
        paged.set(4, 3).unwrap();

        let mut logical = paged.iter();
        assert_eq!(logical.len(), 6);
        assert_eq!(logical.next(), Some(&0));
        assert_eq!(logical.next_back(), Some(&0));
        assert_eq!(logical.len(), 4);
        assert_eq!(logical.copied().collect::<Vec<_>>(), vec![2, 0, 0, 3]);

        assert_eq!(
            (&paged).into_iter().copied().collect::<Vec<_>>(),
            vec![0, 2, 0, 0, 3, 0]
        );
        assert_eq!(
            paged
                .non_default_iter()
                .map(|(index, value)| (index, *value))
                .collect::<Vec<_>>(),
            vec![(1, 2), (4, 3)]
        );
        assert_eq!(
            paged
                .allocated_pages()
                .map(|(index, values)| (index, values.to_vec()))
                .collect::<Vec<_>>(),
            vec![(0, vec![0, 2, 0, 0]), (1, vec![3, 0])]
        );
        assert_eq!(
            paged.allocated_page_indices().collect::<Vec<_>>(),
            vec![0, 1]
        );
    }

    #[test]
    fn conversions_and_allocation_inspection_remain_canonical() {
        let paged = PagedVec::from_vec(vec![0, 1, 0, 2, 0], 0_i32, 4).unwrap();
        assert_eq!(paged.to_vec(), vec![0, 1, 0, 2, 0]);
        assert_eq!(paged.non_default_len(), 2);
        assert_eq!(paged.allocated_page_indices().collect::<Vec<_>>(), vec![0]);
        assert_eq!(paged.allocated_page(0), Some(&[0, 1, 0, 2][..]));
        assert_eq!(paged.allocated_page(1), None);
        assert_eq!(paged.is_page_allocated(0), Some(true));
        assert_eq!(paged.is_page_allocated(1), Some(false));
        assert_eq!(paged.is_page_allocated(2), None);
        assert_eq!(paged.is_allocated(0), Some(true));
        assert_eq!(paged.is_allocated(4), Some(false));
        assert_eq!(paged.is_allocated(5), None);
        assert!(paged.contains(&0));
        assert!(paged.contains(&2));
        assert!(!paged.contains(&9));
        assert_eq!(paged.clone().into_vec(), vec![0, 1, 0, 2, 0]);
        assert_eq!(paged.validate_invariants(), Ok(()));
    }

    #[test]
    fn randomized_model_operations_preserve_logical_values_and_invariants() {
        fn next(state: &mut u64) -> u64 {
            *state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            *state
        }

        const DEFAULT: i32 = 0;
        for (len, page_size, seed) in [(1, 1, 1), (7, 3, 2), (16, 4, 3), (17, 16, 4)] {
            let mut paged = PagedVec::new(len, DEFAULT, page_size);
            let mut model = vec![DEFAULT; len];
            let mut state = seed;

            for _ in 0..2_000 {
                let index = (next(&mut state) as usize) % len;
                if next(&mut state) & 3 == 0 {
                    paged.reset(index).unwrap();
                    model[index] = DEFAULT;
                } else {
                    let value = (next(&mut state) % 11) as i32 - 5;
                    paged.set(index, value).unwrap();
                    model[index] = value;
                }
                assert_matches(&paged, &model, DEFAULT);
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn generated_set_and_reset_operations_match_a_dense_model(
            len in 0_usize..=48,
            page_size in 1_usize..=16,
            default in any::<i32>(),
            operations in prop::collection::vec((any::<bool>(), any::<usize>(), any::<i32>()), 0..96),
        ) {
            let mut paged = PagedVec::new(len, default, page_size);
            let mut model = vec![default; len];

            if len == 0 {
                prop_assert_eq!(paged.to_vec(), model.clone());
                prop_assert_eq!(paged.iter().copied().collect::<Vec<_>>(), model);
                prop_assert_eq!(paged.non_default_iter().count(), 0);
                prop_assert_eq!(paged.validate_invariants(), Ok(()));
                return Ok(());
            }

            for (reset, raw_index, value) in operations {
                let index = raw_index % len;
                if reset {
                    paged.reset(index).unwrap();
                    model[index] = default;
                } else {
                    paged.set(index, value).unwrap();
                    model[index] = value;
                }

                let expected_non_default = model
                    .iter()
                    .enumerate()
                    .filter(|(_, value)| **value != default)
                    .map(|(index, value)| (index, *value))
                    .collect::<Vec<_>>();
                prop_assert_eq!(paged.to_vec(), model.clone());
                prop_assert_eq!(paged.iter().copied().collect::<Vec<_>>(), model.clone());
                prop_assert_eq!(
                    paged.non_default_len(),
                    model.iter().filter(|value| **value != default).count()
                );
                prop_assert_eq!(
                    paged
                        .non_default_iter()
                        .map(|(index, value)| (index, *value))
                        .collect::<Vec<_>>(),
                    expected_non_default
                );
                prop_assert_eq!(paged.validate_invariants(), Ok(()));
            }
        }
    }
}
