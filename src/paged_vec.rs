use alloc::vec::Vec;
use core::num::NonZeroUsize;
use core::ops::Index;

use crate::error::{IndexOutOfBounds, PagedVecError};
use crate::iter::{AllocatedPageIndices, AllocatedPages, Iter, NonDefaultIter};
use crate::page::Page;

/// A vector whose logical contents use lazily allocated fixed-size pages.
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
/// Indexed lookup is O(1). [`Self::allocated_page_count`] is O(number of
/// logical pages) in the current dense page-table backend. Dynamic-length
/// operations can grow or shrink that dense table, and extending an allocated
/// partial final page clones its values and the configured default into
/// replacement storage.
///
/// Methods can propagate panics from `T`'s `Clone`, `PartialEq`, or `Drop`
/// implementations. Mutating methods commit their canonical metadata before
/// dropping detached page storage; method-specific transactional guarantees are
/// documented where a clone or comparison can occur before that commit.
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
    ///
    /// The resulting vector initially has no allocated data pages.
    pub fn try_new(len: usize, default: T, page_size: usize) -> Result<Self, PagedVecError> {
        let page_size = NonZeroUsize::new(page_size).ok_or(PagedVecError::ZeroPageSize)?;
        let page_count = page_count_for(len, page_size);
        let pages = core::iter::repeat_with(|| None).take(page_count).collect();

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
    /// This currently clones every logical value. In particular, it does not
    /// move values from allocated pages; unallocated slots are represented by
    /// cloned default values.
    #[must_use]
    pub fn into_vec(self) -> Vec<T>
    where
        T: Clone,
    {
        self.iter().cloned().collect()
    }

    /// Removes every logical value and releases all page storage.
    ///
    /// This preserves [`Self::page_size`] and [`Self::default_value`]. Use
    /// [`Self::reset_all`] to preserve the logical length instead.
    ///
    /// If dropping released page storage panics, the empty logical state has
    /// already been committed and remains canonical.
    pub fn clear(&mut self) {
        let pages = core::mem::take(&mut self.pages);
        self.len = 0;
        self.non_default = 0;
        self.debug_assert_empty_storage();
        drop(pages);
    }

    /// Restores every logical value to the configured default without changing
    /// the logical length.
    ///
    /// All physical pages are deallocated. Use [`Self::clear`] to set the
    /// logical length to zero instead.
    ///
    /// If dropping released page storage panics, the reset logical state has
    /// already been committed and remains canonical.
    pub fn reset_all(&mut self) {
        let pages = self.empty_page_table(self.len);
        let old_pages = core::mem::replace(&mut self.pages, pages);
        self.non_default = 0;
        self.debug_assert_empty_storage();
        drop(old_pages);
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

    fn empty_page_table(&self, len: usize) -> Vec<Option<Page<T>>> {
        core::iter::repeat_with(|| None)
            .take(page_count_for(len, self.page_size))
            .collect()
    }

    fn ensure_page_table_for_len(&mut self, len: usize) {
        let page_count = page_count_for(len, self.page_size);
        debug_assert!(page_count >= self.pages.len());
        self.pages
            .extend(core::iter::repeat_with(|| None).take(page_count - self.pages.len()));
    }

    #[cfg(debug_assertions)]
    fn debug_assert_empty_storage(&self) {
        debug_assert_eq!(self.pages.len(), page_count_for(self.len, self.page_size));
        debug_assert_eq!(self.non_default, 0);
        debug_assert!(self.pages.iter().all(Option::is_none));
    }

    #[cfg(not(debug_assertions))]
    fn debug_assert_empty_storage(&self) {}
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
    ///
    /// Returns [`PagedVecError::ZeroPageSize`] when `page_size` is zero.
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

    /// Shortens the vector to `new_len` logical values.
    ///
    /// This is a no-op when `new_len` is at least the current length. When
    /// shrinking, only the retained final page is inspected; counters for
    /// discarded whole pages are removed directly from their page metadata.
    /// Default-only retained final pages are deallocated.
    ///
    /// If dropping detached storage panics, the shortened logical state has
    /// already been committed and remains canonical.
    pub fn truncate(&mut self, new_len: usize) {
        if new_len >= self.len {
            return;
        }

        let new_page_count = page_count_for(new_len, self.page_size);
        let final_page_update = if new_page_count == 0 {
            None
        } else {
            let page_index = new_page_count - 1;
            let new_page_len = logical_page_len_for(new_len, self.page_size, page_index)
                .expect("final page index must be valid");
            self.pages[page_index].as_ref().and_then(|page| {
                (page.values.len() != new_page_len).then(|| {
                    let retained_non_default = page.values[..new_page_len]
                        .iter()
                        .filter(|value| *value != &self.default)
                        .count();
                    (
                        page_index,
                        new_page_len,
                        page.non_default,
                        retained_non_default,
                    )
                })
            })
        };

        let (discarded_final_values, retained_default_values) =
            if let Some((page_index, new_page_len, old_non_default, retained_non_default)) =
                final_page_update
            {
                let page = self.pages[page_index]
                    .take()
                    .expect("allocated final page must still be present");
                let mut values = page.values.into_vec();
                let discarded_values = values.split_off(new_page_len);
                self.non_default -= old_non_default;
                self.non_default += retained_non_default;
                if retained_non_default != 0 {
                    self.pages[page_index] = Some(Page {
                        values: values.into_boxed_slice(),
                        non_default: retained_non_default,
                    });
                    (Some(discarded_values), None)
                } else {
                    // Keep these values alive until the logical state has
                    // been committed. Their destructors may panic.
                    (Some(discarded_values), Some(values))
                }
            } else {
                (None, None)
            };

        let removed_pages = self.pages.split_off(new_page_count);
        let removed_non_default = removed_pages
            .iter()
            .filter_map(Option::as_ref)
            .map(|page| page.non_default)
            .sum::<usize>();
        self.non_default -= removed_non_default;
        self.len = new_len;
        self.debug_assert_invariants();
        drop(removed_pages);
        drop(discarded_final_values);
        drop(retained_default_values);
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
    ///
    /// Returns [`IndexOutOfBounds`] when `index` is outside the logical
    /// vector. If replacing a stored value causes `T::Drop` to panic, the new
    /// canonical page state and counters have already been committed.
    pub fn set(&mut self, index: usize, value: T) -> Result<(), IndexOutOfBounds> {
        let (page_index, page_offset) = self.checked_index(index)?;
        self.set_at(page_index, page_offset, value);
        self.debug_assert_invariants();
        Ok(())
    }

    /// Restores the logical value at `index` to [`Self::default_value`].
    ///
    /// Returns [`IndexOutOfBounds`] when `index` is outside the logical
    /// vector. If cloning the configured default panics, the vector is
    /// unchanged.
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
    ///
    /// Returns [`IndexOutOfBounds`] when `index` is outside the logical
    /// vector. Cloning the detached value can also panic before `f` runs, in
    /// which case the vector is unchanged.
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

    /// Changes the logical length, using the configured default for new slots.
    ///
    /// Shrinking is identical to [`Self::truncate`]. Growing does not allocate
    /// pages for the new default-valued slots. If an allocated partial final
    /// page remains in the vector, its physical storage is rebuilt with cloned
    /// default values before the new length is committed.
    ///
    /// Unlike [`Vec::resize`], this method has no fill-value argument because
    /// a `PagedVec` has one configured logical default value.
    ///
    /// If cloning while reconstructing an allocated partial final page panics,
    /// the vector is unchanged. If dropping replaced final-page storage panics,
    /// the grown canonical state has already been committed.
    pub fn resize(&mut self, new_len: usize) {
        if new_len < self.len {
            self.truncate(new_len);
            return;
        }
        if new_len == self.len {
            return;
        }

        let final_page_replacement = if self.len != 0
            && !is_exact_multiple(self.len, self.page_size())
        {
            let page_index = self.page_count() - 1;
            self.pages[page_index].as_ref().map(|page| {
                let new_page_len = (new_len - page_index * self.page_size()).min(self.page_size());
                self.extended_page_with_defaults(page, new_page_len)
            })
        } else {
            None
        };

        self.ensure_page_table_for_len(new_len);
        let replaced_final_page = if let Some(page) = final_page_replacement {
            let page_index = self.page_index(self.len - 1).expect("non-empty vector");
            self.pages[page_index].replace(page)
        } else {
            None
        };
        self.len = new_len;
        self.debug_assert_invariants();
        drop(replaced_final_page);
    }

    /// Appends `value` to the end of the vector.
    ///
    /// Appending a default value to an unallocated page only grows metadata;
    /// no physical page is allocated. Page-table growth, default cloning, and
    /// allocation can make this more expensive than constant time.
    ///
    /// If cloning needed by growth panics, the vector is unchanged. If a later
    /// operation while storing `value` panics, the vector remains canonical,
    /// but can retain the newly appended default-valued slot.
    ///
    /// # Panics
    ///
    /// Panics if the new logical length would exceed [`usize::MAX`]. Panics
    /// from `T`'s `Clone`, `PartialEq`, or `Drop` implementations propagate
    /// with the transactional guarantees described above.
    pub fn push(&mut self, value: T) {
        let index = self.len;
        let new_len = self.len.checked_add(1).expect("PagedVec length overflow");
        self.resize(new_len);
        self.set(index, value)
            .expect("newly appended index must be valid");
        self.debug_assert_invariants();
    }

    /// Removes and returns the last logical value, or `None` when empty.
    ///
    /// Removing an unallocated slot returns a clone of the configured default.
    /// Concrete values are moved out of allocated pages.
    ///
    /// If cloning the configured default panics for an unallocated final slot,
    /// the vector is unchanged.
    pub fn pop(&mut self) -> Option<T> {
        let old_len = self.len;
        let index = old_len.checked_sub(1)?;
        let (page_index, page_offset) = self
            .split_index(index)
            .expect("last logical index must be valid");

        if self.pages[page_index].is_none() {
            let value = self.default.clone();
            self.truncate(index);
            return Some(value);
        }

        let was_non_default = self.pages[page_index]
            .as_ref()
            .expect("allocated page must be present")
            .values[page_offset]
            != self.default;
        let mut page = self.pages[page_index]
            .take()
            .expect("allocated page must be present");
        let mut values = page.values.into_vec();
        let value = values.pop().expect("allocated page must contain last slot");
        let new_len = index;
        let new_page_count = page_count_for(new_len, self.page_size);

        if new_page_count < self.pages.len() {
            self.non_default -= page.non_default;
            let removed_page = self.pages.pop();
            debug_assert!(removed_page.is_some());
        } else {
            page.non_default -= usize::from(was_non_default);
            self.non_default -= usize::from(was_non_default);
            if page.non_default != 0 {
                page.values = values.into_boxed_slice();
                self.pages[page_index] = Some(page);
            }
        }

        self.len = new_len;
        self.debug_assert_invariants();
        Some(value)
    }

    fn extended_page_with_defaults(&self, page: &Page<T>, new_len: usize) -> Page<T> {
        debug_assert!(page.values.len() < new_len);
        let mut values = Vec::with_capacity(new_len);
        values.extend(page.values.iter().cloned());
        values.extend(
            core::iter::repeat_with(|| self.default.clone()).take(new_len - page.values.len()),
        );
        Page {
            values: values.into_boxed_slice(),
            non_default: page.non_default,
        }
    }

    fn set_at(&mut self, page_index: usize, page_offset: usize, value: T) {
        if value == self.default {
            let was_non_default = self.pages[page_index]
                .as_ref()
                .is_some_and(|page| page.values[page_offset] != self.default);
            if !was_non_default {
                return;
            }

            let mut page = self.pages[page_index]
                .take()
                .expect("allocated page must be present");
            let old_value = core::mem::replace(&mut page.values[page_offset], value);
            page.non_default -= 1;
            self.non_default -= 1;
            if page.non_default != 0 {
                self.pages[page_index] = Some(page);
            }

            // Drop after the canonical page and counters have been committed.
            drop(old_value);
            return;
        }

        let was_default = self.pages[page_index]
            .as_ref()
            .is_none_or(|page| page.values[page_offset] == self.default);
        let mut page = if let Some(page) = self.pages[page_index].take() {
            page
        } else {
            let page_len = self
                .logical_page_len(page_index)
                .expect("checked index must have a logical page");
            Page::filled_with(&self.default, page_len)
        };

        let old_value = core::mem::replace(&mut page.values[page_offset], value);
        if was_default {
            page.non_default += 1;
            self.non_default += 1;
        }
        self.pages[page_index] = Some(page);

        // Drop after the canonical page and counters have been committed.
        drop(old_value);
    }
}

/// Indexing panics when `index` is outside the logical vector.
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

/// Extends by repeatedly calling [`PagedVec::push`]. If iteration or handling
/// an element panics, elements appended before the panic remain present in a
/// canonical state.
impl<T: Clone + PartialEq> Extend<T> for PagedVec<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for value in iter {
            self.push(value);
        }
    }
}

pub(crate) fn page_count_for(len: usize, page_size: NonZeroUsize) -> usize {
    let full_pages = len / page_size.get();
    if is_exact_multiple(len, page_size.get()) {
        full_pages
    } else {
        full_pages + 1
    }
}

#[allow(
    clippy::manual_is_multiple_of,
    reason = "usize::is_multiple_of is unstable on the declared Rust 1.85 MSRV"
)]
fn is_exact_multiple(value: usize, divisor: usize) -> bool {
    value % divisor == 0
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
    use std::cell::Cell;
    use std::iter::FusedIterator;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::rc::Rc;
    use std::vec;

    use proptest::prelude::*;

    use super::*;

    fn assert_matches(paged: &PagedVec<i32>, model: &[i32], default: i32) {
        assert_eq!(paged.len(), model.len());
        assert_eq!(paged.is_empty(), model.is_empty());
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

    #[derive(Debug)]
    struct PanicClone {
        value: i32,
        panic_on_clone: Rc<Cell<bool>>,
    }

    impl Clone for PanicClone {
        fn clone(&self) -> Self {
            assert!(!self.panic_on_clone.get(), "clone failed");
            Self {
                value: self.value,
                panic_on_clone: Rc::clone(&self.panic_on_clone),
            }
        }
    }

    impl PartialEq for PanicClone {
        fn eq(&self, other: &Self) -> bool {
            self.value == other.value
        }
    }

    #[derive(Clone, Debug)]
    struct PanicEq {
        value: i32,
        panic_on_eq: Rc<Cell<bool>>,
    }

    impl PartialEq for PanicEq {
        fn eq(&self, other: &Self) -> bool {
            assert!(!self.panic_on_eq.get(), "comparison failed");
            self.value == other.value
        }
    }

    #[derive(Debug)]
    struct PanicDrop {
        value: i32,
        panic_on_drop: Rc<Cell<bool>>,
    }

    impl Clone for PanicDrop {
        fn clone(&self) -> Self {
            Self {
                value: self.value,
                panic_on_drop: Rc::clone(&self.panic_on_drop),
            }
        }
    }

    impl PartialEq for PanicDrop {
        fn eq(&self, other: &Self) -> bool {
            self.value == other.value
        }
    }

    impl Drop for PanicDrop {
        fn drop(&mut self) {
            assert!(!self.panic_on_drop.replace(false), "drop failed");
        }
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
    fn iterator_lengths_and_fused_behavior_survive_dynamic_changes() {
        let mut paged = PagedVec::from_vec(vec![5, 1, 5, 2, 3], 5_i32, 4).unwrap();

        let mut logical = paged.iter();
        assert_eq!(logical.len(), 5);
        assert_eq!(logical.next_back(), Some(&3));
        assert_eq!(logical.len(), 4);
        assert_eq!(logical.next(), Some(&5));
        assert_eq!(logical.next(), Some(&1));
        assert_eq!(logical.next_back(), Some(&2));
        assert_eq!(logical.next(), Some(&5));
        assert_eq!(logical.next(), None);
        assert_eq!(logical.next_back(), None);

        let mut sparse = paged.non_default_iter();
        assert_eq!(sparse.len(), 3);
        assert_eq!(
            sparse.next().map(|(index, value)| (index, *value)),
            Some((1, 1))
        );
        assert_eq!(sparse.len(), 2);
        assert_eq!(
            sparse.next().map(|(index, value)| (index, *value)),
            Some((3, 2))
        );
        assert_eq!(sparse.len(), 1);
        assert_eq!(
            sparse.next().map(|(index, value)| (index, *value)),
            Some((4, 3))
        );
        assert_eq!(sparse.len(), 0);
        assert_eq!(sparse.next(), None);
        assert_eq!(sparse.next(), None);

        let mut pages = paged.allocated_pages();
        assert_eq!(pages.size_hint(), (0, Some(2)));
        assert_eq!(pages.next().map(|(index, _)| index), Some(0));
        assert_eq!(pages.size_hint(), (0, Some(1)));
        assert_eq!(pages.next_back().map(|(index, _)| index), Some(1));
        assert_eq!(pages.next(), None);
        assert_eq!(pages.next_back(), None);

        let mut indices = paged.allocated_page_indices();
        assert_eq!(indices.size_hint(), (0, Some(2)));
        assert_eq!(indices.next_back(), Some(1));
        assert_eq!(indices.size_hint(), (0, Some(1)));
        assert_eq!(indices.next(), Some(0));
        assert_eq!(indices.next(), None);
        assert_eq!(indices.next_back(), None);

        paged.reset_all();
        paged.push(7);
        assert_eq!(
            paged.iter().copied().collect::<Vec<_>>(),
            vec![5, 5, 5, 5, 5, 7]
        );
        assert_eq!(
            paged
                .non_default_iter()
                .map(|(index, value)| (index, *value))
                .collect::<Vec<_>>(),
            vec![(5, 7)]
        );
    }

    #[test]
    fn dynamic_boundaries_preserve_logical_and_physical_views() {
        let mut paged = PagedVec::from_vec(vec![9, 1, 9, 9], 9_i32, 4).unwrap();
        assert_eq!(paged.pop(), Some(9));
        assert_eq!(paged.allocated_page(0), Some(&[9, 1, 9][..]));
        assert_matches(&paged, &[9, 1, 9], 9);

        paged.resize(2);
        assert_eq!(paged.allocated_page(0), Some(&[9, 1][..]));
        paged.resize(3);
        assert_eq!(paged.allocated_page(0), Some(&[9, 1, 9][..]));
        paged.resize(4);
        assert_eq!(paged.allocated_page(0), Some(&[9, 1, 9, 9][..]));
        assert_matches(&paged, &[9, 1, 9, 9], 9);

        paged.truncate(2);
        paged.resize(5);
        assert_matches(&paged, &[9, 1, 9, 9, 9], 9);
        assert_eq!(paged.allocated_page_indices().collect::<Vec<_>>(), vec![0]);

        let mut all_default_retained = PagedVec::from_vec(vec![9, 9, 2], 9_i32, 4).unwrap();
        all_default_retained.truncate(2);
        assert_matches(&all_default_retained, &[9, 9], 9);
        assert_eq!(all_default_retained.allocated_page_count(), 0);

        paged.clear();
        paged.resize(3);
        assert_matches(&paged, &[9, 9, 9], 9);
        paged.reset_all();
        paged.truncate(1);
        assert_matches(&paged, &[9], 9);

        let mut alternating = PagedVec::new(0, 9_i32, 1);
        alternating.push(9);
        alternating.push(1);
        assert_eq!(alternating.pop(), Some(1));
        alternating.push(2);
        assert_matches(&alternating, &[9, 2], 9);
        assert!(alternating.contains(&9));
        assert!(alternating.contains(&2));
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
    fn push_handles_empty_partial_and_boundary_pages() {
        let mut paged = PagedVec::new(0, 0_i32, 2);
        paged.push(0);
        assert_matches(&paged, &[0], 0);
        assert_eq!(paged.allocated_page_count(), 0);

        paged.push(1);
        assert_matches(&paged, &[0, 1], 0);
        assert_eq!(paged.allocated_page(0), Some(&[0, 1][..]));

        paged.push(2);
        assert_matches(&paged, &[0, 1, 2], 0);
        assert_eq!(paged.allocated_page(1), Some(&[2][..]));

        paged.push(0);
        assert_matches(&paged, &[0, 1, 2, 0], 0);
        assert_eq!(paged.allocated_page(1), Some(&[2, 0][..]));

        let mut page_size_one = PagedVec::new(0, 0_i32, 1);
        page_size_one.push(0);
        page_size_one.push(7);
        assert_matches(&page_size_one, &[0, 7], 0);
        assert_eq!(
            page_size_one.allocated_page_indices().collect::<Vec<_>>(),
            vec![1]
        );
    }

    #[test]
    fn pop_moves_values_and_reclaims_pages() {
        let mut paged = PagedVec::from_vec(vec![0, 1, 0, 2, 0], 0_i32, 4).unwrap();
        assert_eq!(paged.pop(), Some(0));
        assert_matches(&paged, &[0, 1, 0, 2], 0);
        assert_eq!(paged.page_count(), 1);

        assert_eq!(paged.pop(), Some(2));
        assert_matches(&paged, &[0, 1, 0], 0);
        assert_eq!(paged.allocated_page(0), Some(&[0, 1, 0][..]));

        assert_eq!(paged.pop(), Some(0));
        assert_eq!(paged.pop(), Some(1));
        assert_eq!(paged.pop(), Some(0));
        assert_eq!(paged.pop(), None);
        assert_matches(&paged, &[], 0);
    }

    #[test]
    fn resize_and_truncate_preserve_partial_page_invariants() {
        let mut paged = PagedVec::new(0, 0_i32, 4);
        paged.resize(9);
        assert_matches(&paged, &[0; 9], 0);
        assert_eq!(paged.allocated_page_count(), 0);

        paged.truncate(12);
        assert_matches(&paged, &[0; 9], 0);
        paged.set(1, 7).unwrap();
        paged.resize(13);
        assert_eq!(paged.allocated_page(0), Some(&[0, 7, 0, 0][..]));
        assert_matches(&paged, &[0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0);

        paged.truncate(3);
        assert_eq!(paged.allocated_page(0), Some(&[0, 7, 0][..]));
        assert_matches(&paged, &[0, 7, 0], 0);

        paged.resize(1);
        assert_matches(&paged, &[0], 0);
        assert_eq!(paged.allocated_page_count(), 0);
        paged.resize(1);
        paged.resize(6);
        assert_matches(&paged, &[0; 6], 0);
        paged.truncate(0);
        assert_matches(&paged, &[], 0);
    }

    #[test]
    fn clear_reset_all_and_extend_preserve_the_default_configuration() {
        let mut paged = PagedVec::from_vec(vec![5, 7, 5, 9, 5], 5_i32, 4).unwrap();
        paged.reset_all();
        assert_matches(&paged, &[5; 5], 5);
        assert_eq!(paged.page_size(), 4);
        assert_eq!(paged.default_value(), &5);

        paged.extend([5, 8, 5, 6]);
        assert_matches(&paged, &[5, 5, 5, 5, 5, 5, 8, 5, 6], 5);
        paged.clear();
        assert_matches(&paged, &[], 5);
        assert_eq!(paged.page_size(), 4);
        assert_eq!(paged.default_value(), &5);

        paged.push(11);
        assert_matches(&paged, &[11], 5);
        paged.extend(std::iter::empty());
        assert_matches(&paged, &[11], 5);
    }

    #[test]
    fn resize_clone_panic_leaves_the_vector_canonical() {
        let panic_on_clone = Rc::new(Cell::new(false));
        let default = PanicClone {
            value: 0,
            panic_on_clone: Rc::clone(&panic_on_clone),
        };
        let mut paged = PagedVec::new(1, default.clone(), 4);
        paged
            .set(
                0,
                PanicClone {
                    value: 1,
                    panic_on_clone: Rc::clone(&panic_on_clone),
                },
            )
            .unwrap();

        panic_on_clone.set(true);
        let result = catch_unwind(AssertUnwindSafe(|| paged.resize(2)));
        assert!(result.is_err());
        panic_on_clone.set(false);
        assert_eq!(paged.len(), 1);
        assert_eq!(paged.non_default_len(), 1);
        assert_eq!(paged.allocated_page(0).unwrap().len(), 1);
        assert_eq!(paged.validate_invariants(), Ok(()));
    }

    #[test]
    fn clone_panics_preserve_documented_dynamic_states() {
        let panic_on_clone = Rc::new(Cell::new(false));
        let default = PanicClone {
            value: 0,
            panic_on_clone: Rc::clone(&panic_on_clone),
        };

        let mut resize = PagedVec::new(1, default.clone(), 4);
        resize
            .set(
                0,
                PanicClone {
                    value: 1,
                    panic_on_clone: Rc::clone(&panic_on_clone),
                },
            )
            .unwrap();
        panic_on_clone.set(true);
        assert!(catch_unwind(AssertUnwindSafe(|| resize.resize(2))).is_err());
        panic_on_clone.set(false);
        assert_eq!(resize.len(), 1);
        assert_eq!(resize.get(0).unwrap().value, 1);
        assert_eq!(resize.validate_invariants(), Ok(()));

        let mut push = PagedVec::new(1, default.clone(), 4);
        push.set(
            0,
            PanicClone {
                value: 1,
                panic_on_clone: Rc::clone(&panic_on_clone),
            },
        )
        .unwrap();
        panic_on_clone.set(true);
        assert!(catch_unwind(AssertUnwindSafe(|| {
            push.push(PanicClone {
                value: 2,
                panic_on_clone: Rc::clone(&panic_on_clone),
            });
        }))
        .is_err());
        panic_on_clone.set(false);
        assert_eq!(push.len(), 1);
        assert_eq!(push.get(0).unwrap().value, 1);
        assert_eq!(push.validate_invariants(), Ok(()));

        let mut pop = PagedVec::new(1, default.clone(), 4);
        panic_on_clone.set(true);
        assert!(catch_unwind(AssertUnwindSafe(|| pop.pop())).is_err());
        panic_on_clone.set(false);
        assert_eq!(pop.len(), 1);
        assert_eq!(pop.non_default_len(), 0);
        assert_eq!(pop.validate_invariants(), Ok(()));

        let mut reset = PagedVec::new(1, default.clone(), 4);
        reset
            .set(
                0,
                PanicClone {
                    value: 1,
                    panic_on_clone: Rc::clone(&panic_on_clone),
                },
            )
            .unwrap();
        panic_on_clone.set(true);
        assert!(catch_unwind(AssertUnwindSafe(|| reset.reset(0))).is_err());
        assert_eq!(reset.get(0).unwrap().value, 1);

        reset.reset_all();
        assert_eq!(reset.len(), 1);
        assert_eq!(reset.non_default_len(), 0);
        panic_on_clone.set(false);
        assert_eq!(reset.validate_invariants(), Ok(()));
    }

    #[test]
    fn comparison_panics_do_not_expose_noncanonical_storage() {
        let panic_on_eq = Rc::new(Cell::new(false));
        let default = PanicEq {
            value: 0,
            panic_on_eq: Rc::clone(&panic_on_eq),
        };

        let mut set = PagedVec::new(2, default.clone(), 4);
        panic_on_eq.set(true);
        assert!(catch_unwind(AssertUnwindSafe(|| {
            let _ = set.set(
                0,
                PanicEq {
                    value: 1,
                    panic_on_eq: Rc::clone(&panic_on_eq),
                },
            );
        }))
        .is_err());
        panic_on_eq.set(false);
        assert_eq!(set.non_default_len(), 0);
        assert_eq!(set.validate_invariants(), Ok(()));

        let mut push = PagedVec::new(0, default.clone(), 4);
        panic_on_eq.set(true);
        assert!(catch_unwind(AssertUnwindSafe(|| {
            push.push(PanicEq {
                value: 1,
                panic_on_eq: Rc::clone(&panic_on_eq),
            });
        }))
        .is_err());
        panic_on_eq.set(false);
        assert_eq!(push.len(), 1);
        assert_eq!(push.non_default_len(), 0);
        assert_eq!(push.validate_invariants(), Ok(()));

        let mut truncate = PagedVec::new(2, default.clone(), 4);
        truncate
            .set(
                1,
                PanicEq {
                    value: 1,
                    panic_on_eq: Rc::clone(&panic_on_eq),
                },
            )
            .unwrap();
        panic_on_eq.set(true);
        assert!(catch_unwind(AssertUnwindSafe(|| truncate.truncate(1))).is_err());
        panic_on_eq.set(false);
        assert_eq!(truncate.len(), 2);
        assert_eq!(truncate.non_default_len(), 1);
        assert_eq!(truncate.validate_invariants(), Ok(()));

        panic_on_eq.set(true);
        assert!(catch_unwind(AssertUnwindSafe(|| {
            let _ = PagedVec::from_vec(
                vec![
                    PanicEq {
                        value: 0,
                        panic_on_eq: Rc::clone(&panic_on_eq),
                    },
                    PanicEq {
                        value: 1,
                        panic_on_eq: Rc::clone(&panic_on_eq),
                    },
                ],
                default.clone(),
                4,
            );
        }))
        .is_err());

        let mut iterator = truncate.non_default_iter();
        assert!(catch_unwind(AssertUnwindSafe(|| iterator.next())).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| truncate.contains(&default))).is_err());
        panic_on_eq.set(false);
        assert_eq!(truncate.validate_invariants(), Ok(()));
    }

    #[test]
    fn drop_panics_after_detaching_storage_leave_canonical_state() {
        fn value(value: i32, panic_on_drop: &Rc<Cell<bool>>) -> PanicDrop {
            PanicDrop {
                value,
                panic_on_drop: Rc::clone(panic_on_drop),
            }
        }

        let panic_on_drop = Rc::new(Cell::new(false));

        let mut clear = PagedVec::new(1, value(0, &panic_on_drop), 4);
        clear.set(0, value(1, &panic_on_drop)).unwrap();
        panic_on_drop.set(true);
        assert!(catch_unwind(AssertUnwindSafe(|| clear.clear())).is_err());
        assert_eq!(clear.len(), 0);
        assert_eq!(clear.page_count(), 0);
        assert_eq!(clear.validate_invariants(), Ok(()));

        let mut set = PagedVec::new(1, value(0, &panic_on_drop), 4);
        set.set(0, value(1, &panic_on_drop)).unwrap();
        panic_on_drop.set(true);
        assert!(catch_unwind(AssertUnwindSafe(|| {
            let _ = set.set(0, value(0, &panic_on_drop));
        }))
        .is_err());
        assert_eq!(set.len(), 1);
        assert_eq!(set.non_default_len(), 0);
        assert_eq!(set.allocated_page_count(), 0);
        assert_eq!(set.validate_invariants(), Ok(()));

        let mut resize = PagedVec::new(1, value(0, &panic_on_drop), 4);
        resize.set(0, value(1, &panic_on_drop)).unwrap();
        panic_on_drop.set(true);
        assert!(catch_unwind(AssertUnwindSafe(|| resize.resize(2))).is_err());
        assert_eq!(resize.len(), 2);
        assert_eq!(resize.non_default_len(), 1);
        assert_eq!(resize.allocated_page(0).unwrap().len(), 2);
        assert_eq!(resize.validate_invariants(), Ok(()));

        let mut reset_all = PagedVec::new(1, value(0, &panic_on_drop), 4);
        reset_all.set(0, value(1, &panic_on_drop)).unwrap();
        panic_on_drop.set(true);
        assert!(catch_unwind(AssertUnwindSafe(|| reset_all.reset_all())).is_err());
        assert_eq!(reset_all.len(), 1);
        assert_eq!(reset_all.non_default_len(), 0);
        assert_eq!(reset_all.validate_invariants(), Ok(()));

        let mut truncate = PagedVec::new(2, value(0, &panic_on_drop), 4);
        truncate.set(1, value(1, &panic_on_drop)).unwrap();
        panic_on_drop.set(true);
        assert!(catch_unwind(AssertUnwindSafe(|| truncate.truncate(1))).is_err());
        assert_eq!(truncate.len(), 1);
        assert_eq!(truncate.non_default_len(), 0);
        assert_eq!(truncate.allocated_page_count(), 0);
        assert_eq!(truncate.validate_invariants(), Ok(()));
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

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn generated_dynamic_operations_match_a_dense_model(
            initial_len in 0_usize..=24,
            page_size in 1_usize..=12,
            default in any::<i32>(),
            operations in prop::collection::vec(
                (
                    0_u8..9,
                    any::<usize>(),
                    any::<i32>(),
                    prop::collection::vec(any::<i32>(), 0..8),
                ),
                0..80,
            ),
        ) {
            let mut paged = PagedVec::new(initial_len, default, page_size);
            let mut model = vec![default; initial_len];

            for (operation, raw_index, value, extension) in operations {
                match operation {
                    0 if !model.is_empty() => {
                        let index = raw_index % model.len();
                        paged.set(index, value).unwrap();
                        model[index] = value;
                    }
                    1 if !model.is_empty() => {
                        let index = raw_index % model.len();
                        paged.reset(index).unwrap();
                        model[index] = default;
                    }
                    2 => {
                        paged.push(value);
                        model.push(value);
                    }
                    3 => prop_assert_eq!(paged.pop(), model.pop()),
                    4 => {
                        let new_len = raw_index % 48;
                        paged.resize(new_len);
                        model.resize(new_len, default);
                    }
                    5 => {
                        let new_len = raw_index % 48;
                        paged.truncate(new_len);
                        model.truncate(new_len);
                    }
                    6 => {
                        paged.clear();
                        model.clear();
                    }
                    7 => {
                        paged.reset_all();
                        model.fill(default);
                    }
                    8 => {
                        paged.extend(extension.clone());
                        model.extend(extension);
                    }
                    _ => {}
                }

                prop_assert_eq!(paged.len(), model.len());
                prop_assert_eq!(paged.is_empty(), model.is_empty());
                prop_assert_eq!(paged.to_vec(), model.clone());
                prop_assert_eq!(paged.iter().copied().collect::<Vec<_>>(), model.clone());
                prop_assert_eq!(
                    paged.non_default_len(),
                    model.iter().filter(|value| **value != default).count()
                );
                let expected_non_default = model
                    .iter()
                    .enumerate()
                    .filter(|(_, value)| **value != default)
                    .map(|(index, value)| (index, *value))
                    .collect::<Vec<_>>();
                prop_assert_eq!(
                    paged
                        .non_default_iter()
                        .map(|(index, value)| (index, *value))
                        .collect::<Vec<_>>(),
                    expected_non_default
                );
                let expected_allocated_pages = (0..paged.page_count())
                    .filter(|page_index| {
                        let start = page_index * paged.page_size();
                        let end = (start + paged.page_size()).min(model.len());
                        model[start..end].iter().any(|value| *value != default)
                    })
                    .map(|page_index| {
                        let start = page_index * paged.page_size();
                        let end = (start + paged.page_size()).min(model.len());
                        (page_index, model[start..end].to_vec())
                    })
                    .collect::<Vec<_>>();
                let expected_allocated_page_count = expected_allocated_pages.len();
                prop_assert_eq!(
                    paged.allocated_page_indices().collect::<Vec<_>>(),
                    expected_allocated_pages
                        .iter()
                        .map(|(page_index, _)| *page_index)
                        .collect::<Vec<_>>()
                );
                prop_assert_eq!(
                    paged
                        .allocated_pages()
                        .map(|(page_index, page)| (page_index, page.to_vec()))
                        .collect::<Vec<_>>(),
                    expected_allocated_pages
                );
                prop_assert_eq!(
                    paged.allocated_page_count(),
                    expected_allocated_page_count
                );
                prop_assert_eq!(paged.contains(&default), model.contains(&default));
                prop_assert_eq!(paged.contains(&value), model.contains(&value));
                prop_assert_eq!(paged.validate_invariants(), Ok(()));
            }
        }
    }
}
