#![forbid(unsafe_code)]
//! A vector whose logical contents use lazily allocated fixed-size pages.
//!
//! [`PagedVec`] has a logical value at every index below [`PagedVec::len`], but
//! only allocates physical storage for a page after that page receives a value
//! different from the configured [`PagedVec::default_value`]. Unallocated pages
//! logically contain the default value.
//!
//! The collection is **not contiguous**. It deliberately does not implement
//! slice conversion, range indexing, or mutable indexing. Those APIs would
//! either misrepresent its layout or allow callers to bypass the bookkeeping
//! required to reclaim default-only pages. Use [`PagedVec::set`],
//! [`PagedVec::reset`], or [`PagedVec::update`] for mutation.
//!
//! [`PagedVec::iter`] visits every logical value, including defaults supplied
//! by unallocated pages. [`PagedVec::non_default_iter`] visits only logical
//! values that differ from the default, while [`PagedVec::allocated_pages`]
//! visits only physical page storage. These are deliberately different views:
//! an allocated page can contain default-valued slots.
//!
//! The logical length can change through [`PagedVec::push`], [`PagedVec::pop`],
//! [`PagedVec::resize`], [`PagedVec::truncate`], and [`PagedVec::clear`].
//! Growing uses the configured default value, so it usually adds only dense
//! page-table metadata rather than physical data pages. [`PagedVec::reset_all`]
//! instead preserves the length while restoring every slot to that default.
//!
//! The current backend uses a dense `Vec<Option<Page<T>>>` page table. Page
//! data is allocated lazily, but page-table metadata is proportional to
//! `ceil(len / page_size)`. It is therefore best suited to workloads where
//! page data dominates page-table overhead; it does not use constant memory
//! for an enormous logical length.
//!
//! # Example
//!
//! ```
//! use pagedvector::PagedVec;
//!
//! let mut values = PagedVec::new(1_000, 0_u32, 64);
//! assert_eq!(values.get(42), Some(&0));
//! assert_eq!(values.allocated_page_count(), 0);
//!
//! values.set(42, 100).unwrap();
//! assert_eq!(values[42], 100);
//! assert_eq!(values.non_default_len(), 1);
//!
//! values.reset(42).unwrap();
//! assert_eq!(values.allocated_page_count(), 0);
//! ```
//!
//! # Read-only views
//!
//! ```
//! use pagedvector::PagedVec;
//!
//! let values = PagedVec::from_vec(vec![0, 5, 0, 7, 0], 0, 4)?;
//!
//! assert_eq!(values.iter().copied().collect::<Vec<_>>(), vec![0, 5, 0, 7, 0]);
//! assert_eq!(
//!     values
//!         .non_default_iter()
//!         .map(|(index, value)| (index, *value))
//!         .collect::<Vec<_>>(),
//!     vec![(1, 5), (3, 7)],
//! );
//! assert_eq!(values.allocated_pages().count(), 1);
//! assert_eq!(values.to_vec(), vec![0, 5, 0, 7, 0]);
//! # Ok::<(), pagedvector::PagedVecError>(())
//! ```
//!
//! # Changing length
//!
//! ```
//! use pagedvector::PagedVec;
//!
//! let mut values = PagedVec::new(0, 0_i32, 4);
//! values.resize(1_000_000);
//! assert_eq!(values.allocated_page_count(), 0);
//!
//! values.push(7);
//! assert_eq!(values.pop(), Some(7));
//! values.truncate(4);
//! values.reset_all(); // keeps four logical default-valued slots
//! values.clear();     // removes every logical slot
//! ```

mod error;
mod iter;
mod page;
mod paged_vec;

#[cfg(feature = "serde")]
mod serde_impl;

pub use crate::error::{IndexOutOfBounds, PagedVecError};
pub use crate::iter::{AllocatedPageIndices, AllocatedPages, Iter, NonDefaultIter};
pub use crate::paged_vec::PagedVec;
