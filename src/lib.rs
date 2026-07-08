// pagedvector.rs
//! This small crate helps creating a paged vector storage for large arrays of data where most of the vector storage is filled with default values (non-initialized).
//! 
//! The idea closely follows virtual memory allocation mechanism in operating systems.
//! Virtual storage is divided in equal-length chunks (pages) with a page book-keeping mechanism. If the user stores a non-default value, the corresponding page becomes allocated.
//! 
//! # Experimental
//!
//! `PagedVec` is currently experimental.
//!
//! Direct mutable access (`get_mut`, `IndexMut`, mutable slices) bypasses the
//! internal sparse bookkeeping. The sparse representation itself remains valid,
//! but allocation statistics and automatic page deallocation are not guaranteed
//! to remain correct after such modifications.
//!
//! Use [`PagedVec::set`] for bookkeeping-safe updates.

use std::ops::Index;
use std::ops::IndexMut;
use std::ops::{Range,RangeFrom,RangeTo,RangeFull,RangeInclusive,RangeToInclusive};

use bincode::Encode;
use serde::{Deserialize, Serialize};

/*********************************************************************************************************************************************************/
/*********************************************************************************************************************************************************/

/// Internal representation of an allocated page.
#[derive(Clone,Debug, Serialize, Deserialize, Encode)]
struct Page<T> {
	data : Vec<T>,
	non_default: usize,
}

impl<T: Clone + PartialEq> Page<T> {
	/// new instance
	pub fn new(default: T, len: usize)->Self {
		return Self {
			data : vec![default; len],
			non_default : 0,
		};
	}
	
}


/*********************************************************************************************************************************************************/
/*********************************************************************************************************************************************************/

/// A vector-like structure with the storage split into equal-sized pages which are allocated only if non-default values are stored inside.
#[derive(Clone,Debug, Serialize, Deserialize, Encode)]
pub struct PagedVec<T> {
	psize : usize,
	vlen : usize,
	default : T,
	default_page : Vec<T>,
	pages : Vec<Option<Page<T>>>,
}

impl<T: Clone + PartialEq> PagedVec<T> {
	
	/// Create a new instance.
	/// 
	/// | Argument | Value |
	/// |---|---|
	/// | vlen | virtual length (similar to `.len()` in `Vec<T>`). |
	/// | default | the default value upon which pages are not allocated. |
	/// | psize | page size (preferrable as 2^N). |
	pub fn new(vlen: usize, default: T, psize: usize)-> Self {
		assert!(psize > 0);
		let npages = vlen.div_ceil(psize);
		return Self {
			psize : psize,
			vlen : vlen,
			default : default.clone(),
			default_page : vec![default;psize],
			pages : vec![None; npages],
		};
	}
	
	/// Virtual length of the vector.
	pub fn len(&self)->usize {
		return self.vlen;
	}
	
	/*********************************************************************************************************************************************************/
	/*********************************************************************************************************************************************************/
	
	pub fn is_default(&self, idx: usize)->bool {
		assert!(idx < self.vlen);
		let (vpn, off) = self.split_index(idx);
		match &self.pages[vpn] {
			Some(page) => {
				return page.data[off] == self.default;
				//return Ordering::Equal == T::total_cmp0(&page.data[off], &self.default);
			}
			None => {return true;}
		}
	}
	
	/// The total number of pages (allocated and empty).
	pub fn number_pages_total(&self)->usize {
		return self.pages.len();
	}
	
	/// Counts the number of allocated pages.
	pub fn number_pages_alloc(&self)->usize {
		let mut count = 0usize;
		for k in 0..self.pages.len(){
			if self.pages[k].is_some(){
				count += 1;
			}
		}
		return count;
	}
	
	/*********************************************************************************************************************************************************/
	/*********************************************************************************************************************************************************/
	
	#[inline]
	fn split_index(&self, idx: usize) -> (usize,usize) {
		return (idx / self.psize, idx % self.psize);
	}
	
	/// Return an immutable reference to a stored value.
	pub fn get(&self, idx: usize)-> &T {
		assert!(idx < self.vlen);
		let (vpn, off) = self.split_index(idx);
		match &self.pages[vpn] {
			Some(page) => {
				return &page.data[off];
			}
			None => {
				return &self.default;
			}
		}
	}
	
	/// Returns a mutable reference to the element at `idx`.
	///
	/// # Warning
	///
	/// This method bypasses the page bookkeeping mechanism. If the value is
	/// modified through the returned reference, the page's internal
	/// `non_default` counter is **not** updated.
	///
	/// Consequently:
	///
	/// - `number_pages_alloc()` may become incorrect;
	/// - pages containing only default values may not be automatically
	///   deallocated;
	/// - future versions of this crate may change this behavior.
	///
	/// If bookkeeping consistency is required, use [`PagedVec::set`] instead.
	///
	/// After arbitrary mutable modifications, call [`PagedVec::cleanup`] (planned)
	/// before relying on allocation statistics.
	#[deprecated(
    since = "0.1.0",
    note = "get_mut() bypasses sparse bookkeeping; use set() instead"
	)]
	pub fn get_mut(&mut self, idx: usize)-> &mut T {
		assert!(idx < self.vlen);
		let (vpn, off) = self.split_index(idx);
		if self.pages[vpn].is_none() {
			self.pages[vpn] = Some(Page::new(self.default.clone(), self.psize));
		}
		return &mut self.pages[vpn].as_mut().unwrap().data[off];
	}
	
	/// Returns a contigous slice of vector memory (allocated or not), panics if the slice spans across more than one page.
	pub fn get_slice_unchecked(&self, start: usize, len: usize)->&[T] {
		assert!(start <= self.vlen);
		assert!(len <= self.vlen - start);
		if len == 0 {
			return &self.default_page[0..0];
		}
		let (vpn0, off0) = self.split_index(start);
		let (vpn1, off1) = self.split_index(start + len - 1);
		assert_eq!(vpn0, vpn1);
		match &self.pages[vpn0] {
			Some(page) => {
				return &page.data[off0..=off1];
			}
			None => {
				return &self.default_page[off0..=off1];
			}
		}
	}
	
	/// Returns a contigous slice of vector memory (preallocates, if necessary), panics if the slice spans across more than one page.
	/// 
	/// # Warning
	///
	/// This method bypasses the page bookkeeping mechanism. If the value is
	/// modified through the returned reference, the page's internal
	/// `non_default` counter is **not** updated.
	///
	/// Consequently:
	///
	/// - `number_pages_alloc()` may become incorrect;
	/// - pages containing only default values may not be automatically
	///   deallocated;
	/// - future versions of this crate may change this behavior.
	#[deprecated(
    since = "0.1.0",
    note = "get_mut() bypasses sparse bookkeeping; use set() instead"
	)]
	pub fn get_slice_unchecked_mut(&mut self, start: usize, len: usize)->&mut [T] {
		assert!(start <= self.vlen);
		assert!(len <= self.vlen - start);
		if len == 0 {
			return &mut self.default_page[0..0];
		}
		let (vpn0, off0) = self.split_index(start);
		let (vpn1, off1) = self.split_index(start + len - 1);
		assert_eq!(vpn0, vpn1);
		if self.pages[vpn0].is_none() {
			self.pages[vpn0] = Some(Page::new(self.default.clone(), self.psize));
		}
		return &mut self.pages[vpn0].as_mut().unwrap().data[off0..=off1];
	}
	
	/// Set a value at a specific index, default and non-default values are handled separately.
	pub fn set(&mut self, idx: usize, value: T) {
		assert!(idx < self.vlen);
		if value == self.default {
			return self.set_default(idx);
		} else {
			return self.set_nondefault(idx, value);
		}
	}
	
	/// Default case.
	fn set_default(&mut self, idx: usize) {
		let (vpn, off) = self.split_index(idx);
		let mut dealloc = false;
		match &mut self.pages[vpn] {
			Some(page) => {
				if page.data[off] != self.default {
					page.data[off] = self.default.clone();
					page.non_default -= 1;
				}
				if page.non_default == 0 {dealloc = true;}
			}
			None => {/*do nothing*/}
		}
		if dealloc {self.pages[vpn] = None;}
	}
	
	/// Non-default case.
	fn set_nondefault(&mut self, idx: usize, value : T) {
		let (vpn, off) = self.split_index(idx);
		if self.pages[vpn].is_none() {
			self.pages[vpn] = Some(Page::new(self.default.clone(),self.psize));
		}
		match &mut self.pages[vpn] {
			Some(page) => {
				if page.data[off] == self.default {
					page.non_default += 1;
				}
				page.data[off] = value;
			}
			None => {/*do nothing*/}
		}
	}
	
}

/*********************************************************************************************************************************************************/
/*********************************************************************************************************************************************************/


impl<T: Clone + PartialEq> Index<usize> for PagedVec<T> {
	type Output = T;
	
	fn index(&self, index: usize) -> &Self::Output {
		return self.get(index);
	}
	
}

/// Mutable indexing.
///
/// # Warning
///
/// Assignments through indexing
///
/// ```ignore
/// vec[i] = value;
/// ```
///
/// bypass the sparse bookkeeping logic in the current implementation.
/// This means allocation statistics and automatic page deallocation may
/// become inconsistent.
///
/// Prefer [`PagedVec::set`] whenever possible.
impl<T: Clone + PartialEq> IndexMut<usize> for PagedVec<T> {
	
	fn index_mut(&mut self, index: usize)-> &mut Self::Output {
		return self.get_mut(index);
	}
	
}

/*********************************************************************************************************************************************************/
/*********************************************************************************************************************************************************/

impl<T: Clone + PartialEq> Index<Range<usize>> for PagedVec<T> {
	type Output = [T];
	
	fn index(&self, range: Range<usize>)->&Self::Output {
		let start = range.start;
		let end = range.end;
		assert!(start <= end);
		let len = end - start;
		return self.get_slice_unchecked(start,len);
	}
}

impl<T: Clone + PartialEq> IndexMut<Range<usize>> for PagedVec<T> {
	
	fn index_mut(&mut self, range: Range<usize>)->&mut Self::Output {
		let start = range.start;
		let end = range.end;
		assert!(start <= end);
		let len = end - start;
		return self.get_slice_unchecked_mut(start,len);
	}
	
}

/*********************************************************************************************************************************************************/
/*********************************************************************************************************************************************************/

impl<T: Clone + PartialEq> Index<RangeFrom<usize>> for PagedVec<T> {
	type Output = [T];
	
	fn index(&self, range: RangeFrom<usize>)->&Self::Output {
		let start = range.start;
		let end = self.len();
		assert!(start <= end);
		let len = end - start;
		return self.get_slice_unchecked(start, len);
	}
	
}

impl<T: Clone + PartialEq> IndexMut<RangeFrom<usize>> for PagedVec<T> {
	
	fn index_mut(&mut self, range: RangeFrom<usize>)->&mut Self::Output {
		let start = range.start;
		let end = self.len();
		assert!(start <= end);
		let len = end - start;
		return self.get_slice_unchecked_mut(start, len);
	}
	
}

/*********************************************************************************************************************************************************/
/*********************************************************************************************************************************************************/

impl<T: Clone + PartialEq> Index<RangeTo<usize>> for PagedVec<T> {
	type Output = [T];
	
	fn index(&self, range: RangeTo<usize>)->&Self::Output {
		let start = 0;
		let end = range.end;
		assert!(end <= self.len());
		let len = end - start;
		return self.get_slice_unchecked(start, len);
	}
	
}

impl<T: Clone + PartialEq> IndexMut<RangeTo<usize>> for PagedVec<T> {
	
	fn index_mut(&mut self, range: RangeTo<usize>)->&mut Self::Output {
		let start = 0;
		let end = range.end;
		assert!(end <= self.len());
		let len = end - start;
		return self.get_slice_unchecked_mut(start, len);
	}
	
}

/*********************************************************************************************************************************************************/
/*********************************************************************************************************************************************************/

impl<T: Clone + PartialEq> Index<RangeFull> for PagedVec<T> {
	type Output = [T];
	
	fn index(&self, _: RangeFull)->&Self::Output {
		let len = self.len();
		return self.get_slice_unchecked(0, len);
	}
	
}

impl<T: Clone + PartialEq> IndexMut<RangeFull> for PagedVec<T> {
	
	fn index_mut(&mut self, _: RangeFull)->&mut Self::Output {
		let len = self.len();
		return self.get_slice_unchecked_mut(0, len);
	}
	
}

/*********************************************************************************************************************************************************/
/*********************************************************************************************************************************************************/

impl<T: Clone + PartialEq> Index<RangeInclusive<usize>> for PagedVec<T> {
	type Output = [T];
	
	fn index(&self, range: RangeInclusive<usize>)->&Self::Output {
		let start = *range.start();
		let end = *range.end();
		assert!(start <= end);
		assert!(end < self.len());
		let len = end - start + 1;
		return self.get_slice_unchecked(start,len);
	}
	
}

impl<T: Clone + PartialEq> IndexMut<RangeInclusive<usize>> for PagedVec<T> {
	
	fn index_mut(&mut self, range: RangeInclusive<usize>)->&mut Self::Output {
		let start = *range.start();
		let end = *range.end();
		assert!(start <= end);
		assert!(end < self.len());
		let len = end - start + 1;
		return self.get_slice_unchecked_mut(start,len);
	}
	
}

/*********************************************************************************************************************************************************/
/*********************************************************************************************************************************************************/

impl<T: Clone + PartialEq> Index<RangeToInclusive<usize>> for PagedVec<T> {
	type Output = [T];
	
	fn index(&self, range: RangeToInclusive<usize>)->&Self::Output {
		let start = 0;
		let end = range.end;
		assert!(end < self.len());
		let len = end - start + 1;
		return self.get_slice_unchecked(start,len);
	}
	
}

impl<T: Clone + PartialEq> IndexMut<RangeToInclusive<usize>> for PagedVec<T> {
	
	fn index_mut(&mut self, range: RangeToInclusive<usize>)->&mut Self::Output {
		let start = 0;
		let end = range.end;
		assert!(end < self.len());
		let len = end - start + 1;
		return self.get_slice_unchecked_mut(start,len);
	}
	
}

/*********************************************************************************************************************************************************/
/*********************************************************************************************************************************************************/
