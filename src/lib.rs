use std::collections::HashMap;
use std::ops::Index;
use std::ops::IndexMut;

use bincode::Encode;
use serde::{Deserialize, Serialize};

// This small crate helps creating a paged vector storage for large arrays of data where most of the vector storage is filled with default values (non-initialized). The idea closely follows virtual memory allocation mechanism in operating systems.
// Virtual storage is divided in equal-length chunks (pages) with a page book-keeping mechanism. If the user stores a non-default value, the corresponding page becomes allocated.

/*********************************************************************************************************************************************************/
/*********************************************************************************************************************************************************/

#[derive(Clone,Debug, Serialize, Deserialize, Encode)]
struct Page<T> {
	data : Vec<T>,
	non_default: usize,
}

impl<T: Clone> Page<T> {
	
	pub fn new(default: T, len: usize)->Self {
		return Self {
			data : vec![default; len],
			non_default : 0,
		};
	}
	
}


/*********************************************************************************************************************************************************/
/*********************************************************************************************************************************************************/

#[derive(Clone,Debug, Serialize, Deserialize, Encode)]
pub struct PagedVec<T> {
	page_size : usize,
	virtual_len : usize,
	default : T,
	pages : Vec<Option<Page<T>>>,
}

impl<T: Clone + PartialEq> PagedVec<T> {
	
	pub fn new(virtual_len: usize, default: T, page_size: usize)-> Self {
		return Self {
			page_size : page_size,
			virtual_len : virtual_len,
			default : default,
			pages : vec![None; virtual_len / page_size + 1],
		};
	}
	
	#[inline]
	fn split_index(&self, idx: usize) -> (usize,usize) {
		return (idx / self.page_size, idx % self.page_size);
	}
	
	pub fn len(&self)->usize {
		return self.virtual_len;
	}
	
	pub fn get(&self, idx: usize)-> &T {
		assert!(idx < self.virtual_len);
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
	
	pub fn get_mut(&mut self, idx: usize)-> &mut T {
		assert!(idx < self.virtual_len);
		let (vpn, off) = self.split_index(idx);
		if self.pages[vpn].is_none() {
			self.pages[vpn] = Some(Page::new(self.default.clone(), self.page_size));
		}
		return &mut self.pages[vpn].as_mut().unwrap().data[off];
	}
	
	pub fn set(&mut self, idx: usize, value: T) {
		assert!(idx < self.virtual_len);
		if value == self.default {
			return self.set_default(idx);
		} else {
			return self.set_nondefault(idx, value);
		}
	}
	
	fn set_default(&mut self, idx: usize) {
		//println!("pagedvector::set_default");
		let (vpn, off) = self.split_index(idx);
		let mut dealloc = false;
		match &mut self.pages[vpn] {
			Some(page) => {
				page.data[off] = self.default.clone();
				page.non_default -= 1;
				if page.non_default == 0 {dealloc = true;}
			}
			None => {/*do nothing*/}
		}
		if dealloc {self.pages[vpn] = None;}
	}
	
	fn set_nondefault(&mut self, idx: usize, value : T) {
		//println!("pagedvector::set_nondefault");
		let (vpn, off) = self.split_index(idx);
		if self.pages[vpn].is_none() {
			self.pages[vpn] = Some(Page::new(self.default.clone(),self.page_size));
		}
		match &mut self.pages[vpn] {
			Some(page) => {
				page.data[off] = value;
				page.non_default += 1;
			}
			None => {/*do nothing*/}
		}
	}
	
	pub fn is_default(&self, idx: usize)->bool {
		let (vpn, off) = self.split_index(idx);
		match &self.pages[vpn] {
			Some(page) => {
				return page.data[off] == self.default;
			}
			None => {return true;}
		}
	}
	
	pub fn number_pages_total(&self)->usize {
		return self.pages.len();
	}
	
	pub fn number_pages_allocated(&self)->usize {
		let mut count = 0usize;
		for k in 0..self.pages.len(){
			if self.pages[k].is_some(){
				count += 1;
			}
		}
		return count;
	}
	
}


impl<T: Clone + PartialEq> Index<usize> for PagedVec<T> {
	type Output = T;
	
	fn index(&self, index: usize) -> &Self::Output {
		return self.get(index);
	}
	
}
