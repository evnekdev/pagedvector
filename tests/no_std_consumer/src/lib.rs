#![no_std]

extern crate alloc;

use pagedvector::PagedVec;

pub fn build_vector() -> PagedVec<u32> {
    let mut values = PagedVec::new(100, 0, 16);
    values.set(7, 42).expect("index is in bounds");
    values.push(9);
    values
}

#[cfg(feature = "serde")]
pub fn assert_serde_support()
where
    PagedVec<u32>: serde::Serialize + for<'de> serde::Deserialize<'de>,
{
}

#[cfg(feature = "bincode")]
pub fn encode_vector() -> alloc::vec::Vec<u8> {
    bincode::serde::encode_to_vec(build_vector(), bincode::config::standard())
        .expect("PagedVec serialization cannot fail")
}
