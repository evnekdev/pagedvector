use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::page::Page;
use crate::paged_vec::{logical_page_len_for, page_count_for, PagedVec};

const FORMAT_VERSION: u8 = 1;

#[derive(Serialize)]
struct PagedVecReprRef<'a, T> {
    version: u8,
    len: usize,
    page_size: usize,
    default: &'a T,
    pages: Vec<Option<&'a [T]>>,
}

#[derive(Deserialize)]
struct PagedVecRepr<T> {
    version: u8,
    len: usize,
    page_size: usize,
    default: T,
    pages: Vec<Option<Vec<T>>>,
}

impl<T: Serialize> Serialize for PagedVec<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let pages = self
            .pages
            .iter()
            .map(|page| page.as_ref().map(|page| page.values.as_ref()))
            .collect();
        PagedVecReprRef {
            version: FORMAT_VERSION,
            len: self.len,
            page_size: self.page_size(),
            default: &self.default,
            pages,
        }
        .serialize(serializer)
    }
}

impl<'de, T> Deserialize<'de> for PagedVec<T>
where
    T: Deserialize<'de> + PartialEq,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = PagedVecRepr::deserialize(deserializer)?;
        if repr.version != FORMAT_VERSION {
            return Err(D::Error::custom(
                "unsupported PagedVec serialization version",
            ));
        }

        let page_size = std::num::NonZeroUsize::new(repr.page_size)
            .ok_or_else(|| D::Error::custom("PagedVec page size must be greater than zero"))?;
        let expected_page_count = page_count_for(repr.len, page_size);
        if repr.pages.len() != expected_page_count {
            return Err(D::Error::custom(
                "PagedVec page table length does not match logical length",
            ));
        }

        let mut non_default = 0usize;
        let mut pages = Vec::with_capacity(expected_page_count);
        for (page_index, values) in repr.pages.into_iter().enumerate() {
            let Some(values) = values else {
                pages.push(None);
                continue;
            };

            let expected_len = logical_page_len_for(repr.len, page_size, page_index)
                .expect("serialized page index was validated against the page count");
            if values.len() != expected_len {
                return Err(D::Error::custom(
                    "PagedVec page length does not match its logical length",
                ));
            }

            let page_non_default = values
                .iter()
                .filter(|value| *value != &repr.default)
                .count();
            non_default += page_non_default;
            if page_non_default == 0 {
                pages.push(None);
            } else {
                pages.push(Some(Page {
                    values: values.into_boxed_slice(),
                    non_default: page_non_default,
                }));
            }
        }

        let vector = PagedVec {
            len: repr.len,
            page_size,
            default: repr.default,
            non_default,
            pages,
        };
        vector.debug_assert_invariants();
        Ok(vector)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "bincode")]
    #[test]
    fn bincode_round_trip_recounts_and_preserves_values() {
        let mut original = PagedVec::new(5, 0_i32, 4);
        original.set(1, 3).unwrap();
        original.set(4, 7).unwrap();

        let encoded =
            bincode::serde::encode_to_vec(&original, bincode::config::standard()).unwrap();
        let (decoded, consumed): (PagedVec<i32>, usize) =
            bincode::serde::decode_from_slice(&encoded, bincode::config::standard()).unwrap();

        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded, original);
        assert_eq!(decoded.non_default_len(), 2);
        assert_eq!(decoded.allocated_page_count(), 2);
        assert_eq!(decoded.validate_invariants(), Ok(()));
    }

    #[cfg(feature = "bincode")]
    #[test]
    fn deserialization_rejects_invalid_structure_and_normalizes_default_pages() {
        #[derive(serde::Serialize)]
        struct RawRepr {
            version: u8,
            len: usize,
            page_size: usize,
            default: i32,
            pages: Vec<Option<Vec<i32>>>,
        }

        let invalid_page_size = RawRepr {
            version: FORMAT_VERSION,
            len: 1,
            page_size: 0,
            default: 0,
            pages: Vec::new(),
        };
        let encoded =
            bincode::serde::encode_to_vec(&invalid_page_size, bincode::config::standard()).unwrap();
        let decoded = bincode::serde::decode_from_slice::<PagedVec<i32>, _>(
            &encoded,
            bincode::config::standard(),
        );
        assert!(decoded.is_err());

        let default_only_page = RawRepr {
            version: FORMAT_VERSION,
            len: 2,
            page_size: 2,
            default: 0,
            pages: vec![Some(vec![0, 0])],
        };
        let encoded =
            bincode::serde::encode_to_vec(&default_only_page, bincode::config::standard()).unwrap();
        let (decoded, _): (PagedVec<i32>, usize) =
            bincode::serde::decode_from_slice(&encoded, bincode::config::standard()).unwrap();
        assert_eq!(decoded.non_default_len(), 0);
        assert_eq!(decoded.allocated_page_count(), 0);
        assert_eq!(decoded.validate_invariants(), Ok(()));
    }
}
