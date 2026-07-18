#[cfg(any(feature = "serde", feature = "bincode"))]
use pagedvector::PagedVec;

#[cfg(feature = "serde")]
#[test]
fn serde_feature_exposes_serialization_traits() {
    fn assert_serde<T>()
    where
        T: serde::Serialize + for<'de> serde::Deserialize<'de>,
    {
    }

    assert_serde::<PagedVec<i32>>();
}

#[cfg(feature = "bincode")]
#[test]
fn bincode_round_trip_uses_the_public_api() {
    let values = PagedVec::from_vec(vec![3, 0, 0, 5, 0], 0_i32, 4).unwrap();
    let encoded = bincode::serde::encode_to_vec(&values, bincode::config::standard()).unwrap();
    let (decoded, consumed): (PagedVec<i32>, usize) =
        bincode::serde::decode_from_slice(&encoded, bincode::config::standard()).unwrap();

    assert_eq!(consumed, encoded.len());
    assert_eq!(decoded, values);
    assert_eq!(decoded.to_vec(), vec![3, 0, 0, 5, 0]);
}

#[cfg(feature = "bincode")]
#[test]
fn bincode_round_trips_dynamic_length_changes() {
    let mut values = PagedVec::new(0, 9_i32, 4);
    values.extend([9, 3, 9, 4, 9]);
    values.push(7);
    assert_eq!(values.pop(), Some(7));
    values.resize(10);
    values.truncate(5);
    values.resize(8);

    let encoded = bincode::serde::encode_to_vec(&values, bincode::config::standard()).unwrap();
    let (decoded, consumed): (PagedVec<i32>, usize) =
        bincode::serde::decode_from_slice(&encoded, bincode::config::standard()).unwrap();
    assert_eq!(consumed, encoded.len());
    assert_eq!(decoded, values);
    assert_eq!(decoded.to_vec(), vec![9, 3, 9, 4, 9, 9, 9, 9]);
    assert_eq!(decoded.allocated_page_count(), 1);

    values.reset_all();
    values.clear();
    let encoded = bincode::serde::encode_to_vec(&values, bincode::config::standard()).unwrap();
    let (decoded, consumed): (PagedVec<i32>, usize) =
        bincode::serde::decode_from_slice(&encoded, bincode::config::standard()).unwrap();
    assert_eq!(consumed, encoded.len());
    assert_eq!(decoded, values);
    assert!(decoded.is_empty());

    values.extend([9, 3, 9, 4]);
    let encoded = bincode::serde::encode_to_vec(&values, bincode::config::standard()).unwrap();
    let (decoded, consumed): (PagedVec<i32>, usize) =
        bincode::serde::decode_from_slice(&encoded, bincode::config::standard()).unwrap();
    assert_eq!(consumed, encoded.len());
    assert_eq!(decoded, values);
    assert_eq!(decoded.to_vec(), vec![9, 3, 9, 4]);
    assert_eq!(decoded.allocated_page_count(), 1);
}
