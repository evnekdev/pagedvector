use pagedvector::{IndexOutOfBounds, PagedVec, PagedVecError};

#[test]
fn construction_and_public_errors_are_explicit() {
    let empty = PagedVec::new(0, 7_i32, 3);
    assert!(empty.is_empty());
    assert_eq!(empty.iter().count(), 0);
    assert_eq!(empty.non_default_iter().count(), 0);
    assert_eq!(empty.allocated_pages().count(), 0);
    assert_eq!(empty.to_vec(), Vec::<i32>::new());
    let from_empty = PagedVec::from_vec(Vec::<i32>::new(), 7, 3).unwrap();
    assert_eq!(from_empty.to_vec(), Vec::<i32>::new());

    let mut values = PagedVec::new(2, 0_i32, 1);
    assert_eq!(values.set(2, 1), Err(IndexOutOfBounds { index: 2, len: 2 }));
    assert!(matches!(
        PagedVec::from_vec(vec![1_i32], 0, 0),
        Err(PagedVecError::ZeroPageSize)
    ));
}

#[test]
fn logical_iteration_and_borrowed_into_iterator_include_default_slots() {
    let mut values = PagedVec::new(5, 9_i32, 2);
    values.set(1, 2).unwrap();
    values.set(4, 3).unwrap();

    assert_eq!(
        values.iter().copied().collect::<Vec<_>>(),
        vec![9, 2, 9, 9, 3]
    );
    assert_eq!(
        (&values).into_iter().copied().collect::<Vec<_>>(),
        vec![9, 2, 9, 9, 3]
    );

    let mut iterator = values.iter();
    assert_eq!(iterator.next_back(), Some(&3));
    assert_eq!(iterator.next(), Some(&9));
    assert_eq!(iterator.copied().collect::<Vec<_>>(), vec![2, 9, 9]);
}

#[test]
fn sparse_and_allocated_page_views_are_distinct() {
    let values = PagedVec::from_vec(vec![10, 20, 10, 30, 40], 10_i32, 4).unwrap();

    assert_eq!(
        values
            .non_default_iter()
            .map(|(index, value)| (index, *value))
            .collect::<Vec<_>>(),
        vec![(1, 20), (3, 30), (4, 40)]
    );
    assert_eq!(
        values
            .allocated_pages()
            .map(|(index, values)| (index, values.to_vec()))
            .collect::<Vec<_>>(),
        vec![(0, vec![10, 20, 10, 30]), (1, vec![40])]
    );
    assert_eq!(
        values.allocated_page_indices().collect::<Vec<_>>(),
        vec![0, 1]
    );

    assert_eq!(values.is_page_allocated(0), Some(true));
    assert_eq!(values.is_page_allocated(1), Some(true));
    assert_eq!(values.is_page_allocated(2), None);
    assert_eq!(values.is_allocated(0), Some(true));
    assert_eq!(values.is_allocated(2), Some(true));
    assert_eq!(values.is_allocated(5), None);
}

#[test]
fn materialization_contains_and_conversion_preserve_logical_values() {
    let values = PagedVec::from_vec(vec![5, 5, 9, 5, 7], 5_i32, 3).unwrap();
    assert_eq!(values.to_vec(), vec![5, 5, 9, 5, 7]);
    assert!(values.contains(&5));
    assert!(values.contains(&9));
    assert!(!values.contains(&8));
    assert_eq!(values.into_vec(), vec![5, 5, 9, 5, 7]);

    let all_non_default = PagedVec::from_vec(vec![1, 2], 0_i32, 2).unwrap();
    assert!(!all_non_default.contains(&0));
}

#[test]
fn equality_uses_default_and_logical_values_not_page_layout_or_history() {
    let left = PagedVec::from_vec(vec![2, 2, 2], 0_i32, 1).unwrap();
    let right = PagedVec::from_vec(vec![2, 2, 2], 0_i32, 3).unwrap();
    assert_eq!(left, right);

    let different_default = PagedVec::from_vec(vec![2, 2, 2], 1_i32, 2).unwrap();
    assert_eq!(left.to_vec(), different_default.to_vec());
    assert_ne!(left, different_default);

    let mut changed_history = PagedVec::new(3, 0_i32, 2);
    changed_history.set(1, 4).unwrap();
    changed_history.reset(1).unwrap();
    assert_eq!(changed_history, PagedVec::new(3, 0_i32, 2));
}
