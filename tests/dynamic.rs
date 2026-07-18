use pagedvector::PagedVec;

#[test]
fn dynamic_workflow_preserves_logical_and_physical_views() {
    let mut values = PagedVec::new(0, 0_i32, 4);

    values.extend([0, 3, 0, 4, 0]);
    values.push(5);
    assert_eq!(values.to_vec(), vec![0, 3, 0, 4, 0, 5]);
    assert_eq!(values.non_default_len(), 3);
    assert_eq!(values.allocated_page_count(), 2);
    assert_eq!(
        values
            .allocated_pages()
            .map(|(index, page)| (index, page.to_vec()))
            .collect::<Vec<_>>(),
        vec![(0, vec![0, 3, 0, 4]), (1, vec![0, 5])],
    );

    assert_eq!(values.pop(), Some(5));
    values.resize(10);
    assert_eq!(values.to_vec(), vec![0, 3, 0, 4, 0, 0, 0, 0, 0, 0]);
    assert_eq!(values.allocated_page_indices().collect::<Vec<_>>(), vec![0]);

    values.truncate(3);
    assert_eq!(values.to_vec(), vec![0, 3, 0]);
    assert_eq!(
        values
            .non_default_iter()
            .map(|(index, value)| (index, *value))
            .collect::<Vec<_>>(),
        vec![(1, 3)],
    );

    values.reset_all();
    assert_eq!(values.len(), 3);
    assert_eq!(values.to_vec(), vec![0, 0, 0]);
    assert_eq!(values.non_default_len(), 0);
    assert_eq!(values.allocated_page_count(), 0);

    values.clear();
    assert!(values.is_empty());
    assert_eq!(values.page_size(), 4);
    assert_eq!(values.default_value(), &0);
    assert_eq!(values.page_count(), 0);
    values.push(9);
    assert_eq!(values.to_vec(), vec![9]);
}

#[test]
fn growth_uses_defaults_without_allocating_and_pop_handles_default_slots() {
    let mut values = PagedVec::new(0, 7_i32, 3);
    values.resize(1_000_000);

    assert_eq!(values.len(), 1_000_000);
    assert_eq!(values.allocated_page_count(), 0);
    assert!(values.contains(&7));
    assert_eq!(values.pop(), Some(7));
    assert_eq!(values.len(), 999_999);
    assert_eq!(values.allocated_page_count(), 0);

    values.push(11);
    assert_eq!(values.non_default_len(), 1);
    assert_eq!(values.allocated_page_count(), 1);
    assert_eq!(values.pop(), Some(11));
    assert_eq!(values.allocated_page_count(), 0);
}

#[test]
fn truncate_and_resize_observe_page_boundaries() {
    let mut values = PagedVec::from_vec(vec![5, 1, 5, 2, 3], 5_i32, 4).unwrap();

    values.truncate(4);
    assert_eq!(values.to_vec(), vec![5, 1, 5, 2]);
    assert_eq!(values.allocated_page(0), Some(&[5, 1, 5, 2][..]));
    assert_eq!(values.allocated_page(1), None);

    values.truncate(2);
    assert_eq!(values.to_vec(), vec![5, 1]);
    assert_eq!(values.allocated_page(0), Some(&[5, 1][..]));
    values.resize(5);
    assert_eq!(values.to_vec(), vec![5, 1, 5, 5, 5]);
    assert_eq!(values.allocated_page(0), Some(&[5, 1, 5, 5][..]));
    assert_eq!(values.allocated_page(1), None);
}

#[test]
fn equality_ignores_dynamic_allocation_history() {
    let mut clear_and_regrow = PagedVec::new(0, 0_i32, 2);
    clear_and_regrow.extend([1, 0, 2]);
    clear_and_regrow.clear();
    clear_and_regrow.extend([0, 3, 0, 4]);

    let expected = PagedVec::from_vec(vec![0, 3, 0, 4], 0_i32, 4).unwrap();
    assert_eq!(clear_and_regrow, expected);

    let mut truncate_and_regrow = PagedVec::from_vec(vec![0, 3, 0, 4, 5], 0_i32, 2).unwrap();
    truncate_and_regrow.truncate(2);
    truncate_and_regrow.extend([0, 4]);
    assert_eq!(truncate_and_regrow, expected);
}
