use pagedvector::PagedVec;

fn main() -> Result<(), pagedvector::PagedVecError> {
    let mut values = PagedVec::new(1_000_000, 0_i32, 1_024);
    values.set(42, 7).expect("sparse index is in bounds");
    values.set(900_000, 9).expect("sparse index is in bounds");

    assert_eq!(
        values
            .non_default_iter()
            .map(|(index, value)| (index, *value))
            .collect::<Vec<_>>(),
        vec![(42, 7), (900_000, 9)],
    );

    values.push(0);
    values.push(3);
    assert_eq!(values.pop(), Some(3));
    values.reset_all();
    assert_eq!(values.allocated_page_count(), 0);
    values.clear();

    let dense = PagedVec::from_vec(vec![0, 4, 0], 0, 2)?;
    assert_eq!(dense.to_vec(), vec![0, 4, 0]);
    Ok(())
}
