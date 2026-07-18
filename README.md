# pagedvector

`pagedvector` provides `PagedVec<T>`, a fixed-length logical vector whose
physical storage is split into fixed-size pages. A page is allocated only after
one of its values differs from a configured default value.

The crate has an invariant-safe core and read-only collection ergonomics. It
is useful when page data dominates metadata and most logical values have a
common default. It is not a drop-in `Vec<T>` replacement and does not claim to
use constant memory for a very large logical length.

## Storage model

Every in-bounds logical index has a value:

- an unallocated page logically contains only the configured default;
- an allocated page owns concrete storage for exactly its logical number of
  slots;
- the final page is partial when `len` is not divisible by `page_size`;
- a page is deallocated as soon as all of its values return to the default.

The current backend has a dense `Vec<Option<Page<T>>>` page table. Therefore
page data is lazy, while page-table metadata remains proportional to
`ceil(len / page_size)`. For an extremely large logical length with few writes,
that metadata can still be substantial. A sparse page-map backend is a future
performance investigation, not a property of this release.

`PagedVec` is non-contiguous. It intentionally does not implement range slices,
`Deref<Target = [T]>`, `AsRef<[T]>`, or `Borrow<[T]>`.

## Safe core API

```rust
use pagedvector::PagedVec;

let mut values = PagedVec::new(1_000_000, 0_u32, 1_024);

assert_eq!(values.len(), 1_000_000);
assert_eq!(values.get(42), Some(&0));
assert_eq!(values.allocated_page_count(), 0);

values.set(42, 100)?;
assert_eq!(values[42], 100);
assert_eq!(values.non_default_len(), 1);

values.reset(42)?;
assert_eq!(values.allocated_page_count(), 0);
# Ok::<(), pagedvector::IndexOutOfBounds>(())
```

`get` follows normal collection conventions: it returns `None` for an
out-of-bounds index. `Index<usize>` is also available and panics out of bounds.
For in-bounds indices in unallocated pages, both return the configured default.

Mutation is controlled through `set`, `reset`, and `update`:

- `set(index, value)` and `reset(index)` return `Result<(), IndexOutOfBounds>`;
- `update(index, f)` invokes `f` on a detached copy and commits only when the
  closure returns normally, so a panicking closure leaves the vector unchanged;
- no safe API returns `&mut T` or `&mut [T]`, because such references could
  bypass the counters required to reclaim pages.

The most relevant inspection methods are `page_size`, `page_count`,
`allocated_page_count`, `non_default_len`, `default_value`, `page_index`,
`page_offset`, and `allocated_page`. `allocated_page` exposes only concrete
physical storage; it does not manufacture a default-filled slice for an
unallocated page.

## Read-only collection ergonomics

`PagedVec` exposes three intentionally different read-only views:

1. **Logical values** exist at every in-bounds index. `iter` (and
   `IntoIterator for &PagedVec`) visits all of them.
2. **Non-default values** are the logical values unequal to `default_value`.
   `non_default_iter` yields their index and reference.
3. **Allocated pages** are concrete physical storage. `allocated_pages` and
   `allocated_page_indices` visit only those pages. An allocated page can still
   contain some default-valued slots.

```rust
use pagedvector::PagedVec;

let values = PagedVec::from_vec(vec![0, 5, 0, 7, 0], 0, 4)?;

assert_eq!(values.iter().copied().collect::<Vec<_>>(), vec![0, 5, 0, 7, 0]);
assert_eq!(
    values
        .non_default_iter()
        .map(|(index, value)| (index, *value))
        .collect::<Vec<_>>(),
    vec![(1, 5), (3, 7)],
);
assert_eq!(
    values
        .allocated_pages()
        .map(|(index, page)| (index, page.to_vec()))
        .collect::<Vec<_>>(),
    vec![(0, vec![0, 5, 0, 7])],
);
assert_eq!(values.to_vec(), vec![0, 5, 0, 7, 0]);
# Ok::<(), pagedvector::PagedVecError>(())
```

`is_page_allocated(page_index)` distinguishes an invalid page index from a
valid but unallocated page. `is_allocated(index)` does the same for logical
indices, but `Some(true)` only says the containing page is physical—it does
not mean the exact value is non-default.

`from_vec` accepts explicit default and page-size policies, and `to_vec` /
`into_vec` materialize the logical sequence. The materialization methods clone
values because unallocated slots share one configured default value.

## Invariants

The implementation maintains these canonical rules after every safe mutation:

1. page size is non-zero;
2. logical indices are valid only below `len`;
3. unallocated pages logically contain only the default;
4. every allocated page has at least one non-default value;
5. each page counter exactly equals its number of non-default values;
6. a page with a zero counter is immediately deallocated;
7. page-table length is `ceil(len / page_size)`;
8. allocated page storage has exactly its logical length, including a partial
   final page;
9. the global non-default counter equals the sum of page counters.

Debug builds check these invariants after every mutation. The test suite also
checks them after every operation in a randomized model test.

## Construction and serialization

`PagedVec::new` panics when `page_size` is zero. Use `try_new` with fallible
input; it returns `PagedVecError::ZeroPageSize`.

Serialization is opt-in:

```toml
[dependencies]
pagedvector = { version = "0.2", features = ["serde"] }
```

The `bincode` feature enables the optional `bincode` dependency in addition to
serde. Serialization uses a private, versioned representation of logical
fields and page values, never trusted page or global counters. Deserialization
validates page sizes, page count, and page lengths; it recounts values and
normalizes default-only pages. The representation is not yet a stable wire
format.

Equality compares logical length, the configured default, and every logical
value. It ignores page size and physical allocation layout; therefore vectors
with matching logical contents can compare equal with different page sizes.
Vectors with different configured defaults compare unequal even if all current
logical values match.

## 0.2 breaking changes

This release removes `get_mut`, `get_page_slice_mut`, `IndexMut`, and all
page-slice/range-slice APIs. It also replaces the panicking `get` with
`Option<&T>`, renames page-count methods, makes `set` fallible, and makes serde
dependencies optional. See [CHANGELOG.md](CHANGELOG.md) for the complete list.

## Roadmap

The roadmap is staged deliberately; none of these items are promises of a
specific release date.

1. **Invariant-safe core** — safe access, set/reset, canonical counters,
   correct final-page sizing, tests, CI. Implemented.
2. **Read-only ergonomics** — iteration, non-default iteration, allocated-page
   iteration, `contains`, materialization, and logical equality documentation.
   Implemented in the current unreleased work.
3. **Dynamic length** — `push`, `pop`, `resize`, `truncate`, `clear`,
   `reset_all`, and `extend`.
4. **Controlled mutation** — richer closure updates, an entry API, mutation
   guards, and transformations, only where bookkeeping remains reliable.
5. **Serialization stability** — compatibility policy, versioned format tests,
   and documented support guarantees.
6. **Performance and backends** — benchmarks and comparison of the dense page
   table with sparse page-map backends.

`iter_mut`, `IndexMut`, arbitrary range slices, and slice-deref conversions are
not roadmap goals because they conflict with the accounting or layout model.

## Development checks

CI runs on stable Rust and executes:

```text
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo test --no-default-features
cargo doc --no-deps --all-features
cargo package
```

An MSRV has not yet been declared; stable Rust is the supported toolchain until
the project establishes and continuously tests one.

## License

Licensed under either of MIT or Apache-2.0, at your option.
