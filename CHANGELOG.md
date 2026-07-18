# Changelog

All notable changes to this project are documented here.

## 0.2.0 — Unreleased

### Breaking changes

- Changed `get(index)` from a panicking `&T` return to `Option<&T>`.
- Changed `is_default(index)` from a panicking `bool` return to `Option<bool>`.
- Changed `set(index, value)` to return `Result<(), IndexOutOfBounds>` and
  added `reset(index)` plus controlled, panic-safe `update(index, closure)`.
- Renamed `number_pages_total` to `page_count` and `number_pages_alloc` to
  `allocated_page_count`.

### Added

- Checked construction through `try_new` and the `PagedVecError` error type.
- `IndexOutOfBounds` for fallible mutation APIs.
- `is_empty`, `page_size`, `page_index`, `page_offset`, `default_value`,
  `non_default_len`, and physical-only `allocated_page` access.
- Immutable logical iteration through `iter` and `IntoIterator for &PagedVec`.
- Sparse `non_default_iter`, physical `allocated_pages`, and
  `allocated_page_indices` iterators.
- `is_page_allocated` and `is_allocated` for explicit physical-allocation
  inspection.
- `to_vec`, `into_vec`, and explicit-policy `from_vec` conversion methods.
- `contains` plus documentation of equality's logical/default-based contract.
- Dynamic-length `push`, `pop`, `resize`, `truncate`, `clear`, `reset_all`,
  and `Extend<T>` support with canonical page reclamation.
- `no_std + alloc` support, including optional Serde and bincode support.

### Changed

- Declared Rust 1.85 as the MSRV and added MSRV, `no_std`, and embedded-target
  CI checks.
- Made `std` an explicit default feature; serialization dependencies now opt
  into `std` only when that feature is enabled.
- Expanded public documentation, examples, package metadata, iterator
  contracts, panic guarantees, and release-facing tests.

### Fixed

- Page and collection non-default counters are now updated together and are
  checked by internal invariants.
- Allocated pages containing only default values are immediately deallocated.
- The final partial page now allocates only its logical number of values.
- Deserialization uses a versioned representation, validates structural data,
  recounts non-default values, and discards default-only pages.
- Dynamic operations commit canonical metadata before detached storage is
  dropped, preserving canonical state if a destructor panics.
- The distributed `LICENSE` now contains both the MIT and Apache-2.0 terms
  declared by the package metadata.

### Removed

- `get_mut`, `get_page_slice_mut`, `IndexMut`, and the commented range indexing
  design. Safe APIs no longer expose mutable references into page storage.
- `default_page` and the fabricated immutable page-slice API.
- Unconditional serde and bincode dependencies. Serde and bincode are now
  opt-in features.
