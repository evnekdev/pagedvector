# Changelog

All notable changes to this project are documented here.

## 0.2.0 — Unreleased

### Breaking changes

- Removed `get_mut`, `get_page_slice_mut`, `IndexMut`, and the commented range
  indexing design. Safe APIs no longer expose mutable references into page
  storage.
- Changed `get(index)` from a panicking `&T` return to `Option<&T>`.
- Changed `set(index, value)` to return `Result<(), IndexOutOfBounds>` and
  added `reset(index)` plus controlled, panic-safe `update(index, closure)`.
- Renamed `number_pages_total` to `page_count` and `number_pages_alloc` to
  `allocated_page_count`.
- Removed `default_page` and the fabricated immutable page-slice API.
- Removed unconditional serde and bincode dependencies. Serde is now opt-in;
  bincode is an additional opt-in feature.
- Bumped the crate version to `0.2.0`.

### Fixed

- Page and collection non-default counters are now updated together and are
  checked by internal invariants.
- Allocated pages containing only default values are immediately deallocated.
- The final partial page now allocates only its logical number of values.
- Deserialization uses a versioned representation, validates structural data,
  recounts non-default values, and discards default-only pages.

### Added

- Checked construction through `try_new` and the `PagedVecError` error type.
- `IndexOutOfBounds` for fallible mutation APIs.
- `is_empty`, `page_size`, `page_index`, `page_offset`, `default_value`,
  `non_default_len`, and physical-only `allocated_page` access.
- Unit, randomized model, and optional serialization round-trip tests.
