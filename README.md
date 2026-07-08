# 📦 pagedvector (early experimental crate)

A lightweight Rust crate implementing a **paged virtual vector** optimized for sparse data.

It mimics how operating systems manage memory: instead of allocating a full vector, it only allocates memory for portions ("pages") that actually contain non-default values.

---

## 🚀 Features

- 💾 Memory-efficient for sparse data
- 📄 Fixed-size paging system
- ⚡ Lazy allocation (allocate on write)
- 🧹 Automatic deallocation (free empty pages)
- 🔍 Familiar API (`get`, `set`, indexing)
- 📦 Serialization support (`serde`, `bincode`)

---

## 🧠 Concept

Instead of storing a full vector of size `N`, `PagedVec`:

1. Splits the vector into equal-sized pages
2. Stores pages as `Option<Page<T>>`
3. Allocates a page only when needed
4. Deallocates when all values return to default

---

## 🛠 Example

```rust
use pagedvector::PagedVec;

fn main() {
    let mut vec = PagedVec::new(1_000_000, 0u32, 1024);

    // No memory allocated yet
    assert_eq!(vec.number_pages_alloc(), 0);

    // Write value
    vec.set(42, 100);

    // Now one page is allocated
    assert!(vec.number_pages_alloc() > 0);

    // Read values
    assert_eq!(vec[42], 100);
    assert_eq!(vec[43], 0);

    // Reset to default
    vec.set(42, 0);
}
```

---

## 📚 API

### Create

```rust
let vec = PagedVec::new(length, default_value, page_size);
```

### Access

```rust
vec.get(i);
vec.get_mut(i);
vec[i];
```

### Modify

```rust
vec.set(i, value);
```

### Inspect

```rust
vec.len();
vec.is_default(i);
vec.number_pages_total();
vec.number_pages_alloc();
```

---

## ⚠️ Notes

- Requires `T: Clone + PartialEq`
- Uses `assert!` for bounds (panics if invalid index)
- Performance depends on `page_size`

---

## 💡 Use Cases

- Sparse arrays
- Simulations
- Game worlds
- Scientific data
- Memory-constrained environments

---
## RoadMap
 Implement more methods to get `Vec<T>`-like feel:

  - is_empty()
  - page_size()
  - capacity_pages()
  - clear()
  - fill(value)
  - contains(&value)

  - get(index) -> Option<&T>
  - get_mut(index) -> Option<&mut T>

  - v[i]      // panics
  - v.get(i) // returns Option

  - iter()
  - iter_mut()
  - enumerate_non_default()
  - count_non_default()
  - allocated_pages()
  - deallocate_empty_pages()


  - resize(new_len, default)
  - truncate(new_len)
  - push(value)
  - pop()
  - extend(iter)

  - Default
  - From<Vec<T>>
  - FromIterator<T>
  - Extend<T>
  - IntoIterator


  - is_allocated(index) -> bool
  - is_page_allocated(page_index) -> bool
  - page_index(index) -> usize
  - page_offset(index) -> usize
  - allocated_page_indices()
  - non_default_len()
  - sparsity()
  - memory_len_allocated()

  - allocated_fraction()
  - default_fraction()

  - to_vec()
  - into_vec()
  - from_vec(vec, default, psize)

  - shrink_pages()
  - recount_page(page_index)
  - recount_all_pages()
  - deallocate_default_pages()

---

## 📦 Dependencies

```toml
serde = { version = "1", features = ["derive"] }
bincode = "2"
```

---

## 📄 License

MIT License (see LICENSE file)
