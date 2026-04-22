# 📦 PagedVec

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
use pagedvec::PagedVec;

fn main() {
    let mut vec = PagedVec::new(1_000_000, 0u32, 1024);

    // No memory allocated yet
    assert_eq!(vec.number_pages_allocated(), 0);

    // Write value
    vec.set(42, 100);

    // Now one page is allocated
    assert!(vec.number_pages_allocated() > 0);

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
vec.number_pages_allocated();
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

## 📦 Dependencies

```toml
serde = { version = "1", features = ["derive"] }
bincode = "2"
```

---

## 📄 License

MIT License (see LICENSE file)
