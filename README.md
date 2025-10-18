# Vec64

High-performance Rust vector type with automatic 64-byte SIMD alignment.

## Overview

`Vec64<T>` is a drop-in replacement for `Vec<T>` that ensures the starting pointer is aligned to a 64-byte boundary. This alignment is useful for optimal performance with SIMD instruction extensions like AVX-512, and helps avoid split loads/stores across cache lines.

Benefits will vary based on one's target architecture.

## Includes

- **Automatic 64-byte alignment** for SIMD throughput.
- **Drop-in replacement** for `std::Vec` with same API
- **Parallel processing** support via Rayon (optional feature)
- **Memory safety** with custom `Alloc64` allocator
- **Zero-cost abstraction** - transparent wrapper over `Vec<T, Alloc64>`

See benchmarks.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
vec64 = "0.1.0"

# Enable parallel processing with Rayon
vec64 = { version = "0.1.0", features = ["parallel_proc"] }
```

## Quick Start

```rust
use vec64::{Vec64, vec64};

// Create a new Vec64
let mut v = Vec64::new();
v.push(42);

// Use the vec64! macro
let v = vec64![1, 2, 3, 4, 5];

// From slice
let data = [1, 2, 3, 4, 5];
let v = Vec64::from_slice(&data);

// All standard Vec operations work
v.extend([6, 7, 8]);
println!("Length: {}", v.len());
```

## SIMD Alignment Benefits

- **AVX-512 compatibility** - Required for optimal performance with 512-bit SIMD instructions
- **Cache line optimisation** - Reduces split loads/stores across cache boundaries
- **Hardware prefetch efficiency** - More predictable memory access patterns
- **SIMD library compatibility** - Works seamlessly with `std::simd` and hand-rolled intrinsics

## When to Use Vec64

Vec64 provides the most benefit for:

- **Complex SIMD kernels** - Distribution PDFs, special functions, transforms with multi-region branching
- **Hand-written SIMD code** - Operations that LLVM cannot auto-vectorize
- **Performance-critical algorithms** - Where guaranteed alignment matters for external SIMD libraries
- **AVX-512 workloads** - Where alignment benefits are more pronounced

Vec64 may not provide significant benefits for:

- **Simple auto-vectorizable loops** - LLVM already optimizes these extremely well
- **Trivial operations** - Modern CPUs have similar performance for aligned vs unaligned loads in many cases
- **Non-SIMD workloads** - Where alignment doesn't impact performance

## Looking for more?
Consider the `Minarrow` crate if you want automatic padding, and other typed but high-performant
foundational data structures, with a focus on high-performance data and systems programming.

## Examples

See the `examples/` directory for benchmarks:

- `hotloop_bench_std.rs` - Demonstrates LLVM auto-vectorization on simple loops
- `hotloop_bench_simd.rs` - Compares hand-written SIMD with aligned vs unaligned loads

These benchmarks show that for simple summation, Vec64's benefits are minimal because LLVM auto-vectorizes effectively. The real value comes from complex SIMD kernels that require guaranteed alignment.

## License

MIT Licensed. See LICENSE for details.