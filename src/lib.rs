//! # Vec64
//!
//! High-performance Rust vector type with automatic 64-byte SIMD alignment.
//!
//! ## Summary
//! `Vec64<T>` is a drop-in replacement for `Vec<T>` that ensures the starting pointer is aligned to a 64-byte boundary.
//! This alignment is useful for optimal performance with SIMD instruction extensions like AVX-512, and helps avoid split loads/stores across cache lines.
//!
//! Benefits will vary based on one's target architecture.

#![feature(allocator_api)]
#![feature(slice_ptr_get)]

pub mod alloc64;
pub mod vec64;

pub use vec64::Vec64;
pub use alloc64::Alloc64;
