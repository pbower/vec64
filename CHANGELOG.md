# Changelog

## [0.5.1] - 2026-08-28

Nightly compatibility release.

The unstable `allocator_api` raw-parts methods were renamed for consistency
with `Box::into_raw_with_allocator`, so `Vec::into_raw_parts_with_alloc` is
now `Vec::into_raw_parts_with_allocator`. This only affects the `mmap`
feature's internal `try_mremap_append` path, so there is no public API
change and this is a patch version bump.

## [0.5.0] - 2026-08-15

Nightly compatibility release.

As of the 2026-08-14 Rust nightly the unstable `allocator_api` `Allocator`
trait no longer provides a `by_ref` method, so the `by_ref` overrides on
`Alloc64` and `MAllocPg64` stopped compiling. Both overrides are removed.
Where code previously called `alloc.by_ref()`, take a reference to the
allocator with `&alloc`, which acts as an allocator through the standard
`impl Allocator for &A` blanket. Removing the methods is a breaking change,
so this is a minor version bump.
