//! # MAllocPg64 - mmap-based 64-byte Aligned Allocator
//!
//! Linux-specific allocator using `mmap` for page-aligned allocations,
//! `mremap` for zero-copy growth, and transparent huge page hints via
//! `madvise(MADV_HUGEPAGE)`.
//!
//! ## Allocation strategy
//!
//! Allocations below `HUGE_PAGE` use the system heap allocator with
//! 64-byte alignment, identical to Alloc64.
//!
//! Allocations at or above `HUGE_PAGE` use mmap with `HUGE_PAGE`
//! rounding. Growth uses `mremap(MREMAP_MAYMOVE)` for zero-copy
//! virtual address remapping - the kernel adjusts page tables
//! without copying physical memory.
//!
//! When a heap allocation grows past `HUGE_PAGE`, it transitions to
//! mmap automatically. The reverse transition occurs on shrink.
//!
//! With the `giant_pages` feature, initial allocations >= 1GB use
//! `MAP_HUGETLB | MAP_HUGE_1GB`. Gigantic pages are only used for
//! the initial allocation; growing into 1GB territory from a smaller
//! mapping stays on the `HUGE_PAGE` rounding.
//!
//! ## Configuring the threshold
//!
//! `HUGE_PAGE` defaults to 256 KiB. Override by setting
//! `VEC64_HUGE_PAGE_BYTES` (a power of two, `>= 4096`) in the build
//! environment; the build script regenerates the constant.
//!
//! ## Platform
//!
//! Linux only. Requires mmap, mremap, munmap, madvise.
//!
//! ## Cross-process safety
//!
//! Allocations are `MAP_PRIVATE | MAP_ANONYMOUS`. The mapping is private to
//! the allocating process. Any in-place mutation method that relies on this
//! allocator's growth, shrink, or splice paths (notably `Vec64::delete_range`
//! on Linux with this feature) is therefore safe to call only from the
//! owning process.
//!
//! If a downstream crate re-exports the underlying pointer across a process
//! boundary, for example via a shared-memory wrapper, the other process holds
//! a separate copy-on-write view. Mutations on either side will not be
//! visible to the other, and in-place page remapping on the owning side
//! invalidates the remote view. Treat such buffers as immutable after
//! sharing, or coordinate the lifecycle externally.
use core::alloc::{AllocError, Allocator, Layout};
use core::ptr::NonNull;
use std::ptr::slice_from_raw_parts_mut;

use crate::alloc64::align_layout;

include!(concat!(env!("OUT_DIR"), "/huge_page.rs"));

#[cfg(feature = "giant_pages")]
const GIGANTIC_PAGE: usize = 1024 * 1024 * 1024;

/// # MAllocPg64
///
/// Zero-sized, hybrid allocator with inherent 64-byte alignment.
///
/// Small allocations (< 2MB) use the system heap with 64-byte
/// alignment. Large allocations (>= 2MB) use anonymous mmap
/// rounded to 2MB huge page boundaries. Growth across the 2MB
/// boundary transitions between heap and mmap automatically.
///
/// mmap returns page-aligned memory, so 64-byte alignment is
/// inherently satisfied. Heap allocations enforce 64-byte
/// alignment via layout adjustment.
#[derive(Copy, Clone, Default, Debug)]
pub struct MAllocPg64;

/// Whether this allocation size should use mmap rather than the heap.
#[inline]
pub(crate) fn uses_mmap(size: usize) -> bool {
    size >= HUGE_PAGE
}

/// Round `size` up to the appropriate page boundary.
///
/// With `giant_pages` enabled, sizes >= 1GB round to 1GB.
/// All other sizes round to 2MB. Zero-sized requests get one
/// 2MB page.
#[inline]
pub(crate) fn mapped_size(size: usize) -> usize {
    #[cfg(feature = "giant_pages")]
    if size >= GIGANTIC_PAGE {
        return (size + GIGANTIC_PAGE - 1) & !(GIGANTIC_PAGE - 1);
    }
    let rounded = (size + HUGE_PAGE - 1) & !(HUGE_PAGE - 1);
    if rounded == 0 { HUGE_PAGE } else { rounded }
}

/// Whether this allocation size qualifies for gigantic 1GB pages.
#[inline]
pub(crate) fn uses_giant_pages(_size: usize) -> bool {
    #[cfg(feature = "giant_pages")]
    { _size >= GIGANTIC_PAGE }
    #[cfg(not(feature = "giant_pages"))]
    { false }
}

/// Perform the mmap syscall with appropriate flags.
pub(crate) fn do_mmap(original_size: usize, mapped: usize) -> Result<NonNull<u8>, AllocError> {
    #[cfg(feature = "giant_pages")]
    let flags = if original_size >= GIGANTIC_PAGE {
        libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_HUGETLB | libc::MAP_HUGE_1GB
    } else {
        libc::MAP_PRIVATE | libc::MAP_ANONYMOUS
    };

    #[cfg(not(feature = "giant_pages"))]
    let flags = {
        let _ = original_size;
        libc::MAP_PRIVATE | libc::MAP_ANONYMOUS
    };

    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            mapped,
            libc::PROT_READ | libc::PROT_WRITE,
            flags,
            -1,
            0,
        )
    };

    if ptr == libc::MAP_FAILED {
        return Err(AllocError);
    }

    NonNull::new(ptr as *mut u8).ok_or(AllocError)
}

/// Hint transparent huge pages on a mapping.
/// Skipped for gigantic page allocations which already use explicit huge pages.
#[inline]
pub(crate) fn hint_thp(ptr: NonNull<u8>, original_size: usize, mapped: usize) {
    if uses_giant_pages(original_size) {
        return;
    }
    unsafe {
        libc::madvise(ptr.as_ptr() as *mut libc::c_void, mapped, libc::MADV_HUGEPAGE);
    }
}

/// Wrap a raw pointer and mapped size into the fat NonNull<[u8]> the Allocator trait expects.
///
/// # Safety
/// `ptr` must be non-null and valid for `mapped` bytes.
#[inline]
pub(crate) unsafe fn fat_ptr(ptr: NonNull<u8>, mapped: usize) -> NonNull<[u8]> {
    unsafe { NonNull::new_unchecked(slice_from_raw_parts_mut(ptr.as_ptr(), mapped)) }
}

// ---------------------------------------------------------------------------
// Heap allocation with 64-byte alignment, for small buffers
// ---------------------------------------------------------------------------

#[inline]
fn heap_alloc(layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
    let layout = align_layout(layout);
    let ptr = unsafe { std::alloc::alloc(layout) };
    NonNull::new(ptr)
        .map(|nn| unsafe { fat_ptr(nn, layout.size()) })
        .ok_or(AllocError)
}

#[inline]
fn heap_alloc_zeroed(layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
    let layout = align_layout(layout);
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    NonNull::new(ptr)
        .map(|nn| unsafe { fat_ptr(nn, layout.size()) })
        .ok_or(AllocError)
}

#[inline]
unsafe fn heap_dealloc(ptr: NonNull<u8>, layout: Layout) {
    unsafe { std::alloc::dealloc(ptr.as_ptr(), align_layout(layout)) };
}

#[inline]
unsafe fn heap_grow(
    ptr: NonNull<u8>,
    old: Layout,
    new: Layout,
) -> Result<NonNull<[u8]>, AllocError> {
    let new_layout = align_layout(new);
    let raw = unsafe { std::alloc::realloc(ptr.as_ptr(), align_layout(old), new_layout.size()) };
    NonNull::new(raw)
        .map(|nn| unsafe { fat_ptr(nn, new_layout.size()) })
        .ok_or(AllocError)
}

#[inline]
unsafe fn heap_shrink(
    ptr: NonNull<u8>,
    old: Layout,
    new: Layout,
) -> Result<NonNull<[u8]>, AllocError> {
    let new_layout = align_layout(new);
    let raw = unsafe { std::alloc::realloc(ptr.as_ptr(), align_layout(old), new_layout.size()) };
    NonNull::new(raw)
        .map(|nn| unsafe { fat_ptr(nn, new_layout.size()) })
        .ok_or(AllocError)
}

unsafe impl Allocator for MAllocPg64 {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let size = layout.size();
        if !uses_mmap(size) {
            return heap_alloc(layout);
        }
        let mapped = mapped_size(size);
        let ptr = do_mmap(size, mapped)?;
        hint_thp(ptr, size, mapped);
        Ok(unsafe { fat_ptr(ptr, mapped) })
    }

    /// mmap anonymous pages are zero-filled by the kernel.
    /// Heap path uses alloc_zeroed.
    #[inline]
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let size = layout.size();
        if !uses_mmap(size) {
            return heap_alloc_zeroed(layout);
        }
        self.allocate(layout)
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        let size = layout.size();
        if !uses_mmap(size) {
            return unsafe { heap_dealloc(ptr, layout) };
        }
        let mapped = mapped_size(size);
        unsafe {
            libc::munmap(ptr.as_ptr() as *mut libc::c_void, mapped);
        }
    }

    /// Grows the allocation. Handles four cases:
    ///
    /// - Heap to heap: realloc
    /// - Mmap to mmap: mremap for zero-copy growth
    /// - Heap to mmap: allocate mmap, copy, free heap
    /// - Mmap to heap: cannot happen during growth since new > old
    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let old_mmap = uses_mmap(old.size());
        let new_mmap = uses_mmap(new.size());

        // Heap to heap
        if !old_mmap && !new_mmap {
            return unsafe { heap_grow(ptr, old, new) };
        }

        // Heap to mmap - transition: allocate mmap, copy old data, free heap
        if !old_mmap && new_mmap {
            let new_mapped = mapped_size(new.size());
            let new_ptr = do_mmap(new.size(), new_mapped)?;
            unsafe {
                core::ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_ptr(), old.size());
                heap_dealloc(ptr, old);
            }
            hint_thp(new_ptr, new.size(), new_mapped);
            return Ok(unsafe { fat_ptr(new_ptr, new_mapped) });
        }

        // Mmap to mmap - use mremap
        let old_mapped = mapped_size(old.size());
        let new_mapped = mapped_size(new.size());

        if old_mapped == new_mapped {
            return Ok(unsafe { fat_ptr(ptr, new_mapped) });
        }

        let raw = unsafe {
            libc::mremap(
                ptr.as_ptr() as *mut libc::c_void,
                old_mapped,
                new_mapped,
                libc::MREMAP_MAYMOVE,
            )
        };

        if raw == libc::MAP_FAILED {
            // mremap requires the source range to be a single mapping, and
            // an in-place splice (`Vec64::delete_range`) or page-level append
            // (`Vec64::extend_from_vec64`) can leave the allocation as
            // several adjacent mappings. Relocate through a fresh mapping.
            let new_ptr = do_mmap(new.size(), new_mapped)?;
            unsafe {
                core::ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_ptr(), old.size());
                libc::munmap(ptr.as_ptr() as *mut libc::c_void, old_mapped);
            }
            hint_thp(new_ptr, new.size(), new_mapped);
            return Ok(unsafe { fat_ptr(new_ptr, new_mapped) });
        }

        let nn = NonNull::new(raw as *mut u8).ok_or(AllocError)?;
        hint_thp(nn, new.size(), new_mapped);
        Ok(unsafe { fat_ptr(nn, new_mapped) })
    }

    /// mremap extends with anonymous pages which are zero-filled by the kernel.
    /// Heap path uses grow then zeroes the new region.
    #[inline]
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let old_mmap = uses_mmap(old.size());
        let new_mmap = uses_mmap(new.size());

        // Heap to heap: allocate zeroed, copy old, free old
        if !old_mmap && !new_mmap {
            let new_block = heap_alloc_zeroed(new)?;
            unsafe {
                core::ptr::copy_nonoverlapping(ptr.as_ptr(), new_block.as_mut_ptr(), old.size());
                heap_dealloc(ptr, old);
            }
            return Ok(new_block);
        }

        // Any path involving mmap: mmap pages are zero-filled by kernel
        unsafe { self.grow(ptr, old, new) }
    }

    /// Shrinks the allocation. Handles four cases:
    ///
    /// - Heap to heap: realloc
    /// - Mmap to mmap: release tail pages via munmap
    /// - Mmap to heap: allocate heap, copy, munmap old
    /// - Heap to mmap: cannot happen during shrink since new < old
    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let old_mmap = uses_mmap(old.size());
        let new_mmap = uses_mmap(new.size());

        // Heap to heap
        if !old_mmap && !new_mmap {
            return unsafe { heap_shrink(ptr, old, new) };
        }

        // Mmap to heap - transition: allocate heap, copy, munmap old
        if old_mmap && !new_mmap {
            let new_block = heap_alloc(new)?;
            unsafe {
                core::ptr::copy_nonoverlapping(ptr.as_ptr(), new_block.as_mut_ptr(), new.size());
                let old_mapped = mapped_size(old.size());
                libc::munmap(ptr.as_ptr() as *mut libc::c_void, old_mapped);
            }
            return Ok(new_block);
        }

        // Mmap to mmap
        let old_mapped = mapped_size(old.size());
        let new_mapped = mapped_size(new.size());

        if old_mapped == new_mapped {
            return Ok(unsafe { fat_ptr(ptr, new_mapped) });
        }

        // Crossing from gigantic to non-gigantic page sizes cannot
        // partially unmap 1GB pages at non-1GB boundaries. Fall back
        // to a fresh allocation with copy.
        if uses_giant_pages(old.size()) && !uses_giant_pages(new.size()) {
            let new_ptr = do_mmap(new.size(), new_mapped)?;
            unsafe {
                core::ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_ptr(), new.size());
                libc::munmap(ptr.as_ptr() as *mut libc::c_void, old_mapped);
            }
            hint_thp(new_ptr, new.size(), new_mapped);
            return Ok(unsafe { fat_ptr(new_ptr, new_mapped) });
        }

        // Release tail pages
        unsafe {
            libc::munmap(
                (ptr.as_ptr() as usize + new_mapped) as *mut libc::c_void,
                old_mapped - new_mapped,
            );
        }

        Ok(unsafe { fat_ptr(ptr, new_mapped) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::{Allocator, Layout};

    #[test]
    fn test_allocate_alignment() {
        let a = MAllocPg64;
        for size in [1, 64, 4096, HUGE_PAGE, HUGE_PAGE + 1] {
            let layout = Layout::from_size_align(size, 1).unwrap();
            let ptr = a.allocate(layout).expect("allocate failed");
            let addr = ptr.as_non_null_ptr().as_ptr() as usize;
            assert_eq!(
                addr % 64, 0,
                "Pointer {:#x} not 64-byte aligned for size {}", addr, size
            );
            unsafe { a.deallocate(ptr.as_non_null_ptr(), layout) };
        }
    }

    #[test]
    fn test_small_alloc_uses_heap() {
        let a = MAllocPg64;
        // Small allocation should not get rounded to 2MB
        let layout = Layout::from_size_align(100, 1).unwrap();
        let ptr = a.allocate(layout).expect("allocate failed");
        // Heap allocations return layout.size, not HUGE_PAGE
        assert!(ptr.len() < HUGE_PAGE, "Small alloc should use heap, not mmap");
        let addr = ptr.as_non_null_ptr().as_ptr() as usize;
        assert_eq!(addr % 64, 0);
        unsafe { a.deallocate(ptr.as_non_null_ptr(), layout) };
    }

    #[test]
    fn test_large_alloc_uses_mmap() {
        let a = MAllocPg64;
        let layout = Layout::from_size_align(HUGE_PAGE + 1, 1).unwrap();
        let ptr = a.allocate(layout).expect("allocate failed");
        // Mmap allocations return mapped_size
        assert!(ptr.len() >= HUGE_PAGE * 2);
        unsafe { a.deallocate(ptr.as_non_null_ptr(), layout) };
    }

    #[test]
    fn test_allocate_zeroed() {
        let a = MAllocPg64;
        for size in [4096, HUGE_PAGE + 1] {
            let layout = Layout::from_size_align(size, 1).unwrap();
            let ptr = a.allocate_zeroed(layout).expect("allocate_zeroed failed");
            let data = unsafe {
                std::slice::from_raw_parts(ptr.as_non_null_ptr().as_ptr(), size)
            };
            assert!(data.iter().all(|&b| b == 0));
            unsafe { a.deallocate(ptr.as_non_null_ptr(), layout) };
        }
    }

    #[test]
    fn test_grow_heap_to_heap() {
        let a = MAllocPg64;
        let small = Layout::from_size_align(1024, 1).unwrap();
        let ptr = a.allocate(small).expect("allocate failed");

        // Write a pattern
        let data = unsafe {
            std::slice::from_raw_parts_mut(ptr.as_non_null_ptr().as_ptr(), 1024)
        };
        for (i, byte) in data.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }

        // Grow within heap range
        let medium = Layout::from_size_align(64 * 1024, 1).unwrap();
        let grown = unsafe {
            a.grow(ptr.as_non_null_ptr(), small, medium).expect("grow failed")
        };

        let addr = grown.as_non_null_ptr().as_ptr() as usize;
        assert_eq!(addr % 64, 0);

        let check = unsafe {
            std::slice::from_raw_parts(grown.as_non_null_ptr().as_ptr(), 1024)
        };
        for (i, &byte) in check.iter().enumerate() {
            assert_eq!(byte, (i % 256) as u8, "Data corrupted at offset {}", i);
        }

        unsafe { a.deallocate(grown.as_non_null_ptr(), medium) };
    }

    #[test]
    fn test_grow_heap_to_mmap() {
        let a = MAllocPg64;
        let small = Layout::from_size_align(1024, 1).unwrap();
        let ptr = a.allocate(small).expect("allocate failed");

        // Write a pattern
        let data = unsafe {
            std::slice::from_raw_parts_mut(ptr.as_non_null_ptr().as_ptr(), 1024)
        };
        for (i, byte) in data.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }

        // Grow past threshold into mmap territory
        let big = Layout::from_size_align(HUGE_PAGE + 4096, 1).unwrap();
        let grown = unsafe {
            a.grow(ptr.as_non_null_ptr(), small, big).expect("grow failed")
        };

        let addr = grown.as_non_null_ptr().as_ptr() as usize;
        assert_eq!(addr % 64, 0);

        let check = unsafe {
            std::slice::from_raw_parts(grown.as_non_null_ptr().as_ptr(), 1024)
        };
        for (i, &byte) in check.iter().enumerate() {
            assert_eq!(byte, (i % 256) as u8, "Data corrupted at offset {}", i);
        }

        unsafe { a.deallocate(grown.as_non_null_ptr(), big) };
    }

    #[test]
    fn test_grow_mmap_to_mmap() {
        let a = MAllocPg64;
        let medium = Layout::from_size_align(HUGE_PAGE, 1).unwrap();
        let ptr = a.allocate(medium).expect("allocate failed");

        let data = unsafe {
            std::slice::from_raw_parts_mut(ptr.as_non_null_ptr().as_ptr(), 1024)
        };
        for (i, byte) in data.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }

        // Grow across huge page boundary to force mremap
        let big = Layout::from_size_align(HUGE_PAGE * 3, 1).unwrap();
        let grown = unsafe {
            a.grow(ptr.as_non_null_ptr(), medium, big).expect("grow failed")
        };

        let check = unsafe {
            std::slice::from_raw_parts(grown.as_non_null_ptr().as_ptr(), 1024)
        };
        for (i, &byte) in check.iter().enumerate() {
            assert_eq!(byte, (i % 256) as u8, "Data corrupted at offset {}", i);
        }

        unsafe { a.deallocate(grown.as_non_null_ptr(), big) };
    }

    #[test]
    fn test_grow_within_mapping_noop() {
        let a = MAllocPg64;
        // Allocate in mmap territory - gets rounded to 4MB
        let initial = Layout::from_size_align(HUGE_PAGE + 1, 1).unwrap();
        let ptr = a.allocate(initial).expect("allocate failed");
        let original_addr = ptr.as_non_null_ptr().as_ptr() as usize;

        // Grow within the same 4MB mapping
        let slightly_bigger = Layout::from_size_align(HUGE_PAGE * 2, 1).unwrap();
        let grown = unsafe {
            a.grow(ptr.as_non_null_ptr(), initial, slightly_bigger).expect("grow failed")
        };
        let grown_addr = grown.as_non_null_ptr().as_ptr() as usize;
        assert_eq!(original_addr, grown_addr, "Should reuse same mapping");

        unsafe { a.deallocate(grown.as_non_null_ptr(), slightly_bigger) };
    }

    #[test]
    fn test_shrink_mmap_to_heap() {
        let a = MAllocPg64;
        let big = Layout::from_size_align(HUGE_PAGE * 2, 1).unwrap();
        let ptr = a.allocate(big).expect("allocate failed");

        // Write pattern
        let data = unsafe {
            std::slice::from_raw_parts_mut(ptr.as_non_null_ptr().as_ptr(), 1024)
        };
        for (i, byte) in data.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }

        // Shrink below threshold back to heap
        let small = Layout::from_size_align(1024, 1).unwrap();
        let shrunk = unsafe {
            a.shrink(ptr.as_non_null_ptr(), big, small).expect("shrink failed")
        };

        let addr = shrunk.as_non_null_ptr().as_ptr() as usize;
        assert_eq!(addr % 64, 0);

        let check = unsafe {
            std::slice::from_raw_parts(shrunk.as_non_null_ptr().as_ptr(), 1024)
        };
        for (i, &byte) in check.iter().enumerate() {
            assert_eq!(byte, (i % 256) as u8, "Data corrupted at offset {}", i);
        }

        unsafe { a.deallocate(shrunk.as_non_null_ptr(), small) };
    }

    #[test]
    fn test_shrink_mmap_to_mmap() {
        let a = MAllocPg64;
        let big = Layout::from_size_align(HUGE_PAGE * 3, 1).unwrap();
        let ptr = a.allocate(big).expect("allocate failed");

        let data = unsafe {
            std::slice::from_raw_parts_mut(ptr.as_non_null_ptr().as_ptr(), 1024)
        };
        for (i, byte) in data.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }

        // Shrink but stay in mmap territory
        let medium = Layout::from_size_align(HUGE_PAGE, 1).unwrap();
        let shrunk = unsafe {
            a.shrink(ptr.as_non_null_ptr(), big, medium).expect("shrink failed")
        };

        let addr = shrunk.as_non_null_ptr().as_ptr() as usize;
        assert_eq!(addr % 64, 0);

        let check = unsafe {
            std::slice::from_raw_parts(shrunk.as_non_null_ptr().as_ptr(), 1024)
        };
        for (i, &byte) in check.iter().enumerate() {
            assert_eq!(byte, (i % 256) as u8, "Data corrupted at offset {}", i);
        }

        unsafe { a.deallocate(shrunk.as_non_null_ptr(), medium) };
    }

    #[test]
    fn test_grow_zeroed_heap() {
        let a = MAllocPg64;
        let small = Layout::from_size_align(1024, 1).unwrap();
        let ptr = a.allocate(small).expect("allocate failed");

        let bigger = Layout::from_size_align(4096, 1).unwrap();
        let grown = unsafe {
            a.grow_zeroed(ptr.as_non_null_ptr(), small, bigger).expect("grow_zeroed failed")
        };

        // New region should be zeroed
        let check = unsafe {
            std::slice::from_raw_parts(grown.as_non_null_ptr().as_ptr().add(1024), 4096 - 1024)
        };
        assert!(check.iter().all(|&b| b == 0), "New region not zeroed");

        unsafe { a.deallocate(grown.as_non_null_ptr(), bigger) };
    }

    #[test]
    fn test_grow_zeroed_mmap() {
        let a = MAllocPg64;
        let small = Layout::from_size_align(HUGE_PAGE, 1).unwrap();
        let ptr = a.allocate(small).expect("allocate failed");

        let big = Layout::from_size_align(HUGE_PAGE * 3, 1).unwrap();
        let grown = unsafe {
            a.grow_zeroed(ptr.as_non_null_ptr(), small, big).expect("grow_zeroed failed")
        };

        // Check that new region past original mapping is zeroed
        let new_start = HUGE_PAGE;
        let check = unsafe {
            std::slice::from_raw_parts(
                grown.as_non_null_ptr().as_ptr().add(new_start),
                4096,
            )
        };
        assert!(check.iter().all(|&b| b == 0), "New region not zeroed after grow_zeroed");

        unsafe { a.deallocate(grown.as_non_null_ptr(), big) };
    }

    /// Classification stability at the heap/mmap boundary: the size a
    /// caller later derives from the returned block must select the same
    /// path the allocation took, and must recompute the same mapped
    /// extent.
    #[test]
    fn test_boundary_classification_stable() {
        let a = MAllocPg64;
        for size in [
            HUGE_PAGE - 64,
            HUGE_PAGE - 8,
            HUGE_PAGE - 1,
            HUGE_PAGE,
            HUGE_PAGE + 1,
        ] {
            let layout = Layout::from_size_align(size, 1).unwrap();
            let ptr = a.allocate(layout).expect("allocate failed");
            let returned = ptr.len();
            assert_eq!(
                uses_mmap(size),
                uses_mmap(returned),
                "classification flipped: requested {size}, returned {returned}"
            );
            if uses_mmap(returned) {
                assert_eq!(mapped_size(returned), returned, "mapped_size not idempotent");
            } else {
                assert_eq!(returned, size, "heap path changed the size");
            }
            unsafe { a.deallocate(ptr.as_non_null_ptr(), layout) };
        }
    }

    #[test]
    fn test_mapped_size_rounding() {
        assert_eq!(mapped_size(0), HUGE_PAGE);
        assert_eq!(mapped_size(1), HUGE_PAGE);
        assert_eq!(mapped_size(HUGE_PAGE), HUGE_PAGE);
        assert_eq!(mapped_size(HUGE_PAGE + 1), HUGE_PAGE * 2);
        assert_eq!(mapped_size(HUGE_PAGE * 3), HUGE_PAGE * 3);
    }
}
