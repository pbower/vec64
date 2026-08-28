//! # **Vec64** - *Special Vector with 64-Byte SIMD Alignment*
//!
//! 64-byte aligned vector type backed by a custom allocator (`Vec64Alloc`).
//!
//! Provides the same API as `Vec`, but guarantees the starting address
//! of the allocation is 64-byte aligned for SIMD, cache line, and
//! low-level hardware optimisations.

use std::borrow::{Borrow, BorrowMut};
use std::fmt::{Debug, Display, Formatter, Result};
use std::mem;
use std::ops::{Deref, DerefMut};
use std::slice::{Iter, IterMut};
use std::vec::Vec;

#[cfg(any(feature = "parallel_proc", feature = "wasm"))]
use rayon::iter::{IntoParallelRefIterator, IntoParallelRefMutIterator};

use crate::Vec64Alloc;

/// # Vec64
///
/// High-performance 64-byte aligned vector.
///
/// ## Purpose
/// A drop-in replacement for `Vec` that ensures the starting pointer is aligned to a
/// 64-byte boundary via a custom `Vec64Alloc` allocator. This predominantly ensures
/// compatibility with SIMD processing instruction extensions such as the AVX-512.
/// These increase CPU throughput when using SIMD-friendly code like `std::simd`, or hand-rolled intrinsics.
///
/// Alignment can help avoid split loads/stores across cache lines and make hardware
/// prefetch more predictable during sequential scans. However, gains are workload- and
/// platform-dependent, and the Rust compiler may generate equally efficient code for
/// ordinary `Vec` in some cases.
///
/// ## Behaviour - Padding
/// This type does not add any padding to your data. Only the first element of the
/// allocation is guaranteed to be aligned. If you construct a buffer that mixes headers,
/// metadata, and then Arrow data pages, and you plan to extract or process the Arrow
/// portion with `Vec64::from_raw_parts` or SIMD at its offset, you must insert your own
/// zero-byte padding so that the Arrow section’s start falls on a 64-byte boundary.
/// Without that manual padding, the middle of the buffer will not be aligned and
/// unaligned access or unsafe reconstitution may fail or force a reallocation.
///
/// All library code in `Minarrow` and `Simd-Kernels` high-performance crates
/// automatically handles such padding, and therefore this is only relevant if you leverage `Vec64` manually.
///
/// ## Notes
/// - All `Vec` APIs remain available - `Vec64` is a tuple wrapper over `Vec<T, Vec64Alloc>`.
/// - When passing to APIs expecting a `Vec`, use `.0` to extract the inner `Vec`.
/// - Avoid mixing `Vec` and `Vec64` unless both use the same custom allocator (`Vec64Alloc`).
/// - Alignment helps with contiguous, stride-friendly access; it does not improve
///   temporal locality or benefit random-access patterns.
#[repr(transparent)]
pub struct Vec64<T>(pub Vec<T, Vec64Alloc>);

impl<T> Vec64<T> {
    #[inline]
    pub fn new() -> Self {
        Self(Vec::new_in(Vec64Alloc::default()))
    }

    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        Self(Vec::with_capacity_in(cap, Vec64Alloc::default()))
    }

    /// Useful when interpreting raw bytes that are buffered
    /// in a Vec64 compatible manner, from network sockets etc.,
    /// to avoid needing to copy.
    ///
    /// # Safety
    /// - `buf` must have come from a `Vec64<u8>` that owns the allocation.
    /// - `T` must be POD (plain old data), properly aligned (which `Vec64` guarantees).
    /// - `buf.len() % size_of::<T>() == 0`
    /// - `buf.capacity() % size_of::<T>() == 0` (to ensure deallocation safety)
    pub unsafe fn from_vec64_u8(buf: Vec64<u8>) -> Vec64<T> {
        let byte_len = buf.len();
        let byte_cap = buf.0.capacity();
        let elem_size = mem::size_of::<T>();

        assert!(
            byte_len % elem_size == 0,
            "Length must be multiple of element size"
        );
        assert!(
            byte_cap % elem_size == 0,
            "Capacity must be multiple of element size for safe deallocation"
        );

        let ptr = buf.0.as_ptr() as *mut T;
        let len = byte_len / elem_size;
        let cap = byte_cap / elem_size;

        // Prevent Vec64<u8> destructor from running - we're transferring ownership to Vec64<T>
        mem::forget(buf);

        let vec = unsafe { Vec::from_raw_parts_in(ptr, len, cap, Vec64Alloc::default()) };
        Vec64(vec)
    }

    /// Takes ownership of a raw allocation.
    ///
    /// # Safety:
    /// - `ptr` must have been allocated by `Vec64Alloc` (or compatible 64-byte aligned allocator)
    /// - `ptr` must be valid for reads and writes for `len * size_of::<T>()` bytes
    /// - `len` must be less than or equal to `capacity`
    /// - The memory must not be aliased elsewhere
    #[inline]
    pub unsafe fn from_raw_parts(ptr: *mut T, len: usize, capacity: usize) -> Self {
        debug_assert_eq!(
            (ptr as usize) % 64,
            0,
            "Vec64::from_raw_parts: pointer is not 64-byte aligned"
        );

        let vec = unsafe { Vec::from_raw_parts_in(ptr, len, capacity, Vec64Alloc::default()) };
        Self(vec)
    }

    /// Splits the collection at the given index.
    ///
    /// Returns a newly allocated vector containing the elements in the range `[at, len)`.
    /// After the call, the original vector will be left containing the elements `[0, at)` with its previous capacity unchanged.
    ///
    /// # Panics
    /// Panics if `at > len`.
    ///
    /// # Examples
    /// ```
    /// use vec64::Vec64;
    ///
    /// let mut vec = Vec64::from(vec![1, 2, 3, 4, 5]);
    /// let vec2 = vec.split_off(2);
    /// assert_eq!(&vec[..], &[1, 2]);
    /// assert_eq!(&vec2[..], &[3, 4, 5]);
    /// ```
    #[inline]
    pub fn split_off(&mut self, at: usize) -> Self {
        Vec64(self.0.split_off(at))
    }

    /// Removes the elements in the range `[start, end)`, shifting any later
    /// elements left so the surviving prefix and suffix become contiguous.
    ///
    /// `end == start` is a no-op. `end == len` short-circuits to `truncate`.
    ///
    /// On Linux with the `mmap` feature, when the allocation is mmap-backed
    /// and the deleted byte span `(end - start) * size_of::<T>()` is a whole
    /// multiple of the system page size, the surviving tail is relocated by
    /// remapping pages rather than copying bytes, and the vacated pages are
    /// returned to the kernel via `madvise(MADV_FREE)`. The position of the
    /// range carries no constraint: a range starting mid-page is handled by
    /// patching the boundary page with a copy of less than one page. The
    /// zero-copy relocation requires Linux 5.7+; older kernels complete
    /// through an equivalent in-place copy. All other inputs go through
    /// `Vec::drain`. Capacity is preserved on every path.
    ///
    /// ## Cross-process safety
    ///
    /// The `mmap` allocation is `MAP_PRIVATE | MAP_ANONYMOUS`. Mutations
    /// apply only within the owning process. If the buffer is re-exported
    /// across a process boundary, the remote view is invalidated by
    /// in-place page remapping on the owning side.
    ///
    /// # Panics
    /// Panics if `start > end` or `end > len`.
    ///
    /// # Examples
    /// ```
    /// use vec64::Vec64;
    ///
    /// let mut v = Vec64::from(vec![1, 2, 3, 4, 5]);
    /// v.delete_range(1, 4);
    /// assert_eq!(&v[..], &[1, 5]);
    /// ```
    pub fn delete_range(&mut self, start: usize, end: usize) {
        assert!(start <= end, "Vec64::delete_range: start ({start}) > end ({end})");
        assert!(
            end <= self.0.len(),
            "Vec64::delete_range: end ({end}) > len ({})",
            self.0.len()
        );

        if start == end {
            return;
        }

        // Tail delete - drop the suffix in place, no shift required.
        if end == self.0.len() {
            self.0.truncate(start);
            return;
        }

        #[cfg(all(feature = "mmap", target_os = "linux"))]
        {
            if unsafe { self.try_mremap_splice(start, end) } {
                return;
            }
        }

        // Fallback path. drain handles drop ordering, panic safety, and the
        // tail memmove. Capacity is preserved.
        self.0.drain(start..end);
    }

    /// mmap fast path for `delete_range`. Returns `true` when the splice
    /// succeeded; `false` to signal the caller should fall back to `drain`.
    ///
    /// Fires when the allocation is mmap-backed and the deleted byte span is
    /// a whole multiple of the system page size. The position of the range
    /// carries no constraint: when `start` sits mid-page, the leading tail
    /// bytes are first copied into the gap so the relocation below starts
    /// and ends on page boundaries i.e. the boundary-page patch costs less
    /// than one page of copying.
    ///
    /// The tail relocates in two `mremap` steps: onto a scratch mapping with
    /// `MREMAP_DONTUNMAP`, which keeps the source range mapped so the
    /// allocation never contains unmapped holes, then onto its final
    /// position. Capacity is preserved, matching `drain`, and the vacated
    /// pages past the new length are handed back to the kernel with
    /// `madvise(MADV_FREE)`. Kernels without `MREMAP_DONTUNMAP` (pre-5.7)
    /// complete through an equivalent in-place copy.
    ///
    /// # Safety
    /// Caller must guarantee `start < end <= self.len()` and `end < self.len()`
    /// i.e. a strict middle/head delete that is not a tail delete and not
    /// empty.
    #[cfg(all(feature = "mmap", target_os = "linux"))]
    unsafe fn try_mremap_splice(&mut self, start: usize, end: usize) -> bool {
        use std::mem::size_of;
        use std::ptr;

        // System page size, queried once. mremap operates at this granularity
        // even when the underlying mapping uses 2 MiB transparent huge pages.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        if page == 0 || !page.is_power_of_two() {
            return false;
        }

        let elem = size_of::<T>();
        if elem == 0 {
            // ZSTs don't touch the allocator. The drain fallback handles len bookkeeping.
            return false;
        }

        let len = self.0.len();
        let cap = self.0.capacity();
        let byte_start = start * elem;
        let byte_end = end * elem;
        let byte_len = len * elem;

        // The tail shifts left by the deleted span, and pages relocate only
        // in whole-page units, so the span must be a page multiple.
        if (byte_end - byte_start) & (page - 1) != 0 {
            return false;
        }

        // The allocation must actually be mmap-backed. The Cargo features here
        // gate the import path, so this lookup only compiles when mmap_alloc
        // is in the module tree.
        if !crate::mmap_alloc::uses_mmap(cap * elem) {
            return false;
        }

        // After this point the splice is committed. Every failure mode below
        // completes the delete through an in-place copy, leaving the Vec valid.

        let base = self.0.as_mut_ptr() as *mut u8;
        let new_len = start + (len - end);

        // Drop the elements in the deleted middle. set_len lower first so that
        // a panic in T::drop does not leave the tail in scope for double-drop.
        // The tail [end..len) is leaked on panic which is preferable to UB.
        unsafe {
            self.0.set_len(start);
            ptr::drop_in_place(ptr::slice_from_raw_parts_mut(
                (base as *mut T).add(start),
                end - start,
            ));
        }

        // Boundary-page patch: when byte_start sits mid-page, copy the
        // leading tail bytes into the gap up to the next page boundary so
        // the relocation below starts and ends page-aligned. The regions
        // cannot overlap: the patch ends at or before byte_end, where the
        // tail begins.
        let seam = byte_start.next_multiple_of(page) - byte_start;
        let tail_total = byte_len - byte_end;
        if seam > 0 {
            unsafe {
                ptr::copy_nonoverlapping(
                    base.add(byte_end) as *const u8,
                    base.add(byte_start),
                    seam.min(tail_total),
                );
            }
            if tail_total <= seam {
                // The whole tail fits inside the patch; no pages move.
                unsafe {
                    self.0.set_len(new_len);
                    let used = (new_len * elem).next_multiple_of(page);
                    let mapped_end = byte_len.next_multiple_of(page);
                    if mapped_end > used {
                        libc::madvise(
                            base.add(used) as *mut libc::c_void,
                            mapped_end - used,
                            libc::MADV_FREE,
                        );
                    }
                }
                return true;
            }
        }

        let splice_src = byte_end + seam;
        let splice_dst = byte_start + seam;
        let tail_bytes = byte_len - splice_src;
        // tail_bytes > 0: the patch-absorbed case returned above.
        let tail_mapped = (tail_bytes + page - 1) & !(page - 1);

        let src = unsafe { base.add(splice_src) } as *mut libc::c_void;
        let dst = unsafe { base.add(splice_dst) } as *mut libc::c_void;

        // The kernel rejects mremap MREMAP_FIXED when source and destination
        // overlap, and mremap MREMAP_MAYMOVE alone is a no-op when the size
        // is unchanged. To relocate the tail without a data copy we therefore
        // need a scratch mapping in unrelated virtual address space, then two
        // FIXED mremap steps that each move into a fresh, non-overlapping slot.
        //
        // Step 1: reserve a scratch region anywhere the kernel chooses.
        let scratch = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                tail_mapped,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if scratch == libc::MAP_FAILED {
            // Scratch reservation failed. Fall back to an in-place memmove.
            unsafe {
                ptr::copy(base.add(splice_src), base.add(splice_dst), tail_bytes);
                self.0.set_len(new_len);
            }
            return true;
        }

        // Step 2: relocate the tail pages into the scratch region. src lives
        // inside the original allocation; scratch is in a freshly mapped slot,
        // so they cannot overlap and MREMAP_FIXED is safe. MREMAP_DONTUNMAP
        // keeps the source range mapped as zero-fill pages, so the move in
        // step 3 only ever replaces pages this allocation owns. Kernels
        // without MREMAP_DONTUNMAP report EINVAL and the copy path completes
        // instead.
        let moved = unsafe {
            libc::mremap(
                src,
                tail_mapped,
                tail_mapped,
                libc::MREMAP_MAYMOVE | libc::MREMAP_FIXED | libc::MREMAP_DONTUNMAP,
                scratch,
            )
        };
        if moved == libc::MAP_FAILED {
            // Roll back via in-place memmove and release the scratch reservation.
            unsafe {
                libc::munmap(scratch, tail_mapped);
                ptr::copy(base.add(splice_src), base.add(splice_dst), tail_bytes);
                self.0.set_len(new_len);
            }
            return true;
        }

        // Step 3: relocate from scratch into the final destination. scratch
        // is in unrelated virtual address space; dst is inside the original
        // allocation; they cannot overlap on any sane kernel layout. If they
        // ever do, fall back to a single memcpy.
        let scratch_lo = scratch as usize;
        let scratch_hi = scratch_lo + tail_mapped;
        let dst_lo = dst as usize;
        let dst_hi = dst_lo + tail_mapped;
        let overlaps = scratch_lo < dst_hi && dst_lo < scratch_hi;

        if overlaps {
            unsafe {
                ptr::copy_nonoverlapping(scratch as *const u8, dst as *mut u8, tail_bytes);
                libc::munmap(scratch, tail_mapped);
                self.0.set_len(new_len);
            }
            return true;
        }

        let placed = unsafe {
            libc::mremap(
                scratch,
                tail_mapped,
                tail_mapped,
                libc::MREMAP_MAYMOVE | libc::MREMAP_FIXED,
                dst,
            )
        };
        if placed == libc::MAP_FAILED {
            unsafe {
                ptr::copy_nonoverlapping(scratch as *const u8, dst as *mut u8, tail_bytes);
                libc::munmap(scratch, tail_mapped);
                self.0.set_len(new_len);
            }
            return true;
        }

        // The allocation's extent and capacity are unchanged; only contents
        // moved. Hand the pages past the new length back to the kernel: the
        // addresses stay mapped and writable, and the kernel reclaims the
        // physical frames under memory pressure.
        unsafe {
            self.0.set_len(new_len);
            let used = (new_len * elem).next_multiple_of(page);
            let mapped_end = byte_len.next_multiple_of(page);
            if mapped_end > used {
                libc::madvise(
                    base.add(used) as *mut libc::c_void,
                    mapped_end - used,
                    libc::MADV_FREE,
                );
            }
        }

        true
    }

    /// Appends `other` to `self`, consuming it.
    ///
    /// On Linux with the `mmap` feature, when both buffers are mmap-backed,
    /// `other` is tightly packed (`cap == len`), its byte size is a
    /// multiple of `HUGE_PAGE`, and `self`'s write position is page-aligned,
    /// the append goes through `mremap`. Other inputs go through a
    /// per-element move.
    ///
    /// ## Cross-process safety
    ///
    /// Mutations on `Vec64` apply only within the owning process. A buffer
    /// re-exported across a process boundary (memfd, shared memory) is
    /// invalidated by in-place page remapping on the owning side.
    ///
    /// # Examples
    /// ```
    /// use vec64::Vec64;
    ///
    /// let mut head: Vec64<u64> = Vec64::from(vec![1, 2, 3]);
    /// let tail: Vec64<u64> = Vec64::from(vec![4, 5, 6]);
    /// head.extend_from_vec64(tail);
    /// assert_eq!(&head[..], &[1, 2, 3, 4, 5, 6]);
    /// ```
    pub fn extend_from_vec64(&mut self, other: Vec64<T>) {
        if other.is_empty() {
            return;
        }

        #[cfg(all(feature = "mmap", target_os = "linux"))]
        {
            match unsafe { self.try_mremap_append(other) } {
                None => return,
                Some(returned) => {
                    self.0.extend(returned.0);
                    return;
                }
            }
        }

        #[cfg(not(all(feature = "mmap", target_os = "linux")))]
        {
            self.0.extend(other.0);
        }
    }

    /// Concatenates the given chunks in order. Each chunk is consumed.
    ///
    /// Preallocates the destination with the total length and forwards
    /// each chunk through `extend_from_vec64`. Returns an empty
    /// `Vec64<T>` when `chunks` is empty.
    ///
    /// # Examples
    /// ```
    /// use vec64::Vec64;
    ///
    /// let a: Vec64<u8> = Vec64::from_slice(&[1, 2]);
    /// let b: Vec64<u8> = Vec64::from_slice(&[3, 4, 5]);
    /// let c: Vec64<u8> = Vec64::from_slice(&[6]);
    /// let joined = Vec64::from_chunks(vec![a, b, c]);
    /// assert_eq!(&joined[..], &[1, 2, 3, 4, 5, 6]);
    /// ```
    pub fn from_chunks(chunks: Vec<Vec64<T>>) -> Vec64<T> {
        let total: usize = chunks.iter().map(|c| c.len()).sum();
        if total == 0 {
            return Vec64::new();
        }
        let mut out = Vec64::with_capacity(total);
        for chunk in chunks {
            out.extend_from_vec64(chunk);
        }
        out
    }

    /// Returns `None` on success (other consumed). Returns `Some(other)`
    /// when the caller must use the move path.
    ///
    /// # Safety
    /// Caller must ensure `other` is non-empty.
    #[cfg(all(feature = "mmap", target_os = "linux"))]
    unsafe fn try_mremap_append(&mut self, other: Vec64<T>) -> Option<Vec64<T>> {
        use std::mem::size_of;

        let elem = size_of::<T>();
        if elem == 0 {
            return Some(other);
        }

        let self_len = self.0.len();
        let self_cap = self.0.capacity();
        let other_len = other.0.len();
        let other_cap = other.0.capacity();

        if self_cap - self_len < other_len {
            return Some(other);
        }
        if other_cap != other_len {
            return Some(other);
        }

        let dst_offset_bytes = self_len * elem;
        let src_bytes = other_len * elem;

        if src_bytes & (crate::mmap_alloc::HUGE_PAGE - 1) != 0 {
            return Some(other);
        }

        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        if page == 0 || !page.is_power_of_two() {
            return Some(other);
        }
        if dst_offset_bytes & (page - 1) != 0 {
            return Some(other);
        }

        if !crate::mmap_alloc::uses_mmap(self_cap * elem)
            || !crate::mmap_alloc::uses_mmap(other_cap * elem)
        {
            return Some(other);
        }

        let dst_ptr = unsafe { (self.0.as_mut_ptr() as *mut u8).add(dst_offset_bytes) };

        let (src_ptr, _src_len, _src_cap, _src_alloc) = other.0.into_raw_parts_with_allocator();
        let src_ptr_u8 = src_ptr as *mut u8;

        let placed = unsafe {
            libc::mremap(
                src_ptr_u8 as *mut libc::c_void,
                src_bytes,
                src_bytes,
                libc::MREMAP_MAYMOVE | libc::MREMAP_FIXED,
                dst_ptr as *mut libc::c_void,
            )
        };
        if placed == libc::MAP_FAILED {
            let recovered = unsafe {
                Vec::from_raw_parts_in(src_ptr, other_len, other_cap, Vec64Alloc::default())
            };
            return Some(Vec64(recovered));
        }

        unsafe { self.0.set_len(self_len + other_len) };
        None
    }
}

// Only require Send+Sync for parallel iterator methods
#[cfg(any(feature = "parallel_proc", feature = "wasm"))]
impl<T: Sync + Send> Vec64<T> {
    #[inline]
    pub fn par_iter(&self) -> rayon::slice::Iter<'_, T> {
        self.0.par_iter()
    }

    #[inline]
    pub fn par_iter_mut(&mut self) -> rayon::slice::IterMut<'_, T> {
        self.0.par_iter_mut()
    }
}

impl<T: Copy> Vec64<T> {
    #[inline]
    pub fn from_slice(slice: &[T]) -> Self {
        let mut v = Self::with_capacity(slice.len());
        // SAFETY: allocated enough capacity, and both
        // pointers are non-overlapping.
        unsafe {
            std::ptr::copy_nonoverlapping(slice.as_ptr(), v.0.as_mut_ptr(), slice.len());
            v.0.set_len(slice.len());
        }
        v
    }
}

impl<T: Clone> Vec64<T> {
    #[inline]
    pub fn from_slice_clone(slice: &[T]) -> Self {
        let mut v = Self::with_capacity(slice.len());
        v.0.extend_from_slice(slice);
        v
    }
}

impl<T> Default for Vec64<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Deref for Vec64<T> {
    type Target = Vec<T, Vec64Alloc>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Vec64<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: Clone> Clone for Vec64<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T: Debug> Debug for Vec64<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        self.0.fmt(f)
    }
}

impl<T: PartialEq> PartialEq for Vec64<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T: Display> Display for Vec64<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "[")?;
        for (i, item) in self.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{item}")?;
        }
        write!(f, "]")
    }
}

impl<T> IntoIterator for Vec64<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T, Vec64Alloc>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a Vec64<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}
impl<'a, T> IntoIterator for &'a mut Vec64<T> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}

impl<T> Extend<T> for Vec64<T> {
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.0.extend(iter)
    }
}

impl<T> FromIterator<T> for Vec64<T> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iterator = iter.into_iter();
        let mut v = if let Some(exact) = iterator.size_hint().1 {
            Vec::with_capacity_in(exact, Vec64Alloc::default())
        } else {
            Vec::with_capacity_in(iterator.size_hint().0, Vec64Alloc::default())
        };
        v.extend(iterator);
        Self(v)
    }
}

impl<T> From<Vec<T, Vec64Alloc>> for Vec64<T> {
    #[inline]
    fn from(v: Vec<T, Vec64Alloc>) -> Self {
        Self(v)
    }
}

impl<T> From<Vec64<T>> for Vec<T, Vec64Alloc> {
    #[inline]
    fn from(v: Vec64<T>) -> Self {
        v.0
    }
}

impl<T> From<Vec<T>> for Vec64<T> {
    #[inline]
    fn from(v: Vec<T>) -> Self {
        let mut vec = Vec::with_capacity_in(v.len(), Vec64Alloc::default());
        vec.extend(v);
        Self(vec)
    }
}

impl<T> From<&[T]> for Vec64<T>
where
    T: Clone,
{
    #[inline]
    fn from(s: &[T]) -> Self {
        let mut v = Vec::with_capacity_in(s.len(), Vec64Alloc::default());
        v.extend_from_slice(s);
        Self(v)
    }
}

impl<T> AsRef<[T]> for Vec64<T> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}
impl<T> AsMut<[T]> for Vec64<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<T> Borrow<[T]> for Vec64<T> {
    #[inline]
    fn borrow(&self) -> &[T] {
        self.0.borrow()
    }
}
impl<T> BorrowMut<[T]> for Vec64<T> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut [T] {
        self.0.borrow_mut()
    }
}

/// Allow `Vec64<u8>` to act as an `io::Write` sink, appending incoming bytes
/// into the 64-byte aligned allocation. Mirrors the standard library's
/// `impl Write for Vec<u8>` so any writer expecting `io::Write` can target
/// an aligned buffer without an intermediate adapter.
impl std::io::Write for Vec64<u8> {
    #[inline]
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        self.0.extend_from_slice(data);
        Ok(data.len())
    }

    #[inline]
    fn write_all(&mut self, data: &[u8]) -> std::io::Result<()> {
        self.0.extend_from_slice(data);
        Ok(())
    }

    #[inline]
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[macro_export]
macro_rules! vec64 {
    // Bool: repetition form
    (bool $elem:expr; $n:expr) => {{
        let len       = $n as usize;
        let byte_len  = (len + 7) / 8;
        let mut v     = $crate::Vec64::<u8>::with_capacity(byte_len);

        // Fill the buffer in one shot.
        let fill = if $elem { 0xFFu8 } else { 0u8 };
        v.0.resize(byte_len, fill);

        // Clear padding bits when fill == 1 and len is not a multiple of 8.
        if $elem && (len & 7) != 0 {
            let mask  = (1u8 << (len & 7)) - 1;
            let last  = byte_len - 1;
            v.0[last] &= mask;
        }
        v
    }};

    // Bool: list form
    (bool $($x:expr),+ $(,)?) => {{
        // Count elements at macro-expansion time.
        let len: usize = 0 $(+ { let _ = &$x; 1 })*;
        let byte_len   = (len + 7) / 8;
        let mut v      = $crate::Vec64::<u8>::with_capacity(byte_len);
        v.0.resize(byte_len, 0);

        // Sequentially set bits - no reallocations.
        let mut _idx = 0usize;
        $(
            if $x {
                let byte_idx = _idx / 8;
                let bit_idx = _idx % 8;
                v.0[byte_idx] |= 1u8 << bit_idx;
            }
            _idx += 1;
        )+
        v
    }};

    // Generic forms
    () => {
        $crate::Vec64::new()
    };

    ($elem:expr; $n:expr) => {{
        let mut v = $crate::Vec64::with_capacity($n);
        v.0.resize($n, $elem);
        v
    }};

    ($($x:expr),+ $(,)?) => {{
        let mut v = $crate::Vec64::with_capacity(0 $(+ { let _ = &$x; 1 })*);
        $(v.push($x);)+
        v
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "parallel_proc")]
    #[test]
    fn test_new_and_default() {
        let v: Vec64<u32> = Vec64::new();
        assert_eq!(v.len(), 0);
        assert_eq!(v.capacity(), 0);

        let d: Vec64<u32> = Default::default();
        assert_eq!(d.len(), 0);
    }

    #[test]
    fn test_with_capacity_and_alignment() {
        let v: Vec64<u64> = Vec64::with_capacity(32);
        assert_eq!(v.len(), 0);
        assert!(v.capacity() >= 32);
        // Underlying allocation must be 64-byte aligned
        assert_eq!(v.0.as_ptr() as usize % 64, 0);
    }

    #[test]
    fn test_from_slice_and_from() {
        let data = [1, 2, 3, 4, 5];
        let v = Vec64::from_slice(&data);
        assert_eq!(v.len(), 5);
        assert_eq!(&v[..], &data);

        let v2: Vec64<_> = Vec64::from(&data[..]);
        assert_eq!(&v2[..], &data);
    }

    #[test]
    fn test_vec_macro() {
        let v = vec64![1, 2, 3, 4, 5];
        assert_eq!(&v[..], &[1, 2, 3, 4, 5]);

        let v2 = vec64![7u8; 4];
        assert_eq!(&v2[..], &[7u8; 4]);
    }

    #[test]
    fn test_extend_and_from_iter() {
        let mut v = Vec64::new();
        v.extend([10, 20, 30]);
        assert_eq!(&v[..], &[10, 20, 30]);

        let v2: Vec64<_> = [100, 200].into_iter().collect();
        assert_eq!(&v2[..], &[100, 200]);
    }

    #[test]
    fn test_push_and_index() {
        let mut v = Vec64::with_capacity(2);
        v.push(123);
        v.push(456);
        assert_eq!(v[0], 123);
        assert_eq!(v[1], 456);
    }

    #[test]
    fn test_as_ref_and_as_mut() {
        let mut v = Vec64::from_slice(&[1, 2, 3]);
        assert_eq!(v.as_ref(), &[1, 2, 3]);
        v.as_mut()[1] = 99;
        assert_eq!(v[1], 99);
    }

    #[test]
    fn test_borrow_traits() {
        use std::borrow::{Borrow, BorrowMut};
        let mut v = Vec64::from_slice(&[4, 5, 6]);
        let r: &[i32] = v.borrow();
        assert_eq!(r, &[4, 5, 6]);
        let r: &mut [i32] = v.borrow_mut();
        r[0] = 42;
        assert_eq!(v[0], 42);
    }

    #[test]
    fn test_clone_partial_eq_debug_display() {
        let v = vec64![1, 2, 3];
        let c = v.clone();
        assert_eq!(v, c);
        let s = format!("{:?}", v);
        assert!(s.contains("1"));
        let s2 = format!("{}", v);
        assert_eq!(s2, "[1, 2, 3]");
    }

    #[test]
    fn test_into_iterator() {
        let v = vec64![2, 4, 6];
        let mut out = Vec::new();
        for x in v {
            out.push(x);
        }
        assert_eq!(out, vec![2, 4, 6]);
    }

    #[test]
    fn test_iter_and_iter_mut() {
        let v = vec64![1, 2, 3];
        let sum: i32 = v.iter().copied().sum();
        assert_eq!(sum, 6);

        let mut v = vec64![0, 0, 0];
        for x in &mut v {
            *x = 7;
        }
        assert_eq!(v[..], [7, 7, 7]);
    }

    #[test]
    fn test_from_std_vec() {
        let std_v = vec![1, 2, 3, 4];
        let v: Vec64<_> = std_v.clone().into();
        assert_eq!(v[..], [1, 2, 3, 4]);
    }

    #[test]
    fn test_into_std_vec() {
        let v = vec64![7, 8, 9];
        let std_v: Vec<_> = v.0.clone().to_vec();
        assert_eq!(std_v, vec![7, 8, 9]);
    }

    #[test]
    fn test_alignment_is_64() {
        let v: Vec64<u8> = Vec64::with_capacity(32);
        assert_eq!(v.0.as_ptr() as usize % 64, 0);
    }

    #[test]
    fn test_zero_sized_types() {
        let v: Vec64<()> = vec64![(); 10];
        assert_eq!(v.len(), 10);
    }

    #[test]
    #[should_panic]
    fn test_index_out_of_bounds() {
        let v: Vec64<i32> = Vec64::new();
        let _ = v[1];
    }

    /// Utility: check that a pointer is 64-byte aligned.
    fn assert_aligned_64<T>(vec: &Vec64<T>) {
        let ptr = vec.as_ptr() as usize;
        assert_eq!(
            ptr % 64,
            0,
            "Pointer {:p} not 64-byte aligned",
            vec.as_ptr()
        );
    }

    #[test]
    fn test_vec64_new_alignment() {
        let v: Vec64<u32> = Vec64::new();
        // Even with capacity 0, allocation should be 64-byte aligned (when not null).
        // (Vec with cap 0 may have dangling non-null but still aligned pointer.)
        if v.capacity() > 0 {
            assert_aligned_64(&v);
        }
    }

    #[test]
    fn test_vec64_with_capacity_alignment() {
        for &n in &[1, 3, 7, 32, 1024, 4096] {
            let v: Vec64<u8> = Vec64::with_capacity(n);
            assert_aligned_64(&v);
        }
    }

    #[test]
    fn test_vec64_from_slice_alignment() {
        let data = [1u64, 2, 3, 4, 5, 6, 7, 8];
        let v = Vec64::from_slice(&data);
        assert_aligned_64(&v);
    }

    #[test]
    fn test_vec64_macro_alignment() {
        let v = vec64![0u32; 64];
        assert_aligned_64(&v);

        let v2 = vec64![1u16, 2, 3, 4, 5];
        assert_aligned_64(&v2);
    }

    #[test]
    fn test_vec64_grow_alignment() {
        let mut v: Vec64<u64> = Vec64::with_capacity(1);
        assert_aligned_64(&v);
        for i in 0..1000 {
            v.push(i);
            assert_aligned_64(&v);
        }
    }

    #[test]
    fn test_vec64_alignment_zst() {
        let v: Vec64<()> = Vec64::with_capacity(100);
        assert_eq!(
            v.capacity(),
            usize::MAX,
            "ZST Vec should have 'infinite' capacity"
        );
    }

    #[test]
    fn test_delete_range_empty_is_noop() {
        let mut v = vec64![1, 2, 3, 4, 5];
        v.delete_range(2, 2);
        assert_eq!(&v[..], &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_delete_range_full_clears() {
        let mut v = vec64![1, 2, 3];
        v.delete_range(0, 3);
        assert!(v.is_empty());
    }

    #[test]
    fn test_delete_range_tail_truncates() {
        let mut v = vec64![1, 2, 3, 4, 5];
        v.delete_range(3, 5);
        assert_eq!(&v[..], &[1, 2, 3]);
    }

    #[test]
    fn test_delete_range_head() {
        let mut v = vec64![1, 2, 3, 4, 5];
        v.delete_range(0, 2);
        assert_eq!(&v[..], &[3, 4, 5]);
    }

    #[test]
    fn test_delete_range_middle() {
        let mut v = vec64![1, 2, 3, 4, 5, 6, 7];
        v.delete_range(2, 5);
        assert_eq!(&v[..], &[1, 2, 6, 7]);
    }

    #[test]
    fn test_delete_range_single_element() {
        let mut v = vec64![10, 20, 30, 40];
        v.delete_range(1, 2);
        assert_eq!(&v[..], &[10, 30, 40]);
    }

    #[test]
    #[should_panic]
    fn test_delete_range_start_gt_end_panics() {
        let mut v = vec64![1, 2, 3];
        v.delete_range(2, 1);
    }

    #[test]
    #[should_panic]
    fn test_delete_range_end_past_len_panics() {
        let mut v = vec64![1, 2, 3];
        v.delete_range(0, 4);
    }

    #[test]
    fn test_delete_range_drops_elements() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

        struct Tracker(#[allow(dead_code)] u32);
        impl Drop for Tracker {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::SeqCst);
            }
        }

        DROP_COUNT.store(0, Ordering::SeqCst);
        let mut v: Vec64<Tracker> = Vec64::new();
        for i in 0..6u32 {
            v.push(Tracker(i));
        }
        DROP_COUNT.store(0, Ordering::SeqCst);

        v.delete_range(1, 4);
        assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 3,
            "exactly the three deleted elements should be dropped");
        assert_eq!(v.len(), 3);

        DROP_COUNT.store(0, Ordering::SeqCst);
        drop(v);
        assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 3,
            "remaining three elements drop when the Vec drops");
    }

    #[test]
    fn test_delete_range_zst() {
        let mut v: Vec64<()> = vec64![(); 10];
        v.delete_range(2, 7);
        assert_eq!(v.len(), 5);
    }

    #[test]
    fn test_delete_range_preserves_alignment() {
        let mut v: Vec64<u64> = (0..1024u64).collect();
        v.delete_range(256, 512);
        assert_eq!(v.len(), 768);
        assert_eq!(v.as_ptr() as usize % 64, 0);
        for i in 0..256 {
            assert_eq!(v[i], i as u64);
        }
        for i in 256..768 {
            assert_eq!(v[i], (i + 256) as u64);
        }
    }

    #[cfg(all(feature = "mmap", target_os = "linux"))]
    #[test]
    fn test_delete_range_mmap_page_aligned_middle() {
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        // Build a Vec<u64> large enough to land in mmap territory (>= 2 MiB).
        // 2 MiB / 8 = 262_144 elements minimum.
        let n: usize = 512 * 1024;
        let mut v: Vec64<u64> = (0..n as u64).collect();
        assert!(v.capacity() * 8 >= crate::mmap_alloc::HUGE_PAGE,
            "expected mmap-backed allocation");

        let base_before = v.as_ptr() as usize;
        let cap_before = v.capacity();
        let elems_per_page = page / 8;
        // Delete one page worth of elements from a page-aligned boundary.
        let start = elems_per_page;
        let end = start + elems_per_page;
        v.delete_range(start, end);

        let base_after = v.as_ptr() as usize;
        assert_eq!(base_before, base_after,
            "head pages should not relocate when the splice fast path fires");
        // Capacity is preserved on the splice path, matching drain; the
        // vacated physical pages return to the kernel via madvise.
        assert_eq!(v.capacity(), cap_before,
            "capacity changed: cap_before {cap_before}, cap_after {}",
            v.capacity());
        assert_eq!(v.len(), n - elems_per_page);
        for i in 0..start {
            assert_eq!(v[i], i as u64, "head bytes corrupted at {i}");
        }
        for i in start..v.len() {
            assert_eq!(v[i], (i + elems_per_page) as u64,
                "tail bytes corrupted at {i}");
        }
        assert_eq!(v.as_ptr() as usize % 64, 0);
    }

    #[cfg(all(feature = "mmap", target_os = "linux"))]
    #[test]
    fn test_delete_range_mmap_span_not_page_multiple_falls_back() {
        // A deleted span that is not a page multiple still has to work,
        // via the drain fallback.
        let n: usize = 512 * 1024;
        let mut v: Vec64<u64> = (0..n as u64).collect();
        v.delete_range(7, 19);
        assert_eq!(v.len(), n - 12);
        assert_eq!(v[0], 0);
        assert_eq!(v[6], 6);
        assert_eq!(v[7], 19);
    }

    #[cfg(all(feature = "mmap", target_os = "linux"))]
    #[test]
    fn test_delete_range_mmap_page_multiple_span_unaligned_start() {
        // The splice condition is on the deleted span, not the endpoints:
        // a page-multiple span starting mid-page takes the fast path with a
        // boundary-page patch.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let elems_per_page = page / 8;
        let n: usize = 512 * 1024;
        let mut v: Vec64<u64> = (0..n as u64).collect();
        let base_before = v.as_ptr() as usize;
        let cap_before = v.capacity();

        // Mid-page start, span of three whole pages.
        let start = elems_per_page + 3;
        let end = start + elems_per_page * 3;
        assert!(unsafe { v.try_mremap_splice(start, end) },
            "page-multiple span at a mid-page position should splice");

        assert_eq!(v.as_ptr() as usize, base_before,
            "head pages should not relocate");
        assert_eq!(v.capacity(), cap_before, "capacity should be preserved");
        assert_eq!(v.len(), n - elems_per_page * 3);
        for i in 0..start {
            assert_eq!(v[i], i as u64, "head bytes corrupted at {i}");
        }
        for i in start..v.len() {
            assert_eq!(v[i], (i + elems_per_page * 3) as u64,
                "tail bytes corrupted at {i}");
        }
        assert_eq!(v.as_ptr() as usize % 64, 0);
    }

    #[cfg(all(feature = "mmap", target_os = "linux"))]
    #[test]
    fn test_try_mremap_splice_rejects_span_not_page_multiple() {
        let n: usize = 512 * 1024;
        let mut v: Vec64<u64> = (0..n as u64).collect();
        // Page-aligned start, span one element short of a page.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let elems_per_page = page / 8;
        let start = elems_per_page;
        let end = start + elems_per_page - 1;
        assert!(!unsafe { v.try_mremap_splice(start, end) });
        // A rejected splice leaves the Vec untouched.
        assert_eq!(v.len(), n);
        assert_eq!(v[start], start as u64);
    }

    #[cfg(all(feature = "mmap", target_os = "linux"))]
    #[test]
    fn test_try_mremap_splice_rejects_heap_backed() {
        let mut v: Vec64<u64> = (0..64u64).collect();
        assert!(!unsafe { v.try_mremap_splice(8, 16) });
        assert_eq!(v.len(), 64);
    }

    #[cfg(all(feature = "mmap", target_os = "linux"))]
    #[test]
    fn test_delete_range_mmap_tail_absorbed_by_patch() {
        // A page-multiple span ending just short of the tail: the surviving
        // tail is smaller than the boundary-page patch and is absorbed by it.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let elems_per_page = page / 8;
        let n: usize = 512 * 1024;
        let mut v: Vec64<u64> = (0..n as u64).collect();
        let base_before = v.as_ptr() as usize;

        let end = n - 2;
        let start = end - elems_per_page;
        assert!(unsafe { v.try_mremap_splice(start, end) },
            "patch-absorbed splice should fire");

        assert_eq!(v.as_ptr() as usize, base_before);
        assert_eq!(v.len(), n - elems_per_page);
        assert_eq!(v[start - 1], (start - 1) as u64);
        assert_eq!(v[start], end as u64);
        assert_eq!(v[start + 1], (end + 1) as u64);
    }

    #[cfg(all(feature = "mmap", target_os = "linux"))]
    #[test]
    fn test_delete_range_mmap_then_grow() {
        // Growth after a splice exercises the allocator with the preserved
        // capacity: the mapping extent still matches what deallocate and
        // grow expect.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let elems_per_page = page / 8;
        let n: usize = 512 * 1024;
        let mut v: Vec64<u64> = (0..n as u64).collect();
        let cap_before = v.capacity();

        v.delete_range(elems_per_page, elems_per_page * 2);
        assert_eq!(v.len(), n - elems_per_page);

        // Extend past the original capacity to force an allocator grow.
        let target = cap_before + elems_per_page;
        while v.len() < target {
            v.push(7);
        }
        assert_eq!(v.len(), target);
        assert_eq!(v[0], 0);
        assert_eq!(v[elems_per_page], (elems_per_page * 2) as u64);
        assert_eq!(v[target - 1], 7);
        assert_eq!(v.as_ptr() as usize % 64, 0);
    }

    #[test]
    fn test_extend_from_vec64_empty_other() {
        let mut a = vec64![1, 2, 3];
        let b: Vec64<i32> = Vec64::new();
        a.extend_from_vec64(b);
        assert_eq!(&a[..], &[1, 2, 3]);
    }

    #[test]
    fn test_extend_from_vec64_basic() {
        let mut a = vec64![1u64, 2, 3];
        let b = vec64![4u64, 5, 6, 7];
        a.extend_from_vec64(b);
        assert_eq!(&a[..], &[1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_extend_from_vec64_into_empty() {
        let mut a: Vec64<u32> = Vec64::new();
        let b = vec64![10u32, 20, 30];
        a.extend_from_vec64(b);
        assert_eq!(&a[..], &[10, 20, 30]);
    }

    #[test]
    fn test_extend_from_vec64_drops_correctly() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);
        struct Tracker(#[allow(dead_code)] u32);
        impl Drop for Tracker {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::SeqCst);
            }
        }

        DROP_COUNT.store(0, Ordering::SeqCst);
        let mut a: Vec64<Tracker> = Vec64::new();
        a.push(Tracker(1));
        a.push(Tracker(2));
        let mut b: Vec64<Tracker> = Vec64::new();
        b.push(Tracker(10));
        b.push(Tracker(20));
        b.push(Tracker(30));
        DROP_COUNT.store(0, Ordering::SeqCst);

        a.extend_from_vec64(b);
        assert_eq!(a.len(), 5);
        assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 0,
            "no elements should be dropped during extend");

        drop(a);
        assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 5,
            "all five elements drop when the merged Vec drops");
    }

    #[test]
    fn test_from_chunks_basic() {
        let chunks: Vec<Vec64<i32>> = vec![
            vec64![1, 2, 3],
            vec64![4, 5],
            vec64![6],
            vec64![7, 8, 9, 10],
        ];
        let merged = Vec64::from_chunks(chunks);
        assert_eq!(&merged[..], &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    }

    #[test]
    fn test_from_chunks_empty() {
        let chunks: Vec<Vec64<u8>> = vec![];
        let merged = Vec64::from_chunks(chunks);
        assert!(merged.is_empty());
    }

    #[test]
    fn test_from_chunks_with_empty_chunks() {
        let chunks: Vec<Vec64<u64>> = vec![
            Vec64::new(),
            vec64![1, 2, 3],
            Vec64::new(),
            vec64![4],
            Vec64::new(),
        ];
        let merged = Vec64::from_chunks(chunks);
        assert_eq!(&merged[..], &[1, 2, 3, 4]);
    }

    #[test]
    fn test_from_chunks_preserves_alignment() {
        let chunks: Vec<Vec64<u64>> = (0..8)
            .map(|i| {
                let v: Vec64<u64> = (i * 100..i * 100 + 50).collect();
                v
            })
            .collect();
        let merged = Vec64::from_chunks(chunks);
        assert_eq!(merged.len(), 8 * 50);
        assert_eq!(merged.as_ptr() as usize % 64, 0);
        for chunk_idx in 0..8 {
            for j in 0..50 {
                let expected = (chunk_idx * 100 + j) as u64;
                assert_eq!(merged[chunk_idx * 50 + j], expected);
            }
        }
    }

    #[cfg(all(feature = "mmap", target_os = "linux"))]
    #[test]
    fn test_extend_from_vec64_mmap_aligned_path() {
        // Each chunk is exactly one HUGE_PAGE (2 MiB) of u64 = 262_144 elements,
        // tightly packed - the mmap path applies.
        let elems = crate::mmap_alloc::HUGE_PAGE / 8;
        let make_chunk = |start: u64| -> Vec64<u64> {
            let mut v: Vec64<u64> = Vec64::with_capacity(elems);
            for i in 0..elems as u64 {
                v.push(start + i);
            }
            assert_eq!(v.len(), v.capacity(), "test invariant: chunk must be tightly packed");
            v
        };

        let total_elems = elems * 3;
        let mut dst: Vec64<u64> = Vec64::with_capacity(total_elems);
        let dst_base = dst.as_ptr() as usize;

        dst.extend_from_vec64(make_chunk(0));
        dst.extend_from_vec64(make_chunk(1_000_000));
        dst.extend_from_vec64(make_chunk(2_000_000));

        assert_eq!(dst.len(), total_elems);
        assert_eq!(dst.as_ptr() as usize, dst_base,
            "destination base must not relocate");
        assert_eq!(dst.as_ptr() as usize % 64, 0);

        for i in 0..elems {
            assert_eq!(dst[i], i as u64, "chunk 0 at {i}");
        }
        for i in 0..elems {
            assert_eq!(dst[elems + i], 1_000_000 + i as u64, "chunk 1 at {i}");
        }
        for i in 0..elems {
            assert_eq!(dst[2 * elems + i], 2_000_000 + i as u64, "chunk 2 at {i}");
        }
    }

    /// Proves the mremap path actually fires when its preconditions hold.
    /// Calls `try_mremap_append` directly and asserts it returned `None`
    /// (which it only does when the mremap syscalls succeeded and the
    /// source was consumed). Also asserts the destination's data is correct.
    #[cfg(all(feature = "mmap", target_os = "linux"))]
    #[test]
    fn test_try_mremap_append_succeeds_under_valid_conditions() {
        let n = crate::mmap_alloc::HUGE_PAGE / 8;

        let mut src: Vec64<u64> = Vec64::with_capacity(n);
        for i in 0..n as u64 {
            src.push(i);
        }
        assert_eq!(src.len(), src.capacity());

        let mut dst: Vec64<u64> = Vec64::with_capacity(n * 2);
        for i in 0..n as u64 {
            dst.push(i * 1000);
        }
        let dst_base = dst.as_ptr() as usize;

        let returned = unsafe { dst.try_mremap_append(src) };
        assert!(returned.is_none(),
            "try_mremap_append returned Some - mremap path did not fire under valid conditions");

        assert_eq!(dst.len(), n * 2);
        assert_eq!(dst.as_ptr() as usize, dst_base);
        for i in 0..n {
            assert_eq!(dst[i], i as u64 * 1000, "head corruption at {i}");
        }
        for i in 0..n {
            assert_eq!(dst[n + i], i as u64, "tail corruption at {i}");
        }
    }

    /// Confirms the path-selection logic correctly rejects sources that
    /// fail any precondition (unaligned byte size, here).
    #[cfg(all(feature = "mmap", target_os = "linux"))]
    #[test]
    fn test_try_mremap_append_rejects_unaligned_source() {
        let n = crate::mmap_alloc::HUGE_PAGE / 8;
        let mut dst: Vec64<u64> = Vec64::with_capacity(n * 2);
        for i in 0..n as u64 {
            dst.push(i);
        }

        let mut src: Vec64<u64> = Vec64::with_capacity(17);
        for i in 0..17u64 {
            src.push(100 + i);
        }
        let src_len = src.len();

        let returned = unsafe { dst.try_mremap_append(src) };
        assert!(returned.is_some(),
            "try_mremap_append returned None for an unaligned source");
        assert_eq!(returned.unwrap().len(), src_len);
    }

    #[cfg(all(feature = "mmap", target_os = "linux"))]
    #[test]
    fn test_extend_from_vec64_falls_back_when_source_unaligned() {
        // Source byte size is not a multiple of HUGE_PAGE: must take the
        // move path, still produce correct data.
        let mut dst: Vec64<u64> = Vec64::with_capacity(100);
        for i in 0..10u64 {
            dst.push(i);
        }
        let src: Vec64<u64> = (100..117u64).collect();
        assert_ne!((src.len() * 8) % crate::mmap_alloc::HUGE_PAGE, 0);
        dst.extend_from_vec64(src);
        assert_eq!(dst.len(), 27);
        for i in 0..10 {
            assert_eq!(dst[i], i as u64);
        }
        for i in 0..17 {
            assert_eq!(dst[10 + i], 100 + i as u64);
        }
    }

    #[cfg(all(feature = "mmap", target_os = "linux"))]
    #[test]
    fn test_from_chunks_mmap_aligned_round_trip() {
        let elems = crate::mmap_alloc::HUGE_PAGE / 8;
        let chunks: Vec<Vec64<u64>> = (0..4u64)
            .map(|chunk_id| {
                let mut v: Vec64<u64> = Vec64::with_capacity(elems);
                for i in 0..elems as u64 {
                    v.push(chunk_id * 10_000_000 + i);
                }
                v
            })
            .collect();
        let merged = Vec64::from_chunks(chunks);
        assert_eq!(merged.len(), 4 * elems);
        for chunk_id in 0..4u64 {
            for i in 0..elems {
                let expected = chunk_id * 10_000_000 + i as u64;
                assert_eq!(merged[chunk_id as usize * elems + i], expected);
            }
        }
    }

    /// Verifies copy-on-write semantics when multiple threads hold Arc
    /// clones of the same Vec64. One or two writer threads call
    /// `Arc::make_mut` to fork off their own copy and apply `delete_range`;
    /// the other threads continue to observe the original through their
    /// own Arc clones. After all threads join, the original allocation
    /// still holds the unmodified data because every reader's Arc kept
    /// it alive.
    #[test]
    fn test_delete_range_arc_shared_threads() {
        use std::sync::{Arc, Barrier};
        use std::thread;

        const N: usize = 4096;
        const READERS: usize = 6;
        const WRITERS: usize = 2;
        const DELETE_START: usize = 1000;
        const DELETE_END: usize = 1500;

        let data: Vec64<u64> = (0..N as u64).collect();
        let shared: Arc<Vec64<u64>> = Arc::new(data);
        let barrier = Arc::new(Barrier::new(READERS + WRITERS));
        let expected_full_sum: u64 = (0..N as u64).sum();

        let mut handles = Vec::with_capacity(READERS + WRITERS);

        // Reader threads: each keeps the original allocation alive via its
        // own Arc clone, so it must continue to see the unmodified data
        // throughout the deleters' work.
        for reader_id in 0..READERS {
            let my_arc = Arc::clone(&shared);
            let b = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                b.wait();
                // Re-read from start to end so a write that aliased the
                // backing storage would surface as a checksum mismatch.
                let observed_len = my_arc.len();
                let observed_sum: u64 = my_arc.iter().sum();
                assert_eq!(observed_len, N,
                    "reader {reader_id} saw len {observed_len}, expected {N}");
                assert_eq!(observed_sum, expected_full_sum,
                    "reader {reader_id} observed modified contents");
                for i in 0..N {
                    assert_eq!(my_arc[i], i as u64,
                        "reader {reader_id} corruption at index {i}");
                }
            }));
        }

        // Writer threads: each takes its own clone, then `Arc::make_mut`
        // forks off a private copy because the refcount is > 1. The local
        // delete then runs against that private copy; the original
        // allocation is untouched.
        for writer_id in 0..WRITERS {
            let mut my_arc = Arc::clone(&shared);
            let b = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                b.wait();
                let mutable = Arc::make_mut(&mut my_arc);
                mutable.delete_range(DELETE_START, DELETE_END);

                let expected_len = N - (DELETE_END - DELETE_START);
                assert_eq!(mutable.len(), expected_len,
                    "writer {writer_id} expected len {expected_len}");
                for i in 0..DELETE_START {
                    assert_eq!(mutable[i], i as u64,
                        "writer {writer_id} head corruption at {i}");
                }
                for i in DELETE_START..mutable.len() {
                    let expected = (i + (DELETE_END - DELETE_START)) as u64;
                    assert_eq!(mutable[i], expected,
                        "writer {writer_id} tail corruption at {i}");
                }
            }));
        }

        for h in handles {
            h.join().expect("worker panicked");
        }

        // Every reader released its Arc when it joined. The original is
        // now uniquely held by `shared`; assert it still mirrors the
        // initial state, proving the writers' deletes never touched it.
        assert_eq!(shared.len(), N);
        let final_sum: u64 = shared.iter().sum();
        assert_eq!(final_sum, expected_full_sum,
            "original allocation was modified through Arc aliasing");
        for i in 0..N {
            assert_eq!(shared[i], i as u64,
                "original corruption at index {i}");
        }
    }
}

#[cfg(test)]
#[cfg(any(feature = "parallel_proc", feature = "wasm"))]
mod parallel_tests {
    use rayon::iter::ParallelIterator;

    use super::*;

    #[test]
    fn test_vec64_par_iter() {
        let v = Vec64::from_slice(&[1u32, 2, 3, 4, 5]);
        let sum: u32 = v.par_iter().sum();
        assert_eq!(sum, 15);
    }
}
