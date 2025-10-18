#![feature(portable_simd)]

/*!
Run with:
    RUSTFLAGS="-C target-cpu=native" cargo run --release --example hotloop_bench_simd

Purpose:
This benchmark compares portable SIMD code on misaligned Vec vs aligned Vec64.
It demonstrates that for simple summation, Vec64's alignment enables using aligned
SIMD loads, while Vec requires unaligned loads.

Methodology:
- Forces misalignment for Vec by allocating with offset
- Vec64 is always 64-byte aligned
- Alternate order, warm up exact slices, black-box inputs/outputs
- Report min/median/p95 over many iterations

Expected Result:
For simple summation, differences may be minimal on modern CPUs because:
1. LLVM auto-vectorizes simple loops extremely well
2. Modern CPUs have similar performance for aligned vs unaligned loads
3. Hand-written SIMD doesn't help much for trivial operations

Value:
Vec64 is not designed for simple auto-vectorizable loops like this benchmark.
Its value comes from:
- Complex SIMD kernels (distribution PDFs, special functions, transforms)
- Multi-region algorithms with branching that LLVM cannot auto-vectorize
- Guaranteeing alignment for performance-critical hand-written SIMD
- AVX-512 workloads where alignment benefits are more pronounced

See the simd-kernels crate in a real-world codebase for examples.
*/

use std::hint::black_box;
use std::mem::align_of;
use std::simd::num::{SimdFloat, SimdInt};
use std::simd::{LaneCount, Simd, SupportedLaneCount};
use std::time::{Duration, Instant};

use vec64::Vec64;

// --------------------------- Configuration ---------------------------

pub(crate) const N: usize = 1_000_000;
pub(crate) const SIMD_LANES: usize = 4;
pub(crate) const ITERS: usize = 100;

// --------------------------- SIMD kernels ----------------------------

#[inline(always)]
fn simd_sum_i64_aligned<const LANES: usize>(data: &[i64]) -> i64
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let n = data.len();
    let simd_width = LANES;
    let simd_chunks = n / simd_width;

    let acc_simd: Simd<i64, LANES>;

    unsafe {
        let data_ptr = data.as_ptr();
        debug_assert!(
            (data_ptr as usize) % align_of::<Simd<i64, LANES>>() == 0,
            "unaligned base pointer passed to aligned kernel"
        );

        let mut acc1 = Simd::<i64, LANES>::splat(0);
        let mut acc2 = Simd::<i64, LANES>::splat(0);
        let mut acc3 = Simd::<i64, LANES>::splat(0);
        let mut acc4 = Simd::<i64, LANES>::splat(0);

        let unroll = 4;
        let unrolled_chunks = simd_chunks / unroll;

        for i in 0..unrolled_chunks {
            let base = i * unroll * simd_width;
            let v1 = std::ptr::read(data_ptr.add(base) as *const Simd<i64, LANES>);
            let v2 = std::ptr::read(data_ptr.add(base + simd_width) as *const Simd<i64, LANES>);
            let v3 = std::ptr::read(data_ptr.add(base + 2 * simd_width) as *const Simd<i64, LANES>);
            let v4 = std::ptr::read(data_ptr.add(base + 3 * simd_width) as *const Simd<i64, LANES>);
            acc1 += v1;
            acc2 += v2;
            acc3 += v3;
            acc4 += v4;
        }

        let processed = unrolled_chunks * unroll;
        for i in processed..simd_chunks {
            let off = i * simd_width;
            let v = std::ptr::read(data_ptr.add(off) as *const Simd<i64, LANES>);
            acc1 += v;
        }

        acc_simd = acc1 + acc2 + acc3 + acc4;
    }

    let mut result = acc_simd.reduce_sum();

    let remainder_start = simd_chunks * simd_width;
    for &x in &data[remainder_start..] {
        result += x;
    }

    result
}

#[inline(always)]
fn simd_sum_i64<const LANES: usize>(data: &[i64]) -> i64
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let n = data.len();
    let simd_width = LANES;
    let simd_chunks = n / simd_width;

    let acc_simd: Simd<i64, LANES>;

    unsafe {
        let data_ptr = data.as_ptr();

        let mut acc1 = Simd::<i64, LANES>::splat(0);
        let mut acc2 = Simd::<i64, LANES>::splat(0);
        let mut acc3 = Simd::<i64, LANES>::splat(0);
        let mut acc4 = Simd::<i64, LANES>::splat(0);

        let unroll = 4;
        let unrolled_chunks = simd_chunks / unroll;

        for i in 0..unrolled_chunks {
            let base = i * unroll * simd_width;
            let v1 = std::ptr::read_unaligned(data_ptr.add(base) as *const Simd<i64, LANES>);
            let v2 = std::ptr::read_unaligned(
                data_ptr.add(base + simd_width) as *const Simd<i64, LANES>
            );
            let v3 = std::ptr::read_unaligned(
                data_ptr.add(base + 2 * simd_width) as *const Simd<i64, LANES>
            );
            let v4 = std::ptr::read_unaligned(
                data_ptr.add(base + 3 * simd_width) as *const Simd<i64, LANES>
            );
            acc1 += v1;
            acc2 += v2;
            acc3 += v3;
            acc4 += v4;
        }

        let processed = unrolled_chunks * unroll;
        for i in processed..simd_chunks {
            let off = i * simd_width;
            let v = std::ptr::read_unaligned(data_ptr.add(off) as *const Simd<i64, LANES>);
            acc1 += v;
        }

        acc_simd = acc1 + acc2 + acc3 + acc4;
    }

    let mut result = acc_simd.reduce_sum();

    let remainder_start = simd_chunks * simd_width;
    for &x in &data[remainder_start..] {
        result += x;
    }

    result
}

#[inline(always)]
fn simd_sum_f64_aligned<const LANES: usize>(data: &[f64]) -> f64
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let n = data.len();
    let simd_width = LANES;
    let simd_chunks = n / simd_width;

    let acc_simd: Simd<f64, LANES>;

    unsafe {
        let data_ptr = data.as_ptr();
        debug_assert!(
            (data_ptr as usize) % align_of::<Simd<f64, LANES>>() == 0,
            "unaligned base pointer passed to aligned kernel"
        );

        let mut acc1 = Simd::<f64, LANES>::splat(0.0);
        let mut acc2 = Simd::<f64, LANES>::splat(0.0);
        let mut acc3 = Simd::<f64, LANES>::splat(0.0);
        let mut acc4 = Simd::<f64, LANES>::splat(0.0);

        let unroll = 4;
        let unrolled_chunks = simd_chunks / unroll;

        for i in 0..unrolled_chunks {
            let base = i * unroll * simd_width;
            let v1 = std::ptr::read(data_ptr.add(base) as *const Simd<f64, LANES>);
            let v2 = std::ptr::read(data_ptr.add(base + simd_width) as *const Simd<f64, LANES>);
            let v3 = std::ptr::read(data_ptr.add(base + 2 * simd_width) as *const Simd<f64, LANES>);
            let v4 = std::ptr::read(data_ptr.add(base + 3 * simd_width) as *const Simd<f64, LANES>);
            acc1 += v1;
            acc2 += v2;
            acc3 += v3;
            acc4 += v4;
        }

        let processed = unrolled_chunks * unroll;
        for i in processed..simd_chunks {
            let off = i * simd_width;
            let v = std::ptr::read(data_ptr.add(off) as *const Simd<f64, LANES>);
            acc1 += v;
        }

        acc_simd = acc1 + acc2 + acc3 + acc4;
    }

    let mut result = acc_simd.reduce_sum();

    let remainder_start = simd_chunks * simd_width;
    for &x in &data[remainder_start..] {
        result += x;
    }

    result
}

#[inline(always)]
fn simd_sum_f64<const LANES: usize>(data: &[f64]) -> f64
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let n = data.len();
    let simd_width = LANES;
    let simd_chunks = n / simd_width;

    let acc_simd: Simd<f64, LANES>;

    unsafe {
        let data_ptr = data.as_ptr();

        let mut acc1 = Simd::<f64, LANES>::splat(0.0);
        let mut acc2 = Simd::<f64, LANES>::splat(0.0);
        let mut acc3 = Simd::<f64, LANES>::splat(0.0);
        let mut acc4 = Simd::<f64, LANES>::splat(0.0);

        let unroll = 4;
        let unrolled_chunks = simd_chunks / unroll;

        for i in 0..unrolled_chunks {
            let base = i * unroll * simd_width;
            let v1 = std::ptr::read_unaligned(data_ptr.add(base) as *const Simd<f64, LANES>);
            let v2 = std::ptr::read_unaligned(
                data_ptr.add(base + simd_width) as *const Simd<f64, LANES>
            );
            let v3 = std::ptr::read_unaligned(
                data_ptr.add(base + 2 * simd_width) as *const Simd<f64, LANES>
            );
            let v4 = std::ptr::read_unaligned(
                data_ptr.add(base + 3 * simd_width) as *const Simd<f64, LANES>
            );
            acc1 += v1;
            acc2 += v2;
            acc3 += v3;
            acc4 += v4;
        }

        let processed = unrolled_chunks * unroll;
        for i in processed..simd_chunks {
            let off = i * simd_width;
            let v = std::ptr::read_unaligned(data_ptr.add(off) as *const Simd<f64, LANES>);
            acc1 += v;
        }

        acc_simd = acc1 + acc2 + acc3 + acc4;
    }

    let mut result = acc_simd.reduce_sum();

    let remainder_start = simd_chunks * simd_width;
    for &x in &data[remainder_start..] {
        result += x;
    }

    result
}

// --------------------------- Dispatch ----------------------------

fn simd_sum_i64_runtime(data: &[i64], lanes: usize) -> i64 {
    match lanes {
        2 => simd_sum_i64::<2>(data),
        4 => simd_sum_i64::<4>(data),
        8 => simd_sum_i64::<8>(data),
        16 => simd_sum_i64::<16>(data),
        _ => panic!("Unsupported SIMD lanes. Only 2, 4, 8, 16 supported."),
    }
}

fn simd_sum_i64_aligned_runtime(data: &[i64], lanes: usize) -> i64 {
    match lanes {
        2 => simd_sum_i64_aligned::<2>(data),
        4 => simd_sum_i64_aligned::<4>(data),
        8 => simd_sum_i64_aligned::<8>(data),
        16 => simd_sum_i64_aligned::<16>(data),
        _ => panic!("Unsupported SIMD lanes. Only 2, 4, 8, 16 supported."),
    }
}

fn simd_sum_f64_runtime(data: &[f64], lanes: usize) -> f64 {
    match lanes {
        2 => simd_sum_f64::<2>(data),
        4 => simd_sum_f64::<4>(data),
        8 => simd_sum_f64::<8>(data),
        16 => simd_sum_f64::<16>(data),
        _ => panic!("Unsupported SIMD lanes. Only 2, 4, 8, 16 supported."),
    }
}

fn simd_sum_f64_aligned_runtime(data: &[f64], lanes: usize) -> f64 {
    match lanes {
        2 => simd_sum_f64_aligned::<2>(data),
        4 => simd_sum_f64_aligned::<4>(data),
        8 => simd_sum_f64_aligned::<8>(data),
        16 => simd_sum_f64_aligned::<16>(data),
        _ => panic!("Unsupported SIMD lanes. Only 2, 4, 8, 16 supported."),
    }
}

// --------------------------- Utilities ----------------------------

#[inline(always)]
fn required_simd_align_i64(lanes: usize) -> usize {
    match lanes {
        2 => align_of::<Simd<i64, 2>>(),
        4 => align_of::<Simd<i64, 4>>(),
        8 => align_of::<Simd<i64, 8>>(),
        16 => align_of::<Simd<i64, 16>>(),
        _ => panic!("Unsupported SIMD lanes."),
    }
}

#[inline(always)]
fn required_simd_align_f64(lanes: usize) -> usize {
    match lanes {
        2 => align_of::<Simd<f64, 2>>(),
        4 => align_of::<Simd<f64, 4>>(),
        8 => align_of::<Simd<f64, 8>>(),
        16 => align_of::<Simd<f64, 16>>(),
        _ => panic!("Unsupported SIMD lanes."),
    }
}

#[derive(Clone, Copy, Debug)]
struct Stats {
    min: Duration,
    med: Duration,
    p95: Duration,
}

fn summarise(mut xs: Vec<Duration>) -> Stats {
    xs.sort_unstable();
    let n = xs.len();
    let min = xs[0];
    let med = xs[n / 2];
    let idx95 = ((n as f64) * 0.95).ceil() as usize - 1;
    let p95 = xs[idx95.min(n - 1)];
    Stats { min, med, p95 }
}

#[inline(always)]
fn kahan_sum(xs: &[f64]) -> f64 {
    let mut sum = 0.0f64;
    let mut c = 0.0f64;
    for &val in xs {
        let y = val - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

#[inline(always)]
fn expected_i64_sum(n: usize) -> i64 {
    let nn = n as i128;
    let exp = (nn - 1) * nn / 2;
    exp as i64
}

fn fmt_dur(d: Duration) -> String {
    let us = d.as_secs_f64() * 1_000_000.0;
    format!("{us:.3} µs")
}

fn create_misaligned_storage_i64(data: &[i64]) -> (Vec<i64>, usize) {
    let n = data.len();
    let mut storage = vec![0i64; n + 1]; // headroom for 1-element offset
    let base = storage.as_ptr() as usize;
    let off = if base % 64 == 0 { 1 } else { 0 }; // ensure misaligned base
    storage[off..off + n].copy_from_slice(data);
    (storage, off)
}

fn create_misaligned_storage_f64(data: &[f64]) -> (Vec<f64>, usize) {
    let n = data.len();
    let mut storage = vec![0f64; n + 1];
    let base = storage.as_ptr() as usize;
    let off = if base % 64 == 0 { 1 } else { 0 };
    storage[off..off + n].copy_from_slice(data);
    (storage, off)
}

// --------------------------- Benchmark harness ----------------------------

pub fn run_benchmark(n: usize, simd_lanes: usize) {
    println!("Running SIMD benchmarks (n = {n}, lanes = {simd_lanes})\n");

    // --------------------------- Prepare data ---------------------------

    // Create reference data
    let reference_i64: Vec<i64> = (0..n as i64).collect();
    let reference_f64: Vec<f64> = (0..n as i64).map(|x| x as f64).collect();

    // Force misalignment for Vec
    let (vec_i64_store, off_i64) = create_misaligned_storage_i64(&reference_i64);
    let vec_i64 = &vec_i64_store[off_i64..off_i64 + reference_i64.len()];

    let (vec_f64_store, off_f64) = create_misaligned_storage_f64(&reference_f64);
    let vec_f64 = &vec_f64_store[off_f64..off_f64 + reference_f64.len()];

    // Vec64 is always 64-byte aligned
    let mut vec64_i64: Vec64<i64> = Vec64::with_capacity(n);
    vec64_i64.extend_from_slice(&reference_i64);
    let mut vec64_f64: Vec64<f64> = Vec64::with_capacity(n);
    vec64_f64.extend_from_slice(&reference_f64);

    // Alignment reporting (Vec is forced misaligned, Vec64 is always aligned)
    let req_i64 = required_simd_align_i64(simd_lanes);
    let req_f64 = required_simd_align_f64(simd_lanes);

    let v_i64_ptr = vec_i64.as_ptr() as usize;
    let v64_i64_ptr = vec64_i64.as_ptr() as usize;
    let v_f64_ptr = vec_f64.as_ptr() as usize;
    let v64_f64_ptr = vec64_f64.as_ptr() as usize;

    println!("Required alignment for i64 Simd<{simd_lanes}>: {req_i64} bytes");
    println!(
        "Vec<i64>   base {:>#018x} aligned: {} [FORCED MISALIGNMENT]",
        v_i64_ptr,
        v_i64_ptr % req_i64 == 0
    );
    println!(
        "Vec64<i64> base {:>#018x} aligned: {}",
        v64_i64_ptr,
        v64_i64_ptr % req_i64 == 0
    );

    println!("Required alignment for f64 Simd<{simd_lanes}>: {req_f64} bytes");
    println!(
        "Vec<f64>   base {:>#018x} aligned: {} [FORCED MISALIGNMENT]",
        v_f64_ptr,
        v_f64_ptr % req_f64 == 0
    );
    println!(
        "Vec64<f64> base {:>#018x} aligned: {}",
        v64_f64_ptr,
        v64_f64_ptr % req_f64 == 0
    );
    println!();

    // Warm-up using the exact slices we will time.
    simd_sum_i64_runtime(black_box(&vec_i64[..]), black_box(simd_lanes));
    simd_sum_i64_aligned_runtime(black_box(&vec64_i64[..]), black_box(simd_lanes));
    simd_sum_f64_runtime(black_box(&vec_f64[..]), black_box(simd_lanes));
    simd_sum_f64_aligned_runtime(black_box(&vec64_f64[..]), black_box(simd_lanes));

    // --------------------------- Correctness ---------------------------

    let exp_i64 = expected_i64_sum(n);
    let got_i64_vec = simd_sum_i64_runtime(&vec_i64[..], simd_lanes);
    let got_i64_vec64 = simd_sum_i64_aligned_runtime(&vec64_i64[..], simd_lanes);
    assert_eq!(got_i64_vec, exp_i64);
    assert_eq!(got_i64_vec64, exp_i64);

    let exp_f64 = kahan_sum(&vec_f64[..]);
    let got_f64_vec = simd_sum_f64_runtime(&vec_f64[..], simd_lanes);
    let got_f64_vec64 = simd_sum_f64_aligned_runtime(&vec64_f64[..], simd_lanes);
    let eps = 1e-6 * exp_f64.abs().max(1.0);
    assert!(
        (got_f64_vec - exp_f64).abs() <= eps,
        "f64 Vec failed: got {got_f64_vec}, exp {exp_f64}"
    );
    assert!(
        (got_f64_vec64 - exp_f64).abs() <= eps,
        "f64 Vec64 failed: got {got_f64_vec64}, exp {exp_f64}"
    );

    // --------------------------- Timed runs ---------------------------

    #[inline(always)]
    fn time_once<F: FnOnce() -> R, R>(f: F) -> Duration {
        let start = Instant::now();
        let r = f();
        black_box(r);
        start.elapsed()
    }

    // Integer benches: Vec (forced misalignment, unaligned loads) vs Vec64 (aligned loads)
    let mut durs_vec_i64 = Vec::with_capacity(ITERS);
    let mut durs_vec64_i64 = Vec::with_capacity(ITERS);

    for i in 0..ITERS {
        if i % 2 == 0 {
            let d1 =
                time_once(|| simd_sum_i64_runtime(black_box(&vec_i64[..]), black_box(simd_lanes)));
            let d2 = time_once(|| {
                simd_sum_i64_aligned_runtime(black_box(&vec64_i64[..]), black_box(simd_lanes))
            });
            durs_vec_i64.push(d1);
            durs_vec64_i64.push(d2);
        } else {
            let d2 = time_once(|| {
                simd_sum_i64_aligned_runtime(black_box(&vec64_i64[..]), black_box(simd_lanes))
            });
            let d1 =
                time_once(|| simd_sum_i64_runtime(black_box(&vec_i64[..]), black_box(simd_lanes)));
            durs_vec64_i64.push(d2);
            durs_vec_i64.push(d1);
        }
    }

    let stats_vec_i64 = summarise(durs_vec_i64);
    let stats_vec64_i64 = summarise(durs_vec64_i64);

    // Float benches: Vec (forced misalignment, unaligned loads) vs Vec64 (aligned loads)
    let mut durs_vec_f64 = Vec::with_capacity(ITERS);
    let mut durs_vec64_f64 = Vec::with_capacity(ITERS);

    for i in 0..ITERS {
        if i % 2 == 0 {
            let d1 =
                time_once(|| simd_sum_f64_runtime(black_box(&vec_f64[..]), black_box(simd_lanes)));
            let d2 = time_once(|| {
                simd_sum_f64_aligned_runtime(black_box(&vec64_f64[..]), black_box(simd_lanes))
            });
            durs_vec_f64.push(d1);
            durs_vec64_f64.push(d2);
        } else {
            let d2 = time_once(|| {
                simd_sum_f64_aligned_runtime(black_box(&vec64_f64[..]), black_box(simd_lanes))
            });
            let d1 =
                time_once(|| simd_sum_f64_runtime(black_box(&vec_f64[..]), black_box(simd_lanes)));
            durs_vec64_f64.push(d2);
            durs_vec_f64.push(d1);
        }
    }

    let stats_vec_f64 = summarise(durs_vec_f64);
    let stats_vec64_f64 = summarise(durs_vec64_f64);

    // --------------------------- Report ---------------------------

    println!("|------------ Integer (i64) ------------|");
    println!(
        "Vec<i64>   (forced misalignment, unaligned loads): min {}  med {}  p95 {}",
        fmt_dur(stats_vec_i64.min),
        fmt_dur(stats_vec_i64.med),
        fmt_dur(stats_vec_i64.p95)
    );
    println!(
        "Vec64<i64> (64-byte aligned loads):                min {}  med {}  p95 {}",
        fmt_dur(stats_vec64_i64.min),
        fmt_dur(stats_vec64_i64.med),
        fmt_dur(stats_vec64_i64.p95)
    );
    println!();

    println!("|------------ Float (f64) -------------|");
    println!(
        "Vec<f64>   (forced misalignment, unaligned loads): min {}  med {}  p95 {}",
        fmt_dur(stats_vec_f64.min),
        fmt_dur(stats_vec_f64.med),
        fmt_dur(stats_vec_f64.p95)
    );
    println!(
        "Vec64<f64> (64-byte aligned loads):                min {}  med {}  p95 {}",
        fmt_dur(stats_vec64_f64.min),
        fmt_dur(stats_vec64_f64.med),
        fmt_dur(stats_vec64_f64.p95)
    );

    println!("\n---------------------- END OF SIMD BENCHMARKS ---------------------------");
}

fn main() {
    println!(
        "SIMD sum benchmarks (n = {}, lanes = {}, iters = {})",
        N, SIMD_LANES, ITERS
    );
    run_benchmark(N, SIMD_LANES);
}
