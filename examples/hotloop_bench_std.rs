//! ---------------------------------------------------------
//! Run with:
//!     RUSTFLAGS="-C target-cpu=native" cargo run --example hotloop_bench_std --release
//!
//! Purpose:
//! Measure a simple summation loop over `Vec` (forced misalignment) vs `Vec64` (64-byte aligned)
//! under normal compiler optimisation.
//!
//! Methodology:
//! - Force misalignment for `Vec` by allocating extra headroom and offsetting the start element.
//! - Compare misaligned `Vec` slice vs aligned `Vec64` on identical data and length.
//! - Warm up the exact slices that are timed.
//! - Alternate measurement order each iteration to reduce cache/order bias.
//! - Use `black_box` on the final accumulator only (not on the slice) to keep the optimiser honest
//!   without blocking potential loop vectorisation.
//! - Report min/median/p95 over many iterations.
//!
//! Expected Result:
//! - LLVM may or may not auto-vectorise this reduction (e.g. `f64` associativity and `i64` overflow
//!   legality can inhibit it). When auto-vectorised, both `Vec` and `Vec64` should be near parity.
//! - On modern CPUs, aligned vs unaligned loads have similar throughput when they do not regularly
//!   straddle cache lines, so differences are typically minimal for this workload.
//!
//! Value:
//! `Vec64` is most useful when you rely on explicit SIMD kernels or third-party SIMD code that
//! assumes/benefits from known alignment, or on AVX-512 workloads where alignment matters more.
//!
//! See `hotloop_bench_simd.rs` for a SIMD comparison.
//! ---------------------------------------------------------

use std::hint::black_box;
use std::time::{Duration, Instant};

use vec64::Vec64;

const N: usize = 1_000_000;
const ITERS: usize = 100;

#[inline(always)]
fn fmt_dur(d: Duration) -> String {
    let us = d.as_secs_f64() * 1_000_000.0;
    format!("{us:.3} µs")
}

#[derive(Clone, Copy)]
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
fn time_once<F: FnOnce() -> R, R>(f: F) -> Duration {
    let start = Instant::now();
    let r = f();
    black_box(r);
    start.elapsed()
}

#[inline(always)]
fn kahan_sum(xs: &[f64]) -> f64 {
    let mut s = 0.0f64;
    let mut c = 0.0f64;
    for &v in xs {
        let y = v - c;
        let t = s + y;
        c = (t - s) - y;
        s = t;
    }
    s
}

fn create_misaligned_storage_i64(data: &[i64]) -> (Vec<i64>, usize) {
    let n = data.len();
    let mut storage = vec![0i64; n + 1];
    let base = storage.as_ptr() as usize;
    let off = if base % 64 == 0 { 1 } else { 0 };
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

pub fn main() {
    println!("hotloop_bench_std: N={N}, iters={ITERS}");

    // ---------------------- Prepare identical data ----------------------
    let reference_i64: Vec<i64> = (0..N as i64).collect();
    let reference_f64: Vec<f64> = (0..N as i64).map(|x| x as f64).collect();

    let (vec_i64_store, off_i64) = create_misaligned_storage_i64(&reference_i64);
    let vec_i64 = &vec_i64_store[off_i64..off_i64 + reference_i64.len()];

    let (vec_f64_store, off_f64) = create_misaligned_storage_f64(&reference_f64);
    let vec_f64 = &vec_f64_store[off_f64..off_f64 + reference_f64.len()];

    let mut vec64_i64: Vec64<i64> = Vec64::with_capacity(N);
    vec64_i64.extend(0..N as i64);
    let mut vec64_f64: Vec64<f64> = Vec64::with_capacity(N);
    vec64_f64.extend((0..N as i64).map(|x| x as f64));

    // Pointer reporting
    println!(
        "Vec<i64>   base={:#018x} (aligned_to_64={}) [FORCED MISALIGNMENT]",
        vec_i64.as_ptr() as usize,
        (vec_i64.as_ptr() as usize) % 64 == 0
    );
    println!(
        "Vec64<i64> base={:#018x} (aligned_to_64={})",
        vec64_i64.as_ptr() as usize,
        (vec64_i64.as_ptr() as usize) % 64 == 0
    );
    println!(
        "Vec<f64>   base={:#018x} (aligned_to_64={}) [FORCED MISALIGNMENT]",
        vec_f64.as_ptr() as usize,
        (vec_f64.as_ptr() as usize) % 64 == 0
    );
    println!(
        "Vec64<f64> base={:#018x} (aligned_to_64={})",
        vec64_f64.as_ptr() as usize,
        (vec64_f64.as_ptr() as usize) % 64 == 0
    );
    println!();

    // ---------------------- Correctness ----------------------
    let mut check_i64_vec = 0i64;
    {
        let s = &vec_i64[..];
        for &v in s {
            check_i64_vec += v;
        }
    }
    let mut check_i64_v64 = 0i64;
    {
        let s = &vec64_i64[..];
        for &v in s {
            check_i64_v64 += v;
        }
    }
    assert_eq!(check_i64_vec, check_i64_v64, "i64 sequences differ");

    let exp_f64 = kahan_sum(&vec_f64[..]);
    let mut check_f64_vec = 0.0f64;
    {
        let s = &vec_f64[..];
        for &v in s {
            check_f64_vec += v;
        }
    }
    let mut check_f64_v64 = 0.0f64;
    {
        let s = &vec64_f64[..];
        for &v in s {
            check_f64_v64 += v;
        }
    }
    let eps = 1e-6 * exp_f64.abs().max(1.0);
    assert!((check_f64_vec - exp_f64).abs() <= eps && (check_f64_v64 - exp_f64).abs() <= eps);

    // ---------------------- Warm-up (no black_box on slices) ----------------------
    {
        let mut acc = 0i64;
        let s = &vec_i64[..];
        for &v in s {
            acc += v;
        }
        black_box(acc);
    }
    {
        let mut acc = 0i64;
        let s = &vec64_i64[..];
        for &v in s {
            acc += v;
        }
        black_box(acc);
    }
    {
        let mut acc = 0.0f64;
        let s = &vec_f64[..];
        for &v in s {
            acc += v;
        }
        black_box(acc);
    }
    {
        let mut acc = 0.0f64;
        let s = &vec64_f64[..];
        for &v in s {
            acc += v;
        }
        black_box(acc);
    }

    // ---------------------- Timed runs (alternate order) ----------------------
    let mut durs_vec_i64 = Vec::with_capacity(ITERS);
    let mut durs_vec64_i64 = Vec::with_capacity(ITERS);
    let mut durs_vec_f64 = Vec::with_capacity(ITERS);
    let mut durs_vec64_f64 = Vec::with_capacity(ITERS);

    for i in 0..ITERS {
        if i % 2 == 0 {
            let d1 = time_once(|| {
                let mut acc = 0i64;
                let s = &vec_i64[..];
                for &v in s {
                    acc += v;
                }
                acc
            });
            let d2 = time_once(|| {
                let mut acc = 0i64;
                let s = &vec64_i64[..];
                for &v in s {
                    acc += v;
                }
                acc
            });
            durs_vec_i64.push(d1);
            durs_vec64_i64.push(d2);

            let d3 = time_once(|| {
                let mut acc = 0.0f64;
                let s = &vec_f64[..];
                for &v in s {
                    acc += v;
                }
                acc
            });
            let d4 = time_once(|| {
                let mut acc = 0.0f64;
                let s = &vec64_f64[..];
                for &v in s {
                    acc += v;
                }
                acc
            });
            durs_vec_f64.push(d3);
            durs_vec64_f64.push(d4);
        } else {
            let d2 = time_once(|| {
                let mut acc = 0i64;
                let s = &vec64_i64[..];
                for &v in s {
                    acc += v;
                }
                acc
            });
            let d1 = time_once(|| {
                let mut acc = 0i64;
                let s = &vec_i64[..];
                for &v in s {
                    acc += v;
                }
                acc
            });
            durs_vec64_i64.push(d2);
            durs_vec_i64.push(d1);

            let d4 = time_once(|| {
                let mut acc = 0.0f64;
                let s = &vec64_f64[..];
                for &v in s {
                    acc += v;
                }
                acc
            });
            let d3 = time_once(|| {
                let mut acc = 0.0f64;
                let s = &vec_f64[..];
                for &v in s {
                    acc += v;
                }
                acc
            });
            durs_vec64_f64.push(d4);
            durs_vec_f64.push(d3);
        }
    }

    let stats_vec_i64 = summarise(durs_vec_i64);
    let stats_vec64_i64 = summarise(durs_vec64_i64);
    let stats_vec_f64 = summarise(durs_vec_f64);
    let stats_vec64_f64 = summarise(durs_vec64_f64);

    // ---------------------- Report ----------------------
    println!("|------------ Integer (i64) ------------|");
    println!(
        "Vec<i64>   (forced misalignment):     min {}  med {}  p95 {}",
        fmt_dur(stats_vec_i64.min),
        fmt_dur(stats_vec_i64.med),
        fmt_dur(stats_vec_i64.p95)
    );
    println!(
        "Vec64<i64> (64-byte aligned):         min {}  med {}  p95 {}",
        fmt_dur(stats_vec64_i64.min),
        fmt_dur(stats_vec64_i64.med),
        fmt_dur(stats_vec64_i64.p95)
    );
    println!();

    println!("|------------ Float (f64) -------------|");
    println!(
        "Vec<f64>   (forced misalignment):     min {}  med {}  p95 {}",
        fmt_dur(stats_vec_f64.min),
        fmt_dur(stats_vec_f64.med),
        fmt_dur(stats_vec_f64.p95)
    );
    println!(
        "Vec64<f64> (64-byte aligned):         min {}  med {}  p95 {}",
        fmt_dur(stats_vec64_f64.min),
        fmt_dur(stats_vec64_f64.med),
        fmt_dur(stats_vec64_f64.p95)
    );
}
