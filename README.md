# Adaptive Parallel MultiMerge Sort 🚀

A high-performance, L1-cache optimized, hybrid stable parallel sorting algorithm designed for Relational Databases (DBMS). Implemented in **Rust** (with zero-allocation optimization for primitive types) and **C++20** (via OpenMP). 

The algorithm is built on a single observation: real database keys are almost never random. They arrive as long monotone stretches — an index being rebuilt, sorted segments being consolidated, a table appended in insertion order. A general-purpose sort throws that structure away and pays `n log n` regardless. This engine detects the structure first, in one linear pass, and only sorts what actually needs sorting.

Against `std::stable_sort(std::execution::par, ...)` — the parallel stable sort of the C++17 standard library — it is **12x faster on reversed data, 2.9x on sorted data and 1.27x on almost-sorted data**, and costs at most 8% where there is no structure at all. Those first three are the shape of a rebuilt index, a table in insertion order, a segment merged by a previous operation: the common case in a database, and the one a general-purpose sort pays full price for.

# 📚 Academic Background & Prior Work

The core theoretical foundation of this parallel architecture is based on the original research and paper:

- **Title:** *Multimerge*
- **Authors:** Fernando B. Couto & Fábio S. Couto
- **Conference:** PDPTA'11 — *The 2011 International Conference on Parallel and Distributed Processing Techniques and Applications*
- **Lecture Series:** *WorldComp'11 (The 2011 World Congress in Computer Science, Computer Engineering, and Applied Computing)*

This engine modernizes the foundational multi-merge paradigms established in the 2011 paper, translating those parallel processing techniques into idiomatic, memory-safe, and highly optimized Rust concurrency using modern work-stealing schedulers.

## 🧠 Architecture & Key Innovations

The engine runs in two phases: a linear **detection** pass that discovers the structure already present in the data, and a **merge** phase that only does the work that structure leaves behind. Five design decisions carry the algorithm.

### 1. O(1) Entropy Shield (Phase 0)

Before anything else, the engine samples roughly 100 consecutive elements near the midpoint of the array and counts direction changes. If the sample looks like noise, there is no structure to exploit and the run detector would only waste memory building metadata for millions of two-element runs. In that case the engine bails out immediately to a parallel fallback: `rayon::par_sort` in Rust, and in C++ one of three, selected at build time.

**Which fallback is chosen matters more than it looks.** Measured at 5M with 8 threads, `std::stable_sort(std::execution::par, ...)` ran 29–48% faster than `__gnu_parallel::stable_sort` on every chaotic scenario. With the GNU extension as the fallback the whole engine trailed the C++17 parallel sort by 62–88% on random input — not because of its own code, but because of where it delegated. The preference order is now PSTL first, the internal `chunk_parallel_sort` second, and `__gnu_parallel` only on explicit request.

This is a constant-cost bet made on a small sample. It is cheap by design — the cost of being wrong is bounded by the fallback, which is a competent parallel sort in its own right.

### 2. Overlapping Block Detection & the Metadata Monoid (Phase 1)

This is the core of the algorithm.

The array is divided into blocks that **overlap by exactly one element**. Each block is scanned independently and in parallel for monotone runs, and each run is recorded as a single signed integer: positive for ascending, negative for descending. Nothing is moved; the pass is read-only.

The one-element overlap is not an implementation detail — it is what makes the whole scheme work. With blocks `[0,11)`, `[10,21)`, `[20,30)`, every adjacent-pair comparison in the array belongs to **exactly one** block: no gaps, no duplicates. Without the overlap, the comparison straddling each boundary would belong to no block at all, and stitching would require a second pass that reads the array again.

Because of that, joining two blocks' metadata is a **pure algebraic operation on the metadata alone** — the array is never touched:

- Same sign → the runs are contiguous and monotone through the shared element, so they merge into one run of length `a + b - 1` (the shared element was counted twice).
- Different signs → the shared element is a peak or a valley. It stays with the left run, and the right run loses one.
- A block reporting a single element is the shared element itself, and is absorbed by its neighbour.

The operation is associative, so it composes as a **monoid under a parallel reduce**, and the identical function stitches micro-blocks into macro-blocks and macro-blocks into the final result. The metadata is self-describing: the lengths encode everything needed to join, so no absolute positions have to be carried through the reduction.

If the whole array collapses to a single run, the answer is immediate: positive means it was already sorted and the engine returns without writing a byte; negative means one parallel reverse and it is done.

**One asymmetry is load-bearing.** Ascending runs are detected with `<=` and descending runs with a strict `>`. Equal elements therefore always fall into ascending runs, which makes every descending run *strictly* decreasing — and reversing a strictly decreasing sequence cannot swap two equal elements. That is what keeps the reverse path stable.

### 3. Bidirectional Merge

Two workers merge the same pair of runs simultaneously: one walks **forward from the start**, one walks **backward from the end**, and they meet in the middle.

They never collide, and no partition point has to be computed:

> A forward merge that breaks ties toward the left run emits exactly the `k` smallest elements in order. A backward merge that breaks ties toward the right run emits exactly the `total - k` largest. For **any** `k`, those two sets partition the multiset.

Each side simply counts its own output and stops. The complementary tie-breaking rules — left-wins going forward, right-wins going backward — are what make the theorem hold and what preserve stability at the same time.

In Rust this falls out of the type system: both halves borrow the source runs immutably and write into disjoint halves of the destination via `split_at_mut`, so the compiler proves the absence of a data race with no `unsafe` and no special cases.

### 4. Cache-Blocked K-Way Merge

Merging `k` sorted runs with a binary tree costs `log₂(k)` full passes over memory. On a bandwidth-saturated machine that pass count — not the comparison count — is what determines the runtime. An 8-way merge costs `log₈(k)` passes instead: for 30,000 runs, 5 levels rather than 15.

The k-way merge is **not** implemented with a loser tree. A loser tree indexes its stream heads with a value only known at runtime, which prevents the compiler from keeping them in registers; measured against the plain binary kernel it cost roughly an order of magnitude more per element, erasing the entire benefit of fewer passes.

Instead, each output tile is partitioned across all `k` runs with **multi-sequence selection**, and the tile is then merged by an ordinary binary tree running **entirely inside L1**. Only the final level writes to the destination, so DRAM sees one read and one write per k-way level while the inner loop keeps the fast, register-resident binary kernel.

The strategy is **hybrid by size**: k-way applies only above a threshold where a merge exceeds cache. Below it, the binary path is faster — it has no partitioning overhead, it keeps the shortcut that turns an already-ordered merge into a copy, and it is where the bidirectional merge operates. Adjacent groups that are already in order are coalesced into a single stream before merging, so the shortcut survives k-way as well.

### 5. Buffer Allocation

The scratch buffer in C++ is allocated with `std::make_unique_for_overwrite`, which for trivial types does not touch the memory: pages are faulted in by the merge itself, and the OS is never asked to zero gigabytes of RAM about to be overwritten.

**The Rust side used to do the same and no longer does.** `Vec::with_capacity` followed by `set_len` was measured at 20M elements and saved 1 ms out of 458 — nothing, because the merge writes every position anyway and the page faults happen either way. It also made the safe public API unsound: `T: Copy` does not imply that every bit pattern is a valid value, and `bool`, `char`, `NonZeroU64` and `&str` are all `Copy` while rejecting most patterns. `vec![arr[0]; n]` is used instead, which compiles to `alloc_zeroed` when the value is zero in bits and costs nothing measurable when it is not.

### Stability

The engine is **stable end to end**, and every phase is built to keep it that way: strictly-decreasing runs make reversal safe, complementary tie-breaking makes the bidirectional merge safe, and multi-sequence selection distributes tied elements in stream order so that the parallel partition and the sequential merge agree on the same total order.



## 📊 Benchmarks

C++ engine, `{key, index}` pairs of 16 bytes, 15M elements, 8 threads. Median of
9 runs with rotating order between engines, a discarded warm-up pass, and a
control row that measures one engine twice to expose the noise floor. Every
output is checked element by element against `std::stable_sort`, index included,
so a wrong answer cannot be mistaken for a fast one.

The reference is **`std::stable_sort(std::execution::par, ...)`** — the parallel
stable sort of the C++17 standard library, backed by Intel TBB. It is the
strongest baseline available. Comparing an 8-thread engine against the
single-threaded `std::stable_sort` would measure parallelism, not algorithm.

| Scenario | `stable_sort` | C++17 `par` | MultiMerge | vs `par` | noise |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Reversed** | 990.8 ms | 455.7 ms | **37.5 ms** | **12.2x** | 2.5% |
| **Already sorted** | 897.2 ms | 33.0 ms | **11.5 ms** | **2.9x** | 0.9% |
| **Almost sorted** (0.1% noise) | 942.4 ms | 549.9 ms | **432.5 ms** | **1.27x** | 1.3% |
| Sawtooth (`i % 1000`) | 958.4 ms | 483.3 ms | 474.6 ms | 1.02x | 2.8% |
| Random, high cardinality | 1932.6 ms | 537.9 ms | 503.8 ms | 1.07x | 1.4% |
| Random, 64 distinct keys | 1313.9 ms | 497.3 ms | 517.9 ms | 0.96x | 0.6% |
| Random, 10 distinct keys | 1193.5 ms | 454.3 ms | 493.0 ms | 0.92x | 3.4% |

### What the numbers say

**Global order is where the engine is untouchable.** Reversed data finishes in
37.5 ms against 455.7 for the standard parallel sort, and already-sorted data in
11.5 against 33.0. Neither is a faster sort: the detection pass finds a single
run and the engine returns without sorting at all, or reverses once in parallel.
No comparison sort can match that, because it must at minimum compare.

This is not a synthetic corner. It is the shape of a rebuilt index, a table read
in insertion order, a segment already merged by a previous operation — the
common case in a database, and the one a general-purpose sort pays full price
for.

**Partial order pays too, and it needs scale.** Almost-sorted data is 21% faster
than the baseline at 15M. At 5M the same scenario was a tie: the k-way merge
engages for too few levels before the sub-ranges drop under cache, and it does
not get the chance to pay for itself. This is a property of the design, not a
tuning accident — the k-way path exists to cut passes over DRAM, and small
arrays never leave cache.

**Chaotic input costs 4 to 8%, and that is the price of asking.** On the two
low-cardinality random scenarios the engine is measurably *behind* the baseline
it delegates to, by more than the noise floor. That gap is the entropy shield
itself: sampling the array, deciding there is no structure, and handing the work
over. It is the honest cost of adaptivity, and it is small enough that the engine
remains a safe general replacement — which is the property that matters most. An
adaptive sort that sometimes loses badly is not usable; one that loses 8% in its
worst case is.

**The fallback choice mattered more than the algorithm.** Before this was
measured the engine delegated to `__gnu_parallel::stable_sort`, and lost 62–88%
to the C++17 `par` on random input — not because its own code was slow, but
because of where it handed off. `__gnu_parallel` uses a loser tree for its
multiway merge, and a loser tree indexes its stream heads with a runtime value,
which stops the compiler from keeping them in registers. This engine measured
that cost directly and rejected the loser tree for the same reason (see the k-way
section). Pointing the shield at the PSTL closed the entire gap.

### Caveats

**These numbers are from one machine**, an 8-core laptop, at one size. The cache
budgets, the block size and the k-way threshold were all tuned there and should
be re-measured on the target hardware.

**The sawtooth pattern (`i % 1000`) is the worst case for adaptive merging**, not
a typical one: every run ends at its maximum and the next begins at its minimum,
so adjacent runs overlap completely in range and the shortcut that turns an
already-ordered merge into a copy never fires. Real workloads, where sorted
segments overlap only partially, land between this and the fully sorted case.

**Measurement noise dominates smaller runs.** At 5M several scenarios showed 10
to 22% spread between repeated measurements of the same engine, which is wider
than most of the effects being measured. The 15M table above was taken when the
machine was quiet, with every noise floor under 3.4%. Treat any published ratio
narrower than its own noise column as a tie.

### Correctness

Every release is checked by four independent layers:

- 9 unit tests plus a 7-case integration suite in Rust, covering the metadata
  monoid, stability under duplicate keys, descending runs with plateaus, block
  boundary sizes, and a 150-case structured/chaotic fuzz
- A C++ stability suite over the same patterns, verifying that equal keys keep
  their original relative order and that the output is byte-identical to
  `std::stable_sort`
- A cross-check binary that sorts identical inputs with all three engines
  (Rayon, MultiMerge Rust, MultiMerge C++) across six data patterns and compares
  every output against `slice::sort` as an independent reference
- A C++ kernel test that validates the bidirectional merge theorem over 4000
  cases with a key domain of 1 to 6 distinct values, saturating tie-breaking

## ⚙️ Getting Started

### Prerequisites (Windows)
To build both the Rust and C++ portions natively, you need the **MSYS2 UCRT64** environment.
```bash
# Inside MSYS2 UCRT64 terminal:
pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-rust

```

### Building & Benchmarking

This project uses a custom `build.rs` script that transparently calls `g++` via `cc-rs` and links the OpenMP library.

1. **Set the default toolchain** (already provided in `.cargo/config.toml`):
```toml
[build]
target = "x86_64-pc-windows-gnu"

```


2. **Run tests**:
```bash
cargo test

```


3. **Run the Benchmark Arena** (Criterion):
```bash
CXX=g++ cargo bench

```

4. **C++ engine build switches**, all read from the environment so A/B
   comparison never requires editing a source file. The active configuration is
   echoed on every build.

| Variable | Effect | Default |
| :--- | :--- | :--- |
| `MULTIMERGE_PSTL` | chaotic fallback via `std::execution::par`; needs PSTL support and TBB linked | off |
| `MULTIMERGE_GNU` | chaotic fallback via `__gnu_parallel::stable_sort` | off |
| *(neither)* | chaotic fallback via the internal `chunk_parallel_sort`, no dependency | — |
| `MULTIMERGE_KWAY` | cache-blocked k-way merge; set to a power of two to change fan-out | on, 8 |
| `MULTIMERGE_BIDIR` | bidirectional leaf merge | on |
| `MULTIMERGE_TBB_SCHED` | merge tree via `tbb::parallel_invoke` | off |
| `MULTIMERGE_TBB_REDUCE` | metadata fold via `tbb::parallel_reduce` | off |
| `MULTIMERGE_TBB_MALLOC` | link `tbbmalloc_proxy` | off |

**The engine does not depend on TBB.** Where PSTL is available it is the best
fallback and worth enabling; where it is absent the internal `chunk_parallel_sort`
covers it with no external dependency. The three `TBB_*` switches were measured
and made no difference on the test machine — they are kept so the experiment can
be repeated on other hardware, not because they are recommended.

**PSTL availability varies, which is why it is off by default.**

| Platform | Parallel policies | Install |
| :--- | :--- | :--- |
| Linux, GCC ≥ 9 + libstdc++ | yes | `apt install libtbb-dev`, link `-ltbb` |
| MSYS2 **UCRT64** | yes | `pacman -S mingw-w64-ucrt-x86_64-tbb`, link `-ltbb12` |
| MSYS2 **MINGW64** | no | — |
| Clang + libc++ | incomplete | — |

The failure mode is quiet: without a working PSTL the `std::execution::par`
overloads still compile and simply run serial, because the standard only says
`par` *may* parallelise. Verify before trusting a measurement — if the parallel
policy is not at least twice as fast as the sequential `std::stable_sort`, it did
not engage. `src/vs_stable.cpp` performs exactly this check and prints a warning
when the ratio looks wrong.



## 💻 Usage in Your Project

### Using the Rust Engine

The engine exposes a single, highly optimized routine for `Copy` types (e.g., Database primary keys, timestamps, `u64`, `i64`).

```rust
use adaptive_parallel_multimerge_sort::sort;

fn main() {
    let mut db_keys: Vec<u64> = vec![9, 5, 1, 3, 7, 8, 2, 6, 4, 0, 15, 12];
    
    // Executes the adaptive algorithm
    sort(&mut db_keys);
    
    assert!(db_keys.windows(2).all(|w| w[0] <= w[1]));
}

```

### Using the C++ Engine (Standalone)

You can take `src/MultiMergeSort.hpp` and drop it directly into your modern C++20 project (like Firebird SQL internals).

```cpp
#include "MultiMergeSort.hpp"
#include <vector>

int main() {
    std::vector<uint64_t> data = {9, 5, 1, 3, 7, 8, 2, 6};
    
    // Pass as std::span
    multimerge::sort(std::span<uint64_t>(data));
    
    return 0;
}

```

*Compile with: `g++ main.cpp -std=c++20 -O3 -fopenmp -o main*`



# 📄 License
This project is licensed under the Apache License 2.0.

You may obtain a copy of the license at:

https://www.apache.org/licenses/LICENSE-2.0

Copyright © Fernando B. Couto

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this project except in compliance with the License.
You may obtain a copy of the License at:

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.