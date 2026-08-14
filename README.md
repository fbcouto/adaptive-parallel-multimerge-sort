# Adaptive Parallel MultiMerge Sort 🚀

A high-performance, L1-cache optimized, hybrid stable parallel sorting algorithm designed for Relational Databases (DBMS). Implemented in **Rust** (with zero-allocation optimization for primitive types) and **C++20** (via OpenMP). 

The algorithm is built on a single observation: real database keys are almost never random. They arrive as long monotone stretches — an index being rebuilt, sorted segments being consolidated, a table appended in insertion order. A general-purpose sort throws that structure away and pays `n log n` regardless. This engine detects the structure first, in one linear pass, and only sorts what actually needs sorting.

On structured datasets it exceeds **1 Gelem/s** (one billion elements per second) on modern CPUs, and it outperforms highly-optimized libraries such as Rust's `rayon` on data containing many overlapping runs — the real-world relational database scenario.

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

Before anything else, the engine samples roughly 100 consecutive elements near the midpoint of the array and counts direction changes. If the sample looks like noise, there is no structure to exploit and the run detector would only waste memory building metadata for millions of two-element runs. In that case the engine bails out immediately to a specialized parallel fallback: `rayon::par_sort` in Rust, and either `__gnu_parallel::stable_sort` or the internal `chunk_parallel_sort` in C++, selected at build time.

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

### 5. Initialization Elision

The scratch buffer for `Copy` types is allocated without being zero-filled (`unsafe { buffer.set_len(n) }` in Rust, `std::make_unique_for_overwrite` in C++). Every position is written by the merge before it is ever read, so the OS is never asked to zero gigabytes of RAM that are about to be overwritten.

### Stability

The engine is **stable end to end**, and every phase is built to keep it that way: strictly-decreasing runs make reversal safe, complementary tie-breaking makes the bidirectional merge safe, and multi-sequence selection distributes tied elements in stream order so that the parallel partition and the sequential merge agree on the same total order.



## 📊 Benchmark Highlights

Tested on modern multi-core hardware against Rust's native `rayon` library, pushing up to **2.4 Gigabytes of raw `u64` keys (300 Million elements)**.

### Scenario A: 100 Million Elements
| Scenario | Rayon (Baseline) | C++ MultiMerge | Rust MultiMerge (Copy) | Winner |
| :--- | :--- | :--- | :--- | :--- |
| **Fully Sorted** | 87.20 ms | **86.11 ms** | 87.37 ms | **Tie (Hardware Bound ~1.16 Gelem/s)** |
| **Reversed** | 161.15 ms | 159.50 ms | **159.09 ms** | **Rust MultiMerge (Copy)** |
| **Sawtooth (Database-like)** | **1.02 s** | 1.07 s | 1.05 s | **Rayon (~3% lead)** |
| **Random Chaos** | 2.09 s | 2.76 s | **1.99 s** | **Rust MultiMerge (Copy)** |
| **Low Cardinality** | **1.95 s** | 2.57 s | 1.96 s | **Tie (Rayon / Rust)** |

### Scenario B: 300 Million Elements (Extreme Scale)
| Scenario | Rayon (Baseline) | C++ MultiMerge | Rust MultiMerge (Copy) | Winner |
| :--- | :--- | :--- | :--- | :--- |
| **Fully Sorted** | 261.53 ms | **257.09 ms** | 257.44 ms | **C++ / Rust (Hardware Bound)** |
| **Reversed** | **446.24 ms** | 450.33 ms | 461.44 ms | **Rayon (~1% lead)** |
| **Sawtooth (Database-like)** | **2.98 s** | 3.14 s | 3.10 s | **Rayon (~3% lead)** |
| **Random Chaos** | **6.42 s** | 9.22 s | 6.50 s | **Rayon / Rust Tie** |
| **Low Cardinality** | **6.36 s** | 8.89 s | 7.02 s | **Rayon** |

> *Note: Both C++ and Rust implementations achieve processing speeds of over **1 Billion elements per second** in highly structured datasets (Sorted), maxing out the physical memory bandwidth. In chaotic and massive datasets, the $O(1)$ Entropy Shield ensures safe bailout to standard parallel fallback engines, maintaining state-of-the-art times safely close to Rayon.*


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
