# Adaptive Parallel MultiMerge Sort 🚀

A high-performance, L1-cache optimized, hybrid stable parallel sorting algorithm designed for Relational Databases (DBMS). Implemented in **Rust** (with zero-allocation optimization for primitive types) and **C++20** (via OpenMP). 

This algorithm achieves processing speeds exceeding **1 Billion elements per second (1 GHz)** on modern CPUs, outperforming standard highly-optimized libraries like Rust's `rayon` in structured and partially sorted datasets (real-world relational database scenarios).

# 📚 Academic Background & Prior Work

The core theoretical foundation of this parallel architecture is based on the original research and paper:

- **Title:** *Multimerge*
- **Authors:** Fernando B. Couto & Fábio S. Couto
- **Conference:** PDPTA'11 — *The 2011 International Conference on Parallel and Distributed Processing Techniques and Applications*
- **Lecture Series:** *WorldComp'11 (The 2011 World Congress in Computer Science, Computer Engineering, and Applied Computing)*

This engine modernizes the foundational multi-merge paradigms established in the 2011 paper, translating those parallel processing techniques into idiomatic, memory-safe, and highly optimized Rust concurrency using modern work-stealing schedulers.

## 🧠 Architecture & Key Innovations

Standard parallel sorting algorithms often suffer from "task explosion" or cache thrashing when merging massive datasets. This engine solves these problems using four architectural pillars:

1. **O(1) Entropy Shield (Phase 0):** Samples 100 central elements to determine if the data is purely random noise. If chaos is detected, it delegates sorting to a highly specialized fallback (`rayon::par_sort` in Rust or `__gnu_parallel::stable_sort` in C++).
2. **Fractal Detection & L1 Cache Optimization (Phase 1):** Scans the dataset dynamically dividing it into `4096-element` micro-slices (perfectly fitting a standard 32KB L1 Data Cache for 64-bit integers). 
3. **Hybrid $O(\log N)$ Co-Rank Merge:** Instead of discovering boundaries sequentially, it computes them beforehand using a double binary search (`co_rank`). This guarantees a collision-free bidirectional parallel merge.
4. **Initialization Elision (Zero-Cost Allocation):** Buffer memory allocation for standard `Copy` types leverages memory bypass techniques (`unsafe { buffer.set_len(n); }` in Rust, `std::make_unique_for_overwrite` in C++) to prevent the OS from zero-filling gigabytes of RAM unnecessarily.

## 📊 Benchmark Highlights

Tested on an Intel CPU against Rayon using a slice of 100,000,000 elements (`u64`):

| Scenario (100M Elements) | Rayon | C++ MultiMerge | Rust MultiMerge (Copy) | Winner |
| :--- | :--- | :--- | :--- | :--- |
| **Fully Sorted** | 86.18 ms | 85.26 ms | **86.19 ms** | **Tie (Hardware Memory Bound ~1.16 Gelem/s)** |
| **Reversed** | 150.27 ms | 155.83 ms | **150.28 ms** | **Tie (Parallel SIMD Reverse)** |
| **Sawtooth (Database-like)** | 1.00 s | 1.01 s | **0.82 s (820 ms)** | **Rust MultiMerge (Copy)** |
| **Random Chaos** | **1.99 s** | 2.87 s | 2.01 s | **Rayon** |

> *Note: In Sawtooth scenarios (structured runs common in databases), the Rust Copy engine obliterates standard libraries due to its zero-copy slice tracking and $O(\log N)$ parallel merges.*

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
