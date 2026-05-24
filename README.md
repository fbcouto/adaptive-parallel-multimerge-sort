# Adaptive Parallel Multimerge Sort in Rust

A high-performance, hybrid, and adaptive parallel sorting architecture designed in Rust. This engine leverages Rayon for work-stealing parallelism and implements dynamic profiling heuristics to optimize sorting strategies based on data distribution, maximizing hardware utilization while avoiding common parallel overhead traps.

---

## Academic Background & Prior Work

The core theoretical foundation of this parallel architecture is based on the original research and paper:

- **Title:** *Multimerge*
- **Authors:** Fernando B. Couto & Fábio S. Couto
- **Conference:** PDPTA'11 — *The 2011 International Conference on Parallel and Distributed Processing Techniques and Applications*
- **Lecture Series:** *WorldComp'11 (The 2011 World Congress in Computer Science, Computer Engineering, and Applied Computing)*

This engine modernizes the foundational multi-merge paradigms established in the 2011 paper, translating those parallel processing techniques into idiomatic, memory-safe, and highly optimized Rust concurrency using modern work-stealing schedulers.

---

# 🚀 Key Features

- **Adaptive Oscillation Heuristic**  
  Dynamically samples data at runtime to detect patterns (sorted, reversed, or highly repetitive/chaotic states) before committing CPU cycles.

- **Work-Stealing Parallelism**  
  Driven by Rayon, partitioning workloads across available logical cores only when data scale justifies the synchronization overhead.

- **ZTrait-Based Polymorphism >   Leverages Rust's trait system (T: Ord) to achieve zero-cost abstractions. The engine achieves native performance for primitives and complex structures alike through compile-time monomorphization, eliminating the need for unsafe casting or manual type dispatching.

- **Memory-Efficient Anchoring**  
  Optimized block thresholds (32,768 elements) to maximize L2/L3 cache locality and prevent memory bus saturation during heavy parallel merge phases.

---

# 🧠 Architecture & Design Decisions

## The Oscillation Heuristic

Parallel sorting algorithms often suffer from performance degradation when applied to low-entropy (already sorted) or ultra-low-range data due to thread allocation overhead.

To mitigate this, this engine runs a lightweight pre-scan:

- **Global Tendency Check**  
  Verifies if the array is already sorted or strictly reversed (`O(N)` early exit).

- **Entropy Sampling**  
  In mid-sized vectors, it samples local directional changes.

  - If high oscillation (pure chaos) is detected, it instantly switches to an unstable parallel quicksort branch.
  - If low oscillation is detected, it deploys a stable parallel merge sort variant using synchronized scratch buffers.

```text
                  [ Input Slice ]
                         |
               Distro Trend Analysis
               /         |          \
     (Sorted/Inverted) (Pure Chaos) (Low-Entropy/Mixed)
           /             |                 \
     Early Exit     Parallel Unstable   Parallel Merge Sort
    (In-place/Rev)    (Work-Stealing)    (Timsort-style + Rayon)
```

---

## Safe Compile-Time Polymorphism
Unlike typical high-level languages that rely on dynamic dispatch (runtime type checking), this engine utilizes Rust's static dispatch mechanism.

By constraining the input slice with T: Ord + Clone + Send, the compiler generates specialized, optimized machine code for every data type processed. This design ensures that:

- Safety: No unsafe pointer casting or manual type reflection is required, guaranteeing memory safety at all times.

- Speed: The sorting logic is "inlined" for the specific types used, providing the same machine-code efficiency as hardcoded implementations.

- Flexibility: The engine natively supports any data type that implements standard comparison traits, without requiring source code modifications to add new types.

---

# 🧪 Benchmarking Methodology CRITERION



---

# 📊 Performance Benchmarks
Our benchmarks indicate that by combining these strategies, the Adaptive MultiMerge engine consistently outperforms the standard Rust parallel sort (par_sort_unstable) by approximately 40% on large datasets.
Environment: 5,000,000 u64 elements, optimized release build.
Algorithm	Mean Time	Improvement
Rayon par_sort_unstable	~98.7 ms	-
MultiMerge (Adaptive)	~58.6 ms	~40.6% Faster
Technical Features
    • Zero-Overhead Abstractions: Uses generic T: Ord + Clone traits with pointer arithmetic for maximum speed.
    • Cache-Friendly: The design minimizes memory writes by avoiding physical data reversals.
    • Adaptive Fallback: Automatically switches to the fastest available parallel sorting method based on data entropy.
    • Production Ready: Fully validated with integration tests covering chaotic, reverse-ordered, and duplicate-heavy datasets.



---
# Usage
Add this to your Cargo.toml:

Ini, TOML
[dependencies]
adaptive-parallel-multimerge-sort = "1.0.0"
Integration example:

Rust
use adaptive_parallel_multimerge_sort::sort;

fn main() {
    let mut data = vec![9, 3, 5, 1, 7, 2, 8, 4, 6];
    sort(&mut data);
    println!("{:?}", data);
}
---
# 📄 License

This project is licensed under the Apache License 2.0.

You may obtain a copy of the license at:

- https://www.apache.org/licenses/LICENSE-2.0

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
