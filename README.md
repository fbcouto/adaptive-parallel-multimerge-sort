# 🚀 Adaptive Parallel Multimerge Sort in Rust

<div align="center">

### High-Performance • Hybrid • Adaptive • Parallel Sorting Engine

A high-performance, hybrid, and adaptive parallel sorting architecture designed in Rust. This engine leverages Rayon for work-stealing parallelism and implements dynamic profiling heuristics to optimize sorting strategies based on data distribution, maximizing hardware utilization while avoiding common parallel overhead traps.

</div>

---

# 📚 Academic Background & Prior Work

The core theoretical foundation of this parallel architecture is based on the original research and paper:

- **Title:** *Multimerge*
- **Authors:** Fernando B. Couto & Fábio S. Couto
- **Conference:** PDPTA'11 — *The 2011 International Conference on Parallel and Distributed Processing Techniques and Applications*
- **Lecture Series:** *WorldComp'11 (The 2011 World Congress in Computer Science, Computer Engineering, and Applied Computing)*

This engine modernizes the foundational multi-merge paradigms established in the 2011 paper, translating those parallel processing techniques into idiomatic, memory-safe, and highly optimized Rust concurrency using modern work-stealing schedulers.

---

# 🚀 Key Features

## Adaptive Oscillation Heuristic
Dynamically samples data at runtime to detect patterns (sorted, reversed, or highly repetitive/chaotic states) before committing CPU cycles.

## Work-Stealing Parallelism
Driven by Rayon, partitioning workloads across available logical cores only when data scale justifies the synchronization overhead.

## Trait-Based Polymorphism
Leverages Rust's trait system (`T: Ord`) to achieve zero-cost abstractions. The engine achieves native performance for primitives and complex structures alike through compile-time monomorphization, eliminating the need for unsafe casting or manual type dispatching.

## Memory-Efficient Anchoring
Optimized block thresholds (32,768 elements) to maximize L2/L3 cache locality and prevent memory bus saturation during heavy parallel merge phases.

---

# 🧠 Architecture & Design Decisions

## Parallel sorting algorithms often suffer from performance degradation when applied to low-entropy (already sorted) or ultra-low-range data due to thread allocation overhead.

To mitigate this, this engine runs a lightweight pre-scan:

- **Global Tendency Check** Verifies if the array is already sorted or strictly reversed (`O(N)` early exit).
- **Topology Mapping** For all other datasets, it maps ascending and descending runs, seamlessly normalizing them into a strictly stable parallel merge pipeline.

```text
                  [ Input Slice ]
                         |
               Distro Trend Analysis
               /         |          \
           (Sorted)  (Reversed)  (Mixed/Chaotic)
             /           |               \
       Early Exit   Reverse Exit   Parallel MultiMerge
      (O(N) return) (O(N) in-place) (Strictly Stable)

```
# 🔒 Safe Compile-Time Polymorphism

Unlike typical high-level languages that rely on dynamic dispatch (runtime type checking), this engine utilizes Rust's static dispatch mechanism.

By constraining the input slice with `T: Ord + Clone + Send + Sync`, the compiler generates specialized, optimized machine code for every data type processed. This design ensures that:

- **Safety:** No unsafe pointer casting or manual type reflection is required, guaranteeing memory safety at all times.
- **Speed:** The sorting logic is "inlined" for the specific types used, providing the same machine-code efficiency as hardcoded implementations.
- **Flexibility:** The engine natively supports any data type that implements standard comparison traits, without requiring source code modifications to add new types.

---

# 🧪 Benchmarking Methodology (CRITERION)

![Performance Comparison Chart](.images/lines.png)
![Violin Plot](.images/violin.png)


---

# # 🛡️ Uncompromised Stability

A crucial distinction of the Adaptive MultiMerge engine is its commitment to **strict stability** (`O(1)` spatial ordering of equal elements). While unstable sorting methods (like quicksort or pattern-defeating quicksort) are traditionally faster because they swap elements without tracking chronological order, they are destructive to complex datasets (e.g., database rows sorted by secondary keys).

This engine competes directly with—and outperforms—Rust's highly optimized standard stable sorting mechanisms (inspired by Timsort).

## Pure Stable Dominance over Chaos
Processing pure entropy (random data) is historically the weak point of metadata-heavy stable algorithms. Traditional stable sorts suffer from severe memory allocation penalties and stack overhead when trying to identify "runs" in completely chaotic data.

Unlike hybrid engines that are forced to downgrade to an unstable sort when they encounter chaos, the Adaptive MultiMerge architecture handles maximum entropy natively. Its Phase 1 metadata generation and Phase 2 directional merging are so mathematically efficient that the engine processes pure randomness **up to 42% faster** than the standard library's stable sort (scaling super-linearly up to 300M+ elements). 

This ensures state-of-the-art speed across all distributions without ever compromising data stability.

---

# 📊 Performance Benchmarks (Criterion V2)

The true measure of a robust stable sorting algorithm is how it scales under maximum entropy (pure random chaos) where hardware memory constraints are severely tested. 

**Environment:** * Hardware: AMD/Intel 8+ Cores
* Scenario: High Entropy (Pure Random `u64`)

| Elements | Standard Library (`rayon::par_sort`) | Adaptive MultiMerge | Delta |
| :--- | :--- | :--- | :--- |
| **1 Million** | ~13.98 ms | **~8.91 ms** | **~36.3% Faster** |
| **10 Million** | ~174.37 ms | **~102.13 ms** | **~41.4% Faster** |
| **100 Million** | ~1.99 s | **~1.16 s** | **~41.7% Faster** |
| **300 Million** | ~6.56 s | **~3.76 s** | **~42.6% Faster** |

> *Note: As the dataset scales to massive proportions (300M+ elements), the MultiMerge architecture increasingly outpaces the standard library, demonstrating superior L2/L3 cache utilization and lower memory bandwidth saturation.*

---

# ⚙️ Technical Features

- **Zero-Overhead Abstractions:** Uses generic `T: Ord + Clone` traits with pointer arithmetic for maximum speed.
- **Cache-Friendly:** The design minimizes memory writes by avoiding physical data reversals.
- **Unified Stable Architecture:** Processes highly structured data and pure entropy with the same strictly stable engine, eliminating the need for unstable fallbacks.
- **Production Ready:** Fully validated with integration tests covering chaotic, reverse-ordered, and duplicate-heavy datasets.

---

# 📦 Usage

## Add this to your `Cargo.toml`

```toml
[dependencies]
adaptive-parallel-multimerge-sort = "1.0.0"
```

## Integration Example
```
use adaptive_parallel_multimerge_sort::sort;

fn main() {
    let mut data = vec![9, 3, 5, 1, 7, 2, 8, 4, 6];
    sort(&mut data);
    println!("{:?}", data);
}
```
---
### 📄 License
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
