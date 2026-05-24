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

# 🧪 Benchmarking Methodology

Performance evaluation is conducted using Criterion, the industry-standard microbenchmarking framework for Rust, ensuring statistical rigor through:

- **100-Sample Iterations**  
  Every data point represents a distribution profile of 100 distinct measurements to isolate environmental noise.

- **Outlier Isolation**  
  Automatic classification of high/low outliers to guarantee the purity of the computed mean.

- **Multi-Distribution Testing**  
  Algorithms are heavily stressed across various layout topologies, including:

  - `random` — Uniformly distributed chaotic data.
  - `random_z1` — Zipf distribution (power-law distribution mimicking real-world database keys).
  - `random_d20` — High-duplication stress tests (elements strictly bound between 0 and 20).

---

# 📊 Performance Highlights

The architecture demonstrates massive scalability compared to the standard library implementations, particularly as vector sizes cross cache boundaries and scale up to `10^7` elements.
<table>
  <tr>
    <td align="center">
      <img src="/images/performance_u64_random_From_10k_Onwards_cold.png" alt="u64 random Cold" width="400">
      <br>
      <em>u64 random Cold</em>
    </td>
    <td align="center">
      <img src="/images/performance_u64_random_From_10k_Onwards_hot.png" alt="u64 random hot" width="400">
      <br>
      <em>u64 random hot</em>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="/images/performance_i32_random_From_10k_Onwards_cold.png" alt="i32 random cold" width="400">
      <br>
      <em>i32 random cold</em>
    </td>
    <td align="center">
      <img src="/images/performance_i32_random_From_10k_Onwards_hot.png" alt="i32 random hot" width="400">
      <br>
      <em>i32 random hot</em>
    </td>
  </tr>
</table>
---

## 📈 Test Case 1: Standard Primitive Scaling (`u64` Random Chaotic Distribution)

| Array Size (Elements) | `rust_std_stable` | `rust_std_unstable (IPNSort)` | `Multimerge_Adaptativo (This Engine)` | Real-World Performance Gain |
|---|---:|---:|---:|---|
| 183,845 | 3.47 ms | 3.47 ms | **1.33 ms** | 🚀 2.6x faster than the state-of-the-art native library |
| 1,000,000 | 44.18 ms | 25.14 ms | **8.37 ms** | 🚀 3.0x faster vs unstable / 5.2x faster vs stable |
| 10,000,000 | 461.67 ms | 249.48 ms | **89.48 ms** | 🚀 2.7x faster vs unstable / 5.1x faster vs stable |

---

## 📉 Test Case 2: Primitive Processing Optimization (`i32` Random Chaotic Distribution)

| Array Size (Elements) | `rust_std_stable` | `rust_std_unstable` | `Multimerge_Adaptativo (This Engine)` | Real-World Performance Gain |
|---|---:|---:|---:|---|
| 10,000,000 | 382.90 ms | 228.56 ms | **85.64 ms** | 🚀 2.6x faster vs unstable / 4.4x faster vs stable |

---

## 🔄 Test Case 3: High Duplication Performance (`u64` Bounded Value Matrix - `random_d20`)

| Array Size (Elements) | `rust_std_stable` | `rust_std_unstable (IPNSort)` | `Multimerge_Adaptativo (This Engine)` | Real-World Performance Gain |
|---|---:|---:|---:|---|
| 400,000 | 4.73 ms | 1.89 ms | **1.54 ms** | 🚀 22% faster due to core multi-threaded execution paths |
| 10,000,000 | 176.38 ms | 72.43 ms | **41.78 ms** | 🚀 Significant scalability gains under duplication-heavy workloads |

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
