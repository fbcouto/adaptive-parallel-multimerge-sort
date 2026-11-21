# Adaptive Parallel MultiMerge Sort 🚀

A high-performance, L1-cache optimized, hybrid stable parallel sorting algorithm designed for Relational Databases (DBMS). Implemented in **Rust** (with zero-allocation optimization for primitive types) and **C++20** (via OpenMP). The Rust crate ships two engines over the same merge tree: `sort`, the default, which is not stable, and `sort_stable`.

The algorithm is built on a single observation: real database keys are almost never random. They arrive as long monotone stretches — an index being rebuilt, sorted segments being consolidated, a table appended in insertion order. A general-purpose sort throws that structure away and pays `n log n` regardless. This engine detects the structure first, in one linear pass, and only sorts what actually needs sorting.

Against `std::stable_sort(std::execution::par, ...)` — the parallel stable sort of the C++17 standard library — it is **12x faster on reversed data, 2.9x on sorted data and 1.27x on almost-sorted data**, and costs at most 8% where there is no structure at all. Those first three are the shape of a rebuilt index, a table in insertion order, a segment merged by a previous operation: the common case in a database, and the one a general-purpose sort pays full price for.

The engine has been fitted into two SQL engines and measured with each one's own tests: an extension operator for Apache DataFusion and a prototype in ClickHouse (see [Integrations](#-integrations)).

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

Before anything else, the engine samples roughly 100 consecutive elements near the midpoint of the array and counts direction changes. If the sample looks like noise, there is no structure to exploit and the run detector would only waste memory building metadata for millions of two-element runs. In that case the engine bails out immediately to a parallel fallback: in C++ one of three, selected at build time. (The Rust engine has since moved to eight windows spread over the array and its own block-and-merge fallback; see section 6.)

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

**One asymmetry is load-bearing in the C++ engine.** Ascending runs are detected with `<=` and descending runs with a strict `>`. Equal elements therefore always fall into ascending runs, which makes every descending run *strictly* decreasing — and reversing a strictly decreasing sequence cannot swap two equal elements. That is what keeps the reverse path stable. The price is that a reversed table with repeated keys is never seen as reversed. The Rust engine lifts that restriction; see section 6.

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

**The Rust side used to do the same and no longer does.** `Vec::with_capacity` followed by `set_len` made the safe public API unsound: `T: Copy` does not imply that every bit pattern is a valid value, and `bool`, `char`, `NonZeroU64` and `&str` are all `Copy` while rejecting most patterns. The Rust engine has no `unsafe` at all.

The measurement that justified dropping it — 1 ms out of 458 — turned out to be confounded. `vec![arr[0]; n]` compiles to `alloc_zeroed` when `arr[0]` is the integer zero, and nearly every synthetic dataset starts at zero, so the "initialized" buffer was being handed over lazily and for free. With `arr[0] != 0` the fill costs about 14 ms per 80 MB (64 runs on 10M `u64`: 64.0 ms with a leading zero, 77.5 ms without). Real data rarely starts with zero.

The buffer is now created as a **parallel copy of the array**. That replaces both the sequential fill and the copy that odd-depth leaves needed, with one parallel pass, and makes the timing independent of the first element. What it cannot do in safe Rust is skip the first touch altogether; the reusable-buffer entry points below are the way to avoid it.

### Stability

The merge machinery is **stable end to end**, and every phase is built to keep it that way: complementary tie-breaking makes the bidirectional merge safe, and multi-sequence selection distributes tied elements in stream order so that the parallel partition and the sequential merge agree on the same total order.

### 6. The Rust engine: two sorts over one tree

The Rust crate exposes two engines that share the run detector and the merge tree.

| | | |
| :--- | :--- | :--- |
| `sort` | **not stable** — the default | equal keys may be reordered |
| `sort_stable` | stable | for a DBMS, or a cascade of sorts over several columns |

Each has a `_by` (comparator) and a `_by_key` variant, and a `_by_with_buffer` variant that reuses the caller's working memory.

They differ in exactly three places.

**A descending run with repeated keys.** `9 9 7 7 3` is descending, but reversing it swaps the equal elements. The engine used to refuse: ascending runs took `<=`, descending runs a strict `>`, so every plateau became its own ascending run and a reversed table with duplicate keys was never seen as reversed. Run direction is now decided by the first *strictly different* pair, so the whole stretch is one run. `sort` reverses it. `sort_stable` reverses it and then reverses each block of equal keys back, which restores their original order — the blocks are found by binary search, because the stretch is already sorted. Both are `O(n)`.

**A stretch with no structure.** `sort_stable` hands it to the standard library's stable sort, `sort` to the unstable one.

**Chaotic input.** `sort_stable` sorts 65,536-element blocks and merges them through the tree. `sort` delegates to Rayon's parallel quicksort, which works in place and allocates nothing — but only when the sample says the data is really random, or globally ascending with cheap comparisons. Interleaved runs are a bad pivot pattern for it and with expensive keys the damage is an order of magnitude (8 runs of 250,000 strings: 2,040 ms there against 112 ms through the tree), so everything else goes through blocks and the tree.

Four things changed in the shared machinery.

- **Runs already in global order are joined in the metadata, before the tree.** The "already ordered" shortcut used to live inside the tree, where it was a `memcpy` per level. A thousand teeth in disjoint value ranges paid `log(1000)` copy passes; they now cost one comparison per boundary. The greedy scan also hands the *maximum* of a descending run to the ascending run before it, which blocks the join; that element is given back.
- **Short runs are grouped and sorted as one block.** Merging run by run only pays when runs are long. Random data that slips past the sample becomes a sequence of blocks instead of millions of two-element runs.
- **The tree only runs when it pays**: when, after joining, runs average at least 256 elements for cheap comparisons or 48 for expensive ones. Below that the block path ties or wins. Detection gives up early once the data proves too fragmented, so asking costs a fraction of a pass.
- **The sample is eight windows spread over the array**, not one in the middle, and ties do not count as a direction — `4 4 3 3 2 2` used to read as pure chaos. The same sample times its own 512 comparisons to tell a cheap key from an expensive one. The windows are read once before the clock starts, so the clock measures the comparison and not the cache misses of fetching the array. Until 2026-11-20 the cold reading was inside the measurement, and a one-instruction `u64` comparison, right after the array was copied, looked expensive in 25 to 36 of 60 calls on 10M elements on this laptop (in 19 to 42 of 60 on 100M on a 12-core EPYC). In those calls the unstable engine lost its quicksort shortcut: 10M almost-sorted `u64` took 105 to 162 ms instead of about 70 here, 100M about 735 ms instead of 295 there. What a comparison fetches outside the element — the bytes of a string — stays cold and still counts: `&str` reads as expensive in 60 of 60 calls either way, six to thirteen times over the limit. A preemption can only make a cheap key look expensive, which only withholds a shortcut; and for plain integers the key is cheap by definition, clock or no clock.

Measured on 10M `u64`, 8 threads (i7-7700HQ, 4 cores), by `examples/vs_rayon.rs`: median of 15 rounds after a discarded warm-up, a fresh copy of the data outside the timer, and the order of the contenders drawn at random in every round. Each table of this README was measured twice, in two separate processes, and each cell is the lower of the two medians, because interference only ever adds time. Each row has two bands. The first is the stable pair: Rayon's `par_sort` against this crate's `sort_stable`, shown as `MM_sort_stable`. The second is the unstable pair: Rayon's `par_sort_unstable` against this crate's `sort`, shown as `MM_sort`. The column after each pair says how many times faster the `MM_` entry is. `sort_stable` runs twice in every round and the gap between its two medians is the noise floor of the row, 0.7% to 8%; a ratio reads *tie* when the difference fits inside that floor (5% at least) or is under 1 ms. In each band the winner is in large bold blue (rendered by GitHub's math extension; other Markdown viewers may show the raw markup); where the ratio reads *tie*, the blue only marks the lower of two numbers the measurement cannot tell apart:

| Scenario | `par_sort` | `MM_sort_stable` | | `par_sort_unstable` | `MM_sort` | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| already sorted | 4.4 ms | ${\color{blue}\Large\textbf{3.8 ms}}$ | tie | 8.7 ms | ${\color{blue}\Large\textbf{3.8 ms}}$ | **2.3x** |
| reversed | 13.8 ms | ${\color{blue}\Large\textbf{11.9 ms}}$ | 1.16x | 18.8 ms | ${\color{blue}\Large\textbf{11.7 ms}}$ | **1.61x** |
| reversed, repeated keys | 123.4 ms | ${\color{blue}\Large\textbf{22.5 ms}}$ | **5.5x** | 52.9 ms | ${\color{blue}\Large\textbf{13.4 ms}}$ | **3.9x** |
| sawtooth, disjoint ranges | 111.8 ms | ${\color{blue}\Large\textbf{9.3 ms}}$ | **12x** | 50.4 ms | ${\color{blue}\Large\textbf{8.9 ms}}$ | **5.7x** |
| 2 interleaved runs | 43.8 ms | ${\color{blue}\Large\textbf{41.4 ms}}$ | 1.06x | 103.4 ms | ${\color{blue}\Large\textbf{40.7 ms}}$ | **2.5x** |
| 64 interleaved runs | 105.5 ms | ${\color{blue}\Large\textbf{87.2 ms}}$ | **1.21x** | 151.0 ms | ${\color{blue}\Large\textbf{87.4 ms}}$ | **1.73x** |
| sorted + 400 chaotic elements | ${\color{blue}\Large\textbf{36.4 ms}}$ | 39.2 ms | 0.93x | 91.3 ms | ${\color{blue}\Large\textbf{40.2 ms}}$ | **2.3x** |
| almost sorted (1% swapped) | 162.0 ms | ${\color{blue}\Large\textbf{147.9 ms}}$ | 1.10x | 62.3 ms | ${\color{blue}\Large\textbf{62.2 ms}}$ | tie |
| random, 64 distinct keys | 204.8 ms | ${\color{blue}\Large\textbf{99.7 ms}}$ | **2.1x** | ${\color{blue}\Large\textbf{67.9 ms}}$ | 68.4 ms | tie |
| random, high cardinality | 217.2 ms | ${\color{blue}\Large\textbf{180.0 ms}}$ | **1.21x** | ${\color{blue}\Large\textbf{123.7 ms}}$ | 127.2 ms | tie |
| random + 400 sorted in the middle | 216.7 ms | ${\color{blue}\Large\textbf{176.7 ms}}$ | **1.23x** | 126.3 ms | ${\color{blue}\Large\textbf{125.3 ms}}$ | tie |

On 2M `&str` keys (median of 9 rounds) the gap against the unstable quicksort is wider, because comparisons dominate:

| Scenario | `par_sort` | `MM_sort_stable` | | `par_sort_unstable` | `MM_sort` | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 interleaved runs | 90.1 ms | ${\color{blue}\Large\textbf{78.8 ms}}$ | 1.14x | 556.6 ms | ${\color{blue}\Large\textbf{78.9 ms}}$ | **7.1x** |
| 8 interleaved runs | 176.2 ms | ${\color{blue}\Large\textbf{113.8 ms}}$ | **1.55x** | 2,040 ms | ${\color{blue}\Large\textbf{111.8 ms}}$ | **18x** |
| 64 interleaved runs | 324.9 ms | ${\color{blue}\Large\textbf{183.0 ms}}$ | **1.78x** | 1,223 ms | ${\color{blue}\Large\textbf{183.8 ms}}$ | **6.7x** |
| almost sorted (1% swapped) | 389.5 ms | ${\color{blue}\Large\textbf{287.3 ms}}$ | **1.36x** | 640.2 ms | ${\color{blue}\Large\textbf{283.8 ms}}$ | **2.3x** |
| random, 64 distinct strings | 113.2 ms | ${\color{blue}\Large\textbf{67.9 ms}}$ | **1.67x** | ${\color{blue}\Large\textbf{59.7 ms}}$ | 61.8 ms | tie |
| random | ${\color{blue}\Large\textbf{296.8 ms}}$ | 308.4 ms | tie | ${\color{blue}\Large\textbf{316.6 ms}}$ | 320.0 ms | tie |

Where there is no structure, `sort` *is* Rayon's quicksort plus a 512-comparison sample, so a tie is the ceiling. Where there is, it does not sort: it reverses, joins or merges. The ceiling goes away when the caller tells the engine what the key bits are: see section 7.

**Reusing the buffer is worth 10–38% wherever the engine merges.** A fresh 80 MB buffer costs about 20 ms on this machine, most of it page faults. A caller that sorts repeatedly — a server — keeps the buffer between calls with `sort_by_with_buffer` / `sort_stable_by_with_buffer`: in the same measurement, 2 runs drop from 41 to 26 ms, 64 runs from 87 to 63 ms, random from 180 to 160 ms. Sorted, reversed and chained input never touch the buffer, so there is nothing to save there.

### 7. Integer and prefix keys: radix on the chaotic path

A comparison sort cannot beat a parallel quicksort on random data by much. A sort that knows the **bits** of the key can. Six more entries tell the engine what those bits are; they return exactly what the entries above return, and only the path taken by chaotic input changes.

| | | |
| :--- | :--- | :--- |
| `sort_radix`, `sort_stable_radix` | the element is an integer | any type implementing `RadixKey`; implemented for the primitive integers |
| `sort_by_radix_key`, `sort_stable_by_radix_key` | a `u64` key decides the whole order | the element may carry data along: a `{key, row}` pair |
| `sort_by_prefix_key`, `sort_stable_by_prefix_key` | a `u64` prefix plus a comparator that breaks ties | keys longer than 64 bits, such as text: the prefix is the first 8 bytes, big-endian |

Nothing changes for ordered input: the sample, the run detector and the merge tree come first, and a sorted array still costs one pass. The radix routine runs when the sample says "random" and, when the comparison is cheap, also where comparison used to be the choice: on short runs, on fragmented data that is not globally ascending, and when more than 32 runs are left after joining (from 262K elements). Merging run by run moves the whole array once per level of the tree; the radix finishes in two passes whatever the order. Below 262K the tree works inside the cache and wins. The minimum size for the radix depends on what it competes with: against the block path (the stable engine; the unstable one on short runs or fragmented data) 8K elements, or 1K per thread when there are more than eight threads, up to 32K; against the parallel quicksort (the unstable engine on random data) 8K on one thread and 16K per thread above that — 32K per thread for elements wider than 8 bytes — capped at 131K for elements of up to 4 bytes and 524K for wider ones.

1. **A 1,024-element sample**, one position drawn at random inside each stride, so that periodic data — runs of equal length — cannot line up with it. If one bucket would hold more than a quarter of the data, the keys are skewed and the routine gives up before scanning anything. The stable engine with an exact key keeps going in two cases where a single pass finishes the job: when one digit covers every varying bit (a few values in a small range; from 524K elements), and when one key holds about half the data or more (from 262K): that key is split off into smaller, equal and greater in one pass, the equal part is done, and the two sides re-enter the routine.
2. **Which bits vary.** A parallel AND/OR over the keys. The digit is taken from the highest bits *that vary*, so small, shifted or common-prefix keys do not all land in one bucket.
3. **Counting**, when the element is the key and at most 16 bits vary: a parallel histogram and a parallel fill, with no buffer.
4. **One MSD pass** on the top 8 varying bits, into 256 buckets, in parallel. The destination is carved beforehand into disjoint slices, one per bucket per chunk of input, so the scatter needs no `unsafe`.
5. **Each bucket inside the cache**, one task per bucket: an LSD radix when few bits remain, the standard library's sort otherwise. A bucket that is still large is cut once more.

The routine returns without having moved a single element whenever it gives up — skewed keys, a dominant bucket, no memory for the buffer — and the comparison path takes over as before. One MSD pass rather than a full LSD sort is what makes it independent of key width: a pure LSD sort sweeps the whole array twice per key byte and loses to the quicksort on `u64`.

Measured on 30M elements, 8 threads (i7-7700HQ, 4 cores), by `examples/vs_rayon_radix.rs`, in ms: the minimum of 5 rounds after a discarded warm-up, the order of the contenders drawn at random in every round, a fresh copy of the data outside the timer; measured twice, each cell the lower of the two. One contender runs twice in every round: that pair differs by 0.4% to 2.3% on the rows above 100 ms and by up to 9% on the rows of a few tens of milliseconds. The columns are the same two bands, stable and then unstable, each with a third column for the radix entry; `MM_` marks the entries of this crate, and the winner of each band is in blue.

| 30M elements | `par_sort` | `MM_sort_stable` | `MM_sort_stable_radix` | | `par_sort_unstable` | `MM_sort` | `MM_sort_radix` | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `i32`, random | 482 | 468 | ${\color{blue}\Large\textbf{155}}$ | **3.0x** | 364 | 362 | ${\color{blue}\Large\textbf{160}}$ | **2.3x** |
| `i32`, 64 distinct values | 446 | 190 | ${\color{blue}\Large\textbf{27}}$ | **7.0x** | 128 | 130 | ${\color{blue}\Large\textbf{25}}$ | **5.2x** |
| `i32`, skewed (90% of the keys in 0..1000) | 476 | 275 | ${\color{blue}\Large\textbf{273}}$ | tie | 180 | ${\color{blue}\Large\textbf{179}}$ | 181 | tie |
| `i32`, already sorted | 8.9 | 7.9 | ${\color{blue}\Large\textbf{7.6}}$ | tie | 17.9 | ${\color{blue}\Large\textbf{7.2}}$ | 7.8 | tie |
| `u64`, random | 701 | 572 | ${\color{blue}\Large\textbf{326}}$ | **1.75x** | 394 | 398 | ${\color{blue}\Large\textbf{293}}$ | **1.36x** |
| `u64`, 64 distinct values | 641 | 274 | ${\color{blue}\Large\textbf{43}}$ | **6.3x** | 187 | 181 | ${\color{blue}\Large\textbf{44}}$ | **4.2x** |
| `u64`, skewed (90% of the keys in 0..1000) | 682 | ${\color{blue}\Large\textbf{359}}$ | 361 | tie | 278 | ${\color{blue}\Large\textbf{277}}$ | 282 | tie |
| `{u64 key, u64 row}`, random, by key | 1240 | 1282 | ${\color{blue}\Large\textbf{612}}$ | **2.1x** | 639 | 642 | ${\color{blue}\Large\textbf{527}}$ | **1.22x** |
| `{u64 key, u64 row}`, 64 distinct keys, by key | 1121 | 694 | ${\color{blue}\Large\textbf{322}}$ | **2.2x** | 344 | 344 | ${\color{blue}\Large\textbf{310}}$ | 1.11x |

The ratio column of each band compares the radix entry with the comparison entry of the same engine. For the `{key, row}` rows the entries are `sort_by_key` / `sort_by_radix_key` and their stable counterparts.

What the table says:

- **The price of stability disappears on random input.** The scatter visits the input chunks in order and the LSD pass is stable by construction, so the stable engine runs the same routine. Sorting 30M `{key, row}` pairs stably drops from 1.28 s to 0.61 s.
- **The gain shrinks as the key widens.** A 32-bit key leaves 24 bits after the first pass, which a short LSD finishes inside the cache. A 64-bit key leaves 56, and the bucket falls back to a comparison sort.
- **Skewed keys tie, by design.** The sample sees the skew and hands the array to the comparison path untouched; what is left is the cost of having asked, which the table cannot tell from noise. The exception is the stable engine with an exact key, which since 2026-11-20 takes the two one-pass cases described above (two or three values; a dominant key): measured on another machine (AMD EPYC 7B13, 1 and 8 threads), 1.0x to 3.9x over the comparison path in those cases, from 300K to 10M `{key, row}` pairs. Skew spread over a wide range — the row above — still gives up. The table above was not re-measured.
- **Already-sorted input ties**, because the radix is never reached. That row is the control.

**Where the radix also pays now.** The routing changes of 2026-11-20 — short runs, fragmented data and many runs going to the radix when the comparison is cheap — act on data with structure, which the table above does not have. Measured on an AMD EPYC 7B13 with 8 threads, with `vs_rayon u64` and the radix entries added to it, the engine before and after the changes side by side in one process, lowest median of two passes: `u64` almost sorted (1% swapped), `sort_stable_radix`: 5.2 → 2.6 ms at 1M, 59.8 → 26.5 ms at 10M, 616 → 279 ms at 100M; `u64`, 64 interleaved runs: 41.9 → 27.1 ms (stable) and 42.1 → 27.3 (unstable) at 10M, 365 → 271 and 365 → 273 at 100M. These were the two rows where the engine lost to Rayon's `par_sort` on that machine (29.2 and 52.4 ms at 10M). On `&str` through the prefix entries carried in pairs, stable, almost sorted 1%: 13.6 → 8.1 ms at 1M and 200 → 125 ms at 10M; 64 runs 13.7 → 9.5 and 167 → 146. The comparison entries `sort` and `sort_stable` measured equal; the shield fix of section 6 changes their path only in the calls where the old cost verdict was wrong.

With the README's own programs at the README's sizes — the tables of this section and of section 6, whose data has no structure for the radix entries — the crate before and after the changes, each built on its own and run in alternating processes three times on the same EPYC, are equal: 143 of 147 cells within 5%, three faster by 6–7% and one slower by 6%, while Rayon's sorts, the same code in both binaries, varied by up to 24% between the two builds.

**Text goes through the prefix entries.** A string is longer than 64 bits, so the engine gets its first 8 bytes as a `u64` prefix and the string comparison to break the ties. What makes it pay is computing the prefix once and carrying it next to the reference, as a `(prefix, &str)` pair: almost every comparison is then settled without touching the memory of the string. 5M `&str`, 8 threads, in ms (`vs_rayon prefix`): median of 7 rounds, measured twice like the tables above; the time to build the pairs and to write the sorted references back is inside the measurement:

| 5M `&str` | `par_sort` | `MM_sort_stable` | `MM_sort_stable_by_prefix_key` | | `par_sort_unstable` | `MM_sort` | `MM_sort_by_prefix_key` | |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random, 32 hex characters | 1032 | 707 | ${\color{blue}\Large\textbf{252}}$ | **2.8x** | 920 | 913 | ${\color{blue}\Large\textbf{217}}$ | **4.2x** |
| random, 12 to 27 letters | 982 | 857 | ${\color{blue}\Large\textbf{225}}$ | **3.8x** | 1008 | 1004 | ${\color{blue}\Large\textbf{200}}$ | **5.0x** |
| random, 64 distinct strings | 252 | ${\color{blue}\Large\textbf{152}}$ | 263 | 0.58x | ${\color{blue}\Large\textbf{137}}$ | 139 | 180 | 0.78x |
| random, a common 10-byte prefix | 1191 | ${\color{blue}\Large\textbf{1085}}$ | 1175 | 0.92x | ${\color{blue}\Large\textbf{1305}}$ | 1314 | 1468 | 0.89x |
| already sorted | ${\color{blue}\Large\textbf{72}}$ | 74 | 100 | 0.74x | 161 | ${\color{blue}\Large\textbf{73}}$ | 99 | 0.74x |

Random text is sorted 4 to 5 times faster than by comparison, and the stable entry gains 2.8x to 3.8x. The entry **loses** wherever the prefix cannot do the work: when the strings are few and repeated, when they all start with the same 8 bytes — every comparison then reads the string anyway, on top of the cost of the pairs — and when the input is already sorted, where building the pairs and writing them back is 26 ms added to a 73 ms job. An engine used on a column of unknown content has a safer form: pass the prefix as a function of the reference, `sort_by_prefix_key(v, |s| prefix(s), |a, b| a.cmp(b))`, with no pairs. In the same measurement it never lost — 1.5x and 1.8x on the two random rows, 1.3x on 64 distinct strings, a tie on the other two.

What it costs: a buffer of `n` elements, where the unstable engine used to sort random input in place (if the allocation fails, the quicksort takes over); and for keys longer than 64 bits only the first 8 bytes take part, so a common prefix of 8 bytes or more leaves everything to the comparator.


### 8. The C++ engine: chained runs and the radix path

`MultiMergeSort.hpp` is stable-only and keeps its own structure — the single-point shield, the overlapping blocks, the metadata monoid of sections 1 and 2. It received the two changes that matter most to it.

**Runs are normalized before the tree.** An ascending scan advances while `<=`, so at an ascending-to-descending turn it swallows the first element of the descending tooth, which is that tooth's maximum:

```text
... 3 9 | 8 7 6 5 4        the 9 belongs to the tooth, not to the run before it
... 3 9 9 | 8 7 6 5 4      with a tie on top, both 9s
```

With the 9 left in the ascending run, that run ends in 9 and the tooth, once reversed, starts at 4: the two cannot be chained and every pair of teeth pays a merge. Three steps between detection and the merge tree fix it:

1. **The stolen plateau goes back to the tooth** — all of it, not one element: when the top of a descending tooth has repeated keys, returning a single one leaves the break in place.
2. **Descending runs are reversed in place, in parallel.** A descending run is strict, so reversing it is stable. The returned plateau is the exception: it enters the run with equal keys, the reversal swaps them, and a second reversal of just the plateau restores their order.
3. **Runs already in order are chained.** The "already ordered" shortcut used to live inside the tree, where it cost a copy of the stretch per level; a boundary with `a[p-1] <= a[p]` now simply disappears.

A thousand teeth in disjoint value ranges become one segment: no buffer, no merge, no copy. On 15M `{key, index}` pairs that input went from 193 ms to 27 ms, with or without a repeated key on top of a tooth, measured with the old and the new header in the same binary.

**Three entries take the radix path of section 7**, with the same routine — sample, range of the varying bits, counting for narrow bare keys, one MSD pass, each bucket finished inside the cache:

| | |
| :--- | :--- |
| `multimerge::sort_radix(span<T>)` | `T` is an integer type |
| `multimerge::sort_by_radix_key(span<T>, key)` | `key(x)` is a `uint64_t` that decides the whole order; `T` may carry data along |
| `multimerge::sort_by_prefix_key(span<T>, prefix)` | `prefix(x)` is the start of the order and `T`'s operators break the ties: keys longer than 64 bits |

The output is exactly that of `multimerge::sort`, stability included: the scatter visits the input chunks in order, the LSD pass is stable, and the comparison leaf is `std::stable_sort`. Where the Rust engine carves the destination into slices to stay free of `unsafe`, the C++ scatter writes through pointers at the offsets the histogram gives; everything runs as OpenMP tasks inside one parallel region, so a bucket that re-enters the routine shares its work with the idle threads.

The structure is the Rust one; three thresholds are not, because they depend on what the comparison alternative is, and they were measured again here:

- **The LSD leaf goes up to 56 remaining bits** (24 and 40 in Rust). `std::sort` and `std::stable_sort` lose to the LSD pass at every width: on 15M `uint64_t`, 147 ms against 250.
- **The second cut comes at 16K elements** (64K in Rust): 242 ms against 391 on 15M 16-byte rows with 64-bit keys.
- **The minimum size depends on the fallback the engine was built with**: 65K elements against the internal `chunk_parallel_sort`; 131K (elements of up to 4 bytes) and 524K (larger) against `std::execution::par`.


## 📊 Benchmarks

C++ engine, `{key, index}` pairs of 16 bytes, 15M elements, 8 threads. Median of
9 runs after a discarded warm-up pass, with the order of the engines drawn at
random in every run, and a control that measures one engine twice to expose the
noise floor. Like the tables above, it was measured twice and each cell is the
lower of the two medians; the noise column keeps the larger of the two floors.
Every output is checked element by element against `std::stable_sort`, index
included, so a wrong answer cannot be mistaken for a fast one. The C++ engine
has a single, stable sort, so in the two tables below the blue marks the fastest
time of the whole row. The engine is built the way
`cargo` builds it — k-way merge and bidirectional leaf merge on — with the
chaotic fallback pointed at the PSTL; the harness is `src/vs_stable.cpp`.

The reference is **`std::stable_sort(std::execution::par, ...)`** — the parallel
stable sort of the C++17 standard library, backed by Intel TBB. It is the
strongest baseline available. Comparing an 8-thread engine against the
single-threaded `std::stable_sort` would measure parallelism, not algorithm.

| Scenario | `stable_sort` | C++17 `par` | `sort` | `sort_by_radix_key` | `sort` vs `par` | radix vs `par` | noise |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Reversed** | 1162.3 ms | 492.7 ms | 37.3 ms | ${\color{blue}\Large\textbf{36.3 ms}}$ | **13x** | **14x** | 4.2% |
| **Already sorted** | 922.2 ms | 33.3 ms | ${\color{blue}\Large\textbf{11.0 ms}}$ | 11.1 ms | **3.0x** | **3.0x** | 0.5% |
| **Sawtooth, disjoint ranges** (odd blocks reversed) | 955.5 ms | 272.4 ms | 23.7 ms | ${\color{blue}\Large\textbf{23.1 ms}}$ | **11x** | **12x** | 1.8% |
| **Same, one repeated key on top of a tooth** | 955.3 ms | 276.7 ms | ${\color{blue}\Large\textbf{23.2 ms}}$ | 23.3 ms | **12x** | **12x** | 3.9% |
| **Almost sorted** (0.1% noise) | 986.7 ms | 562.8 ms | 256.6 ms | ${\color{blue}\Large\textbf{249.1 ms}}$ | **2.2x** | **2.3x** | 4.1% |
| **Sawtooth** (`i % 1000`) | 997.0 ms | 508.6 ms | 262.1 ms | ${\color{blue}\Large\textbf{252.6 ms}}$ | **1.94x** | **2.0x** | 2.2% |
| Random, 64-bit keys | 2112.6 ms | 553.4 ms | 541.1 ms | ${\color{blue}\Large\textbf{225.2 ms}}$ | tie | **2.5x** | 3.0% |
| Random, high cardinality | 2129.8 ms | 600.8 ms | 600.8 ms | ${\color{blue}\Large\textbf{219.8 ms}}$ | tie | **2.7x** | 0.7% |
| Random, 64 distinct keys | 1406.3 ms | 524.5 ms | 530.4 ms | ${\color{blue}\Large\textbf{130.3 ms}}$ | tie | **4.0x** | 2.1% |
| Random, 10 distinct keys | 1275.0 ms | 516.6 ms | 511.5 ms | ${\color{blue}\Large\textbf{133.9 ms}}$ | tie | **3.9x** | 1.0% |

On plain integers the element is the key, which adds the counting path. 30M
elements, same machine and same method, median of 9 runs, in ms; the harness is
`src/vs_integers.cpp`:

| 30M elements | `std::sort(par)` | `std::stable_sort(par)` | `sort` | `sort_radix` | radix vs `std::sort(par)` |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `int32_t`, random | 741 | 774 | 770 | ${\color{blue}\Large\textbf{133}}$ | **5.6x** |
| `int32_t`, 64 distinct values | 448 | 507 | 504 | ${\color{blue}\Large\textbf{40}}$ | **11x** |
| `int32_t`, skewed (90% of the keys in 0..1000) | ${\color{blue}\Large\textbf{655}}$ | 789 | 693 | 699 | 0.94x |
| `int32_t`, already sorted | 84 | 87 | ${\color{blue}\Large\textbf{9.5}}$ | 9.6 | **8.7x** |
| `uint64_t`, random | 880 | 923 | 895 | ${\color{blue}\Large\textbf{321}}$ | **2.7x** |
| `uint64_t`, 64 distinct values | 713 | 781 | 754 | ${\color{blue}\Large\textbf{65}}$ | **11x** |
| `uint64_t`, skewed (90% of the keys in 0..1000) | 837 | 850 | 841 | ${\color{blue}\Large\textbf{805}}$ | tie |
| `uint64_t`, already sorted | 69 | 86 | ${\color{blue}\Large\textbf{15.2}}$ | 15.3 | **4.5x** |

### What the numbers say

**Global order is where the engine is untouchable.** Reversed data finishes in
37.3 ms against 492.7 for the standard parallel sort, and already-sorted data in
11.0 against 33.3. Neither is a faster sort: the detection pass finds a single
run and the engine returns without sorting at all, or reverses once in parallel.
No comparison sort can match that, because it must at minimum compare.

This is not a synthetic corner. It is the shape of a rebuilt index, a table read
in insertion order, a segment already merged by a previous operation — the
common case in a database, and the one a general-purpose sort pays full price
for.

**Teeth that already fit together cost one pass.** A sorted array with every
other block reversed is ordered blocks in disjoint value ranges: after reversing
the descending ones nothing is left to merge. The engine sorts it in 23.7 ms,
11 times faster than the baseline. Before the run normalization of section 8
the same input took 193 ms, because each descending tooth had lost its maximum
to the run before it and every pair of teeth paid a merge. A repeated key on
top of a tooth, which used to break the chain for the whole array, changes
nothing: 23.2 ms.

**Partial order pays too.** Almost-sorted data is 2.2x faster than the baseline
and the classic sawtooth 1.9x. Here there is real merging to do — 15,000 runs —
and what wins is the cache-blocked k-way merge of section 4: it cuts the number
of passes over DRAM, which is what this workload is bound by.

**Chaotic input by comparison is a tie; with the key bits it is not.** On random
data `sort` hands the array to the standard parallel sort and lands within 3%
of it in all four rows, inside the noise floor. `sort_by_radix_key` does not
compare: 2.5x on 64-bit keys, 4x when the keys are few. On plain integers the
gap is wider still — 30M random `int32_t` in 133 ms against 741 for
`std::sort(par)`.

**Skewed keys and ordered input tie between the two entries, by design.** When
the sample sees that one bucket would hold more than a quarter of the data, the
radix routine gives up before moving anything; when the shield sees order, it
is never reached: sorted input reads 11.0 and 11.1 ms for the two entries in
the first table, 9.5 and 9.6 in the second. On skewed `int32_t` both entries
trail `std::sort(par)` by about 6%, because what they fall back to is the
*stable* parallel sort.

**Why the order of the contenders is drawn at random.** An earlier version of
these tables rotated the contenders cyclically, which leaves every one of them
behind the same neighbour in every round. On sorted input that made the radix
entry look 1 to 2 ms slower than `sort`, although both execute the same code.
With the order drawn in each round the difference is gone.

**The fallback choice mattered more than the algorithm.** Before this was
measured the engine delegated to `__gnu_parallel::stable_sort`, and lost 62–88%
to the C++17 `par` on random input — not because its own code was slow, but
because of where it handed off. `__gnu_parallel` uses a loser tree for its
multiway merge, and a loser tree indexes its stream heads with a runtime value,
which stops the compiler from keeping them in registers. This engine measured
that cost directly and rejected the loser tree for the same reason (see the k-way
section). Pointing the shield at the PSTL closed the entire gap.

### Caveats

**These numbers are from one machine**, an 8-thread laptop (4 cores), at one
size per table. All the tables were measured back to back in a single session
of about forty minutes, so the machine was warm throughout: a short isolated
run of the same harness reads 10 to 20% lower in absolute terms, with nearly
the same ratios. The cache budgets, the block size, the k-way threshold and the
radix thresholds were all tuned there and should be re-measured on the target
hardware. The thresholds added on 2026-11-20 — the 32 runs and 262K elements
above which the tree gives way to the radix, the 524K and 262K of the stable
skew cases, the per-thread minimum size — were measured on an AMD EPYC 7B13
(12 cores, 64 MB of L3) with 1 to 24 threads, not on this laptop.

**An earlier version of the first table understated the engine.** It was
measured with the k-way merge and the bidirectional leaf merge switched off: the
documented compile line did not pass the two flags that `build.rs` passes, and
the header left both off unless asked. Almost-sorted data read 432 ms there and
reads 257 ms here. The header now enables both by default
(`-DMULTIMERGE_NO_KWAY` and `-DMULTIMERGE_NO_BIDIR` turn them off).

**The sawtooth pattern (`i % 1000`) is the worst case for adaptive merging**, not
a typical one: every run ends at its maximum and the next begins at its minimum,
so adjacent runs overlap completely in range and nothing can be chained. Real
workloads, where sorted segments overlap only partially, land between this and
the disjoint sawtooth.

**The radix entries need a buffer of `n` elements** on chaotic input, and
`sort_by_prefix_key` was checked for correctness but not measured in C++.

**Measurement noise dominates smaller runs.** At 5M several scenarios showed 10
to 22% spread between repeated measurements of the same engine, which is wider
than most of the effects being measured. In the first table every noise floor
is at most 4.2%; the integer table repeats one contender and that pair differs by
up to 11% on the skewed rows, which are the noisiest of all. Treat any published ratio narrower than its own
noise as a tie.

### Correctness

Every release is checked by four independent layers:

- 25 unit tests in Rust, which force every route and every radix leaf through
  miniature thresholds — including, since 2026-11-20, the stable engine on
  dominant and narrow keys (checked pair by pair against the standard stable
  sort), the decisions of the radix routine on skewed keys (it takes the two
  one-pass cases and gives up on wide skew without moving anything), the
  classification of the skew sample, the "globally ascending" check on
  interleaved runs, and the radix minimum size per competitor and thread
  count; five deliberate mutations of that code are each caught — plus
  integration suites over the public API: stability under duplicate keys,
  descending runs with plateaus, block boundary sizes, a 150-case
  structured/chaotic fuzz, and the radix entries on bare integers, keys with a
  payload and prefix keys
- A C++ stability suite over the same patterns, verifying that equal keys keep
  their original relative order and that the output is byte-identical to
  `std::stable_sort` — for `sort`, for the three radix entries, and for the
  disjoint sawtooth with plateaus of 1, 2, 3 and 17 equal keys on top of each
  tooth, where it also asserts that nothing was merged. It is built twice more:
  against the PSTL fallback and with the radix thresholds in miniature. Eight
  deliberate mutations of the new code were each caught by it
- Tests through the FFI bridge that run the C++ radix and prefix entries from
  Rust against the standard library's sort
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

4. **Reproduce the Rust tables.** The programs that measured them are examples
   of the crate and use its public API; each one checks every output against a
   reference before timing anything:
```bash
cargo run --release --example vs_rayon -- u64 10000000 15      # section 6, u64
cargo run --release --example vs_rayon -- text 2000000 9       # section 6, &str
cargo run --release --example vs_rayon -- prefix 5000000 7     # section 7, text by prefix
cargo run --release --example vs_rayon_radix -- 30000000 5     # section 7, radix

```

5. **Reproduce the C++ tables** (MSYS2 UCRT64 or Linux, where the C++17 parallel
   policies engage; link `-ltbb` instead of `-ltbb12` on Linux):
```bash
cd src
g++ -std=c++20 -O3 -fopenmp -DMULTIMERGE_PSTL -DMULTIMERGE_TEST_PAR vs_stable.cpp -ltbb12 -o vs_stable
g++ -std=c++20 -O3 -fopenmp -DMULTIMERGE_PSTL vs_integers.cpp -ltbb12 -o vs_integers
./vs_stable 15000000      # {key, index} pairs
./vs_integers 30000000 9  # plain integers

```

6. **C++ engine build switches**, all read from the environment so A/B
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

The engine sorts slices of `Copy` types (database keys, timestamps, `{key, row}` pairs, `&str`). `sort` is the default and is **not stable**; `sort_stable` preserves the order of equal keys.

```rust
use adaptive_parallel_multimerge_sort::{sort, sort_by, sort_stable, sort_stable_by_key};
use adaptive_parallel_multimerge_sort::sort_stable_by_with_buffer;

fn main() {
    // Default engine: not stable.
    let mut keys: Vec<u64> = vec![9, 5, 1, 3, 7, 8, 2, 6, 4, 0, 15, 12];
    sort(&mut keys);
    sort_by(&mut keys, |a, b| b.cmp(a)); // descending

    // Stable engine: rows with equal keys keep their order.
    let mut rows: Vec<(u32, u64)> = vec![(7, 0), (3, 1), (7, 2), (3, 3)];
    sort_stable(&mut rows);
    sort_stable_by_key(&mut rows, |r| r.0);
    assert_eq!(rows, [(3, 1), (3, 3), (7, 0), (7, 2)]);

    // A server that sorts repeatedly keeps the working memory between calls.
    let mut scratch: Vec<(u32, u64)> = Vec::new();
    sort_stable_by_with_buffer(&mut rows, &mut scratch, |a, b| a.0.cmp(&b.0));
}
```

When the key is an integer, or starts with one, say so: the result is the same and random input is sorted by radix instead of by comparison (see section 7).

```rust
use adaptive_parallel_multimerge_sort::{sort_by_prefix_key, sort_radix, sort_stable_by_radix_key};

fn main() {
    // Integers: same result as `sort`, radix on random input.
    let mut ids: Vec<i32> = vec![42, -7, 0, 19, -7, 3];
    sort_radix(&mut ids);
    assert_eq!(ids, [-7, -7, 0, 3, 19, 42]);

    // Rows ordered by an integer key; the stable engine keeps equal keys in order.
    let mut rows: Vec<(u32, &str)> = vec![(7, "a"), (3, "b"), (7, "c"), (3, "d")];
    sort_stable_by_radix_key(&mut rows, |r| r.0 as u64);
    assert_eq!(rows, [(3, "b"), (3, "d"), (7, "a"), (7, "c")]);

    // Keys longer than 64 bits: the first 8 bytes as the radix prefix, the
    // full comparison to break ties.
    let mut names: Vec<[u8; 12]> = vec![*b"multimerge-b", *b"multimerge-a", *b"adaptive----"];
    sort_by_prefix_key(&mut names, |n| u64::from_be_bytes(n[..8].try_into().unwrap()), |a, b| a.cmp(b));
    assert_eq!(names, [*b"adaptive----", *b"multimerge-a", *b"multimerge-b"]);
}
```

### Using the C++ Engine (Standalone)

You can take `src/MultiMergeSort.hpp` and drop it directly into your modern C++20 project (like Firebird SQL internals). The element type needs `<`, `<=`, `>` and `>=`; the radix entries also need it to be trivially copyable.

```cpp
#include "MultiMergeSort.hpp"
#include <vector>

// A {key, row} pair ordered by its key. The engine asks the type for <, <=, > and >=.
struct Row { uint64_t key; uint64_t row; };
inline bool operator< (const Row& a, const Row& b) { return a.key <  b.key; }
inline bool operator<=(const Row& a, const Row& b) { return a.key <= b.key; }
inline bool operator> (const Row& a, const Row& b) { return a.key >  b.key; }
inline bool operator>=(const Row& a, const Row& b) { return a.key >= b.key; }

int main() {
    std::vector<uint64_t> data = {9, 5, 1, 3, 7, 8, 2, 6};

    // Pass as std::span. Stable.
    multimerge::sort(std::span<uint64_t>(data));

    // Integers: same output, with random input sorted by radix.
    multimerge::sort_radix(std::span<uint64_t>(data));

    // Rows: the lambda tells the engine where the key bits are.
    // Rows with equal keys keep their order.
    std::vector<Row> rows = {{7, 0}, {3, 1}, {7, 2}, {3, 3}};
    multimerge::sort_by_radix_key(std::span<Row>(rows), [](const Row& r) { return r.key; });
    // rows is now {3,1} {3,3} {7,0} {7,2}

    return 0;
}
```

*Compile with: `g++ main.cpp -std=c++20 -O3 -fopenmp -o main`*. The header needs
nothing else: the k-way merge and the bidirectional leaf merge are on by
default, and chaotic input falls back to the internal sort unless
`-DMULTIMERGE_PSTL` (plus `-ltbb`) points it at the standard parallel one.

## 🔌 Integrations

The engine has been fitted into two SQL engines. Each carries a copy of `src/multimerge.rs` (version 2.1.0, without its tests), and each was measured with the host engine's own tests. Both are experimental, and both have been put to that engine's maintainers.

### Apache DataFusion: an extension operator

[`multimerge-datafusion`](https://github.com/fbcouto/multimerge_apache_Datafusion) is a Rust crate with `MultimergeSortExec`, a sort operator that a physical optimizer rule puts in place of DataFusion's final merge (`SortPreservingMergeExec`), or of the whole sort. DataFusion itself does not change.

- On the sort queries of DataFusion's `sort_tpch` benchmark, replacing only the final merge is 3.98x faster than DataFusion's own plan at TPC-H scale factor 1 with the table in memory, 3.17x reading Parquet, and 3.48x at scale factor 10 (geometric means, 24 partitions).
- DataFusion's SQL test suite (sqllogictest, 502 files) ran with the rule installed: in the final-merge mode, the only differences are `EXPLAIN` output.
- A [5-minute video](https://youtu.be/6IBn_Cl8D_Q) presents it, and the measurements are offered in [apache/datafusion#21381](https://github.com/apache/datafusion/issues/21381#issuecomment-7062844145).

### ClickHouse: a prototype on a fork

The branch [`multimerge-sort`](https://github.com/fbcouto/ClickHouse/tree/multimerge-sort/multimerge) adds `MultimergeSortedTransform`, which can take the place of ClickHouse's single-threaded final merge of an `ORDER BY` without `LIMIT` (`MergingSortedTransform`), or do the whole sort. It is off by default, behind experimental settings.

- On ClickHouse's own sort performance tests (81 queries, 24 vCPUs), it is 2.15x faster as the final merge and 3.22x as the whole sort (geometric means over the queries it handles).
- ClickHouse's 856 sort-related functional tests ran with it forced on: no result changed.
- A [4½-minute video](https://youtu.be/mEXKimDtu-8) presents it, and the questions for the maintainers are in [ClickHouse/ClickHouse#133443](https://github.com/ClickHouse/ClickHouse/issues/133443).

Both videos are narrated with a synthetic voice reading the author's script.



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