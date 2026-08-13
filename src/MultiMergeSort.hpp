#ifndef MULTIMERGESORT_HPP
#define MULTIMERGESORT_HPP

#include <vector>
#include <algorithm>
#include <span>
#include <cmath>
#include <numeric>
#include <cstdint>
#include <omp.h> 
#include <memory>

#ifdef MULTIMERGE_USE_GNU_PARALLEL
#include <parallel/algorithm>
#endif

namespace multimerge {

    // ==========================================
    // PHASE 0: LOCAL ENTROPY SHIELD O(1)
    // ==========================================
    template <typename T>
    bool evaluate_local_entropy(std::span<T> arr) {
        size_t n = arr.size();
        if (n < 120) return false;
        
        size_t mid = n / 2;
        int direction_changes = 0;
        bool is_ascending = arr[mid] <= arr[mid + 1];
        
        size_t limit = std::min(mid + 100, n - 1);
        for (size_t i = mid + 1; i < limit; ++i) {
            bool current_direction = arr[i] <= arr[i + 1];
            if (current_direction != is_ascending) {
                direction_changes++;
                is_ascending = current_direction;
            }
        }
        return direction_changes > 15;
    }

    // ==========================================
    // PHASE 1: FRACTAL DETECTION & RUN IDENTIFICATION
    // ==========================================
    template <typename T>
    std::vector<int64_t> generate_sequential_metadata(std::span<T> arr) {
        size_t n = arr.size();
        if (n == 0) return {};
        if (n == 1) return {1};

        std::vector<int64_t> metadata;
        metadata.reserve(n / 64 + 1);
        size_t head = 0;

        while (head < n - 1) {
            size_t tail = head + 1;
            if (arr[head] <= arr[tail]) {
                while (tail < n && arr[tail - 1] <= arr[tail]) [[likely]] tail++;
                metadata.push_back(static_cast<int64_t>(tail - head));
            } else {
                while (tail < n && arr[tail - 1] > arr[tail]) [[likely]] tail++;
                metadata.push_back(-static_cast<int64_t>(tail - head));
            }
            head = tail;
        }
        
        if (head == n - 1) metadata.push_back(1);
        return metadata;
    }

    inline std::vector<int64_t> merge_metadata_pure(std::vector<int64_t> left, const std::vector<int64_t>& right) {
        if (left.empty()) return right;
        if (right.empty()) return left;

        int64_t last_left = left.back();
        left.pop_back();
        int64_t first_right = right[0];
        auto uabs = [](int64_t x) -> uint64_t { return x < 0 ? (uint64_t)(-x) : (uint64_t)x; };

        if (uabs(last_left) == 1) {
            left.insert(left.end(), right.begin(), right.end());
            return left;
        }
        if (uabs(first_right) == 1) {
            left.push_back(last_left);
            left.insert(left.end(), right.begin() + 1, right.end());
            return left;
        }

        bool left_asc = last_left > 0;
        bool right_asc = first_right > 0;
        if (left_asc == right_asc) {
            int64_t sign = left_asc ? 1 : -1;
            int64_t combined_mag = (int64_t)(uabs(last_left) + uabs(first_right) - 1);
            left.push_back(combined_mag * sign);
            left.insert(left.end(), right.begin() + 1, right.end());
        } else {
            left.push_back(last_left);
            int64_t right_mag = (int64_t)(uabs(first_right) - 1);
            int64_t sign = right_asc ? 1 : -1;
            left.push_back(right_mag * sign);
            left.insert(left.end(), right.begin() + 1, right.end());
        }
        return left;
    }

    template <typename T>
    std::vector<int64_t> process_macro_block(std::span<T> arr) {
        size_t n = arr.size();
        
        // Tuned to 4096 (approx. 32KB for uint64_t) representing standard L1D Cache size.
        const size_t micro_slice_len = 4096;
        
        if (n <= micro_slice_len) {
            if (evaluate_local_entropy(arr)) {
                std::stable_sort(arr.begin(), arr.end());
                return { static_cast<int64_t>(n) };
            }
            return generate_sequential_metadata(arr);
        }

        // Step is `len - 1` to ensure 1 element overlap for metadata boundary detection.
        size_t micro_step = micro_slice_len - 1;
        long long num_micro_blocks = (n + micro_step - 1) / micro_step;
        std::vector<std::vector<int64_t>> local_meta(num_micro_blocks);

        for (long long i = 0; i < num_micro_blocks; ++i) {
            size_t start = i * micro_step;
            size_t end = std::min(start + micro_slice_len, n);
            auto span = arr.subspan(start, end - start);
            
            if (evaluate_local_entropy(span)) {
                std::stable_sort(span.begin(), span.end());
                local_meta[i] = { static_cast<int64_t>(span.size()) };
            } else {
                local_meta[i] = generate_sequential_metadata(span);
            }
        }

        std::vector<int64_t> combined;
        for (auto& meta : local_meta) {
            combined = merge_metadata_pure(std::move(combined), meta);
        }
        return combined;
    }

    template <typename T>
    std::vector<int64_t> detect_global_trend(std::span<T> arr) {
        size_t n = arr.size();
        if (n <= 1) return n == 1 ? std::vector<int64_t>{1} : std::vector<int64_t>{};

        const size_t macro_slice_len = 32768; // L1 Macro Block
        if (n <= macro_slice_len) return process_macro_block(arr);

        size_t macro_step = macro_slice_len - 1;
        long long num_macro_blocks = (n + macro_step - 1) / macro_step;
        std::vector<std::vector<int64_t>> local_meta(num_macro_blocks);

        #pragma omp parallel for schedule(dynamic)
        for (long long i = 0; i < num_macro_blocks; ++i) {
            size_t start = i * macro_step;
            size_t end = std::min(start + macro_slice_len, n);
            local_meta[i] = process_macro_block(arr.subspan(start, end - start));
        }

        std::vector<int64_t> combined;
        for (auto& meta : local_meta) {
            combined = merge_metadata_pure(std::move(combined), meta);
        }
        return combined;
    }

    // ==========================================
    // HYBRID BIDIRECTIONAL MERGE
    // ==========================================
    template <typename T>
    std::pair<size_t, size_t> co_rank(size_t k, std::span<T> a, std::span<T> b) {
        size_t m = a.size();
        size_t n = b.size();
        size_t i_lo = (k > n) ? k - n : 0;
        size_t i_hi = std::min(k, m);
        while (i_lo < i_hi) {
            size_t i = i_lo + (i_hi - i_lo + 1) / 2;
            size_t j = k - i;
            if (j == n || a[i - 1] <= b[j]) i_lo = i;
            else i_hi = i - 1;
        }
        return {i_lo, k - i_lo};
    }

    template <typename T>
    void merge_seq(std::span<T> a, std::span<T> b, std::span<T> dest) {
        size_t i = 0, j = 0, k = 0;
        while (i < a.size() && j < b.size()) {
            if (a[i] <= b[j]) dest[k++] = std::move(a[i++]);
            else dest[k++] = std::move(b[j++]);
        }
        if (i < a.size()) std::move(a.begin() + i, a.end(), dest.begin() + k);
        if (j < b.size()) std::move(b.begin() + j, b.end(), dest.begin() + k);
    }

    template <typename T>
    void merge_front(std::span<T> a, std::span<T> b, std::span<T> dest) {
        size_t pa = 0, pb = 0, count = dest.size();
        for (size_t k = 0; k < count; ++k) {
            bool take_a = pb >= b.size() || (pa < a.size() && a[pa] <= b[pb]);
            if (take_a) dest[k] = std::move(a[pa++]);
            else dest[k] = std::move(b[pb++]);
        }
    }

    template <typename T>
    void merge_back(std::span<T> a, std::span<T> b, std::span<T> dest) {
        size_t qa = a.size(), qb = b.size(), count = dest.size();
        for (size_t k = 0; k < count; ++k) {
            bool take_b = qa == 0 || (qb > 0 && b[qb - 1] >= a[qa - 1]);
            if (take_b) {
                qb--;
                dest[count - 1 - k] = std::move(b[qb]);
            } else {
                qa--;
                dest[count - 1 - k] = std::move(a[qa]);
            }
        }
    }

    template <typename T>
    void bidirectional_merge(std::span<T> a, std::span<T> b, std::span<T> dest, size_t leaf_size) {
        if (a.empty()) { std::move(b.begin(), b.end(), dest.begin()); return; }
        if (b.empty()) { std::move(a.begin(), a.end(), dest.begin()); return; }
        
        size_t total = a.size() + b.size();
        if (total <= leaf_size) { merge_seq(a, b, dest); return; }

        size_t k = total / 2;
        auto [split_a, split_b] = co_rank(k, a, b);

        auto a_front = a.subspan(0, split_a);
        auto a_back  = a.subspan(split_a);
        auto b_front = b.subspan(0, split_b);
        auto b_back  = b.subspan(split_b);
        auto dest_front = dest.subspan(0, k);
        auto dest_back  = dest.subspan(k);

        // Prevents task explosion by enforcing a size threshold
        #pragma omp task if(k > 32768)
        merge_front(a_front, b_front, dest_front);
        
        merge_back(a_back, b_back, dest_back);
        
        #pragma omp taskwait
    }

    const size_t CORANK_SPLIT_FACTOR = 16;

    template <typename T>
    void parallel_merge(std::span<T> a, std::span<T> b, std::span<T> dest, size_t leaf_size) {
        size_t total = a.size() + b.size();
        
        if (total > leaf_size * CORANK_SPLIT_FACTOR) {
            size_t k = total / 2;
            auto [i, j] = co_rank(k, a, b);
            
            auto a_left = a.subspan(0, i);
            auto a_right = a.subspan(i);
            auto b_left = b.subspan(0, j);
            auto b_right = b.subspan(j);
            auto dest_left = dest.subspan(0, k);
            auto dest_right = dest.subspan(k);
            
            #pragma omp task if(k > 32768)
            parallel_merge(a_left, b_left, dest_left, leaf_size);
            
            parallel_merge(a_right, b_right, dest_right, leaf_size);
            
            #pragma omp taskwait
            return;
        }
        bidirectional_merge(a, b, dest, leaf_size);
    }

    template <typename T>
    size_t get_leaf_size() {
        const size_t L1_CACHE_BYTES = 32768;
        size_t element_size = std::max<size_t>(sizeof(T), 1);
        size_t val = L1_CACHE_BYTES / element_size;
        return std::clamp<size_t>(val, (size_t)4096, (size_t)8192);
    }

    // ==========================================
    // STRUCTURED PATH: BOTTOM-UP MERGE
    // ==========================================
    inline std::vector<size_t> block_offsets(const std::vector<int64_t>& metadata) {
        std::vector<size_t> offsets;
        offsets.reserve(metadata.size() + 1);
        size_t off = 0;
        offsets.push_back(0);
        for (int64_t m : metadata) {
            off += std::abs(m);
            offsets.push_back(off);
        }
        return offsets;
    }

    template <typename T>
    void bottom_up_merge(std::span<T> v, std::span<T> buf, std::span<const int64_t> metadata, std::span<const size_t> offsets, size_t leaf_size, bool into_buf) {
        size_t num_blocks = metadata.size();

        if (num_blocks == 1) {
            bool is_desc = metadata[0] < 0;
            if (into_buf) {
                if (is_desc) std::reverse(v.begin(), v.end());
                std::move(v.begin(), v.end(), buf.begin());
            } else if (is_desc) {
                std::reverse(v.begin(), v.end());
            }
            return;
        }

        size_t base = offsets[0];
        size_t split_idx = num_blocks / 2;
        size_t mid = offsets[split_idx] - base;
        
        auto left_meta = metadata.subspan(0, split_idx);
        auto right_meta = metadata.subspan(split_idx);
        auto left_offsets = offsets.subspan(0, split_idx + 1);
        auto right_offsets = offsets.subspan(split_idx); 

        auto v_l = v.subspan(0, mid);
        auto v_r = v.subspan(mid);
        auto buf_l = buf.subspan(0, mid);
        auto buf_r = buf.subspan(mid);

        #pragma omp task
        bottom_up_merge<T>(v_l, buf_l, left_meta, left_offsets, leaf_size, !into_buf);
        
        bottom_up_merge<T>(v_r, buf_r, right_meta, right_offsets, leaf_size, !into_buf);
        
        #pragma omp taskwait

        if (into_buf) {
            if (v[mid - 1] <= v[mid]) std::move(v.begin(), v.end(), buf.begin());
            else parallel_merge<T>(v_l, v_r, buf, leaf_size);
        } else {
            if (buf[mid - 1] <= buf[mid]) std::move(buf.begin(), buf.end(), v.begin());
            else parallel_merge<T>(buf_l, buf_r, v, leaf_size);
        }
    }

    template <typename T>
    void parallel_reverse(std::span<T> arr) {
        size_t n = arr.size();
        
        if (n <= 100000) {
            std::reverse(arr.begin(), arr.end());
            return;
        }

        #pragma omp parallel for simd schedule(static)
        for (long long i = 0; i < (long long)(n / 2); ++i) {
            std::swap(arr[i], arr[n - 1 - i]);
        }
    }

    template <typename T>
    void chunk_parallel_sort(std::span<T> arr, size_t leaf_size) {
        size_t n = arr.size();
        const size_t CHUNK_LENGTH = 2000; 

        size_t num_chunks = (n + CHUNK_LENGTH - 1) / CHUNK_LENGTH;

        #pragma omp parallel for schedule(dynamic)
        for (long long i = 0; i < (long long)num_chunks; ++i) {
            size_t start = (size_t)i * CHUNK_LENGTH;
            size_t end = std::min(start + CHUNK_LENGTH, n);
            std::stable_sort(arr.begin() + start, arr.begin() + end);
        }

        std::vector<int64_t> metadata(num_chunks);
        for (size_t i = 0; i < num_chunks; ++i) {
            size_t start = i * CHUNK_LENGTH;
            size_t end = std::min(start + CHUNK_LENGTH, n);
            metadata[i] = (int64_t)(end - start); 
        }
        std::vector<size_t> offsets = block_offsets(metadata);
        
        auto buffer_ptr = std::make_unique_for_overwrite<T[]>(n);
        std::span<T> buffer(buffer_ptr.get(), n);

        #pragma omp parallel
        {
            #pragma omp single
            {
                bottom_up_merge<T>(arr, buffer, metadata, offsets, leaf_size, false);
            }
        }
    }

    // ==========================================
    // MAIN ENTRY POINT
    // ==========================================
    template <typename T>
    void sort(std::span<T> arr) {
        size_t n = arr.size();
        size_t leaf_size = get_leaf_size<T>();

        if (n <= leaf_size) {
            std::stable_sort(arr.begin(), arr.end()); 
            return;
        }

        // Failsafe for complete chaos (random sequences)
        if (evaluate_local_entropy(arr)) {
#if defined(MULTIMERGE_USE_GNU_PARALLEL)
            __gnu_parallel::stable_sort(arr.begin(), arr.end());
#else
            chunk_parallel_sort(arr, leaf_size);
#endif
            return;
        }

        std::vector<int64_t> metadata = detect_global_trend(arr);

        if (metadata.size() == 1) {
            if (metadata[0] > 0) return;
            parallel_reverse(arr);
            return;
        }

        std::vector<size_t> offsets = block_offsets(metadata);

        // Zero-initialization elision: Memory allocated without being cleared
        auto buffer_ptr = std::make_unique_for_overwrite<T[]>(n);
        std::span<T> buffer(buffer_ptr.get(), n);
        
        // Bootstraps the OpenMP Thread Pool exactly once
        #pragma omp parallel
        {
            #pragma omp single
            {
                bottom_up_merge<T>(arr, buffer, metadata, offsets, leaf_size, false);
            }
        }
    }
}

#endif // MULTIMERGESORT_HPP