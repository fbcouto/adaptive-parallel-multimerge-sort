use rayon::prelude::*;

// ==========================================
// PARALLEL METADATA GENERATOR (PHASE 1)
// ==========================================

/// Scans the array in parallel chunks to generate the topology metadata.
/// Positive values = Ascending runs. Negative values = Descending runs.
pub fn detect_global_trend<T: Ord + Sync>(arr: &[T]) -> Vec<i64> {
    let n = arr.len();
    if n == 0 { return Vec::new(); }
    if n == 1 { return vec![1]; }

    let chunk_size = 32_768; 
    
    // Fast path for small arrays: compute sequentially
    if n <= chunk_size {
        return generate_sequential_metadata(arr);
    }

    let num_blocks = (n + chunk_size - 1) / chunk_size;

    // Map-Reduce implementation for parallel metadata generation
    let final_metadata = (0..num_blocks)
        .into_par_iter()
        .map(|i| {
            let start = i * chunk_size;
            let end = std::cmp::min(start + chunk_size, n);
            // MAP: Generate metadata for this specific chunk
            let local_meta = generate_sequential_metadata(&arr[start..end]);
            (start, end, local_meta)
        })
        .reduce(
            || (0, 0, Vec::new()), // Identity value
            |left, right| {
                if left.2.is_empty() { return right; }
                if right.2.is_empty() { return left; }

                let mut merged_meta = left.2;
                let right_meta = right.2;

                let boundary_index = right.0; // The seam between Left and Right chunks

                let last_left = merged_meta.pop().unwrap();
                let first_right = right_meta[0];

                // REDUCE: Check the seam to see if the runs connect
                if arr[boundary_index - 1] <= arr[boundary_index] {
                    // Boundary is ASCENDING
                    if last_left > 0 && first_right > 0 {
                        // Stitch them together seamlessly
                        merged_meta.push(last_left + first_right);
                        merged_meta.extend_from_slice(&right_meta[1..]);
                    } else {
                        // They don't form a continuous run, just append
                        merged_meta.push(last_left);
                        merged_meta.extend_from_slice(&right_meta);
                    }
                } else {
                    // Boundary is STRICTLY DESCENDING
                    if last_left < 0 && first_right < 0 {
                        // Stitch them together seamlessly (addition preserves the negative sign)
                        merged_meta.push(last_left + first_right);
                        merged_meta.extend_from_slice(&right_meta[1..]);
                    } else {
                        merged_meta.push(last_left);
                        merged_meta.extend_from_slice(&right_meta);
                    }
                }

                (left.0, right.1, merged_meta)
            }
        );

    final_metadata.2
}

/// Helper function to generate metadata for a single chunk sequentially
fn generate_sequential_metadata<T: Ord>(arr: &[T]) -> Vec<i64> {
    let n = arr.len();
    if n == 0 { return Vec::new(); }
    if n == 1 { return vec![1]; }

    let mut metadata = Vec::with_capacity(n / 64); 
    let mut head = 0;

    while head < n - 1 {
        let mut tail = head + 1;
        
        if arr[head] <= arr[tail] {
            while tail < n && arr[tail - 1] <= arr[tail] { tail += 1; }
            metadata.push((tail - head) as i64);
        } else {
            while tail < n && arr[tail - 1] > arr[tail] { tail += 1; }
            metadata.push(-((tail - head) as i64));
        }
        head = tail;
    }
    
    if head == n - 1 { metadata.push(1); }
    metadata
}

fn evaluate_local_entropy<T: Ord>(arr: &[T]) -> bool {
    let n = arr.len();
    if n < 120 { return false; } 
    
    let mid = n / 2;
    let mut direction_changes = 0;
    let mut is_ascending = arr[mid] <= arr[mid + 1];
    
    for i in (mid + 1)..(mid + 100).min(n - 1) {
        let current_direction = arr[i] <= arr[i + 1];
        if current_direction != is_ascending {
            direction_changes += 1;
            is_ascending = current_direction;
        }
    }
    direction_changes > 15 
}

/// Phase 1: Maps the array topology in O(N).
/// Generates positive sizes for ascending runs, negative sizes for descending.
fn generate_chunk_metadata<T: Ord>(arr: &[T]) -> Vec<i64> {
    let n = arr.len();
    if n == 0 { return Vec::new(); }
    if n == 1 { return vec![1]; }

    let mut metadata = Vec::with_capacity(n / 64); 
    let mut head = 0;

    while head < n - 1 {
        let mut tail = head + 1;
        
        if arr[head] <= arr[tail] {
            while tail < n && arr[tail - 1] <= arr[tail] { tail += 1; }
            metadata.push((tail - head) as i64);
        } else {
            // Strictly descending to preserve stability during bidirectional read
            while tail < n && arr[tail - 1] > arr[tail] { tail += 1; }
            metadata.push(-((tail - head) as i64));
        }
        head = tail;
    }
    if head == n - 1 { metadata.push(1); }
    metadata
}

// ==========================================
// 2. ZERO-COST DIRECTIONAL MERGE
// ==========================================

fn directional_merge<T: Ord + Clone>(
    src: &[T], dest: &mut [T],
    start_a: usize, len_a: usize, is_desc_a: bool,
    start_b: usize, len_b: usize, is_desc_b: bool,
) {
    let (mut i, mut j, mut k) = (0, 0, 0);
    while i < len_a && j < len_b {
        let idx_a = if is_desc_a { start_a + len_a - 1 - i } else { start_a + i };
        let idx_b = if is_desc_b { start_b + len_b - 1 - j } else { start_b + j };

        if src[idx_a] <= src[idx_b] {
            dest[start_a + k] = src[idx_a].clone();
            i += 1;
        } else {
            dest[start_a + k] = src[idx_b].clone();
            j += 1;
        }
        k += 1;
    }
    while i < len_a {
        let idx_a = if is_desc_a { start_a + len_a - 1 - i } else { start_a + i };
        dest[start_a + k] = src[idx_a].clone();
        i += 1;
        k += 1;
    }
    while j < len_b {
        let idx_b = if is_desc_b { start_b + len_b - 1 - j } else { start_b + j };
        dest[start_a + k] = src[idx_b].clone();
        j += 1;
        k += 1;
    }
}

fn copy_directional<T: Clone>(src: &[T], dest: &mut [T], start: usize, len: usize, is_desc: bool) {
    if is_desc {
        for i in 0..len { dest[start + i] = src[start + len - 1 - i].clone(); }
    } else {
        dest[start..start + len].clone_from_slice(&src[start..start + len]);
    }
}

/// Phase 2: Resolves all negative chunks. Buffer becomes strictly ascending.
fn phase_2_normalizing_pass<T: Ord + Clone + Send>(arr: &[T], buffer: &mut [T], metadata: &[i64]) {
    let num_chunks = metadata.len();
    let mut current_offset = 0;
    let mut c = 0;

    while c < num_chunks {
        let len_a = metadata[c].abs() as usize;
        let is_desc_a = metadata[c] < 0;

        if c + 1 < num_chunks {
            let len_b = metadata[c + 1].abs() as usize;
            let is_desc_b = metadata[c + 1] < 0;

            directional_merge(
                arr, buffer,
                current_offset, len_a, is_desc_a,
                current_offset + len_a, len_b, is_desc_b,
            );
            current_offset += len_a + len_b;
            c += 2; 
        } else {
            copy_directional(arr, buffer, current_offset, len_a, is_desc_a);
            current_offset += len_a;
            c += 1;
        }
    }
}

// ==========================================
// 3. ADAPTIVE RECURSIVE ENGINE
// ==========================================


fn parallel_recursive_sort<T: Ord + Clone + Send>(arr: &mut [T], buffer: &mut [T], threshold: usize) {
    let n = arr.len();
    
     // 1. LEAF NODE HYBRIDIZATION
    if n < get_dynamic_threshold::<T>() {
        if evaluate_local_entropy(arr) { arr.sort_unstable(); } 
        else { arr.sort(); }
        return;
    }
    
    let mid = n / 2;
    
    // 2. THE SEAM BOUNDARY ISOLATION
    // Scopes the mutable borrows to just the parallel phase
    {
        let (left_chunk, rest) = arr.split_at_mut(mid - 1);
        let (seam, right_chunk) = rest.split_at_mut(2);
        
        let (buf_left, rest_buf) = buffer.split_at_mut(mid - 1);
        let (_, buf_right) = rest_buf.split_at_mut(2);
        
        rayon::join(
            || parallel_recursive_sort(left_chunk, buf_left, threshold),
            || parallel_recursive_sort(right_chunk, buf_right, threshold)
        );
        
        // 3. THE O(1) SEAM SWAP
        if seam[0] > seam[1] { seam.swap(0, 1); }
        
        let left_max_ok = left_chunk.is_empty() || left_chunk.last().unwrap() <= &seam[0];
        let right_min_ok = right_chunk.is_empty() || &seam[1] <= right_chunk.first().unwrap();
        
        // SUCCESS: The boundary is perfectly stitched.
        if left_max_ok && right_min_ok { return; }
    }
    
    // 4. FALLBACK RESOLUTION
    // If the seam was highly interwoven, we trigger Rust's O(N) near-sorted fast path
    // to absorb the seam elements back into the left and right halves before merging.
    arr[..mid].sort();
    arr[mid..].sort();
    stable_merge(arr, mid, buffer);
}

fn stable_merge<T: Ord + Clone>(arr: &mut [T], mid: usize, buffer: &mut [T]) {
    if arr[mid - 1] <= arr[mid] { return; }
    buffer[..arr.len()].clone_from_slice(arr);
    let (mut i, mut j, mut k) = (0, mid, 0);
    let n = arr.len();
    while i < mid && j < n {
        if buffer[i] <= buffer[j] { arr[k] = buffer[i].clone(); i += 1; }
        else { arr[k] = buffer[j].clone(); j += 1; }
        k += 1;
    }
    if i < mid { arr[k..n].clone_from_slice(&buffer[i..mid]); }
}

use std::sync::OnceLock;

static DYNAMIC_THRESHOLD: OnceLock<usize> = OnceLock::new();

fn get_dynamic_threshold<T>() -> usize {
    *DYNAMIC_THRESHOLD.get_or_init(|| {
        let l1_cache_bytes = 32_768; // Or use a crate later
        let element_size = std::mem::size_of::<T>();
        let read_elements = l1_cache_bytes / element_size;
        
        // We now know from the bench that 8192 is the sweet spot for structured data
        // We clamp it between 4096 and 8192
        read_elements.clamp(4096, 8192) 
    })
}

// ==========================================
// 4. MAIN ENTRY POINT
// ==========================================

pub fn multi_merge_sort<T: Ord + Clone + Send + Sync>(arr: &mut [T]) {
    let n = arr.len();
    
    // 1. HARDWARE FALLBACK
    if n < get_dynamic_threshold::<T>() {
        arr.sort();
        return;
    }
    
    // 2. THE RANDOM SCENARIO SAVIOR
    // If the data is chaotic, bail out immediately to the unstable sort.
    if evaluate_local_entropy(arr) {
        arr.par_sort_unstable();
        return;
    }

    // 3. TOPOLOGY MAPPING (Only runs if data is NOT pure chaos)
    let metadata = detect_global_trend(arr);

    if metadata.len() == 1 {
        if metadata[0] > 0 { return; } 
        else { arr.reverse(); return; }
    }
    
    // 4. ZERO-COST SAWTOOTH NORMALIZATION
    if metadata.iter().any(|&m| m < 0) {
        let mut buffer = vec![arr[0].clone(); n];
        phase_2_normalizing_pass(arr, &mut buffer, &metadata);
        arr.clone_from_slice(&buffer); 
    }

    // 5. PARALLEL RECURSION
    let mut buffer = vec![arr[0].clone(); n];
    let num_threads = rayon::current_num_threads();
    let threshold = (n / num_threads).max(get_dynamic_threshold::<T>()); 
    
    parallel_recursive_sort(arr, &mut buffer, threshold);
}
