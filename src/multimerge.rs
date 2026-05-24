use rayon::prelude::*;

// --- Generic Utilities ---

fn calculate_minrun(mut n: usize) -> usize {
    let mut r = 0;
    while n >= 64 { 
        r |= n & 1; 
        n >>= 1; 
    }
    n + r
}

pub fn insertion_sort<T: Ord>(arr: &mut [T]) {
    for i in 1..arr.len() {
        let mut j = i;
        while j > 0 && arr[j - 1] > arr[j] {
            arr.swap(j - 1, j);
            j -= 1;
        }
    }
}

fn detect_global_trend<T: Ord + Sync>(arr: &mut [T]) -> bool {
    let n = arr.len();
    if n < 100_000 { return false; } 

    let chunk_size = 32768; 
    let num_blocks = (n + chunk_size - 1) / chunk_size;
    let immutable_arr: &[T] = arr;

    let (ascending, descending) = (0..num_blocks)
        .into_par_iter()
        .map(|i| {
            let start = i * chunk_size;
            let end = std::cmp::min(start + chunk_size + 1, n);
            let overlapping_chunk = &immutable_arr[start..end];
            
            let mut asc = 0;
            let mut desc = 0;
            for j in 1..overlapping_chunk.len() {
                if overlapping_chunk[j - 1] < overlapping_chunk[j] { asc += 1; }
                else if overlapping_chunk[j - 1] > overlapping_chunk[j] { desc += 1; }
            }
            (asc, desc)
        })
        .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));

    if descending == 0 { return true; }
    if ascending == 0 { arr.reverse(); return true; }
    if descending > 0 && descending < (n / 20) { return false; }
    
    false
}

// --- Adaptive Main Engine ---

pub fn multi_merge_sort_generic<T: Ord + Sync + Send + Clone>(arr: &mut [T]) {
    let n = arr.len();
    
    // Safety check: Prevent panic if the array is empty
    if n == 0 { return; } 

    if n < 1024 {
        insertion_sort(arr);
        return;
    }
    
    if detect_global_trend(arr) { return; }

    let mut is_pure_chaos = false;
    if n > 120 {
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
        if direction_changes > 15 { is_pure_chaos = true; }
    }

    if !is_pure_chaos {
        let mut buffer = vec![arr[0].clone(); n];
        let num_threads = rayon::current_num_threads();
        let threshold = (n / num_threads).max(1_000_000); 
        parallel_recursive_sort(arr, &mut buffer, threshold);
    } else {
        arr.par_sort_unstable();
    }
}

// --- Processing Core (Path A) ---

fn sequential_timsort_style<T: Ord + Clone>(arr: &mut [T], buffer: &mut [T]) {
    let n = arr.len();
    let minrun = calculate_minrun(n);

    for i in (0..n).step_by(minrun) {
        let end = (i + minrun).min(n);
        insertion_sort(&mut arr[i..end]);
    }

    let mut block_size = minrun;
    while block_size < n {
        for left in (0..n).step_by(block_size * 2) {
            let mid = (left + block_size).min(n);
            let right = (left + block_size * 2).min(n);
            if mid < right {
                stable_merge(&mut arr[left..right], buffer, mid - left);
            }
        }
        block_size *= 2;
    }
}

fn parallel_recursive_sort<T: Ord + Clone + Send>(arr: &mut [T], buffer: &mut [T], threshold: usize) {
    let n = arr.len();
    if n <= threshold {
        sequential_timsort_style(arr, buffer);
        return;
    }

    let mid = n / 2;
    let (arr_left, arr_right) = arr.split_at_mut(mid);
    let (buf_left, buf_right) = buffer.split_at_mut(mid);

    rayon::join(
        || parallel_recursive_sort(arr_left, buf_left, threshold),
        || parallel_recursive_sort(arr_right, buf_right, threshold),
    );

    stable_merge(arr, buffer, mid);
}

fn stable_merge<T: Ord + Clone>(arr: &mut [T], buffer: &mut [T], mid: usize) {
    let n = arr.len();
    buffer[..n].clone_from_slice(&arr[..n]);

    let mut i = 0;
    let mut j = mid;
    let mut k = 0;

    while i < mid && j < n {
        if buffer[i] <= buffer[j] { 
            arr[k] = buffer[i].clone(); 
            i += 1; 
        } else { 
            arr[k] = buffer[j].clone(); 
            j += 1; 
        }
        k += 1;
    }

    if i < mid { 
        arr[k..k + (mid - i)].clone_from_slice(&buffer[i..mid]); 
    } else if j < n { 
        arr[k..k + (n - j)].clone_from_slice(&buffer[j..n]); 
    }
}
