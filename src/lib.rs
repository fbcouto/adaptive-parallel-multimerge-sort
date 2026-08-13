pub mod multimerge;

// Exposes the highly optimized Copy-based sort as the default routine
pub use multimerge::multi_merge_sort as sort;

// Exposes the C++ engine via FFI for benchmarking or external usage
extern "C" {
    fn cxx_multimerge_sort_u64(ptr: *mut u64, len: usize);
}

/// Safe Rust wrapper to invoke the C++ (OpenMP) engine.
pub fn cpp_sort_u64(arr: &mut [u64]) {
    unsafe {
        cxx_multimerge_sort_u64(arr.as_mut_ptr(), arr.len());
    }
}