pub mod multimerge;

// Re-exporting the main engine function with the 'sort' alias for cleaner API usage.
// This allows users to simply call: multimerge::sort(...)
pub use multimerge::multi_merge_sort_generic as sort;
