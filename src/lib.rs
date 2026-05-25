pub mod multimerge;

// 2. Makes the function publicly visible under the alias "sort"
pub use multimerge::multi_merge_sort as sort;