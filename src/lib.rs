pub mod multimerge;

use std::os::raw::{c_char, c_int};

// `sort` e o motor padrao: NAO estavel. `sort_stable` preserva a ordem
// relativa de chaves iguais, para quem precisa disso (SGBD, ordenacao em
// cascata). Cada um tem as variantes `_by` (comparador), `_by_key` (chave) e
// `_by_with_buffer` (comparador + memoria de trabalho do chamador, para quem
// ordena repetidamente e nao quer pagar o buffer a cada chamada).
pub use multimerge::{
    multi_merge_sort as sort, multi_merge_sort_by as sort_by,
    multi_merge_sort_by_key as sort_by_key,
    multi_merge_sort_by_with_buffer as sort_by_with_buffer,
    multi_merge_sort_stable as sort_stable, multi_merge_sort_stable_by as sort_stable_by,
    multi_merge_sort_stable_by_key as sort_stable_by_key,
    multi_merge_sort_stable_by_with_buffer as sort_stable_by_with_buffer,
};

// As entradas que dizem ao motor quais sao os BITS da chave. O resultado e o
// mesmo das de cima; o que muda e o caminho do dado aleatorio, que passa a ser
// ordenado por radix em vez de comparacao, nos dois motores.
//   `_radix`          o elemento e um inteiro (`RadixKey`).
//   `_by_radix_key`   uma chave u64 decide a ordem toda; o elemento pode levar
//                     dado junto.
//   `_by_prefix_key`  um prefixo u64 mais um comparador que desempata: e a
//                     porta de chave maior que 64 bits, como texto.
pub use multimerge::{
    multi_merge_sort_by_prefix_key as sort_by_prefix_key,
    multi_merge_sort_by_radix_key as sort_by_radix_key, multi_merge_sort_radix as sort_radix,
    multi_merge_sort_stable_by_prefix_key as sort_stable_by_prefix_key,
    multi_merge_sort_stable_by_radix_key as sort_stable_by_radix_key,
    multi_merge_sort_stable_radix as sort_stable_radix, RadixKey,
};

extern "C" {
    fn cxx_multimerge_sort_u64(ptr: *mut u64, len: usize);
    fn cxx_multimerge_sort_str(ptr: *mut *const c_char, len: usize);
    fn cxx_multimerge_sort_radix_u64(ptr: *mut u64, len: usize);
    fn cxx_multimerge_sort_radix_str(ptr: *mut *const c_char, len: usize);
    fn cxx_seq_sort_u64(ptr: *mut u64, len: usize);
    fn cxx_pstl_sort_u64(ptr: *mut u64, len: usize);
    fn cxx_pstl_sort_str(ptr: *mut *const c_char, len: usize);
    fn cxx_has_pstl() -> c_int;
    fn cxx_tbb_limit(threads: c_int) -> c_int;
}

/// C++ (OpenMP) MultiMerge engine over `u64`.
pub fn cpp_sort_u64(arr: &mut [u64]) {
    unsafe { cxx_multimerge_sort_u64(arr.as_mut_ptr(), arr.len()) }
}

/// C++ (OpenMP) MultiMerge engine over NUL-terminated C strings.
///
/// # Safety
/// Every pointer must be non-null and point to a NUL-terminated string that
/// stays alive and unmodified for the duration of the call.
pub unsafe fn cpp_sort_str(arr: &mut [*const c_char]) {
    cxx_multimerge_sort_str(arr.as_mut_ptr(), arr.len())
}

/// C++ engine over `u64` through `multimerge::sort_radix`: same output as
/// [`cpp_sort_u64`], with chaotic input sorted by radix instead of comparison.
pub fn cpp_sort_radix_u64(arr: &mut [u64]) {
    unsafe { cxx_multimerge_sort_radix_u64(arr.as_mut_ptr(), arr.len()) }
}

/// C++ engine over NUL-terminated C strings through
/// `multimerge::sort_by_prefix_key`: the first 8 bytes are the radix prefix and
/// `strcmp` breaks the ties. Same output as [`cpp_sort_str`].
///
/// # Safety
/// Same contract as [`cpp_sort_str`].
pub unsafe fn cpp_sort_radix_str(arr: &mut [*const c_char]) {
    cxx_multimerge_sort_radix_str(arr.as_mut_ptr(), arr.len())
}

/// Plain sequential `std::stable_sort` over `u64` — the denominator of the
/// "did `par` actually engage?" check.
pub fn seq_sort_u64(arr: &mut [u64]) {
    unsafe { cxx_seq_sort_u64(arr.as_mut_ptr(), arr.len()) }
}

/// `std::stable_sort(std::execution::par, ...)` over `u64`.
///
/// Runs SERIAL unless the C++ side was built with `MULTIMERGE_PSTL=1` against a
/// libstdc++ with PSTL support and TBB linked. Check [`has_pstl`] first.
pub fn pstl_sort_u64(arr: &mut [u64]) {
    unsafe { cxx_pstl_sort_u64(arr.as_mut_ptr(), arr.len()) }
}

/// `std::stable_sort(std::execution::par, ...)` over NUL-terminated C strings.
///
/// # Safety
/// Same contract as [`cpp_sort_str`].
pub unsafe fn pstl_sort_str(arr: &mut [*const c_char]) {
    cxx_pstl_sort_str(arr.as_mut_ptr(), arr.len())
}

/// True when the C++ side was compiled with the parallel execution policies
/// available. Necessary, not sufficient: the standard only says `par` *may*
/// parallelize, so the benchmark still checks the clock.
pub fn has_pstl() -> bool {
    unsafe { cxx_has_pstl() != 0 }
}

/// Fixa o teto de paralelismo da TBB pelo resto do processo.
///
/// A TBB nao le `OMP_NUM_THREADS` nem `RAYON_NUM_THREADS`: abre
/// `hardware_concurrency()` sozinha. Num bench que ja mantem uma pool do Rayon
/// e uma do OpenMP, isso triplica as threads sobre o mesmo numero de nucleos.
///
/// `threads <= 0` deixa a TBB decidir. Devolve `false` quando esta build nao
/// tem o cabecalho da TBB — o chamador deve avisar, nao assumir.
pub fn tbb_limit(threads: i32) -> bool {
    unsafe { cxx_tbb_limit(threads as c_int) != 0 }
}
