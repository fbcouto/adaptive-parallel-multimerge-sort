#include <span>
#include <cstdint>
#include "MultiMergeSort.hpp"

extern "C" {
    // Exportamos a função especificamente para arrays de u64 (uint64_t)
    void cxx_multimerge_sort_u64(uint64_t* ptr, size_t len) {
        // Transforma o ponteiro bruto do Rust em um std::span e aciona o motor!
        multimerge::sort(std::span<uint64_t>(ptr, len));
    }
}