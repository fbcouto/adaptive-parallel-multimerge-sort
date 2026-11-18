// ============================================================
// PONTE FFI — Rust <-> motor C++
//
// Expoe DOIS motores, para que o criterion possa medir os quatro contendores
// no mesmo processo, na mesma maquina, com o mesmo dado:
//
//   cxx_multimerge_sort_*   o motor deste projeto
//   cxx_pstl_sort_*         std::stable_sort(std::execution::par, ...)
//
// O segundo e a referencia mais forte disponivel em C++. Ele so paraleliza de
// verdade quando a libstdc++ foi construida com suporte a PSTL E a Intel TBB
// esta lincada -- no MSYS2 isso e o ambiente UCRT64; no MINGW64 os overloads
// compilam e rodam SERIAL, em silencio (medido: 0,95x contra o sequencial no
// MINGW64, 3,62x no UCRT64).
//
// Por isso `cxx_has_pstl()` existe: o bench pergunta antes de medir, e pula o
// contendor em vez de publicar um numero que nao mede o que diz medir.
// ============================================================
#include <span>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <compare>
#include <algorithm>
#include "MultiMergeSort.hpp"

#if defined(MULTIMERGE_PSTL)
#include <execution>
#endif


// Limite de paralelismo da TBB.  Ela NAO le OMP_NUM_THREADS: abre
// hardware_concurrency() por conta propria.  No bench isso soma as pools do
// Rayon (8) e do OpenMP (8) as threads dela -- 24 disputando 8 nucleos.
#if __has_include(<tbb/global_control.h>)
#include <tbb/global_control.h>
#include <memory>
#define FFI_TEM_TBB_CONTROL 1
#endif

namespace {

// Chave de texto vista do lado C++: ponteiro para C-string terminada em NUL.
//
// O lado Rust ordena `&str` (ponteiro gordo de 16 bytes, comparacao por memcmp
// com comprimento conhecido); o lado C++ ordena `const char*` (8 bytes, strcmp).
// Sao as representacoes idiomaticas de cada lado, e a TAREFA e a mesma -- mas o
// dado movido nao e, e isso precisa ser lido junto com o resultado.
struct CStr {
    const char* p;

    friend std::strong_ordering operator<=>(const CStr& a, const CStr& b) noexcept {
        const int c = std::strcmp(a.p, b.p);
        return c < 0 ? std::strong_ordering::less
             : c > 0 ? std::strong_ordering::greater
                     : std::strong_ordering::equal;
    }
    friend bool operator==(const CStr& a, const CStr& b) noexcept {
        return std::strcmp(a.p, b.p) == 0;
    }
};

static_assert(sizeof(CStr) == sizeof(const char*));
static_assert(alignof(CStr) == alignof(const char*));
static_assert(std::is_trivially_copyable_v<CStr>);

}  // namespace

extern "C" {

void cxx_multimerge_sort_u64(uint64_t* ptr, size_t len) {
    multimerge::sort(std::span<uint64_t>(ptr, len));
}

void cxx_multimerge_sort_str(const char** ptr, size_t len) {
    multimerge::sort(std::span<CStr>(reinterpret_cast<CStr*>(ptr), len));
}

// O mesmo motor, com a chave radix: mesma saida, e o dado caotico deixa de
// passar por comparacao.
void cxx_multimerge_sort_radix_u64(uint64_t* ptr, size_t len) {
    multimerge::sort_radix(std::span<uint64_t>(ptr, len));
}

// Texto e chave maior que 64 bits: entra o prefixo. Sao os 8 primeiros bytes
// da C-string em big-endian, com zeros depois do NUL -- o que preserva a ordem
// do strcmp (uma string que e prefixo de outra vem antes dela). O desempate
// entre prefixos iguais continua sendo o strcmp do `operator<=>`.
void cxx_multimerge_sort_radix_str(const char** ptr, size_t len) {
    multimerge::sort_by_prefix_key(
        std::span<CStr>(reinterpret_cast<CStr*>(ptr), len),
        [](const CStr& s) noexcept {
            uint64_t p = 0;
            int i = 0;
            for (; i < 8; ++i) {
                const unsigned char c = static_cast<unsigned char>(s.p[i]);
                if (c == 0) break;
                p = (p << 8) | c;
            }
            for (; i < 8; ++i) p <<= 8;
            return p;
        });
}

// 1 quando os overloads paralelos existem E a TBB esta lincada. Nao e prova de
// que engatou -- o padrao so diz que `par` PODE paralelizar -- mas e a condicao
// necessaria, e o bench confere o resto pelo relogio.
int cxx_has_pstl(void) {
#if defined(MULTIMERGE_PSTL)
    return 1;
#else
    return 0;
#endif
}

// Sequencial puro. Existe para UM proposito: o bench divide o tempo deste pelo
// tempo do `par` e recusa a rodada se a razao nao chegar a 2x -- a checagem que
// o CLAUDE.md exige, porque a falha do PSTL e silenciosa.
void cxx_seq_sort_u64(uint64_t* ptr, size_t len) {
    std::stable_sort(ptr, ptr + len);
}

void cxx_pstl_sort_u64(uint64_t* ptr, size_t len) {
#if defined(MULTIMERGE_PSTL)
    std::stable_sort(std::execution::par, ptr, ptr + len);
#else
    std::stable_sort(ptr, ptr + len);
#endif
}

void cxx_pstl_sort_str(const char** ptr, size_t len) {
    CStr* first = reinterpret_cast<CStr*>(ptr);
#if defined(MULTIMERGE_PSTL)
    std::stable_sort(std::execution::par, first, first + len);
#else
    std::stable_sort(first, first + len);
#endif
}

// Fixa o teto de paralelismo da TBB pelo resto do processo. Chamado UMA vez
// pelo bench, antes de medir. Sem isto a pool da TBB nao respeita nenhuma das
// variaveis de ambiente que fixam os outros dois lados, e a comparacao entre
// rodadas separadas deixa de valer -- alem de sobrecarregar a maquina.
//
// Retorna 1 quando o limite foi aplicado, 0 quando esta build nao tem o
// cabecalho da TBB (ai o bench avisa em vez de assumir).
int cxx_tbb_limit(int threads) {
#if defined(FFI_TEM_TBB_CONTROL)
    if (threads <= 0) return 1;  // 0 = deixar a TBB decidir, pedido valido
    // O limite vale enquanto o objeto existir: este nunca e destruido.
    static std::unique_ptr<tbb::global_control> trava;
    trava = std::make_unique<tbb::global_control>(
        tbb::global_control::max_allowed_parallelism,
        static_cast<size_t>(threads));
    return 1;
#else
    (void)threads;
    return 0;
#endif
}

}  // extern "C"
