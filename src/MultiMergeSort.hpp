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
#include <type_traits>
#include <bit>
#include <concepts>
#include <new>
#include <utility>

// ============================================================
// CONFIGURACAO PADRAO
//
// O merge de K vias com blocagem de cache e o merge bidirecional nas folhas
// vem LIGADOS, que e como o build.rs sempre compilou o motor. Antes dependiam
// de -DMULTIMERGE_KWAY e -DMULTIMERGE_BIDIR, e quem incluisse o cabecalho sem
// as duas flags -- inclusive o vs_stable.cpp, de onde saia a tabela do README
// -- media a arvore binaria com merge sequencial nas folhas, que nao e o que o
// projeto entrega. Medido em 15M de pares {chave, idx}, quase ordenado: 460 ms
// sem as duas, 262 ms com elas.
//
//   -DMULTIMERGE_NO_KWAY    arvore binaria (para comparar)
//   -DMULTIMERGE_NO_BIDIR   merge sequencial nas folhas (para comparar)
//   -DKWAY_FANOUT=<k>       fan-out do merge de K vias; potencia de 2, padrao 8
// ============================================================
#if defined(MULTIMERGE_NO_KWAY)
#undef MULTIMERGE_KWAY
#elif !defined(MULTIMERGE_KWAY)
#define MULTIMERGE_KWAY
#endif
#if defined(MULTIMERGE_NO_BIDIR)
#undef MULTIMERGE_BIDIR
#elif !defined(MULTIMERGE_BIDIR)
#define MULTIMERGE_BIDIR
#endif

// ============================================================
// FALLBACK PARA ENTRADA CAOTICA
//
// Quando o escudo de entropia decide que nao ha estrutura a explorar, o motor
// entrega o array a um sort estavel paralelo de terceiros. A escolha importa
// mais do que parece: medido em 5M com 8 threads, as tres opcoes ficaram longe
// uma da outra.
//
//   std::execution::par   126-161 ms   <- melhor
//   __gnu_parallel        228-240 ms   +29 a +48%
//   chunk_parallel_sort   (proprio, entre os dois)
//
// Com o __gnu_parallel o motor perdia 62 a 88% para o par do C++17 em dado
// aleatorio -- nao pelo algoritmo, mas por delegar para o fallback errado.
//
// A ordem de preferencia e: PSTL, depois o interno, e o __gnu_parallel so por
// pedido explicito. O PSTL exige libstdc++ com suporte E a Intel TBB lincada;
// no MSYS2, por exemplo, o ambiente UCRT64 tem e o MINGW64 nao. Por isso ele e
// opcional, e o caminho interno cobre quem nao tiver.
//
//   -DMULTIMERGE_PSTL          usa std::execution::par   (precisa de -ltbb)
//   -DMULTIMERGE_USE_GNU_PARALLEL  usa __gnu_parallel
//   nenhum dos dois                usa chunk_parallel_sort
// ============================================================
#if defined(MULTIMERGE_PSTL)
#include <execution>
#endif

// ============================================================
// CAMADA OPCIONAL DE TBB
//
// O motor NAO depende da TBB. Onde ela existe, alguns pontos podem usa-la;
// onde nao existe, o caminho OpenMP original continua valendo, byte por byte.
// Cada ponto e um interruptor separado, para poder ser medido isolado -- foi
// assim que descobrimos que a escolha do fallback caotico valia mais que o
// algoritmo do proprio motor.
//
//   -DMULTIMERGE_TBB          liga a camada (precisa de -ltbb / -ltbb12)
//   -DMULTIMERGE_TBB_SCHED    arvore de merge com tbb::parallel_invoke
//   -DMULTIMERGE_TBB_REDUCE   fold do metadata com tbb::parallel_reduce
//
// E ha um ganho que NAO precisa de macro nenhuma: lincar -ltbbmalloc_proxy
// troca malloc/free globais pelo alocador escalavel da TBB. Isso importa
// porque os std::stable_sort das folhas alocam POR CHAMADA, e com oito threads
// isso vira contencao no alocador do sistema.
//
// POR QUE NAO E TUDO OU NADA. Trocar o escalonador do OpenMP pelo da TBB
// transformaria uma dependencia opcional em obrigatoria. Com os interruptores
// separados, quem nao tiver TBB compila e roda exatamente como antes.
#if defined(MULTIMERGE_TBB)
#include <tbb/parallel_invoke.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#endif

#ifdef MULTIMERGE_USE_GNU_PARALLEL
#include <parallel/algorithm>
#endif

namespace multimerge {

std::vector<int64_t> merge_metadata_pure(std::vector<int64_t> left,
                                         const std::vector<int64_t>& right);

namespace detail {

/// Executa duas funcoes, em paralelo quando `paralelo` e verdadeiro.
///
/// Existe para que os pontos fork-join da arvore de merge nao precisem saber
/// qual escalonador esta ativo. Sem TBB o corpo e identico ao `#pragma omp
/// task` de antes, incluindo o `if(paralelo)` que decide inline vs tarefa.
///
/// O escalonador da TBB usa roubo de trabalho com fila dupla: a thread consome
/// o proprio topo (LIFO, dados quentes em cache) e rouba da base alheia (FIFO,
/// tarefa mais proxima da raiz, logo mais gorda). O `schedule(dynamic)` do
/// OpenMP distribui de uma fila central -- balanceia igual, mas sem essa
/// propriedade de localidade e com contencao no ponto central.
///
/// Se isso rende alguma coisa AQUI e uma pergunta empirica. Meça antes de
/// concluir.
template <typename F1, typename F2>
inline void bifurca(bool paralelo, F1&& f1, F2&& f2) {
#if defined(MULTIMERGE_TBB) && defined(MULTIMERGE_TBB_SCHED)
    if (paralelo) {
        tbb::parallel_invoke(std::forward<F1>(f1), std::forward<F2>(f2));
    } else {
        f1();
        f2();
    }
#else
    #pragma omp task if(paralelo)
    { f1(); }
    f2();
    #pragma omp taskwait
#endif
}

/// Combina os metadados dos blocos em um so.
///
/// A operacao e associativa -- e a propriedade que faz do metadata um MONOIDE,
/// e ela existe justamente para permitir reducao em arvore. A versao sequencial
/// percorre em ordem enquanto todos os nucleos esperam; a paralela reduz de
/// fato em arvore.
///
/// Para dado bem estruturado o metadata e pequeno e o fold nao pesa. Para dado
/// com muitos runs curtos, pesa. Meça antes de ligar.
inline std::vector<int64_t> junta_metadata(std::vector<std::vector<int64_t>>& partes) {
#if defined(MULTIMERGE_TBB) && defined(MULTIMERGE_TBB_REDUCE)
    return tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, partes.size()),
        std::vector<int64_t>{},
        [&](const tbb::blocked_range<size_t>& r, std::vector<int64_t> acc) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                acc = merge_metadata_pure(std::move(acc), partes[i]);
            }
            return acc;
        },
        [](std::vector<int64_t> a, std::vector<int64_t> b) {
            return merge_metadata_pure(std::move(a), b);
        });
#else
    std::vector<int64_t> combinado;
    for (auto& parte : partes) {
        combinado = merge_metadata_pure(std::move(combinado), parte);
    }
    return combinado;
#endif
}

}  // namespace detail

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
            // OVERLAP CONTRACT: arr[0] and arr[n-1] are shared with the
            // neighbouring blocks, which may already have measured them (and,
            // at macro level, may be measuring them concurrently). They are
            // READ-ONLY here. Only the interior may be reordered.
            if (evaluate_local_entropy(arr) && n > 2) {
                std::stable_sort(arr.begin() + 1, arr.end() - 1);
            }
            // Always reclassify. After the interior sort this normally collapses
            // to 2-3 runs, and the scan is over data still hot in L1.
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
            
            // OVERLAP CONTRACT: span[0] belongs to micro-block i-1 and
            // span[size-1] belongs to micro-block i+1. Writing to them would
            // invalidate metadata already emitted by the neighbour. Sorting only
            // the interior keeps every write disjoint, so this stays stable and
            // race-free.
            if (evaluate_local_entropy(span) && span.size() > 2) {
                std::stable_sort(span.begin() + 1, span.end() - 1);
            }
            local_meta[i] = generate_sequential_metadata(span);
        }

        return detail::junta_metadata(local_meta);
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

        return detail::junta_metadata(local_meta);
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

#ifdef MULTIMERGE_BIDIR
        // MERGE BIDIRECIONAL, com o limiar corrigido.
        //
        // O `if(k > 32768)` original era inalcancavel: parallel_merge so chama
        // esta funcao com total <= leaf_size*16 = 65536, logo k <= 32768 e a
        // comparacao e estrita. A task nunca era diferida - merge_front rodava
        // ate o fim e so entao merge_back comecava. As duas threads nunca
        // existiram. Amarrando o limiar a leaf_size, ela passa a disparar.
        const size_t k = total / 2;
        const bool spawn = (k > leaf_size);

        if constexpr (std::is_trivially_copyable_v<T>) {
            // TEOREMA DO MERGE BIDIRECIONAL
            // merge_front (empate -> A) emite os k menores, em ordem.
            // merge_back  (empate -> B) emite os total-k maiores, em ordem.
            // Os dois conjuntos particionam o multiconjunto para QUALQUER k,
            // entao nenhum ponto de corte precisa ser calculado - cada lado so
            // conta as proprias saidas. O co_rank aqui era trabalho redundante.
            //
            // Seguranca: os dois lados leem a e b INTEIROS. Para T trivialmente
            // copiavel, std::move e copia e nunca muta a origem, entao as
            // leituras sobrepostas no ponto de encontro sao leituras puras.
            // As escritas caem em metades disjuntas de dest. Sem race.
            #pragma omp task if(spawn)
            merge_front(a, b, dest.subspan(0, k));

            merge_back(a, b, dest.subspan(k));

            #pragma omp taskwait
        } else {
            // Tipos cujo move MUTA a origem: as leituras sobrepostas no ponto
            // de encontro seriam race de verdade. Mantem o corte disjunto.
            auto [split_a, split_b] = co_rank(k, a, b);

            #pragma omp task if(spawn)
            merge_front(a.subspan(0, split_a), b.subspan(0, split_b), dest.subspan(0, k));

            merge_back(a.subspan(split_a), b.subspan(split_b), dest.subspan(k));

            #pragma omp taskwait
        }
#else
        // SEM o bidirecional (-DMULTIMERGE_NO_BIDIR): uma unica passada para frente.
        // Sem co_rank redundante, sem overhead de task, e sem a caminhada
        // reversa que atrapalha o prefetcher.
        merge_seq(a, b, dest);
#endif
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

    // =====================================================================
    //  K-WAY MERGE
    //
    //  Motivo: o merge binario paga log2(runs) passadas sobre a memoria e a
    //  maquina esta saturada em banda. Um merge de K vias paga logK(runs).
    //  Com K=8 e 30000 runs: 15 niveis viram 5.
    //
    //  Tres pecas:
    //    1. WinnerTree      - merge sequencial de K vias, estavel
    //    2. multiseq_partition - corte por rank em K sequencias (paralelismo)
    //    3. bottom_up_merge_kway - recursao K-aria sobre o metadata
    //
    //  ESTABILIDADE: empates sempre vencem pelo MENOR indice de stream, tanto
    //  na WinnerTree quanto na distribuicao de empates do multiseq_partition.
    //  As duas regras precisam concordar, senao o corte paralelo e o merge
    //  sequencial produzem ordens diferentes.
    // =====================================================================

    #ifndef KWAY_FANOUT
#define KWAY_FANOUT 8
#endif
    constexpr int KWAY_K = KWAY_FANOUT;   // potencia de 2

    template <typename T, int K>
    struct WinnerTree {
        T*  head[K];
        T*  fin[K];
        int node[2 * K];
        int ns = 0;

        // stream a vence stream b?
        // INVARIANTE: em node[j] = beats(node[2j], node[2j+1]), o slot 2j vem
        // sempre da subarvore ESQUERDA, logo indice(a) < indice(b) sempre.
        // O desempate estavel e portanto sempre `<=`, sem branch de indice.
        inline bool beats(int a, int b) const {
            if (b < 0) return true;
            if (a < 0) return false;
            if (head[a] == fin[a]) return false;   // exaurido sempre perde
            if (head[b] == fin[b]) return true;
            return *head[a] <= *head[b];
        }

        void build() {
            for (int i = 0; i < K; ++i) node[K + i] = (i < ns) ? i : -1;
            for (int j = K - 1; j >= 1; --j)
                node[j] = beats(node[2 * j], node[2 * j + 1]) ? node[2 * j] : node[2 * j + 1];
        }

        inline void update(int s) {
            for (int j = (K + s) / 2; j >= 1; j /= 2)
                node[j] = beats(node[2 * j], node[2 * j + 1]) ? node[2 * j] : node[2 * j + 1];
        }

        void merge_into(T* dest, size_t count) {
            build();
            for (size_t i = 0; i < count; ++i) {
                const int w = node[1];
                dest[i] = std::move(*head[w]);
                ++head[w];
                update(w);
            }
        }
    };

    // Acha out[i] com sum(out) == rank, tal que os prefixos sao exatamente os
    // `rank` menores elementos. Empates distribuidos em ordem de stream.
    template <typename T, int K>
    void multiseq_partition(T* const* seq, const size_t* len, int ns,
                            size_t rank, size_t* out) {
        size_t lo[K], hi[K], plt[K], ple[K];
        for (int i = 0; i < ns; ++i) { lo[i] = 0; hi[i] = len[i]; }

        while (true) {
            int m = -1; size_t w = 0;
            for (int i = 0; i < ns; ++i)
                if (hi[i] - lo[i] > w) { w = hi[i] - lo[i]; m = i; }
            if (m < 0) { for (int i = 0; i < ns; ++i) out[i] = lo[i]; return; }

            const T P = seq[m][lo[m] + (hi[m] - lo[m]) / 2];

            size_t lt = 0, le = 0;
            for (int i = 0; i < ns; ++i) {
                plt[i] = (size_t)(std::lower_bound(seq[i], seq[i] + len[i], P) - seq[i]);
                ple[i] = (size_t)(std::upper_bound(seq[i], seq[i] + len[i], P) - seq[i]);
                lt += plt[i]; le += ple[i];
            }

            if (lt <= rank && rank <= le) {
                size_t need = rank - lt;
                for (int i = 0; i < ns; ++i) {
                    const size_t ties = ple[i] - plt[i];
                    const size_t take = (ties < need) ? ties : need;
                    out[i] = plt[i] + take;
                    need -= take;
                }
                return;
            }
            if (lt > rank) { for (int i = 0; i < ns; ++i) if (plt[i] < hi[i]) hi[i] = plt[i]; }
            else           { for (int i = 0; i < ns; ++i) if (ple[i] > lo[i]) lo[i] = ple[i]; }
        }
    }


    // TILE em ELEMENTOS. O otimo medido fica em L1 (512 em maquina com L1=32KB,
    // 1024 em outra). Derivado do L1 em vez de fixo: 3 buffers ativos por tile
    // (entrada streamada + s1 + s2), entao L1/(3*sizeof(T)) e o alvo.
    template <typename T>
    inline size_t tile_elems() {
        const size_t L1 = 32768;
        // /8 e nao /3. O calculo ingenuo ("3 buffers ativos: entrada, s1, s2")
        // da 1365 para u64, mas o otimo medido e 512. A diferenca e que os
        // dados de entrada streamados tambem disputam L1 com s1 e s2, entao o
        // orcamento real por buffer e bem menor que um terco.
        // Medido em duas maquinas independentes: 512 elementos para u64
        // (490 Melem/s contra ~460 em 1365).
        size_t t = L1 / (8 * std::max<size_t>(sizeof(T), 1));
        return std::clamp<size_t>(t, 128, 2048);
    }


    // Scratch por thread, alocado uma unica vez. Alocar por chamada colocava
    // malloc no caminho critico: com milhares de merges pequenos isso sozinho
    // custava mais do que a blocagem economizava.
    template <typename T>
    T* tiled_scratch() {
        static thread_local std::unique_ptr<T[]> buf;
        static thread_local size_t cap = 0;
        const size_t need = 2 * tile_elems<T>();
        if (cap < need) { buf = std::make_unique_for_overwrite<T[]>(need); cap = need; }
        return buf.get();
    }

    template <typename T>
    void bottom_up_merge(std::span<T>, std::span<T>, std::span<const int64_t>,
                         std::span<const size_t>, size_t, bool);

    // Acima deste tamanho o merge estoura o cache e a blocagem paga. Abaixo,
    // o merge ja e cache-resident e o caminho binario simples ganha, porque
    // nao paga particionamento nem tiles.
    constexpr size_t KWAY_MIN_BYTES = 2u << 20;   // 2 MB

    // Merge de ns <= K runs com BLOCAGEM DE CACHE.
    //
    // Em vez de log2(K) passadas na DRAM, corta a saida em tiles que cabem em
    // L1 via multiseq_partition e roda a arvore binaria de log2(K) niveis
    // inteiramente dentro do cache. A DRAM ve 1 leitura + 1 escrita.
    //
    // O kernel continua sendo o merge_seq com indices diretos - e por isso que
    // isso ganha da winner tree, que paga indirecao por elemento.
    template <typename T, int K>
    void kway_merge_tiled(T* const* runs, const size_t* lens, int ns,
                          T* dest, size_t total) {
        if (ns == 1) { std::move(runs[0], runs[0] + lens[0], dest); return; }
        if (ns == 2) {
            merge_seq<T>(std::span<T>(runs[0], lens[0]), std::span<T>(runs[1], lens[1]),
                         std::span<T>(dest, total));
            return;
        }

        const size_t TILE = tile_elems<T>();
        T* s1 = tiled_scratch<T>();      // thread_local, alocado UMA vez
        T* s2 = s1 + TILE;

        size_t cur[K] = {0};
        // Inicializados: o GCC nao prova que o laco `for (i < ns)` abaixo os
        // preenche antes do uso apos inlinar. Falso positivo, mas warning que
        // fica escondendo os de verdade. As escritas mortas somem no -O3.
        T*     win[K]  = {};
        size_t wlen[K] = {}, cut[K] = {}, seglen[K] = {};
        size_t done = 0;

        while (done < total) {
            const size_t want = std::min(TILE, total - done);

            // Janela restrita: um tile consome no maximo `want` de qualquer run,
            // entao a busca binaria roda sobre `want`, nao sobre o run inteiro.
            for (int i = 0; i < ns; ++i) {
                win[i]  = runs[i] + cur[i];
                wlen[i] = std::min(lens[i] - cur[i], want);
            }
            multiseq_partition<T, K>(win, wlen, ns, want, cut);

            // arvore binaria sobre os ns segmentos, toda em s1/s2 (L1)
            int    cnt = ns;
            T*     srcp[K] = {};
            size_t srcl[K] = {};
            for (int i = 0; i < ns; ++i) { srcp[i] = win[i]; srcl[i] = cut[i]; }

            T* buf = s1;
            T* alt = s2;
            while (cnt > 2) {
                int  out = 0;
                size_t off = 0;
                for (int i = 0; i < cnt; i += 2) {
                    if (i + 1 == cnt) {                       // sobra impar: so copia
                        std::move(srcp[i], srcp[i] + srcl[i], buf + off);
                        seglen[out] = srcl[i];
                    } else {
                        merge_seq<T>(std::span<T>(srcp[i], srcl[i]),
                                     std::span<T>(srcp[i + 1], srcl[i + 1]),
                                     std::span<T>(buf + off, srcl[i] + srcl[i + 1]));
                        seglen[out] = srcl[i] + srcl[i + 1];
                    }
                    srcp[out] = buf + off;
                    off += seglen[out];
                    ++out;
                }
                for (int i = 0; i < out; ++i) srcl[i] = seglen[i];
                cnt = out;
                std::swap(buf, alt);
            }
            // ultimo nivel escreve DIRETO no destino: a unica escrita na DRAM
            if (cnt == 2)
                merge_seq<T>(std::span<T>(srcp[0], srcl[0]), std::span<T>(srcp[1], srcl[1]),
                             std::span<T>(dest + done, want));
            else
                std::move(srcp[0], srcp[0] + srcl[0], dest + done);

            for (int i = 0; i < ns; ++i) cur[i] += cut[i];
            done += want;
        }
    }

    template <typename T, int K>
    void parallel_kway_merge(T* const* heads, const size_t* lens, int ns,
                             T* dest, size_t total, size_t leaf_size) {
        if (ns == 1) { std::move(heads[0], heads[0] + lens[0], dest); return; }

        if (total <= leaf_size * CORANK_SPLIT_FACTOR) {
            kway_merge_tiled<T, K>(heads, lens, ns, dest, total);
            return;
        }

        size_t cut[K] = {};
        multiseq_partition<T, K>(heads, lens, ns, total / 2, cut);

        T*     lh[K] = {}; size_t ll[K] = {};
        T*     rh[K] = {}; size_t rl[K] = {};
        size_t ltot = 0;
        for (int i = 0; i < ns; ++i) {
            lh[i] = heads[i];           ll[i] = cut[i];              ltot += cut[i];
            rh[i] = heads[i] + cut[i];  rl[i] = lens[i] - cut[i];
        }

        #pragma omp task firstprivate(lh, ll, ns, dest, ltot, leaf_size)
        parallel_kway_merge<T, K>(lh, ll, ns, dest, ltot, leaf_size);

        parallel_kway_merge<T, K>(rh, rl, ns, dest + ltot, total - ltot, leaf_size);

        #pragma omp taskwait
    }

    template <typename T>
    void bottom_up_merge_kway(std::span<T> v, std::span<T> buf,
                              std::span<const int64_t> metadata,
                              std::span<const size_t> offsets,
                              size_t leaf_size, bool into_buf) {
        constexpr int K = KWAY_K;
        const size_t num_blocks = metadata.size();

        if (num_blocks == 1) {
            const bool is_desc = metadata[0] < 0;
            if (into_buf) {
                if (is_desc) std::reverse(v.begin(), v.end());
                std::move(v.begin(), v.end(), buf.begin());
            } else if (is_desc) {
                std::reverse(v.begin(), v.end());
            }
            return;
        }

        const size_t base  = offsets[0];
        const size_t total = offsets[num_blocks] - base;

        // HIBRIDO: k-vias so nos niveis de topo, que sao os que vao a DRAM.
        // Abaixo do limiar todo o subtrecho cabe em cache e o binario e mais
        // rapido. Aplicar blocagem onde nao ha cache miss so custa.
        if (total * sizeof(T) <= KWAY_MIN_BYTES) {
            bottom_up_merge<T>(v, buf, metadata, offsets, leaf_size, into_buf);
            return;
        }

        const int    g     = (int)std::min<size_t>(K, num_blocks);

        // Cortes BALANCEADOS POR ELEMENTO, nao por contagem de runs.
        // (o binario usa num_blocks/2, que desequilibra quando os runs tem
        //  tamanhos muito diferentes)
        size_t cm[K + 1];
        cm[0] = 0; cm[g] = num_blocks;
        for (int i = 1; i < g; ++i) {
            const size_t target = base + total * (size_t)i / (size_t)g;
            size_t idx = (size_t)(std::lower_bound(offsets.begin(), offsets.end(), target)
                                  - offsets.begin());
            if (idx < cm[i - 1] + 1) idx = cm[i - 1] + 1;
            if (idx > num_blocks - (size_t)(g - i)) idx = num_blocks - (size_t)(g - i);
            cm[i] = idx;
        }

        for (int i = 0; i < g; ++i) {
            const size_t lo = cm[i], hi = cm[i + 1];
            const size_t s = offsets[lo] - base, e = offsets[hi] - base;
            auto sub_v = v.subspan(s, e - s);
            auto sub_b = buf.subspan(s, e - s);
            auto sub_m = metadata.subspan(lo, hi - lo);
            auto sub_o = offsets.subspan(lo, hi - lo + 1);
            if (i < g - 1) {
                #pragma omp task firstprivate(sub_v, sub_b, sub_m, sub_o, leaf_size, into_buf)
                bottom_up_merge_kway<T>(sub_v, sub_b, sub_m, sub_o, leaf_size, !into_buf);
            } else {
                bottom_up_merge_kway<T>(sub_v, sub_b, sub_m, sub_o, leaf_size, !into_buf);
            }
        }
        #pragma omp taskwait

        // Origem = o buffer onde os filhos escreveram
        std::span<T> src  = into_buf ? v : buf;
        std::span<T> dst  = into_buf ? buf : v;

        // ATALHO PRESERVADO: grupos adjacentes ja em ordem viram UM stream so.
        // Se todos colarem, ns==1 e o merge inteiro vira um move.
        T*     heads[K] = {};
        size_t lens[K] = {};
        int    ns = 0;
        size_t gs = 0;
        for (int i = 0; i < g; ++i) {
            const size_t e = offsets[cm[i + 1]] - base;
            const bool join_next = (i + 1 < g) && (src[e - 1] <= src[e]);
            if (!join_next) {
                heads[ns] = src.data() + gs;
                lens[ns]  = e - gs;
                ++ns; gs = e;
            }
        }

        if (ns == 1) { std::move(src.begin(), src.end(), dst.begin()); return; }
        parallel_kway_merge<T, K>(heads, lens, ns, dst.data(), total, leaf_size);
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

    /// Comprimento do bloco pre-ordenado no caminho caotico.
    ///
    /// CONSTANTE, nao formula. A varredura de 36 combinacoes (bloco x N x
    /// threads, ruido de 0,1% a 15%) mostrou que a superficie e RASA: qualquer
    /// bloco entre ~8K e ~131K fica a menos de 10% do otimo, e o "otimo" de
    /// cada linha varia sem tendencia -- em 10M com 8 threads o caso aleatorio
    /// escolhe 32768 e o de baixa cardinalidade escolhe 1024, os dois extremos.
    /// Uma formula com L3 e contagem de threads erraria por 0 a 22% um alvo que
    /// nao existe, e exigiria consultar o tamanho da L3, que nao tem forma
    /// portavel de ser obtido.
    ///
    /// 65536 foi o valor mais frequente entre os otimos medidos e fica dentro
    /// de 10% do melhor em quase todas as combinacoes.
    ///
    /// O que o bloco NAO deve ser e pequeno demais: ele multiplica o numero de
    /// runs, e cada nivel de merge a mais custa uma varredura inteira da DRAM,
    /// que e o gargalo desta carga. O valor anterior era 2000.
    ///
    /// Para fixar outro valor e comparar: -DMULTIMERGE_CHUNK=<n>
    template <typename T>
    inline size_t chunk_length_for(size_t n) {
#if defined(MULTIMERGE_CHUNK)
        (void)n;
        return (size_t)(MULTIMERGE_CHUNK);
#else
        constexpr size_t ALVO = 65536;
        // Teto de balanceamento: ao menos dois blocos por thread, senao arrays
        // pequenos geram menos blocos que nucleos e a ultima rodada deixa
        // gente parada.
        const size_t threads = (size_t)std::max(1, omp_get_max_threads());
        const size_t por_balanco = n / (threads * 2 + 1);
        return std::clamp<size_t>(std::min(ALVO, por_balanco), 1024, ALVO);
#endif
    }

    template <typename T>
    void chunk_parallel_sort(std::span<T> arr, size_t leaf_size) {
        size_t n = arr.size();
        const size_t CHUNK_LENGTH = chunk_length_for<T>(n);

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

    // =====================================================================
    //  FASE 2: NORMALIZACAO DOS RUNS
    //
    //  Entre a deteccao e a arvore. Deixa em `arr` so runs ASCENDENTES e
    //  junta os que ja estao em ordem entre si, para a arvore so receber o
    //  que precisa mesmo de merge.
    //
    //  1. DEVOLVE O PLATO ROUBADO. A varredura de um run ascendente avanca
    //     enquanto `<=`. Na virada ascendente -> descendente ela leva junto o
    //     primeiro elemento do dente descendente, que e o MAIOR dele:
    //
    //         ... 3 9 | 8 7 6 5 4          o 9 e do dente, nao do run anterior
    //         ... 3 9 9 | 8 7 6 5 4        com empate no topo, os DOIS 9
    //
    //     Com o 9 no run ascendente, ele termina em 9 e o dente, depois de
    //     invertido, comeca em 4: os dois nao encadeiam e cada par de dentes
    //     paga um merge. Devolver so um elemento nao basta quando o topo do
    //     dente tem chaves repetidas -- devolve-se o plato inteiro.
    //
    //  2. INVERTE OS DESCENDENTES no lugar, em paralelo. Um run descendente e
    //     ESTRITO, logo inverter e estavel. O plato devolvido e a excecao: ele
    //     entra na frente do run com chaves iguais, a inversao troca a ordem
    //     delas, e uma segunda inversao so do plato a devolve.
    //
    //  3. ENCADEIA. Fronteira com `arr[p-1] <= arr[p]` deixa de existir. O
    //     atalho "ja em ordem" que ficava DENTRO da arvore custava um memcpy
    //     do trecho por nivel; aqui custa uma comparacao por fronteira. Mil
    //     dentes em faixas disjuntas viram um segmento so: sem buffer, sem
    //     merge, sem copia.
    // =====================================================================
namespace detail {

#if defined(MULTIMERGE_STATS)
    /// So para os testes (-DMULTIMERGE_STATS): quantos segmentos ascendentes a
    /// ultima chamada entregou a arvore. 1 quer dizer que nao houve merge.
    /// Variavel global sem trava: nao ligue isto em codigo de producao.
    inline size_t g_segmentos = 0;
    /// Idem: 1 quando a ultima chamada foi resolvida pela rotina de radix.
    inline int g_radix = 0;
#endif

    /// Igualdade na ordem do motor. O motor so pede `<=` e `>` do tipo, e o
    /// `==` de T pode olhar mais do que a chave (um par {chave, linha}).
    template <typename T>
    inline bool mesma_chave(const T& a, const T& b) { return a <= b && b <= a; }

    /// Passo 1. Ajusta `meta` no lugar. Em `platos` ficam os runs descendentes
    /// que receberam mais de um elemento: (indice do run, tamanho do plato).
    template <typename T>
    void devolve_roubados(std::span<T> arr, std::vector<int64_t>& meta,
                          std::vector<std::pair<size_t, size_t>>& platos) {
        size_t off = 0;   // inicio do run i
        for (size_t i = 0; i + 1 < meta.size(); ++i) {
            const size_t len = (size_t)std::abs(meta[i]);
            const size_t head = off + len;            // inicio do run i+1
            if (meta[i] > 1 && meta[i + 1] < 0) {     // ascendente seguido de descendente
                const size_t dlen = (size_t)(-meta[i + 1]);
                const size_t tail = head + dlen;      // fim do run i+1
                if (!(arr[head - 1] <= arr[head])) {
                    // Quantos elementos iguais ao maximo ficaram no fim do
                    // ascendente. O limite deixa ao menos um elemento nele.
                    size_t k = 1;
                    while (k < len - 1 && mesma_chave(arr[head - 1 - k], arr[head - 1])) ++k;
                    // So vale devolver se o que sobra do ascendente encadeia
                    // com o dente: seu novo ultimo <= o minimo do descendente.
                    if (arr[head - 1 - k] <= arr[tail - 1]) {
                        meta[i]     = (int64_t)(len - k);
                        meta[i + 1] = -(int64_t)(dlen + k);
                        if (k > 1) platos.emplace_back(i + 1, k);
                        off += len - k;
                        continue;
                    }
                }
            }
            off += len;
        }
    }

    /// Passo 2. Runs grandes um de cada vez, com a inversao paralela por
    /// dentro; os pequenos em paralelo entre si.
    template <typename T>
    void inverte_descendentes(std::span<T> arr, const std::vector<int64_t>& meta,
                              const std::vector<size_t>& off,
                              const std::vector<std::pair<size_t, size_t>>& platos) {
        constexpr size_t GRANDE = (size_t)1 << 18;
        std::vector<size_t> pequenos;
        for (size_t i = 0; i < meta.size(); ++i) {
            if (meta[i] >= 0) continue;
            if (off[i + 1] - off[i] > GRANDE) parallel_reverse(arr.subspan(off[i], off[i + 1] - off[i]));
            else pequenos.push_back(i);
        }
        #pragma omp parallel for schedule(dynamic, 64) if(pequenos.size() > 64)
        for (long long j = 0; j < (long long)pequenos.size(); ++j) {
            const size_t i = pequenos[(size_t)j];
            std::reverse(arr.begin() + off[i], arr.begin() + off[i + 1]);
        }
        // O plato devolvido esta agora no FIM do run, com os iguais em ordem
        // trocada. Inverter so ele devolve a ordem original entre eles.
        #pragma omp parallel for schedule(dynamic, 64) if(platos.size() > 64)
        for (long long j = 0; j < (long long)platos.size(); ++j) {
            const size_t fim = off[platos[(size_t)j].first + 1];
            std::reverse(arr.begin() + (fim - platos[(size_t)j].second), arr.begin() + fim);
        }
    }

    /// Passo 3. Limites dos segmentos ascendentes que sobram depois de juntar
    /// os runs vizinhos ja em ordem. `off` tem R+1 valores; o resultado, S+1.
    template <typename T>
    std::vector<size_t> encadeia(std::span<T> arr, const std::vector<size_t>& off) {
        std::vector<size_t> seg;
        seg.push_back(0);
        for (size_t i = 1; i + 1 < off.size(); ++i) {
            const size_t p = off[i];
            if (!(arr[p - 1] <= arr[p])) seg.push_back(p);
        }
        seg.push_back(arr.size());
        return seg;
    }

}  // namespace detail

    // =====================================================================
    //  CAMINHO CAOTICO COM CHAVE: RADIX
    //
    //  Um sort por comparacao nao passa muito de um quicksort paralelo em dado
    //  aleatorio. Quem conhece os BITS da chave passa. As entradas
    //  `sort_radix`, `sort_by_radix_key` e `sort_by_prefix_key` dizem ao motor
    //  quais sao esses bits; devolvem exatamente o que `sort` devolve, e so o
    //  caminho do dado caotico muda:
    //
    //    1. AMOSTRA: 1024 elementos espalhados. Se um balde dos bits altos
    //       ficaria com mais de 1/4 deles, a chave e enviesada e a rotina
    //       desiste antes de varrer o array.
    //    2. FAIXA: AND e OR de todas as chaves dizem quais bits variam. Chave
    //       "nua" (o elemento e a chave) com ate 16 bits variando: CONTAGEM,
    //       histograma e preenchimento paralelos, sem buffer.
    //    3. HISTOGRAMA dos 8 bits mais altos QUE VARIAM, antes de alocar. Se
    //       um balde passaria de 1/4 do dado, desiste: nada foi movido.
    //    4. BUFFER e ESPALHAMENTO paralelo em 256 baldes. Cada pedaco da
    //       entrada escreve nas suas proprias faixas do destino, calculadas
    //       do histograma: sem disputa entre threads e na ordem que mantem os
    //       iguais como vieram.
    //    5. Cada BALDE dentro do cache, uma tarefa por balde: radix LSD se
    //       sobram poucos bits, o sort da std se sobram muitos.
    //    6. Balde grande leva um segundo corte.
    //
    //  A rotina devolve `false` SEM TER MOVIDO NADA quando desiste -- chave
    //  enviesada, balde dominante, falta de memoria -- e o caminho por
    //  comparacao segue como antes.
    //
    //  ESTABILIDADE. O espalhamento percorre os pedacos em ordem, o LSD e
    //  estavel por construcao, a folha por comparacao e o std::stable_sort, e
    //  a contagem so existe para chave nua, em que iguais sao indistinguiveis
    //  (e por isso ali a folha por comparacao pode ser o std::sort).
    //
    //  PARALELISMO. Tudo roda como tarefas dentro de UMA regiao paralela: um
    //  balde que reentra na rotina divide o trabalho com as threads livres,
    //  em vez de abrir uma regiao aninhada que o OpenMP serializaria.
    //
    //  LIMIARES. A estrutura e a do motor Rust; tres limiares NAO sao, porque
    //  dependem de quem e a alternativa por comparacao, e a do C++ e outra.
    //  Medido em 2026-11-18, 15M, 8 threads, minimo de 5, duas passadas:
    //
    //    folha LSD ate 56 bits (no Rust: 24 e 40). O std::sort e o
    //      std::stable_sort perdem do LSD em qualquer largura: u64 puro 147 ms
    //      com LSD ate 56 bits, 250 com LSD so ate 24; linha de 16 bytes com
    //      chave de 64 bits, 242 contra 530.
    //    segundo corte em 16 mil elementos (no Rust: 65 mil). Linha de 16
    //      bytes: 242 ms contra 391 sem o corte; 4 mil empata com 16 mil.
    //    tamanho minimo: ver `radix_min`.
    //
    //  Cada um pode ser trocado na linha de compilacao para medir:
    //    -DMULTIMERGE_RADIX_MIN=<n>        abaixo disto, so comparacao
    //    -DMULTIMERGE_RADIX_LSD_BARE=<b>   folha LSD ate b bits, chave nua
    //    -DMULTIMERGE_RADIX_LSD_ROW=<b>    idem, chave com dado junto
    //    -DMULTIMERGE_RADIX_RECUT=<n>      balde acima disto leva outro corte
    // =====================================================================
#ifndef MULTIMERGE_RADIX_LSD_BARE
#define MULTIMERGE_RADIX_LSD_BARE 56
#endif
#ifndef MULTIMERGE_RADIX_LSD_ROW
#define MULTIMERGE_RADIX_LSD_ROW 56
#endif
#ifndef MULTIMERGE_RADIX_RECUT
#define MULTIMERGE_RADIX_RECUT 16384
#endif
// Os de baixo so mudam nos TESTES, que os poem em miniatura para a rotina
// reentrar e passar por todas as folhas com poucos milhares de elementos.
#ifndef MULTIMERGE_RADIX_DIGIT
#define MULTIMERGE_RADIX_DIGIT 8
#endif
#ifndef MULTIMERGE_RADIX_COUNT_BITS
#define MULTIMERGE_RADIX_COUNT_BITS 16
#endif
#ifndef MULTIMERGE_RADIX_PIECE
#define MULTIMERGE_RADIX_PIECE 65536
#endif
#ifndef MULTIMERGE_RADIX_BIG
#define MULTIMERGE_RADIX_BIG (2u << 20)
#endif
#ifndef MULTIMERGE_RADIX_LEAF_MIN
#define MULTIMERGE_RADIX_LEAF_MIN 128
#endif
#ifndef MULTIMERGE_RADIX_SKEW
#define MULTIMERGE_RADIX_SKEW 4
#endif

namespace detail {

    constexpr unsigned RADIX_DIGIT      = MULTIMERGE_RADIX_DIGIT;       // bits da passada MSD
    constexpr unsigned RADIX_COUNT_BITS = MULTIMERGE_RADIX_COUNT_BITS;  // chave nua ate aqui: contagem
    constexpr unsigned RADIX_LSD_NARROW = 24;       // folha LSD "curta": ate 3 passadas
    constexpr size_t   RADIX_PIECE      = MULTIMERGE_RADIX_PIECE;       // elementos por pedaco paralelo
    constexpr size_t   RADIX_BIG        = MULTIMERGE_RADIX_BIG;
    constexpr size_t   RADIX_BIG_DIV    = 16;       // balde > max(n/16, RADIX_BIG) reentra sempre
    constexpr size_t   RADIX_RECUT      = MULTIMERGE_RADIX_RECUT;
    constexpr size_t   RADIX_SKEW       = MULTIMERGE_RADIX_SKEW;        // balde > 1/4 do dado: desiste
    constexpr size_t   RADIX_LEAF_MIN   = MULTIMERGE_RADIX_LEAF_MIN;    // balde menor: direto ao sort da std
    constexpr int      RADIX_MAX_DEPTH  = 6;
    static_assert(RADIX_DIGIT >= 1 && RADIX_DIGIT <= 8, "o espalhamento usa cursores de 256 baldes na pilha");

    /// Abaixo deste tamanho o caminho por comparacao ganha -- e o piso
    /// depende de QUEM e esse caminho. Medido em 2026-11-18, 8 threads, minimo
    /// de 40 rodadas alternadas, dado aleatorio:
    ///
    ///   fallback interno   o radix ganha de 65 mil em diante (u64 1,28x; i32
    ///                      1,72x; linha de 16 bytes 1,08x). Em 32 mil ainda
    ///                      perde na linha de 16 bytes (0,86x).
    ///   std par (PSTL)     i32 ganha de 131 mil (1,24x). u64 e linha de 16
    ///                      bytes perdem em 393 mil (0,90x e 0,75x) e ganham
    ///                      em 524 mil (1,35x e 1,34x).
    ///   __gnu_parallel     nao medido: vale o piso do PSTL, que e o mais alto.
    template <typename T>
    constexpr size_t radix_min() {
#if defined(MULTIMERGE_RADIX_MIN)
        return (size_t)(MULTIMERGE_RADIX_MIN);
#elif defined(MULTIMERGE_PSTL) || defined(MULTIMERGE_USE_GNU_PARALLEL)
        return sizeof(T) <= 4 ? (size_t)131072 : (size_t)524288;
#else
        return (size_t)65536;
#endif
    }

    // ---- Como o caminho caotico enxerga a chave de um elemento.
    //   enabled  ha chave? `NoKey` tira o radix do codigo gerado.
    //   exact    chaves iguais implicam elementos iguais na ordem. Libera a
    //            folha LSD. Falso para prefixo: ali o desempate e comparacao.
    //   bare     o elemento E a chave e se reconstroi dos bits. Libera a
    //            contagem, e o std::sort como folha.
    //   lsd_max  folha LSD quando sobram ate estes bits.

    /// Sem chave: so comparacao. E o que `sort` usa.
    struct NoKey {
        static constexpr bool enabled = false, exact = false, bare = false;
        static constexpr unsigned lsd_max = 0;
    };

    /// O elemento e um inteiro: ele e a chave. Com sinal, inverte-se o bit de
    /// sinal para a ordem dos bits ser a ordem dos valores.
    template <typename T>
    struct BareKey {
        static constexpr bool enabled = true, exact = true, bare = true;
        static constexpr unsigned lsd_max = MULTIMERGE_RADIX_LSD_BARE;
        using U = std::make_unsigned_t<T>;
        static constexpr U SINAL = std::is_signed_v<T> ? (U)((U)1 << (sizeof(T) * 8 - 1)) : (U)0;
        uint64_t bits(const T& x) const noexcept { return (uint64_t)(U)((U)x ^ SINAL); }
        T rebuild(uint64_t b) const noexcept { return (T)(U)((U)b ^ SINAL); }
    };

    /// A chave extraida decide a ordem inteira; o elemento pode levar dado.
    template <typename F>
    struct ExactKey {
        static constexpr bool enabled = true, exact = true, bare = false;
        static constexpr unsigned lsd_max = MULTIMERGE_RADIX_LSD_ROW;
        F f;
        template <typename T> uint64_t bits(const T& x) const { return (uint64_t)f(x); }
    };

    /// A chave extraida e so o comeco da ordem; a comparacao desempata.
    template <typename F>
    struct PrefixKey {
        static constexpr bool enabled = true, exact = false, bare = false;
        static constexpr unsigned lsd_max = 0;
        F f;
        template <typename T> uint64_t bits(const T& x) const { return (uint64_t)f(x); }
    };

    /// A chave e enviesada? Olha 1024 elementos espalhados: se um balde dos
    /// bits altos fica com mais de 1/RADIX_SKEW da amostra, o espalhamento nao
    /// dividiria o trabalho.
    template <typename T, typename Keyer>
    bool radix_skewed(const T* a, size_t n, const Keyer& key) {
        const size_t m = std::min<size_t>(1024, n);
        const size_t step = n / m;
        uint64_t x_and = ~(uint64_t)0, x_or = 0;
        for (size_t i = 0; i < m; ++i) {
            const uint64_t b = key.bits(a[i * step]);
            x_and &= b; x_or |= b;
        }
        const uint64_t diff = x_and ^ x_or;
        const unsigned hi = diff ? 64u - (unsigned)std::countl_zero(diff) : 0u;
        if (Keyer::bare && hi <= RADIX_COUNT_BITS) return false;   // candidato a contagem
        if (hi == 0) return true;                                  // a amostra inteira tem a mesma chave
        const unsigned b = std::min<unsigned>(RADIX_DIGIT, hi);
        const unsigned shift = hi - b;
        const uint64_t mask = ((uint64_t)1 << b) - 1;
        size_t h[256] = {};
        for (size_t i = 0; i < m; ++i) ++h[(key.bits(a[i * step]) >> shift) & mask];
        for (size_t d = 0; d <= mask; ++d)
            if (h[d] * RADIX_SKEW > m) return true;
        return false;
    }

    /// AND e OR de todas as chaves: os bits em que diferem sao os que variam.
    template <typename T, typename Keyer>
    void radix_range(const T* a, size_t n, const Keyer& key, uint64_t& out_and, uint64_t& out_or) {
        const size_t pieces = std::clamp<size_t>(n >> 16, 1, 256);
        const size_t chunk = (n + pieces - 1) / pieces;
        std::vector<uint64_t> pa(pieces, ~(uint64_t)0), po(pieces, 0);
        #pragma omp taskloop grainsize(1) default(shared) if(pieces > 1)
        for (long long q = 0; q < (long long)pieces; ++q) {
            const size_t s = (size_t)q * chunk, e = std::min(s + chunk, n);
            uint64_t x_and = ~(uint64_t)0, x_or = 0;
            for (size_t i = s; i < e; ++i) {
                const uint64_t b = key.bits(a[i]);
                x_and &= b; x_or |= b;
            }
            pa[(size_t)q] = x_and; po[(size_t)q] = x_or;
        }
        out_and = ~(uint64_t)0; out_or = 0;
        for (size_t q = 0; q < pieces; ++q) { out_and &= pa[q]; out_or |= po[q]; }
    }

    /// Chave nua de faixa pequena: conta e preenche, sem buffer. `x_or` e o OU
    /// de todas as chaves e `hi` quantos bits de baixo variam.
    template <typename T, typename Keyer>
    void radix_count(T* a, size_t n, const Keyer& key, uint64_t x_or, unsigned hi) {
        const size_t nb = (size_t)1 << hi;
        const uint64_t low = (uint64_t)nb - 1;
        const uint64_t base = x_or & ~low;   // os bits de cima sao iguais em todas as chaves
        const size_t pieces = std::clamp<size_t>(n >> 20, 1, 16);
        const size_t chunk = (n + pieces - 1) / pieces;
        std::vector<size_t> h(pieces * nb, 0);
        #pragma omp taskloop grainsize(1) default(shared) if(pieces > 1)
        for (long long q = 0; q < (long long)pieces; ++q) {
            size_t* hq = h.data() + (size_t)q * nb;
            const size_t s = (size_t)q * chunk, e = std::min(s + chunk, n);
            for (size_t i = s; i < e; ++i) ++hq[key.bits(a[i]) & low];
        }
        std::vector<size_t> ini(nb + 1, 0);   // onde cada valor comeca na saida
        for (size_t d = 0; d < nb; ++d) {
            size_t c = 0;
            for (size_t q = 0; q < pieces; ++q) c += h[q * nb + d];
            ini[d + 1] = ini[d] + c;
        }
        // Cada pedaco da SAIDA e preenchido por uma tarefa, comecando no valor
        // que cai no seu inicio: equilibrado mesmo com um valor dominante.
        #pragma omp taskloop grainsize(1) default(shared) if(pieces > 1)
        for (long long q = 0; q < (long long)pieces; ++q) {
            size_t s = (size_t)q * chunk;
            const size_t e = std::min(s + chunk, n);
            size_t d = (size_t)(std::upper_bound(ini.begin(), ini.end(), s) - ini.begin()) - 1;
            while (s < e) {
                const size_t fim = std::min(ini[d + 1], e);
                std::fill(a + s, a + fim, key.rebuild(base | (uint64_t)d));
                s = fim; ++d;
            }
        }
    }

    /// Histograma da passada MSD de um trecho.
    struct RadixPlan {
        unsigned shift = 0;          // bits abaixo do digito: o que sobra para a folha
        uint64_t mask = 0;
        size_t nb = 0, chunk = 0, pieces = 0;
        std::vector<size_t> hist;    // pieces x nb; o espalhamento o transforma em cursores
        std::vector<size_t> bounds;  // limites dos baldes no destino (nb + 1)
    };

    /// Conta o digito MSD: os RADIX_DIGIT bits mais altos entre os `hi` que
    /// variam. Falso se um balde ficaria com mais de 1/RADIX_SKEW do trecho.
    template <typename T, typename Keyer>
    bool radix_plan(const T* a, size_t n, const Keyer& key, unsigned hi, RadixPlan& p) {
        const unsigned b = std::min<unsigned>(RADIX_DIGIT, hi);
        p.shift = hi - b;
        p.nb = (size_t)1 << b;
        p.mask = (uint64_t)p.nb - 1;
        p.pieces = std::clamp<size_t>(n / RADIX_PIECE, 1, 128);
        p.chunk = (n + p.pieces - 1) / p.pieces;
        p.hist.assign(p.pieces * p.nb, 0);
        const unsigned shift = p.shift;
        const uint64_t mask = p.mask;
        const size_t nb = p.nb, chunk = p.chunk, pieces = p.pieces;
        size_t* hist = p.hist.data();
        #pragma omp taskloop grainsize(1) default(shared) if(pieces > 1)
        for (long long q = 0; q < (long long)pieces; ++q) {
            size_t* h = hist + (size_t)q * nb;
            const size_t s = (size_t)q * chunk, e = std::min(s + chunk, n);
            for (size_t i = s; i < e; ++i) ++h[(key.bits(a[i]) >> shift) & mask];
        }
        p.bounds.assign(nb + 1, 0);
        for (size_t d = 0; d < nb; ++d) {
            size_t c = 0;
            for (size_t q = 0; q < pieces; ++q) c += hist[q * nb + d];
            p.bounds[d + 1] = p.bounds[d] + c;
        }
        const size_t limite = n / RADIX_SKEW;
        for (size_t d = 0; d < nb; ++d)
            if (p.bounds[d + 1] - p.bounds[d] > limite) return false;
        return true;
    }

    /// Espalha `src` em `dst` conforme o plano, em paralelo. Para cada balde,
    /// e dentro dele para cada pedaco, o histograma da a faixa exata do
    /// destino: o pedaco q escreve o balde d a partir de
    /// bounds[d] + (o que os pedacos anteriores tem do balde d).
    template <typename T, typename Keyer>
    void radix_scatter(const T* src, T* dst, size_t n, RadixPlan& p, const Keyer& key) {
        const size_t nb = p.nb, chunk = p.chunk, pieces = p.pieces;
        const unsigned shift = p.shift;
        const uint64_t mask = p.mask;
        size_t* hist = p.hist.data();
        for (size_t d = 0; d < nb; ++d) {
            size_t acc = p.bounds[d];
            for (size_t q = 0; q < pieces; ++q) {
                const size_t c = hist[q * nb + d];
                hist[q * nb + d] = acc;
                acc += c;
            }
        }
        #pragma omp taskloop grainsize(1) default(shared) if(pieces > 1)
        for (long long q = 0; q < (long long)pieces; ++q) {
            size_t cur[256];   // copia local: sem falso compartilhamento entre pedacos
            std::copy(hist + (size_t)q * nb, hist + (size_t)q * nb + nb, cur);
            const size_t s = (size_t)q * chunk, e = std::min(s + chunk, n);
            for (size_t i = s; i < e; ++i) {
                const T x = src[i];
                dst[cur[(key.bits(x) >> shift) & mask]++] = x;
            }
        }
    }

    /// Radix LSD sequencial de um balde, sobre os `low_bits` de baixo da chave
    /// (os de cima sao iguais no balde inteiro). O dado esta em `src`; o
    /// resultado termina em `dst`. Os histogramas de todas as passadas saem de
    /// UMA leitura; passada de digito constante e pulada.
    template <typename T, typename Keyer>
    void radix_lsd(T* src, T* dst, size_t n, const Keyer& key, unsigned low_bits) {
        const unsigned passes = (low_bits + 7) / 8;   // <= 8
        size_t h[8][256] = {};
        for (size_t i = 0; i < n; ++i) {
            const uint64_t b = key.bits(src[i]);
            for (unsigned p = 0; p < passes; ++p) ++h[p][(b >> (8 * p)) & 0xFF];
        }
        T* a = src;
        T* b = dst;
        for (unsigned p = 0; p < passes; ++p) {
            size_t* hp = h[p];
            bool constante = false;
            size_t acc = 0;
            for (unsigned d = 0; d < 256; ++d) {
                const size_t c = hp[d];
                if (c == n) constante = true;
                hp[d] = acc;
                acc += c;
            }
            if (constante) continue;
            const unsigned sh = 8 * p;
            for (size_t i = 0; i < n; ++i) {
                const T x = a[i];
                b[hp[(key.bits(x) >> sh) & 0xFF]++] = x;
            }
            std::swap(a, b);
        }
        if (a != dst) std::copy(a, a + n, dst);
    }

    /// Folha por comparacao. Com chave nua, iguais sao indistinguiveis e o
    /// std::sort (que nao aloca) serve; fora disso, o estavel.
    template <typename T, typename Keyer>
    inline void radix_leaf_sort(T* a, size_t n) {
        if constexpr (Keyer::bare) std::sort(a, a + n);
        else std::stable_sort(a, a + n);
    }

    /// Sort por comparacao de um trecho GRANDE em que o radix nao serve
    /// (prefixo que nao distingue, balde que continua enviesado). Blocos em
    /// tarefas e a arvore de merge do motor; roda dentro da regiao paralela.
    template <typename T, typename Keyer>
    void radix_fallback(T* a, size_t n) {
        constexpr size_t BLOCO = 65536;
        if (n <= 2 * BLOCO) { radix_leaf_sort<T, Keyer>(a, n); return; }
        std::unique_ptr<T[]> buf(new (std::nothrow) T[n]);
        if (!buf) { radix_leaf_sort<T, Keyer>(a, n); return; }
        const size_t blocos = (n + BLOCO - 1) / BLOCO;
        #pragma omp taskloop grainsize(1) default(shared)
        for (long long i = 0; i < (long long)blocos; ++i) {
            const size_t s = (size_t)i * BLOCO, e = std::min(s + BLOCO, n);
            radix_leaf_sort<T, Keyer>(a + s, e - s);
        }
        std::vector<int64_t> meta(blocos);
        for (size_t i = 0; i < blocos; ++i)
            meta[i] = (int64_t)(std::min((i + 1) * BLOCO, n) - i * BLOCO);
        const std::vector<size_t> off = block_offsets(meta);
        bottom_up_merge<T>(std::span<T>(a, n), std::span<T>(buf.get(), n),
                           std::span<const int64_t>(meta), std::span<const size_t>(off),
                           get_leaf_size<T>(), false);
    }

    template <typename T, typename Keyer>
    void radix_bucket(T* a, T* s, size_t n, const Keyer& key, int depth);

    /// Ordena cada balde. O dado esta em `buf`; o resultado termina em `arr`.
    template <typename T, typename Keyer>
    void radix_buckets(T* arr, T* buf, size_t n, const RadixPlan& p, const Keyer& key, int depth) {
        const size_t big = std::max<size_t>(n / RADIX_BIG_DIV, RADIX_BIG);
        const unsigned shift = p.shift;
        // Folha estreita = LSD de poucas passadas: ali so reentra o balde
        // grande de verdade. Folha larga (muitas passadas, ou comparacao): um
        // segundo corte MSD sai mais barato que ordenar um balde que nao cabe
        // no cache.
        const bool narrow = Keyer::exact && shift <= RADIX_LSD_NARROW;
        const size_t recut = narrow ? big : std::min<size_t>(RADIX_RECUT, big);
        for (size_t d = 0; d < p.nb; ++d) {
            const size_t lo = p.bounds[d];
            const size_t len = p.bounds[d + 1] - lo;
            if (len == 0) continue;
            T* a = arr + lo;
            T* s = buf + lo;
            #pragma omp task firstprivate(a, s, len, shift, recut, big, depth) shared(key) if(len > 2048)
            {
                if (Keyer::exact && shift == 0) {
                    // O digito cobriu todos os bits que variam: chaves identicas.
                    std::copy(s, s + len, a);
                } else if (len > recut && depth < RADIX_MAX_DEPTH) {
                    // De volta para `arr` e a rotina de novo, com o proprio
                    // pedaco do buffer como rascunho.
                    std::copy(s, s + len, a);
                    radix_bucket(a, s, len, key, depth + 1);
                } else if (Keyer::exact && len >= RADIX_LEAF_MIN && shift <= Keyer::lsd_max) {
                    radix_lsd(s, a, len, key, shift);
                } else {
                    std::copy(s, s + len, a);
                    if (len > big) radix_fallback<T, Keyer>(a, len);
                    else radix_leaf_sort<T, Keyer>(a, len);
                }
            }
        }
        #pragma omp taskwait
    }

    /// A rotina de novo em um balde grande. O dado esta em `a`; `s` e rascunho.
    template <typename T, typename Keyer>
    void radix_bucket(T* a, T* s, size_t n, const Keyer& key, int depth) {
        uint64_t x_and, x_or;
        radix_range(a, n, key, x_and, x_or);
        const uint64_t diff = x_and ^ x_or;
        if (diff == 0) {
            if (!Keyer::exact) radix_fallback<T, Keyer>(a, n);
            return;
        }
        const unsigned hi = 64u - (unsigned)std::countl_zero(diff);
        if constexpr (Keyer::bare) {
            if (hi <= RADIX_COUNT_BITS) { radix_count(a, n, key, x_or, hi); return; }
        }
        RadixPlan p;
        if (radix_plan(a, n, key, hi, p)) {
            radix_scatter(a, s, n, p, key);
            radix_buckets(a, s, n, p, key, depth);
        } else {
            radix_fallback<T, Keyer>(a, n);
        }
    }

    /// Ordena `arr` por radix. Devolve `false`, sem ter movido nenhum
    /// elemento, quando a chave e enviesada, quando o prefixo nao distingue
    /// nada ou quando nao ha memoria para o buffer.
    template <typename T, typename Keyer>
    bool radix_sort(std::span<T> arr, const Keyer& key) {
        static_assert(std::is_trivially_copyable_v<T>,
                      "as entradas com radix pedem um tipo trivialmente copiavel");
        T* const a = arr.data();
        const size_t n = arr.size();
        if (n < 2) return true;
        if (radix_skewed(a, n, key)) return false;

        bool feito = false;
        #pragma omp parallel
        #pragma omp single
        {
            feito = [&]() -> bool {
                uint64_t x_and, x_or;
                radix_range(a, n, key, x_and, x_or);
                const uint64_t diff = x_and ^ x_or;
                // Todas as chaves iguais. Com chave exata nao ha o que ordenar;
                // com prefixo, ele nao distingue nada e a comparacao assume.
                if (diff == 0) return Keyer::exact;
                const unsigned hi = 64u - (unsigned)std::countl_zero(diff);
                if constexpr (Keyer::bare) {
                    if (hi <= RADIX_COUNT_BITS) { radix_count(a, n, key, x_or, hi); return true; }
                }
                // O histograma vem ANTES do buffer: se a chave for enviesada,
                // desistir nao custou nem a alocacao.
                RadixPlan p;
                if (!radix_plan(a, n, key, hi, p)) return false;
                std::unique_ptr<T[]> buf(new (std::nothrow) T[n]);
                if (!buf) return false;
                radix_scatter(a, buf.get(), n, p, key);
                radix_buckets(a, buf.get(), n, p, key, 0);
                return true;
            }();
        }
        return feito;
    }

    // ==========================================
    // ENTRADA COMUM
    // ==========================================
    template <typename T, typename Keyer>
    void sort_with_key(std::span<T> arr, const Keyer& key) {
        size_t n = arr.size();
        size_t leaf_size = get_leaf_size<T>();
#if defined(MULTIMERGE_STATS)
        g_segmentos = 0;
        g_radix = 0;
#endif

        if (n <= leaf_size) {
            std::stable_sort(arr.begin(), arr.end());
            return;
        }

        // Failsafe for complete chaos (random sequences).
        if (evaluate_local_entropy(arr)) {
            // RADIX: so com chave conhecida. Se a rotina desistir, nada foi
            // movido e o caminho por comparacao abaixo segue igual.
            if constexpr (Keyer::enabled) {
                const bool feito = n >= radix_min<T>() && radix_sort(arr, key);
#if defined(MULTIMERGE_STATS)
                g_radix = feito ? 1 : 0;
#endif
                if (feito) return;
            }
            // Ver a nota no topo do arquivo: a ordem de preferencia foi medida.
#if defined(MULTIMERGE_PSTL)
            std::stable_sort(std::execution::par, arr.begin(), arr.end());
#elif defined(MULTIMERGE_USE_GNU_PARALLEL)
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

        // FASE 2: so sobram runs ascendentes, e os que ja estao em ordem
        // entre si viram um segmento.
        std::vector<std::pair<size_t, size_t>> platos;
        devolve_roubados(arr, metadata, platos);
        std::vector<size_t> offsets = block_offsets(metadata);
        inverte_descendentes(arr, metadata, offsets, platos);
        offsets = encadeia(arr, offsets);
#if defined(MULTIMERGE_STATS)
        g_segmentos = offsets.size() - 1;
#endif
        if (offsets.size() == 2) return;   // um segmento: ja esta ordenado

        metadata.resize(offsets.size() - 1);
        for (size_t i = 0; i + 1 < offsets.size(); ++i)
            metadata[i] = (int64_t)(offsets[i + 1] - offsets[i]);

        // Zero-initialization elision: Memory allocated without being cleared
        auto buffer_ptr = std::make_unique_for_overwrite<T[]>(n);
        std::span<T> buffer(buffer_ptr.get(), n);

        // Bootstraps the OpenMP Thread Pool exactly once
        #pragma omp parallel
        {
            #pragma omp single
            {
#ifdef MULTIMERGE_KWAY
                bottom_up_merge_kway<T>(arr, buffer, metadata, offsets, leaf_size, false);
#else
                bottom_up_merge<T>(arr, buffer, metadata, offsets, leaf_size, false);
#endif
            }
        }
    }

}  // namespace detail

    // ==========================================
    // MAIN ENTRY POINT
    // ==========================================

    /// Ordenacao estavel. O tipo precisa de `<`, `<=`, `>` e `>=`.
    template <typename T>
    void sort(std::span<T> arr) {
        detail::sort_with_key(arr, detail::NoKey{});
    }

    /// `sort` para inteiros. Mesmo resultado; o dado caotico vai por radix.
    /// Esse caminho aloca um buffer de n elementos, salvo quando as chaves
    /// cabem em 16 bits (contagem). Sem memoria, ou com chave enviesada
    /// demais para o radix render, o caminho por comparacao assume.
    template <typename T>
        requires std::integral<T> && (!std::same_as<T, bool>)
    void sort_radix(std::span<T> arr) {
        detail::sort_with_key(arr, detail::BareKey<T>{});
    }

    /// `sort` quando uma chave de ate 64 bits decide a ordem inteira:
    /// `key(a) < key(b)` exatamente quando `a < b`, e chaves iguais quando
    /// nenhum dos dois vem antes. O elemento pode levar dado junto (um par
    /// {chave, linha}); a ordem entre chaves iguais e preservada.
    template <typename T, typename KeyFn>
    void sort_by_radix_key(std::span<T> arr, KeyFn key) {
        detail::sort_with_key(arr, detail::ExactKey<KeyFn>{std::move(key)});
    }

    /// `sort` quando a chave e maior que 64 bits, como texto. `prefix` da o
    /// comeco da ordem -- `prefix(a) < prefix(b)` implica `a < b` -- e os
    /// operadores de T desempatam. Para texto: os 8 primeiros bytes, em
    /// big-endian.
    template <typename T, typename PrefixFn>
    void sort_by_prefix_key(std::span<T> arr, PrefixFn prefix) {
        detail::sort_with_key(arr, detail::PrefixKey<PrefixFn>{std::move(prefix)});
    }
}

#endif // MULTIMERGESORT_HPP