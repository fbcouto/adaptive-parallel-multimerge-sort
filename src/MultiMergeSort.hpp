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
        // VARIANTE C (padrao): uma unica passada para frente.
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

        // Failsafe for complete chaos (random sequences).
        // Ver a nota no topo do arquivo: a ordem de preferencia foi medida.
        if (evaluate_local_entropy(arr)) {
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

        std::vector<size_t> offsets = block_offsets(metadata);

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
}

#endif // MULTIMERGESORT_HPP