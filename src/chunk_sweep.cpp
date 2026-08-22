// VARREDURA DO BLOCO DO CAMINHO CAOTICO — ESCALA E THREADS
//
// A rodada anterior (10M, 8 threads, ruido de 0,8%) mostrou duas coisas:
//
//   1. O chunk_parallel_sort JA batia o __gnu_parallel com o bloco de 2.000:
//      -8,5% em aleatorio e -24,8% em baixa cardinalidade. O MULTIMERGE_GNU
//      esta ON por padrao, entao o motor usa o fallback pior.
//
//   2. O otimo estava em 8.192, nao nos 86.480 da formula. A extrapolacao do
//      PIM estava errada: la o caminho aleatorio SUBSTITUI a fase de deteccao,
//      aqui ele alimenta o bottom_up_merge com k-way, que ja economiza
//      passadas. As duas otimizacoes competem pelo mesmo recurso.
//
// Esta versao varre BLOCO x N x THREADS para achar a forma da superficie antes
// de fixar qualquer formula. Cada combinacao imprime o otimo medido e o que a
// formula atual escolheria, para dar o erro dela em cada ponto.
//
// Compilar:
//   g++ -std=c++20 -O3 -fopenmp -DMULTIMERGE_USE_GNU_PARALLEL chunk_sweep.cpp -o chunk_sweep
//
// DISCIPLINA DE MEDICAO (aprendida a duras penas no PIM):
//   - buffer reutilizado: alocar centenas de MB dentro do laco mede o alocador
//   - ordem rotativa entre motores: neutraliza aquecimento e deriva termica
//   - LINHA DE CONTROLE: a mesma configuracao medida duas vezes. A diferenca
//     entre elas e o piso de ruido. Nada abaixo disso e legivel.

#include "MultiMergeSort.hpp"
#include <vector>
#include <cstdio>
#include <chrono>
#include <random>
#include <algorithm>
#include <string>
#include <span>

#if defined(MULTIMERGE_USE_GNU_PARALLEL)
#include <parallel/algorithm>
#endif

using u64 = uint64_t;
using clk = std::chrono::high_resolution_clock;

static const int REPS = 9;

static double mediana(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Copia do chunk_parallel_sort com o bloco como PARAMETRO, para varrer sem
// recompilar. Fora o bloco, e o mesmo codigo do header.
template <typename T>
static void chunk_sort_param(std::span<T> arr, size_t leaf_size, size_t chunk) {
    size_t n = arr.size();
    size_t num_chunks = (n + chunk - 1) / chunk;

    #pragma omp parallel for schedule(dynamic)
    for (long long i = 0; i < (long long)num_chunks; ++i) {
        size_t start = (size_t)i * chunk;
        size_t end = std::min(start + chunk, n);
        std::stable_sort(arr.begin() + start, arr.begin() + end);
    }

    std::vector<int64_t> metadata(num_chunks);
    for (size_t i = 0; i < num_chunks; ++i) {
        size_t start = i * chunk;
        size_t end = std::min(start + chunk, n);
        metadata[i] = (int64_t)(end - start);
    }
    std::vector<size_t> offsets = multimerge::block_offsets(metadata);

    auto buffer_ptr = std::make_unique_for_overwrite<T[]>(n);
    std::span<T> buffer(buffer_ptr.get(), n);

    #pragma omp parallel
    {
        #pragma omp single
        {
            multimerge::bottom_up_merge<T>(arr, buffer, metadata, offsets, leaf_size, false);
        }
    }
}

struct Motor {
    std::string nome;
    size_t chunk;      // 0 = nao usa chunk_sort_param
    bool gnu;
};

static size_t roda(const char* cenario, const std::vector<u64>& base, bool detalhe) {
    const size_t n = base.size();
    const size_t leaf = multimerge::get_leaf_size<u64>();
    const size_t formula = multimerge::chunk_length_for<u64>(n);
    const int threads = omp_get_max_threads();

    std::vector<Motor> motores;
    motores.push_back({"std::stable_sort (1 thread)", 0, false});
#if defined(MULTIMERGE_USE_GNU_PARALLEL)
    motores.push_back({"__gnu_parallel::stable_sort", 0, true});
    motores.push_back({"__gnu_parallel (CONTROLE)",   0, true});
#endif
    for (size_t c : {1024ull, 2000ull, 4096ull, 8192ull, 16384ull, 32768ull, 65536ull, 131072ull}) {
        if (c < n) motores.push_back({"chunk=" + std::to_string(c), (size_t)c, false});
    }
    if (formula < n) motores.push_back({"chunk=" + std::to_string(formula) + " (FORMULA)", formula, false});

    std::vector<std::vector<double>> t(motores.size());
    std::vector<u64> v(n);
    std::vector<u64> ref = base;
    std::stable_sort(ref.begin(), ref.end());

    for (int rep = 0; rep < REPS + 1; ++rep) {
        for (size_t slot = 0; slot < motores.size(); ++slot) {
            size_t id = (rep + slot) % motores.size();
            std::copy(base.begin(), base.end(), v.begin());   // fora do cronometro
            auto t0 = clk::now();
            if (motores[id].chunk) {
                chunk_sort_param<u64>(std::span<u64>(v), leaf, motores[id].chunk);
            } else if (motores[id].gnu) {
#if defined(MULTIMERGE_USE_GNU_PARALLEL)
                __gnu_parallel::stable_sort(v.begin(), v.end());
#endif
            } else {
                std::stable_sort(v.begin(), v.end());
            }
            double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
            if (v != ref) { printf("  ERRO: %s nao ordenou corretamente\n", motores[id].nome.c_str()); }
            if (rep > 0) t[id].push_back(ms);
        }
    }

    std::vector<double> med(motores.size());
    for (size_t i = 0; i < motores.size(); ++i) med[i] = mediana(t[i]);

    size_t ref_id = 0;
    for (size_t i = 0; i < motores.size(); ++i) if (motores[i].gnu) { ref_id = i; break; }

    size_t ctrl = (size_t)-1;
    for (size_t i = 0; i < motores.size(); ++i)
        if (motores[i].nome.find("CONTROLE") != std::string::npos) ctrl = i;
    double ruido = (ctrl != (size_t)-1)
        ? 100.0 * std::abs(med[ctrl] - med[ref_id]) / med[ref_id] : 0.0;

    // melhor bloco (ignora os motores que nao usam bloco)
    size_t melhor_id = (size_t)-1;
    for (size_t i = 0; i < motores.size(); ++i)
        if (motores[i].chunk && (melhor_id == (size_t)-1 || med[i] < med[melhor_id])) melhor_id = i;
    size_t melhor_bloco = melhor_id == (size_t)-1 ? 0 : motores[melhor_id].chunk;

    if (detalhe) {
        printf("\n=== %s ===  n=%zu  threads=%d  leaf=%zu  formula=%zu  ruido=%.1f%%\n",
               cenario, n, threads, leaf, formula, ruido);
        printf("  %-34s %10s %10s\n", "motor", "mediana", "vs ref");
        printf("  %s\n", std::string(56, '-').c_str());
        for (size_t i = 0; i < motores.size(); ++i) {
            printf("  %-34s %8.1f ms %+9.1f%%%s\n", motores[i].nome.c_str(), med[i],
                   100.0 * (med[i] - med[ref_id]) / med[ref_id],
                   i == melhor_id ? "   <- melhor" : "");
        }
        if (ruido > 10.0) printf("  AVISO: ruido %.1f%%. Diferencas menores nao sao legiveis.\n", ruido);
    } else {
        // uma linha por combinacao
        double err = melhor_bloco ? 100.0 * (med[melhor_id] - med[ref_id]) / med[ref_id] : 0.0;
        size_t f_id = (size_t)-1;
        for (size_t i = 0; i < motores.size(); ++i)
            if (motores[i].nome.find("FORMULA") != std::string::npos) f_id = i;
        double perda_formula = (f_id != (size_t)-1 && melhor_id != (size_t)-1)
            ? 100.0 * (med[f_id] - med[melhor_id]) / med[melhor_id] : 0.0;
        printf("  %-26s %3d %10zu %10zu %9.1f %9.1f%% %10.1f%% %8.1f%%\n",
               cenario, threads, melhor_bloco, formula, med[melhor_id], err,
               perda_formula, ruido);
    }
    return melhor_bloco;
}

static std::vector<u64> gera(const char* tipo, size_t n, std::mt19937_64& rng) {
    std::vector<u64> v(n);
    if (std::string(tipo) == "aleatorio") {
        for (auto& x : v) x = rng();
    } else if (std::string(tipo) == "baixa_card") {
        for (auto& x : v) x = rng() % 64;
    } else {
        for (size_t i = 0; i < n; ++i) v[i] = (u64)(i % 1000);
    }
    return v;
}

int main(int argc, char** argv) {
    // Sem argumento: varredura completa N x threads, uma linha por combinacao.
    // Com argumento: tabela detalhada naquele N, nas threads atuais.
    bool detalhe = (argc > 1);
    size_t n_unico = detalhe ? std::stoull(argv[1]) : 0;

    printf("VARREDURA DO BLOCO CAOTICO — %d threads OpenMP disponiveis\n", omp_get_max_threads());
#if defined(MULTIMERGE_USE_GNU_PARALLEL)
    printf("build: MULTIMERGE_USE_GNU_PARALLEL = ON (a referencia esta disponivel)\n");
#else
    printf("build: MULTIMERGE_USE_GNU_PARALLEL = OFF — recompile com -DMULTIMERGE_USE_GNU_PARALLEL\n");
    printf("       para comparar contra o fallback que o motor usa por padrao.\n");
#endif
    printf("\nA formula e L3/(threads * 1,5 * sizeof(T)), limitada por n/(2*threads).\n");
    printf("Se a linha FORMULA ficar abaixo do gnu_parallel, o C++ deixa de perder\n");
    printf("os 35-47%% medidos no Criterion E se livra da extensao GNU.\n");

    std::mt19937_64 rng(0xA11EA701);
    const char* tipos[] = {"aleatorio", "baixa_card", "dente_serra"};
    const int max_thr = omp_get_max_threads();

    if (detalhe) {
        for (const char* tp : tipos) {
            auto v = gera(tp, n_unico, rng);
            roda(tp, v, true);
        }
        return 0;
    }

    printf("\n  %-26s %3s %10s %10s %9s %10s %11s %8s\n",
           "cenario", "thr", "otimo", "formula", "ms", "vs gnu", "perda form", "ruido");
    printf("  %s\n", std::string(96, '-').c_str());

    for (size_t n : {1'000'000ull, 10'000'000ull, 50'000'000ull}) {
        for (int thr : {1, 2, 4, max_thr}) {
            if (thr > max_thr) continue;
            omp_set_num_threads(thr);
            for (const char* tp : tipos) {
                auto v = gera(tp, (size_t)n, rng);
                char rot[64];
                snprintf(rot, sizeof rot, "%s/%zuM", tp, (size_t)(n / 1'000'000));
                roda(rot, v, false);
            }
        }
        printf("\n");
    }
    omp_set_num_threads(max_thr);

    printf("COMO LER:\n");
    printf("  otimo      = bloco mais rapido da varredura naquela combinacao\n");
    printf("  formula    = o que chunk_length_for() escolheria hoje\n");
    printf("  vs gnu     = o otimo contra __gnu_parallel::stable_sort (negativo = melhor)\n");
    printf("  perda form = quanto a formula perde para o otimo (0%% = acertou)\n");
    printf("  ruido      = mesma config medida duas vezes; ignore deltas menores\n\n");
    printf("Se o otimo NAO se mover com N nem com threads, uma constante basta e a\n");
    printf("formula com L3 e threads e complicacao desnecessaria. Se se mover, a\n");
    printf("coluna diz em que direcao.\n");
    return 0;
}
