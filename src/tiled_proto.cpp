// PROTOTIPO: merge de 8 runs, tres estrategias, mesmo resultado.
//
//   A) 3 niveis binarios sobre os arrays INTEIROS  -> 3 passadas na DRAM (hoje)
//   B) 8-vias direto com winner tree               -> 1 passada, kernel lento
//   C) TILED: multiseq corta em tiles de L2, e a arvore binaria de 3 niveis
//      roda dentro do cache                        -> 1 passada, kernel rapido
//
//   g++ -std=c++20 -O3 -fopenmp -DMULTIMERGE_KWAY tiled_proto.cpp -o tiled_proto

#include "MultiMergeSort.hpp"
#include <vector>
#include <cstdio>
#include <chrono>
#include <random>
#include <algorithm>
#include <cstring>

using u64 = uint64_t;
static const int K = 8;

// ---------- A: arvore binaria completa sobre arrays inteiros ----------
static void merge_full_tree(u64* r[K], size_t len[K], u64* dest, size_t total,
                            u64* s1, u64* s2) {
    size_t o = 0, l1[4];
    for (int p = 0; p < 4; ++p) {                       // nivel 1: 4 merges
        size_t a = len[2*p], b = len[2*p+1];
        multimerge::merge_seq<u64>(std::span<u64>(r[2*p], a), std::span<u64>(r[2*p+1], b),
                                   std::span<u64>(s1 + o, a + b));
        l1[p] = a + b; o += a + b;
    }
    size_t l2[2]; o = 0;
    for (int p = 0; p < 2; ++p) {                       // nivel 2: 2 merges
        size_t a = l1[2*p], b = l1[2*p+1];
        multimerge::merge_seq<u64>(std::span<u64>(s1 + o, a), std::span<u64>(s1 + o + a, b),
                                   std::span<u64>(s2 + o, a + b));
        l2[p] = a + b; o += a + b;
    }
    multimerge::merge_seq<u64>(std::span<u64>(s2, l2[0]), std::span<u64>(s2 + l2[0], l2[1]),
                               std::span<u64>(dest, total));   // nivel 3
}

// ---------- B: 8-vias direto ----------
static void merge_winner_tree(u64* r[K], size_t len[K], u64* dest, size_t total) {
    multimerge::WinnerTree<u64, K> t;
    t.ns = K;
    for (int i = 0; i < K; ++i) { t.head[i] = r[i]; t.fin[i] = r[i] + len[i]; }
    t.merge_into(dest, total);
}

// ---------- C: tiled ----------
static void merge_tiled(u64* r[K], size_t len[K], u64* dest, size_t total,
                        size_t TILE, u64* s1, u64* s2) {
    size_t cur[K] = {0};
    size_t done = 0;
    u64*   win[K];
    size_t wlen[K], cut[K];

    while (done < total) {
        const size_t want = std::min(TILE, total - done);

        // Janela restrita: um tile consome no maximo `want` de qualquer run,
        // entao a busca binaria roda sobre `want` e nao sobre o array inteiro.
        for (int i = 0; i < K; ++i) {
            win[i]  = r[i] + cur[i];
            wlen[i] = std::min(len[i] - cur[i], want);
        }
        multimerge::multiseq_partition<u64, K>(win, wlen, K, want, cut);

        // arvore binaria de 3 niveis, toda em s1/s2 (residentes em L2)
        size_t o = 0, l1[4];
        for (int p = 0; p < 4; ++p) {
            size_t a = cut[2*p], b = cut[2*p+1];
            multimerge::merge_seq<u64>(std::span<u64>(win[2*p], a),
                                       std::span<u64>(win[2*p+1], b),
                                       std::span<u64>(s1 + o, a + b));
            l1[p] = a + b; o += a + b;
        }
        size_t l2[2]; o = 0;
        for (int p = 0; p < 2; ++p) {
            size_t a = l1[2*p], b = l1[2*p+1];
            multimerge::merge_seq<u64>(std::span<u64>(s1 + o, a),
                                       std::span<u64>(s1 + o + a, b),
                                       std::span<u64>(s2 + o, a + b));
            l2[p] = a + b; o += a + b;
        }
        multimerge::merge_seq<u64>(std::span<u64>(s2, l2[0]),
                                   std::span<u64>(s2 + l2[0], l2[1]),
                                   std::span<u64>(dest + done, want));

        for (int i = 0; i < K; ++i) cur[i] += cut[i];
        done += want;
    }
}

int main() {
    const size_t PER = 1'000'000;              // por run
    const size_t TOTAL = PER * K;              // 8M elementos = 64 MB
    printf("8 runs x %zu = %zu elementos (%.0f MB de entrada + %.0f MB de saida)\n\n",
           PER, TOTAL, TOTAL * 8.0 / 1e6, TOTAL * 8.0 / 1e6);

    std::vector<u64> pool(TOTAL), out(TOTAL), sa(TOTAL), sb(TOTAL), ref(TOTAL);
    std::mt19937_64 rng(42);
    u64*   r[K];
    size_t len[K];
    for (int i = 0; i < K; ++i) {
        r[i] = pool.data() + (size_t)i * PER;
        len[i] = PER;
        for (size_t j = 0; j < PER; ++j) r[i][j] = rng() % 1000;   // muitos empates
        std::sort(r[i], r[i] + PER);
    }
    // referencia
    ref = pool; std::stable_sort(ref.begin(), ref.end());

    auto bench = [&](const char* nome, auto fn) {
        double best = 1e18;
        bool ok = true;
        for (int t = 0; t < 3; ++t) {
            std::fill(out.begin(), out.end(), 0);
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            double s = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - t0).count();
            best = std::min(best, s);
            if (out != ref) ok = false;
        }
        printf("  %-42s %7.1f ms  %7.0f Melem/s   %s\n",
               nome, best * 1e3, TOTAL / best / 1e6, ok ? "correto" : "ERRO <<<<");
        return best;
    };

    double a = bench("A) 3 niveis binarios, arrays inteiros",
        [&]{ merge_full_tree(r, len, out.data(), TOTAL, sa.data(), sb.data()); });
    double b = bench("B) 8-vias winner tree",
        [&]{ merge_winner_tree(r, len, out.data(), TOTAL); });

    printf("\n");
    double bestC = 1e18; size_t bestT = 0;
    for (size_t TILE : {256ul, 512ul, 1024ul, 2048ul, 4096ul}) {
        char nome[96];
        snprintf(nome, sizeof nome, "C) tiled, TILE=%-6zu (~%zu KB em L2)",
                 TILE, TILE * 8 * 3 / 1024);
        double c = bench(nome,
            [&]{ merge_tiled(r, len, out.data(), TOTAL, TILE, sa.data(), sb.data()); });
        if (c < bestC) { bestC = c; bestT = TILE; }
    }

    printf("\n--- veredicto ---\n");
    printf("A (3 passadas na DRAM) : %.1f ms\n", a * 1e3);
    printf("B (8-vias winner tree) : %.1f ms   -> %.2fx vs A\n", b * 1e3, a / b);
    printf("C (tiled, TILE=%zu)  : %.1f ms   -> %.2fx vs A\n", bestT, bestC * 1e3, a / bestC);
    return 0;
}
