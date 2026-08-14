// Mede o teto de banda da maquina E o efeito do numero de streams simultaneos.
//
//   g++ -std=c++20 -O3 -fopenmp bandwidth.cpp -o bandwidth.exe
//   .\bandwidth.exe
//
// Todos os testes movem EXATAMENTE o mesmo total de bytes (le N + escreve N).
// A unica variavel e de quantos streams de entrada a leitura vem:
//
//   2 streams  = 1 leitura + 1 escrita   -> teto puro da maquina
//   3 streams  = 2 leituras + 1 escrita  -> o que o merge binario faz HOJE
//   5 streams  = 4 leituras + 1 escrita  -> como seria um merge 4-vias
//   9 streams  = 8 leituras + 1 escrita  -> como seria um merge 8-vias
//
// Se a banda cair muito de 3 para 9 streams, o ganho de trafego do k-vias
// e comido pela perda de eficiencia (prefetcher + paginas DRAM + TLB).

#include <vector>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <omp.h>

static const size_t N = 50'000'000;          // 400 MB de destino
static const int    REPS = 5;

using clk = std::chrono::high_resolution_clock;

static double gbs(double seconds) {
    return (double)N * 16.0 / seconds / 1e9;  // le 8 bytes + escreve 8 bytes por elemento
}

int main() {
    printf("threads OpenMP: %d\n", omp_get_max_threads());
    printf("buffer de destino: %.0f MB   |   total movido por teste: %.0f MB\n\n",
           N * 8.0 / 1e6, N * 16.0 / 1e6);

    std::vector<uint64_t> dst(N);
    std::vector<uint64_t> pool(N);

    // first touch em paralelo, para as paginas ficarem no lugar certo
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)N; ++i) { pool[i] = (uint64_t)i; dst[i] = 0; }

    printf("%-44s %10s %12s\n", "teste", "melhor", "banda");
    printf("%-44s %10s %12s\n", "-----", "------", "-----");

    // ---- 2 streams: copia pura, o teto ----
    {
        double best = 1e18;
        for (int r = 0; r < REPS; ++r) {
            auto t0 = clk::now();
            #pragma omp parallel for schedule(static)
            for (long long i = 0; i < (long long)N; ++i) dst[i] = pool[i];
            double s = std::chrono::duration<double>(clk::now() - t0).count();
            best = std::min(best, s);
        }
        printf("%-44s %8.1f ms %9.1f GB/s\n", "2 streams (copia pura) = TETO", best * 1e3, gbs(best));
    }

    // ---- k streams intercalados ----
    auto kway = [&](int k, const char* rotulo) {
        const uint64_t* p[8];
        size_t chunk = N / (size_t)k;
        for (int s = 0; s < k; ++s) p[s] = pool.data() + (size_t)s * chunk;

        double best = 1e18;
        for (int r = 0; r < REPS; ++r) {
            auto t0 = clk::now();
            #pragma omp parallel for schedule(static)
            for (long long b = 0; b < (long long)chunk; ++b) {
                uint64_t* out = dst.data() + (size_t)b * k;
                for (int s = 0; s < k; ++s) out[s] = p[s][b];   // k streams avancando juntos
            }
            double s = std::chrono::duration<double>(clk::now() - t0).count();
            best = std::min(best, s);
        }
        printf("%-44s %8.1f ms %9.1f GB/s\n", rotulo, best * 1e3, gbs(best));
        return gbs(best);
    };

    double b2 = 0, b4 = 0, b8 = 0;
    b2 = kway(2, "3 streams (2 leituras) = merge BINARIO hoje");
    b4 = kway(4, "5 streams (4 leituras) = merge 4-VIAS");
    b8 = kway(8, "9 streams (8 leituras) = merge 8-VIAS");

    printf("\n--- o que isso significa para o k-vias ---\n");
    printf("4-vias: trafego cai 2,0x, banda vai a %.0f%% -> ganho liquido estimado %.2fx\n",
           100.0 * b4 / b2, 2.0 * b4 / b2);
    printf("8-vias: trafego cai 3,0x, banda vai a %.0f%% -> ganho liquido estimado %.2fx\n",
           100.0 * b8 / b2, 3.0 * b8 / b2);
    printf("\n(Sawtooth/30M hoje trafega ~23,7 GB/s. Compare com o TETO acima:\n");
    printf(" perto do teto = saturado, k-vias ajuda. Bem abaixo = o gargalo e outro.)\n");
    return 0;
}
