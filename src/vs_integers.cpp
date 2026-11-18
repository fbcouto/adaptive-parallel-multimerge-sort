// Inteiros puros no motor C++: multimerge::sort x multimerge::sort_radix x os
// sorts paralelos da biblioteca padrao. E a segunda tabela da secao
// "Benchmarks" do README.
//
// Compilar (UCRT64 do MSYS2, que e onde o `par` do C++17 engata):
//   g++ -std=c++20 -O3 -fopenmp -DMULTIMERGE_PSTL vs_integers.cpp -ltbb12 -o vs_integers
//   vs_integers [n] [rodadas]         padrao: 30000000 7
//
// 1 rodada de aquecimento descartada (nela a saida de cada competidor e
// conferida contra o std::sort), ordem dos competidores SORTEADA a cada rodada
// (semente fixa; ver o porque em vs_stable.cpp), copia nova fora do
// cronometro, mediana e minimo. sort_radix aparece duas vezes: a diferenca
// entre as duas medianas e o piso de ruido da linha. "ja ordenado" e a linha
// de controle: ali o radix nem e chamado.
#include "MultiMergeSort.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <execution>
#include <random>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;
enum { PAR_INST, PAR_EST, MM, MM_RADIX, MM_RADIX_AA, N };
static const char* nome[] = {"std::sort(par)", "stable_sort(par)", "multimerge::sort", "sort_radix", "sort_radix(A/A)"};

template <typename T>
static void roda(const std::string& cenario, const std::vector<T>& base, int rodadas) {
    std::vector<T> ref = base;
    std::sort(std::execution::par, ref.begin(), ref.end());
    std::vector<std::vector<double>> t(N);
    bool ok[N] = {true, true, true, true, true};
    std::vector<T> v(base.size());
    int ordem[N];
    for (int i = 0; i < N; ++i) ordem[i] = i;
    std::mt19937 sorteio(0x0DE5u + (unsigned)base.size());
    for (int r = 0; r <= rodadas; ++r) {
        std::shuffle(ordem, ordem + N, sorteio);
        for (int slot = 0; slot < N; ++slot) {
            const int m = ordem[slot];
            std::copy(std::execution::par, base.begin(), base.end(), v.begin());
            const auto t0 = clk::now();
            switch (m) {
                case PAR_INST: std::sort(std::execution::par, v.begin(), v.end()); break;
                case PAR_EST:  std::stable_sort(std::execution::par, v.begin(), v.end()); break;
                case MM:       multimerge::sort(std::span<T>(v)); break;
                default:       multimerge::sort_radix(std::span<T>(v)); break;
            }
            const double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
            if (r > 0) t[m].push_back(ms);
            if (r == 0 && !(v == ref)) ok[m] = false;
        }
    }
    printf("%-34s", cenario.c_str());
    for (int m = 0; m < N; ++m) {
        std::sort(t[m].begin(), t[m].end());
        printf(" | %s: %7.1f (%7.1f)%s", nome[m], t[m][t[m].size() / 2], t[m][0], ok[m] ? "" : " ERRADO");
    }
    printf("\n");
    fflush(stdout);
}

template <typename T>
static void tipo(const char* nome_tipo, size_t n, int rodadas) {
    std::mt19937_64 rng(42);
    auto gera = [&](int forma) {
        std::vector<T> v(n);
        for (size_t i = 0; i < n; ++i) {
            const uint64_t z = rng();
            switch (forma) {
                case 0:  v[i] = (T)z; break;
                case 1:  v[i] = (T)(z % 64); break;
                case 2:  v[i] = (T)((z % 10 != 0) ? (z >> 8) % 1000 : z); break;
                default: v[i] = (T)z; break;
            }
        }
        if (forma == 3) std::sort(std::execution::par, v.begin(), v.end());
        return v;
    };
    roda(std::string(nome_tipo) + " aleatorio", gera(0), rodadas);
    roda(std::string(nome_tipo) + " 64 valores", gera(1), rodadas);
    roda(std::string(nome_tipo) + " enviesado (90% em 0..1000)", gera(2), rodadas);
    roda(std::string(nome_tipo) + " ja ordenado (CONTROLE)", gera(3), rodadas);
}

int main(int argc, char** argv) {
    const size_t n = argc > 1 ? std::stoull(argv[1]) : 30'000'000;
    const int rodadas = argc > 2 ? std::stoi(argv[2]) : 7;
    printf("n = %zu; %d threads; %d rodadas + aquecimento; ms: MEDIANA (minimo)\n", n, omp_get_max_threads(), rodadas);
    tipo<int32_t>("i32", n, rodadas);
    tipo<uint64_t>("u64", n, rodadas);
    return 0;
}
