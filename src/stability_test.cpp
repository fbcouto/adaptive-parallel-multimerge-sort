// Teste de estabilidade e correcao para multimerge::sort
//
// Compilar:
//   g++ -std=c++20 -O3 -fopenmp -DMULTIMERGE_USE_GNU_PARALLEL stability_test.cpp -o stability_test
//   g++ -std=c++20 -O3 -fopenmp stability_test.cpp -o stability_test_nognu
//
// Cada elemento carrega sua posicao original (idx). Uma ordenacao estavel deve
// produzir, para chaves iguais, idx estritamente crescente.

#include "MultiMergeSort.hpp"
#include <vector>
#include <random>
#include <algorithm>
#include <cstdio>
#include <string>

struct Keyed {
    uint64_t key;
    uint64_t idx;   // posicao original — nunca participa da comparacao
};

// O motor usa <=, >, >= ; std::stable_sort usa < . Todos comparam SO a chave.
inline bool operator< (const Keyed& a, const Keyed& b) { return a.key <  b.key; }
inline bool operator<=(const Keyed& a, const Keyed& b) { return a.key <= b.key; }
inline bool operator> (const Keyed& a, const Keyed& b) { return a.key >  b.key; }
inline bool operator>=(const Keyed& a, const Keyed& b) { return a.key >= b.key; }
inline bool operator==(const Keyed& a, const Keyed& b) { return a.key == b.key && a.idx == b.idx; }

static int g_failures = 0;

static void check(const std::string& name, std::vector<Keyed> input) {
    for (size_t i = 0; i < input.size(); ++i) input[i].idx = i;

    std::vector<Keyed> reference = input;
    std::stable_sort(reference.begin(), reference.end());

    std::vector<Keyed> work = input;
    multimerge::sort(std::span<Keyed>(work));

    // 1. ordenado por chave?
    size_t bad_order = (size_t)-1;
    for (size_t i = 1; i < work.size(); ++i)
        if (work[i-1].key > work[i].key) { bad_order = i; break; }

    // 2. estavel? chaves iguais devem manter idx crescente
    size_t bad_stable = (size_t)-1;
    for (size_t i = 1; i < work.size(); ++i)
        if (work[i-1].key == work[i].key && work[i-1].idx > work[i].idx) { bad_stable = i; break; }

    // 3. identico ao std::stable_sort (pega tambem perda/duplicacao de elementos)
    bool exact = (work == reference);

    bool ok = (bad_order == (size_t)-1) && (bad_stable == (size_t)-1) && exact;
    if (!ok) g_failures++;

    printf("  %-42s n=%-9zu ordem:%-4s estavel:%-4s identico:%-4s %s\n",
           name.c_str(), input.size(),
           bad_order  == (size_t)-1 ? "OK" : "FALHA",
           bad_stable == (size_t)-1 ? "OK" : "FALHA",
           exact ? "OK" : "FALHA",
           ok ? "" : "  <<<<<< PROBLEMA");

    if (bad_stable != (size_t)-1) {
        size_t i = bad_stable;
        printf("      inversao de estabilidade em %zu: key=%llu idx %llu veio ANTES de idx %llu\n",
               i, (unsigned long long)work[i].key,
               (unsigned long long)work[i-1].idx, (unsigned long long)work[i].idx);
    }
}

int main() {
    printf("threads OpenMP: %d\n", omp_get_max_threads());
#ifdef MULTIMERGE_USE_GNU_PARALLEL
    printf("build: MULTIMERGE_USE_GNU_PARALLEL = ON\n\n");
#else
    printf("build: MULTIMERGE_USE_GNU_PARALLEL = OFF\n\n");
#endif

    std::mt19937_64 rng(20260814);
    auto make = [](size_t n) { return std::vector<Keyed>(n); };

    printf("[A] Estabilidade com baixa cardinalidade (muitas chaves iguais)\n");
    {
        // todos iguais — o caso extremo
        auto v = make(200000);
        for (auto& e : v) e.key = 42;
        check("todos os elementos iguais", v);
    }
    {
        // 8 chaves distintas, distribuidas ao acaso
        auto v = make(200000);
        for (auto& e : v) e.key = rng() % 8;
        check("8 chaves distintas, aleatorio", v);
    }
    {
        // baixa cardinalidade + trecho caotico posicionado na sonda do micro-bloco
        auto v = make(200000);
        for (size_t i = 0; i < v.size(); ++i) v[i].key = i / 1000;   // estruturado
        for (size_t i = 34700; i < 34950; ++i) v[i].key = rng() % 4; // caos na sonda
        check("estruturado + caos na fronteira macro", v);
    }

    printf("\n[B] Estabilidade em runs DESCENDENTES com repeticoes\n");
    printf("    (testa o invariante: run descendente e ESTRITO, logo reverter e estavel)\n");
    {
        auto v = make(200000);
        for (size_t i = 0; i < v.size(); ++i) v[i].key = (v.size() - i) / 500; // desc c/ platos
        check("descendente com platos de 500 iguais", v);
    }
    {
        auto v = make(200000);
        for (size_t i = 0; i < v.size(); ++i) v[i].key = v.size() - i;
        for (size_t i = 0; i < v.size(); i += 3) v[i].key = v[i - (i%3)].key;  // duplicatas
        check("descendente estrito + duplicatas", v);
    }

    printf("\n[C] Padroes do benchmark do README\n");
    {
        auto v = make(300000);
        for (size_t i = 0; i < v.size(); ++i) v[i].key = i % 1000;   // sawtooth
        check("sawtooth (database-like)", v);
    }
    {
        auto v = make(300000);
        for (size_t i = 0; i < v.size(); ++i) v[i].key = i;
        for (int p = 0; p < 20; ++p) {                                // 99% sorted
            size_t s = rng() % (v.size() - 100);
            for (size_t i = s; i < s + 100; ++i) v[i].key = rng() % v.size();
        }
        check("99% ordenado com ruido", v);
    }
    {
        auto v = make(300000);
        for (auto& e : v) e.key = rng() % 1000;                       // random chaos
        check("caos aleatorio, cardinalidade 1000", v);
    }
    {
        auto v = make(300000);
        for (size_t i = 0; i < v.size(); ++i) v[i].key = i / 7;
        check("totalmente ordenado com platos", v);
    }

    printf("\n[D] Fuzz: estruturado + N trechos caoticos, chaves repetidas\n");
    {
        int local_fail_before = g_failures;
        const int TRIALS = 150;
        for (int t = 0; t < TRIALS; ++t) {
            size_t n = 5000 + (rng() % 250000);
            std::vector<Keyed> v(n);
            uint64_t card = 1 + (rng() % 500);
            for (size_t i = 0; i < n; ++i) v[i].key = (i / 13) % card;
            int patches = 1 + (int)(rng() % 6);
            for (int p = 0; p < patches; ++p) {
                size_t s = rng() % n;
                size_t len = 200 + (rng() % 6000);
                for (size_t i = s; i < std::min(s + len, n); ++i) v[i].key = rng() % card;
            }
            for (size_t i = 0; i < n; ++i) v[i].idx = i;

            std::vector<Keyed> ref = v;  std::stable_sort(ref.begin(), ref.end());
            std::vector<Keyed> w = v;    multimerge::sort(std::span<Keyed>(w));
            if (!(w == ref)) g_failures++;
        }
        printf("  %-42s %d/%d falharam\n", "fuzz estruturado+caotico",
               g_failures - local_fail_before, TRIALS);
    }

    printf("\n[E] Tamanhos de borda (fronteiras de bloco)\n");
    {
        for (size_t n : {0u, 1u, 2u, 3u, 119u, 120u, 4095u, 4096u, 4097u,
                         8190u, 8191u, 32767u, 32768u, 32769u, 65534u}) {
            std::vector<Keyed> v(n);
            for (size_t i = 0; i < n; ++i) v[i].key = rng() % 16;
            check("n = " + std::to_string(n), v);
        }
    }

    printf("\n===================================================\n");
    printf("RESULTADO: %d falha(s)\n", g_failures);
    printf("===================================================\n");
    return g_failures == 0 ? 0 : 1;
}
