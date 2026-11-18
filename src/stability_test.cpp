// Teste de estabilidade e correcao para multimerge::sort e para as entradas
// com chave radix (sort_radix, sort_by_radix_key, sort_by_prefix_key)
//
// Compilar:
//   g++ -std=c++20 -O3 -fopenmp -DMULTIMERGE_STATS stability_test.cpp -o stability_test
//   g++ -std=c++20 -O3 -fopenmp -DMULTIMERGE_USE_GNU_PARALLEL stability_test.cpp -o stability_test_gnu
//
// -DMULTIMERGE_STATS liga dois contadores so de teste: quantos segmentos foram
// para a arvore (o dente de serra tem de dar 1) e se o radix entrou mesmo.
//
// Uma terceira build poe os limiares do radix em miniatura, para a rotina
// reentrar e passar por todas as folhas (contagem, LSD, comparacao, recuo) com
// os mesmos dados (uma linha so):
//   g++ -std=c++20 -O3 -fopenmp -DMULTIMERGE_STATS -DMULTIMERGE_RADIX_DIGIT=2
//       -DMULTIMERGE_RADIX_COUNT_BITS=2 -DMULTIMERGE_RADIX_RECUT=12 -DMULTIMERGE_RADIX_BIG=24
//       -DMULTIMERGE_RADIX_PIECE=64 -DMULTIMERGE_RADIX_LEAF_MIN=4 -DMULTIMERGE_RADIX_MIN=64
//       -DMULTIMERGE_RADIX_SKEW=1 stability_test.cpp -o stability_test_mini
// (com digito de 2 bits ha so 4 baldes, e a regra "um balde com mais de 1/4 do
// dado denuncia chave enviesada" desistiria sempre: o SKEW=1 a desliga.)
//
// Cada elemento carrega sua posicao original (idx). Uma ordenacao estavel deve
// produzir, para chaves iguais, idx estritamente crescente.

#include "MultiMergeSort.hpp"
#include <vector>
#include <random>
#include <algorithm>
#include <cstdio>
#include <string>
#include <cstring>

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

// Ordenado, com cada bloco IMPAR de `dente` elementos invertido: dentes em
// faixas de valor disjuntas, direcao alternada (o "Sawtooth" dos benchmarks
// Java e Python). `plato` = quantas vezes o maior de cada dente aparece.
static std::vector<Keyed> serra_disjunta(size_t n, size_t dente, size_t plato) {
    std::vector<Keyed> v(n);
    for (size_t i = 0; i < n; ++i) {
        const size_t d = i / dente;
        const size_t len = std::min(dente, n - d * dente);
        size_t k = i % dente;
        if (d % 2 == 1) k = len - 1 - k;                    // dente impar: descendente
        const size_t topo = len > plato ? len - plato : 0;  // os `plato` ultimos repetem o maximo
        v[i].key = (uint64_t)(d * dente + std::min(k, topo));
    }
    return v;
}

// Formatos de chave que mandam a rotina de radix por caminhos diferentes.
static const char* nome_formato(int forma) {
    switch (forma) {
        case 0:  return "radix: faixa inteira";
        case 1:  return "radix: faixa de 50 mil (contagem)";
        case 2:  return "radix: 64 valores";
        case 3:  return "radix: enviesada (a rotina desiste)";
        case 4:  return "radix: 40 bits, prefixo comum";
        case 5:  return "radix: ordenada (nem chega ao radix)";
        default: return "radix: invertida com repetidos";
    }
}
static std::vector<uint64_t> formato(size_t n, int forma, std::mt19937_64& rng) {
    std::vector<uint64_t> c(n);
    for (size_t i = 0; i < n; ++i) {
        const uint64_t z = rng();
        switch (forma) {
            case 0:  c[i] = z; break;
            case 1:  c[i] = z % 50000; break;
            case 2:  c[i] = (z % 64) * 1000003; break;
            case 3:  c[i] = (z % 10 == 0) ? z : (z >> 8) % 1000; break;
            case 4:  c[i] = 0xABCD000000000000ull | (z & 0xFFFFFFFFFFull); break;
            case 5:  c[i] = i; break;
            default: c[i] = (n - i) / 3; break;
        }
    }
    return c;
}

// Inteiro nu: sort_radix tem de dar o mesmo que o std::sort. A conversao para
// tipos estreitos e com sinal trunca de proposito: metade vira negativa.
template <typename T>
static bool confere_inteiros(const std::vector<uint64_t>& chaves) {
    std::vector<T> v(chaves.size());
    for (size_t i = 0; i < v.size(); ++i) v[i] = (T)chaves[i];
    std::vector<T> ref = v;
    std::sort(ref.begin(), ref.end());
    multimerge::sort_radix(std::span<T>(v));
    return v == ref;
}

// Chave maior que 64 bits: texto de 12 bytes mais a posicao original.
struct Texto {
    char t[12];
    uint32_t idx;
};
inline bool operator< (const Texto& a, const Texto& b) { return std::memcmp(a.t, b.t, 12) <  0; }
inline bool operator<=(const Texto& a, const Texto& b) { return std::memcmp(a.t, b.t, 12) <= 0; }
inline bool operator> (const Texto& a, const Texto& b) { return std::memcmp(a.t, b.t, 12) >  0; }
inline bool operator>=(const Texto& a, const Texto& b) { return std::memcmp(a.t, b.t, 12) >= 0; }
inline bool operator==(const Texto& a, const Texto& b) { return std::memcmp(a.t, b.t, 12) == 0 && a.idx == b.idx; }

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

    printf("\n[F] Dente de serra em faixas disjuntas, direcao alternada\n");
    printf("    (o maximo de cada dente descendente e roubado pelo run anterior;\n");
    printf("     devolvido, os dentes encadeiam e nao ha merge nenhum)\n");
    {
        // plato = quantas vezes o MAIOR de cada dente aparece. Com 2 ou mais,
        // devolver um elemento so nao basta, e o plato devolvido precisa
        // voltar a ordem original depois da inversao do dente.
        for (size_t dente : {100u, 1000u, 4096u, 4097u, 5000u, 33000u, 100000u}) {
            for (size_t plato : {1u, 2u, 3u, 17u}) {
                const size_t n = 600000;
                check("dente " + std::to_string(dente) + ", plato " + std::to_string(plato),
                      serra_disjunta(n, dente, plato));
#if defined(MULTIMERGE_STATS)
                if (multimerge::detail::g_segmentos != 1) {
                    g_failures++;
                    printf("      <<<<<< PROBLEMA: %zu segmentos foram para a arvore; esperado 1 (sem merge)\n",
                           multimerge::detail::g_segmentos);
                }
#endif
            }
        }
        // O mesmo com um UNICO dente defeituoso no meio de um array grande: e
        // o caso que custava um merge do array inteiro.
        // Chaves todas distintas e UMA duplicata, no topo de um dente descendente.
        auto v = serra_disjunta(900000, 1000, 1);
        v[451001].key = v[451000].key;   // o dente 451 e impar: desce a partir de 451000
        check("uma unica duplicata no topo de um dente", v);
#if defined(MULTIMERGE_STATS)
        if (multimerge::detail::g_segmentos != 1) {
            g_failures++;
            printf("      <<<<<< PROBLEMA: %zu segmentos foram para a arvore; esperado 1 (sem merge)\n",
                   multimerge::detail::g_segmentos);
        }
#endif
    }

    printf("\n[G] Radix: mesmas saidas de multimerge::sort, pelo caminho com chave\n");
    {
        // n acima de MULTIMERGE_RADIX_MIN, senao o radix nem entra.
        const size_t n = 600000;
        for (int forma = 0; forma < 7; ++forma) {
            const std::vector<uint64_t> chaves = formato(n, forma, rng);

            // (1) chave u64 com dado junto: tem de bater, par a par, com o
            //     std::stable_sort -- ordem, estabilidade, perda e duplicacao.
            std::vector<Keyed> v(n);
            for (size_t i = 0; i < n; ++i) { v[i].key = chaves[i]; v[i].idx = i; }
            std::vector<Keyed> ref = v;
            std::stable_sort(ref.begin(), ref.end());
            std::vector<Keyed> w = v;
            multimerge::sort_by_radix_key(std::span<Keyed>(w), [](const Keyed& k) { return k.key; });
            const bool ok_linha = (w == ref);

            // (2) inteiros nus, com e sem sinal, largos e estreitos.
            const bool ok_int = confere_inteiros<uint64_t>(chaves) && confere_inteiros<int64_t>(chaves)
                             && confere_inteiros<uint32_t>(chaves) && confere_inteiros<int32_t>(chaves)
                             && confere_inteiros<int16_t>(chaves)  && confere_inteiros<uint8_t>(chaves);

            // (3) prefixo: texto de 12 bytes; o radix so ve os 8 primeiros.
            bool ok_prefixo = true;
            for (int livres : {8, 1, 0}) {
                std::vector<Texto> t(n);
                for (size_t i = 0; i < n; ++i) {
                    uint64_t z = chaves[i] * 0x9E3779B97F4A7C15ull + i;
                    for (int j = 0; j < 12; ++j) {
                        t[i].t[j] = 'A';
                        if (j >= 8 - livres) { t[i].t[j] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"[z % 36]; z /= 36; }
                    }
                    t[i].idx = (uint32_t)i;
                }
                std::vector<Texto> tref = t;
                std::stable_sort(tref.begin(), tref.end());
                multimerge::sort_by_prefix_key(std::span<Texto>(t), [](const Texto& x) {
                    uint64_t p = 0;
                    for (int j = 0; j < 8; ++j) p = (p << 8) | (unsigned char)x.t[j];
                    return p;
                });
                if (!(t == tref)) ok_prefixo = false;
            }

            const bool ok = ok_linha && ok_int && ok_prefixo;
            if (!ok) g_failures++;
            printf("  %-42s n=%-9zu linha:%-5s inteiros:%-5s prefixo:%-5s %s\n",
                   nome_formato(forma), n, ok_linha ? "OK" : "FALHA", ok_int ? "OK" : "FALHA",
                   ok_prefixo ? "OK" : "FALHA", ok ? "" : "  <<<<<< PROBLEMA");
        }

        // Tamanhos em volta dos limiares.
        int antes = g_failures;
        for (size_t m : {8193u, 32767u, 32768u, 32769u, 65535u, 65536u, 65537u, 131072u, 262145u}) {
            std::vector<uint64_t> c(m);
            for (auto& x : c) x = rng();
            if (!confere_inteiros<uint64_t>(c) || !confere_inteiros<int32_t>(c)) g_failures++;
            std::vector<Keyed> v(m);
            for (size_t i = 0; i < m; ++i) { v[i].key = c[i] % 5000; v[i].idx = i; }
            std::vector<Keyed> ref = v;  std::stable_sort(ref.begin(), ref.end());
            multimerge::sort_by_radix_key(std::span<Keyed>(v), [](const Keyed& k) { return k.key; });
            if (!(v == ref)) g_failures++;
        }
        printf("  %-42s %d falha(s)\n", "tamanhos em volta dos limiares", g_failures - antes);

#if defined(MULTIMERGE_STATS)
        // O radix entrou mesmo? Sem isto o teste passaria igual com a rotina
        // desligada, porque o caminho por comparacao da a mesma saida.
        {
            std::vector<uint64_t> c(n);
            for (auto& x : c) x = rng();
            multimerge::sort_radix(std::span<uint64_t>(c));
            const bool entrou = multimerge::detail::g_radix == 1;
            for (auto& x : c) x = rng();
            multimerge::sort(std::span<uint64_t>(c));
            const bool ficou_fora = multimerge::detail::g_radix == 0;
            if (!entrou || !ficou_fora) g_failures++;
            printf("  %-42s %s\n", "sort_radix usa o radix; sort nao usa",
                   (entrou && ficou_fora) ? "OK" : "FALHA  <<<<<< PROBLEMA");
        }
#endif
    }

    printf("\n===================================================\n");
    printf("RESULTADO: %d falha(s)\n", g_failures);
    printf("===================================================\n");
    return g_failures == 0 ? 0 : 1;
}
