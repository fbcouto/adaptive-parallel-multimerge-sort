// multimerge::sort CONTRA os sorts estaveis da biblioteca padrao
//
// A pergunta: o motor paga o proprio custo de deteccao?
//
// AVISO SOBRE A COMPARACAO. std::stable_sort roda em UMA thread; o multimerge
// usa todas. Comparar os dois diretamente mede paralelismo, nao algoritmo. Por
// isso a tabela traz as duas leituras:
//
//   vs std   = ganho total, inclui o paralelismo (o numero "de folder")
//   vs gnu   = ganho contra um sort estavel PARALELO, mesmo numero de threads
//              (esta e a comparacao honesta do algoritmo)
//
// Compilar com a referencia paralela:
//   g++ -std=c++20 -O3 -fopenmp -DMULTIMERGE_USE_GNU_PARALLEL vs_stable.cpp -o vs_stable
//
// DISCIPLINA. Uma medicao unica nao vale nada nesta carga: ja vimos 15 a 50%
// de variacao entre execucoes identicas. Aqui ha mediana de N repeticoes,
// ordem rotativa entre motores, buffer reutilizado, e uma LINHA DE CONTROLE
// que mede o mesmo motor duas vezes para expor o piso de ruido.

#include "MultiMergeSort.hpp"
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <span>

#if defined(MULTIMERGE_USE_GNU_PARALLEL)
#include <parallel/algorithm>
#endif

// Politicas de execucao do C++17. Com libstdc++ elas EXIGEM a Intel TBB
// lincada; sem ela o compilador aceita o codigo e roda em UMA THREAD, em
// silencio. Por isso a tabela compara o tempo do 'par' com o do sequencial e
// avisa quando os dois batem -- ver a checagem no fim de roda().
//
//   MSYS2:   pacman -S mingw-w64-x86_64-tbb
//   Debian:  sudo apt install libtbb-dev
//   compilar com -ltbb no fim da linha
#if defined(MULTIMERGE_TEST_PAR)
#include <execution>
#endif

using clk = std::chrono::high_resolution_clock;
static const int REPS = 9;

struct Keyed {
    uint64_t key;
    uint64_t idx;   // posicao original: nunca participa da comparacao
};
inline bool operator< (const Keyed& a, const Keyed& b) { return a.key <  b.key; }
inline bool operator<=(const Keyed& a, const Keyed& b) { return a.key <= b.key; }
inline bool operator> (const Keyed& a, const Keyed& b) { return a.key >  b.key; }
inline bool operator>=(const Keyed& a, const Keyed& b) { return a.key >= b.key; }
inline bool operator==(const Keyed& a, const Keyed& b) { return a.key == b.key && a.idx == b.idx; }

/// Ordenado por chave E com idx crescente dentro de cada grupo de chaves iguais.
static bool estavel(const std::vector<Keyed>& v) {
    for (size_t i = 1; i < v.size(); ++i) {
        if (v[i-1].key > v[i].key) return false;
        if (v[i-1].key == v[i].key && v[i-1].idx > v[i].idx) return false;
    }
    return true;
}

static double mediana(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

enum Motor { STD, PAR, GNU, MULTI, GNU_CTRL };

static const char* nome_motor(Motor m) {
    switch (m) {
        case STD:      return "std::stable_sort (1 thr)";
        case PAR:      return "std::stable_sort(par) C++17";
        case GNU:      return "__gnu_parallel::stable_sort";
        case GNU_CTRL: return "  ^ CONTROLE (mesmo motor)";
        default:       return "multimerge::sort";
    }
}

static void aplica(Motor m, std::vector<Keyed>& v) {
    switch (m) {
        case STD: std::stable_sort(v.begin(), v.end()); break;
        case PAR:
#if defined(MULTIMERGE_TEST_PAR)
            std::stable_sort(std::execution::par, v.begin(), v.end());
#else
            std::stable_sort(v.begin(), v.end());
#endif
            break;
        case GNU: case GNU_CTRL:
#if defined(MULTIMERGE_USE_GNU_PARALLEL)
            __gnu_parallel::stable_sort(v.begin(), v.end());
#else
            std::stable_sort(v.begin(), v.end());
#endif
            break;
        default: multimerge::sort(std::span<Keyed>(v)); break;
    }
}

/// Gera `n` elementos. `card` = numero de chaves distintas.
/// `forma`: 0 aleatorio, 1 dente de serra, 2 quase ordenado, 3 ordenado, 4 invertido
static std::vector<Keyed> gera(size_t n, uint64_t card, int forma, std::mt19937_64& rng) {
    std::vector<Keyed> v(n);
    switch (forma) {
        case 1: for (size_t i = 0; i < n; ++i) v[i].key = (uint64_t)(i % 1000); break;
        case 2:
            for (size_t i = 0; i < n; ++i) v[i].key = (uint64_t)i;
            for (size_t p = 0; p < n / 1000; ++p) v[rng() % n].key = rng() % n;
            break;
        case 3: for (size_t i = 0; i < n; ++i) v[i].key = (uint64_t)i; break;
        case 4: for (size_t i = 0; i < n; ++i) v[i].key = (uint64_t)(n - i); break;
        default: for (auto& e : v) e.key = rng() % card; break;
    }
    for (size_t i = 0; i < n; ++i) v[i].idx = i;
    return v;
}

static void roda(const std::string& nome, const std::vector<Keyed>& base) {
    const size_t n = base.size();

    std::vector<Keyed> ref = base;
    std::stable_sort(ref.begin(), ref.end());

    std::vector<Motor> motores = {STD, MULTI};
#if defined(MULTIMERGE_TEST_PAR)
    motores.insert(motores.begin() + 1, PAR);
#endif
#if defined(MULTIMERGE_USE_GNU_PARALLEL)
    motores.insert(motores.end() - 1, GNU);
    motores.push_back(GNU_CTRL);
#endif

    std::vector<std::vector<double>> t(motores.size());
    std::vector<bool> ok(motores.size(), true), est(motores.size(), true);
    std::vector<Keyed> v(n);

    // uma passada de aquecimento fora da conta: paga as faltas de pagina
    for (size_t i = 0; i < motores.size(); ++i) {
        std::copy(base.begin(), base.end(), v.begin());
        aplica(motores[i], v);
    }

    for (int rep = 0; rep < REPS; ++rep) {
        for (size_t slot = 0; slot < motores.size(); ++slot) {
            size_t id = (rep + slot) % motores.size();
            std::copy(base.begin(), base.end(), v.begin());   // fora do cronometro
            auto t0 = clk::now();
            aplica(motores[id], v);
            t[id].push_back(std::chrono::duration<double, std::milli>(clk::now() - t0).count());
            if (v != ref)     ok[id]  = false;
            if (!estavel(v))  est[id] = false;
        }
    }

    std::vector<double> med(motores.size());
    for (size_t i = 0; i < motores.size(); ++i) med[i] = mediana(t[i]);

    size_t i_std = 0, i_gnu = (size_t)-1, i_multi = 0, i_ctrl = (size_t)-1, i_par = (size_t)-1;
    for (size_t i = 0; i < motores.size(); ++i) {
        if (motores[i] == STD)      i_std = i;
        if (motores[i] == PAR)      i_par = i;
        if (motores[i] == GNU)      i_gnu = i;
        if (motores[i] == MULTI)    i_multi = i;
        if (motores[i] == GNU_CTRL) i_ctrl = i;
    }

    printf("\n=== %s ===  n=%zu  threads=%d\n", nome.c_str(), n, omp_get_max_threads());
    printf("  %-30s %10s %9s %9s %8s %9s\n",
           "motor", "mediana", "vs std", "vs gnu", "exato", "estavel");
    printf("  %s\n", std::string(82, '-').c_str());
    for (size_t i = 0; i < motores.size(); ++i) {
        char vs_gnu[16] = "     -";
        if (i_gnu != (size_t)-1)
            snprintf(vs_gnu, sizeof vs_gnu, "%+8.1f%%", 100.0 * (med[i] - med[i_gnu]) / med[i_gnu]);
        printf("  %-30s %8.1f ms %+8.1f%% %s %8s %9s\n",
               nome_motor(motores[i]), med[i],
               100.0 * (med[i] - med[i_std]) / med[i_std], vs_gnu,
               ok[i] ? "OK" : "FALHA", est[i] ? "OK" : "FALHA");
    }
    printf("  aceleracao do multimerge sobre std::stable_sort: %.2fx\n",
           med[i_std] / med[i_multi]);
    if (i_par != (size_t)-1) {
        double razao = med[i_std] / med[i_par];
        printf("  aceleracao do C++17 par sobre std::stable_sort:   %.2fx", razao);
        if (razao < 1.25) {
            printf("   <- SUSPEITO: a politica 'par' esta SERIALIZANDO.\n");
            printf("     Com libstdc++ ela exige a Intel TBB lincada (-ltbb); sem ela o\n");
            printf("     codigo compila e roda em uma thread so, sem avisar. Esta linha\n");
            printf("     nao mede paralelismo do C++17, mede o sequencial de novo.");
        }
        printf("\n");
    }
    if (i_ctrl != (size_t)-1) {
        double ruido = 100.0 * std::abs(med[i_ctrl] - med[i_gnu]) / med[i_gnu];
        printf("  PISO DE RUIDO: %.1f%%", ruido);
        if (ruido > 10.0) printf("   <- ALTO: diferencas menores que isto nao sao legiveis");
        printf("\n");
    }
}

int main(int argc, char** argv) {
    size_t n = (argc > 1) ? std::stoull(argv[1]) : 5'000'000;

    printf("multimerge::sort CONTRA OS SORTS ESTAVEIS DA STD\n");
    printf("%d threads OpenMP | mediana de %d, ordem rotativa, aquecimento descartado\n",
           omp_get_max_threads(), REPS);
#if defined(MULTIMERGE_USE_GNU_PARALLEL)
    printf("referencia paralela: __gnu_parallel::stable_sort DISPONIVEL\n");
#else
    printf("AVISO: sem -DMULTIMERGE_USE_GNU_PARALLEL nao ha referencia PARALELA.\n");
    printf("A coluna 'vs std' compara 8 threads contra 1 e mede paralelismo,\n");
    printf("nao algoritmo. Recompile com a flag para a comparacao honesta.\n");
#endif
    printf("\nExato   = saida identica ao std::stable_sort (pega perda e duplicacao)\n");
    printf("Estavel = chaves iguais mantem a ordem original de idx\n");

    std::mt19937_64 rng(42);
    roda("aleatorio, cardinalidade alta",   gera(n, 10'000'000, 0, rng));
    roda("aleatorio, cardinalidade 10",     gera(n,         10, 0, rng));
    roda("aleatorio, cardinalidade 64",     gera(n,         64, 0, rng));
    roda("dente de serra (i % 1000)",       gera(n,          0, 1, rng));
    roda("quase ordenado (0,1% de ruido)",  gera(n,          0, 2, rng));
    roda("ja ordenado",                     gera(n,          0, 3, rng));
    roda("invertido",                       gera(n,          0, 4, rng));

    printf("\nOs quatro ultimos cenarios sao ESTRUTURADOS: e onde o motor deveria\n");
    printf("ganhar, porque ha estrutura para detectar. Os tres primeiros passam\n");
    printf("pelo escudo de entropia e caem no fallback, entao medem o fallback.\n");
    return 0;
}
