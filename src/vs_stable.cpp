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
// Compilar (UCRT64 do MSYS2, que e onde o `par` do C++17 engata):
//   g++ -std=c++20 -O3 -fopenmp -DMULTIMERGE_PSTL -DMULTIMERGE_TEST_PAR vs_stable.cpp -ltbb12 -o vs_stable
//
// O cabecalho ja liga o merge de K vias e o merge bidirecional, que e como o
// cargo compila o motor. (Ate 2026-11-18 era preciso pedir os dois por flag, e
// a tabela do README saiu durante meses de um binario sem eles: media a arvore
// binaria.) MULTIMERGE_PSTL manda o caminho caotico para o `par`; sem ele vale
// o chunk_parallel_sort interno. Para acrescentar a referencia __gnu_parallel:
// -DMULTIMERGE_USE_GNU_PARALLEL (que tambem TROCA o fallback do motor quando o
// PSTL nao esta ligado).
//
// DISCIPLINA. Uma medicao unica nao vale nada nesta carga: ja vimos 15 a 50%
// de variacao entre execucoes identicas. Aqui ha mediana de N repeticoes,
// ordem dos motores SORTEADA a cada rodada, buffer reutilizado, e uma LINHA DE
// CONTROLE que mede o mesmo motor duas vezes para expor o piso de ruido.
//
// A ordem e sorteada (semente fixa), nao girada: na rotacao ciclica cada motor
// roda sempre atras do mesmo vizinho, e em cenarios de poucos milissegundos
// isso fez sort_by_radix_key parecer 1 a 2 ms mais lento que sort no dado ja
// ordenado, onde os dois executam o mesmo codigo (medido em 2026-11-18).
//
//   vs_stable [n] [repeticoes]        padrao: 5000000 9

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
static int REPS = 9;

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

enum Motor { STD, PAR, GNU, MULTI, MULTI_CTRL, MULTI_RADIX, GNU_CTRL };

static const char* nome_motor(Motor m) {
    switch (m) {
        case STD:      return "std::stable_sort (1 thr)";
        case PAR:      return "std::stable_sort(par) C++17";
        case GNU:      return "__gnu_parallel::stable_sort";
        case GNU_CTRL: return "  ^ CONTROLE (mesmo motor)";
        case MULTI_CTRL:  return "  ^ CONTROLE (mesmo motor)";
        case MULTI_RADIX: return "multimerge::sort_by_radix_key";
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
        case MULTI_RADIX:
            // Mesma saida de multimerge::sort; o dado caotico vai por radix.
            multimerge::sort_by_radix_key(std::span<Keyed>(v), [](const Keyed& k) { return k.key; });
            break;
        default: multimerge::sort(std::span<Keyed>(v)); break;
    }
}

/// Gera `n` elementos. `card` = numero de chaves distintas.
/// `forma`: 0 aleatorio, 1 dente de serra, 2 quase ordenado, 3 ordenado, 4 invertido,
///          5 dentes em faixas disjuntas (ordenado, com cada bloco impar invertido;
///            1000 dentes -- o "Sawtooth" dos benchmarks Java e Python),
///          6 o mesmo, com UMA chave repetida no topo de um dente descendente
/// `card` = 0 no aleatorio quer dizer a faixa inteira de 64 bits.
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
        case 5: case 6: {
            const size_t dente = std::max<size_t>(n / 1000, 2);
            for (size_t i = 0; i < n; ++i) {
                const size_t d = i / dente;
                const size_t len = std::min(dente, n - d * dente);
                size_t k = i % dente;
                if (d % 2 == 1) k = len - 1 - k;
                v[i].key = (uint64_t)(d * dente + k);
            }
            if (forma == 6 && n > 502 * dente + 1) v[501 * dente + 1].key = v[501 * dente].key;
            break;
        }
        default: for (auto& e : v) e.key = card ? rng() % card : rng(); break;
    }
    for (size_t i = 0; i < n; ++i) v[i].idx = i;
    return v;
}

static void roda(const std::string& nome, const std::vector<Keyed>& base) {
    const size_t n = base.size();

    std::vector<Keyed> ref = base;
    std::stable_sort(ref.begin(), ref.end());

    std::vector<Motor> motores = {STD, MULTI, MULTI_CTRL, MULTI_RADIX};
#if defined(MULTIMERGE_TEST_PAR)
    motores.insert(motores.begin() + 1, PAR);
#endif
#if defined(MULTIMERGE_USE_GNU_PARALLEL)
    motores.push_back(GNU);
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

    std::vector<size_t> ordem(motores.size());
    for (size_t i = 0; i < ordem.size(); ++i) ordem[i] = i;
    std::mt19937 sorteio(0x0DE5u + (unsigned)n);

    for (int rep = 0; rep < REPS; ++rep) {
        std::shuffle(ordem.begin(), ordem.end(), sorteio);
        for (size_t slot = 0; slot < motores.size(); ++slot) {
            size_t id = ordem[slot];
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
    size_t i_mctrl = (size_t)-1, i_radix = (size_t)-1;
    for (size_t i = 0; i < motores.size(); ++i) {
        if (motores[i] == STD)         i_std = i;
        if (motores[i] == PAR)         i_par = i;
        if (motores[i] == GNU)         i_gnu = i;
        if (motores[i] == MULTI)       i_multi = i;
        if (motores[i] == GNU_CTRL)    i_ctrl = i;
        if (motores[i] == MULTI_CTRL)  i_mctrl = i;
        if (motores[i] == MULTI_RADIX) i_radix = i;
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
    if (i_par != (size_t)-1) {
        printf("  multimerge::sort contra o par: %.2fx", med[i_par] / med[i_multi]);
        if (i_radix != (size_t)-1) printf("   |   com chave radix: %.2fx", med[i_par] / med[i_radix]);
        printf("\n");
    }
    if (i_mctrl != (size_t)-1) {
        // O mesmo motor medido duas vezes em cada rodada: a diferenca entre
        // as duas medianas e o piso de ruido desta tabela.
        double ruido = 100.0 * std::abs(med[i_mctrl] - med[i_multi]) / med[i_multi];
        printf("  PISO DE RUIDO: %.1f%%", ruido);
        if (ruido > 10.0) printf("   <- ALTO: diferencas menores que isto nao sao legiveis");
        printf("\n");
    } else if (i_ctrl != (size_t)-1) {
        double ruido = 100.0 * std::abs(med[i_ctrl] - med[i_gnu]) / med[i_gnu];
        printf("  PISO DE RUIDO: %.1f%%", ruido);
        if (ruido > 10.0) printf("   <- ALTO: diferencas menores que isto nao sao legiveis");
        printf("\n");
    }
}

int main(int argc, char** argv) {
    size_t n = (argc > 1) ? std::stoull(argv[1]) : 5'000'000;
    if (argc > 2) REPS = std::max(1, std::stoi(argv[2]));

    printf("multimerge::sort CONTRA OS SORTS ESTAVEIS DA STD\n");
    printf("%d threads OpenMP | mediana de %d, ordem sorteada a cada rodada, aquecimento descartado\n",
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
    roda("aleatorio, chave de 64 bits",      gera(n,          0, 0, rng));
    roda("aleatorio, cardinalidade alta",    gera(n, 10'000'000, 0, rng));
    roda("aleatorio, cardinalidade 64",      gera(n,         64, 0, rng));
    roda("aleatorio, cardinalidade 10",      gera(n,         10, 0, rng));
    roda("dente de serra (i % 1000)",        gera(n,          0, 1, rng));
    roda("dentes em faixas disjuntas",       gera(n,          0, 5, rng));
    roda("idem, uma chave repetida no topo", gera(n,          0, 6, rng));
    roda("quase ordenado (0,1% de ruido)",   gera(n,          0, 2, rng));
    roda("ja ordenado",                      gera(n,          0, 3, rng));
    roda("invertido",                        gera(n,          0, 4, rng));

    printf("\nOs seis ultimos cenarios sao ESTRUTURADOS: e onde o motor ganha por\n");
    printf("detectar a estrutura, e ali sort e sort_by_radix_key fazem o mesmo\n");
    printf("caminho. Os quatro primeiros passam pelo escudo de entropia: sort cai\n");
    printf("no fallback por comparacao, sort_by_radix_key vai por radix.\n");
    return 0;
}
