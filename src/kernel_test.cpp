#include "MultiMergeSort.hpp"
#include <vector>
#include <random>
#include <algorithm>
#include <cstdio>
#include <chrono>

struct Keyed { uint64_t key; uint64_t idx; };
inline bool operator< (const Keyed&a,const Keyed&b){return a.key< b.key;}
inline bool operator<=(const Keyed&a,const Keyed&b){return a.key<=b.key;}
inline bool operator> (const Keyed&a,const Keyed&b){return a.key> b.key;}
inline bool operator>=(const Keyed&a,const Keyed&b){return a.key>=b.key;}
inline bool operator==(const Keyed&a,const Keyed&b){return a.key==b.key&&a.idx==b.idx;}

int main(){
    std::mt19937_64 rng(20260814);
    int fails = 0, trials = 0;

    // Ataca bidirectional_merge DIRETO, com total > leaf_size para forcar o split,
    // e dominio de chaves minusculo para saturar de empates.
    for (int t = 0; t < 4000; ++t) {
        size_t m = 1 + rng() % 9000;
        size_t n = 1 + rng() % 9000;
        uint64_t card = 1 + (rng() % 6);          // 1..6 chaves distintas: empates em massa
        std::vector<Keyed> a(m), b(n);
        uint64_t id = 0;
        for (auto&e:a) e = { rng()%card, id++ };
        for (auto&e:b) e = { rng()%card, id++ };
        std::stable_sort(a.begin(),a.end());
        std::stable_sort(b.begin(),b.end());

        std::vector<Keyed> expected; expected.reserve(m+n);
        std::merge(a.begin(),a.end(),b.begin(),b.end(),std::back_inserter(expected));

        std::vector<Keyed> got(m+n);
        auto aa=a, bb=b;
        #pragma omp parallel
        #pragma omp single
        multimerge::bidirectional_merge<Keyed>(std::span<Keyed>(aa), std::span<Keyed>(bb),
                                               std::span<Keyed>(got), 4096);
        trials++;
        if (got != expected) {
            if (++fails <= 3) {
                for (size_t i=0;i<got.size();++i)
                    if (!(got[i]==expected[i])) {
                        printf("  FALHA t=%d m=%zu n=%zu card=%llu @%zu: got{%llu,%llu} exp{%llu,%llu}\n",
                          t,m,n,(unsigned long long)card,i,
                          (unsigned long long)got[i].key,(unsigned long long)got[i].idx,
                          (unsigned long long)expected[i].key,(unsigned long long)expected[i].idx);
                        break;
                    }
            }
        }
    }
    printf("kernel bidirectional_merge: %d/%d falharam (empates pesados)\n", fails, trials);

    // ---- custo do co_rank removido: mede so o kernel, 1 thread ----
    const size_t M = 32768, N = 32768;
    std::vector<uint64_t> a(M), b(N), d(M+N);
    for (auto&x:a) x = rng(); for (auto&x:b) x = rng();
    std::sort(a.begin(),a.end()); std::sort(b.begin(),b.end());
    auto bench=[&](const char* nm){
        auto t0=std::chrono::high_resolution_clock::now();
        for(int r=0;r<3000;++r){
            auto aa=a, bb=b;
            multimerge::bidirectional_merge<uint64_t>(std::span<uint64_t>(aa),std::span<uint64_t>(bb),
                                                      std::span<uint64_t>(d),4096);
        }
        auto t1=std::chrono::high_resolution_clock::now();
        printf("%s: %.2f ms / 3000 merges de 65536 elem\n", nm,
               std::chrono::duration<double,std::milli>(t1-t0).count());
    };
    bench("tempo total (inclui copia dos inputs)");
    return fails? 1:0;
}
