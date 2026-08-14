#include "MultiMergeSort.hpp"
#include <vector>
#include <cstdio>
#include <chrono>
#include <algorithm>
int main(){
#ifdef MULTIMERGE_KWAY
    printf(">>> K-VIAS (K=%d)\n", multimerge::KWAY_K);
#else
    printf(">>> BINARIO\n");
#endif
    for (size_t n : {5000000ul, 20000000ul}) {
        std::vector<uint64_t> base(n);
        for(size_t i=0;i<n;++i) base[i]=i%1000;
        double best=1e18; bool ok=true;
        for(int r=0;r<3;++r){
            auto v=base;
            auto t0=std::chrono::high_resolution_clock::now();
            multimerge::sort(std::span<uint64_t>(v));
            double s=std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count();
            best=std::min(best,s);
            if(!std::is_sorted(v.begin(),v.end())) ok=false;
        }
        size_t runs=n/1000;
        int nb=(int)std::ceil(std::log2((double)runs)), nk=(int)std::ceil(std::log2((double)runs)/3.0);
        printf("  n=%-9zu runs=%-6zu  %7.1f ms   %6.1f Melem/s   niveis: bin=%d kway=%d   %s\n",
               n, runs, best*1e3, n/best/1e6, nb, nk, ok?"ok":"ERRO");
    }
    return 0;
}
