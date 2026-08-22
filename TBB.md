# Camada opcional de TBB — como medir

O motor **não depende** da TBB. Onde ela existe, quatro pontos podem usá-la;
onde não existe, o caminho OpenMP original vale byte por byte. Cada ponto é um
interruptor separado, para ser medido isolado.

Isso importa porque nesta investigação a escolha do fallback caótico acabou
valendo mais que o algoritmo do próprio motor — e só descobrimos porque os
caminhos eram comparáveis um a um.

## Os quatro pontos

| variável | o que muda | custo se ligado |
| :--- | :--- | :--- |
| `MULTIMERGE_TBB_MALLOC` | linka `tbbmalloc_proxy`: troca `malloc`/`free` globais pelo alocador escalável | só link, zero código |
| `MULTIMERGE_TBB_SCHED` | árvore de merge com `tbb::parallel_invoke` em vez de `#pragma omp task` | dependência dura |
| `MULTIMERGE_TBB_REDUCE` | fold do metadata com `tbb::parallel_reduce` em vez de laço sequencial | dependência dura |
| `MULTIMERGE_PSTL` | fallback caótico com `std::execution::par` | dependência dura |

## Ordem sugerida de medição

**1. `MULTIMERGE_TBB_MALLOC` primeiro.** É o único que não muda código nenhum, e
tem a hipótese mais concreta: os `std::stable_sort` das folhas alocam **por
chamada** — cada bloco do `chunk_parallel_sort`, cada micro-bloco caótico do
`process_macro_block`. Com oito threads isso vira contenção no alocador do
sistema.

```bash
cd src
g++ -std=c++20 -O3 -fopenmp -DMULTIMERGE_PSTL -DMULTIMERGE_TEST_PAR \
    -DMULTIMERGE_USE_GNU_PARALLEL vs_stable.cpp -ltbb12 -o ../vs_base.exe
g++ -std=c++20 -O3 -fopenmp -DMULTIMERGE_PSTL -DMULTIMERGE_TEST_PAR \
    -DMULTIMERGE_USE_GNU_PARALLEL vs_stable.cpp -ltbb12 -ltbbmalloc_proxy \
    -o ../vs_malloc.exe
cd ..
./vs_base.exe 5000000
./vs_malloc.exe 5000000
```

Compare **dente de serra** e **quase ordenado**, que são os que passam pela Fase
1 com muitos `stable_sort` de folha. Os cenários aleatórios delegam e não
deveriam responder — servem de controle.

**2. `MULTIMERGE_TBB_SCHED` depois.** A árvore de merge é recursão fork-join,
que é exatamente o que o escalonador da TBB foi desenhado para fazer. A fila
dupla com LIFO local e FIFO no roubo mantém dados quentes em cache de um jeito
que a fila central do OpenMP não faz.

```bash
g++ -std=c++20 -O3 -fopenmp -DMULTIMERGE_PSTL -DMULTIMERGE_TEST_PAR \
    -DMULTIMERGE_USE_GNU_PARALLEL -DMULTIMERGE_TBB -DMULTIMERGE_TBB_SCHED \
    vs_stable.cpp -ltbb12 -o ../vs_sched.exe
```

**3. `MULTIMERGE_TBB_REDUCE` por último.** O fold é sequencial hoje, e a
operação é associativa — é o monoide que o README destaca, e ele existe
justamente para permitir redução em árvore. Mas para dado bem estruturado o
metadata é pequeno e o ganho é limitado. O cenário que responderia é dado com
**muitos runs curtos**.

## Pelo cargo

```powershell
$env:MULTIMERGE_TBB_MALLOC = "1"
cargo build --release
```

A linha que o `build.rs` imprime mostra o que está ativo:

```
tbb=off
tbb=malloc
tbb=sched+reduce+malloc
```

## O que NÃO fazer

Ligar tudo de uma vez. Já aconteceu duas vezes nesta investigação de uma
combinação ficar pior que qualquer um dos componentes isolados — as variáveis
interagem, e a otimização isolada não compõe.

E antes de aceitar qualquer resultado: olhe o **piso de ruído** que o
`vs_stable` imprime. Na rodada de 50M ele chegou a 25,8% em três cenários, e ali
nada abaixo disso era legível.
