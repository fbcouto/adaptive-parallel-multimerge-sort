// ============================================================
// SEIS CONTENDORES, O MESMO DADO, O MESMO PROCESSO
//
//   ESTAVEIS
//   1_Rayon_par_sort          rayon par_sort                        Rust
//   2_MultiMerge_Stable_Rust  este motor, sort_stable               Rust
//   3_MultiMerge_Cpp          este motor                            C++20 / OpenMP
//   4_PSTL_TBB_Cpp            std::stable_sort(execution::par, ...) C++ / TBB
//
//   INSTAVEIS
//   5_Rayon_par_sort_unstable rayon par_sort_unstable               Rust
//   6_MultiMerge_Rust         este motor, sort (o padrao)           Rust
//
// Compare dentro de cada bloco. Um sort instavel tem liberdade que o estavel
// nao tem, e cruzar os dois blocos mede essa liberdade, nao o motor.
//
// O quarto e a referencia mais forte disponivel, e a unica que precisa ser
// verificada antes de valer: sem uma libstdc++ com PSTL E a TBB lincada os
// overloads `par` compilam e rodam SERIAL, em silencio. Por isso o bench mede a
// razao seq/par em 5M antes de qualquer coisa e RECUSA a rodada abaixo de 2x.
//
// Tamanhos e cenarios por ambiente:
//   MM_SIZES=1000000,3000000,5000000     (padrao)
//   MM_TBB_THREADS=8                     teto de paralelismo da TBB
//   MM_LARGA=1                           liga os cenarios de chave de 16 bytes
//
// MM_TBB_THREADS nao e opcional na pratica: este processo mantem TRES pools de
// threads -- Rayon, OpenMP e TBB -- e so as duas primeiras leem variaveis de
// ambiente. Sem o teto a TBB abre hardware_concurrency() por cima das outras
// duas, e em 8 nucleos sao 24 threads disputando a maquina.
//
// Texto: o lado Rust ordena `&str` (ponteiro gordo de 16 bytes, memcmp com
// comprimento conhecido); o lado C++ ordena `const char*` (8 bytes, strcmp).
// Sao as representacoes idiomaticas de cada lado e a TAREFA e identica -- mas o
// dado movido nao e, e a coluna de C++ move metade dos bytes. Leia as duas
// linguagens como blocos separados, nao como uma corrida direta.
// ============================================================
use adaptive_parallel_multimerge_sort as mm;
use criterion::measurement::WallTime;
use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkGroup, BenchmarkId, Criterion,
    Throughput,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::ffi::CString;
use std::os::raw::c_char;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

const SEG_MEDICAO: u64 = 10;

/// Menos amostras quando cada iteracao ja custa quase um segundo. O piso do
/// criterion e 10; abaixo disso ele recusa.
fn amostras(n: usize) -> usize {
    if n >= 3_000_000 { 10 } else { 20 }
}

/// `LargeInput` deixa o criterion agrupar varias copias antes de medir; a
/// partir de 3M cada copia ja e dezenas de MB (80 MB no texto do lado Rust, que
/// carrega ponteiro gordo) e o agrupamento estoura a RAM antes do relogio.
fn lote(n: usize) -> BatchSize {
    if n >= 3_000_000 { BatchSize::PerIteration } else { BatchSize::LargeInput }
}

fn tamanhos() -> Vec<usize> {
    std::env::var("MM_SIZES")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect::<Vec<_>>())
        .filter(|v: &Vec<usize>| !v.is_empty())
        .unwrap_or_else(|| vec![1_000_000, 3_000_000, 5_000_000])
}

fn ligado(nome: &str) -> bool {
    std::env::var(nome).map(|v| !v.is_empty() && v != "0").unwrap_or(false)
}

fn rng(semente: u64) -> StdRng {
    StdRng::seed_from_u64(semente)
}

// ============================================================
// PREPARO: as tres pools de threads deste processo
// ============================================================

/// Fecha o terceiro lado do paralelismo antes de qualquer medicao.
///
/// `RAYON_NUM_THREADS` e `OMP_NUM_THREADS` fixam dois dos tres motores de
/// thread deste processo. O terceiro e a TBB, que nao le nenhuma das duas e
/// abre `hardware_concurrency()` sozinha -- em 8 nucleos isso vira 24 threads
/// disputando a maquina, e sobre arrays grandes e o que a trava.
///
/// Roda UMA vez, antes do primeiro sort, porque o limite so vale para as pools
/// que a TBB abrir depois dele.
fn trava_tbb() {
    static UMA_VEZ: OnceLock<()> = OnceLock::new();
    UMA_VEZ.get_or_init(|| {
        let n: i32 = std::env::var("MM_TBB_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        if n <= 0 {
            eprintln!(
                "\n[TBB] MM_TBB_THREADS nao definido -- a TBB abre \
                 hardware_concurrency() por conta propria,\n\
                 [TBB] somando as {} threads do Rayon e as do OpenMP. \
                 Defina MM_TBB_THREADS para fixar.\n",
                rayon::current_num_threads()
            );
        } else if mm::tbb_limit(n) {
            eprintln!("\n[TBB] paralelismo limitado a {n} threads.");
        } else {
            eprintln!(
                "\n[TBB] esta build nao tem <tbb/global_control.h>: \
                 MM_TBB_THREADS={n} IGNORADO.\n"
            );
        }
    });
}

// ============================================================
// O PORTAO: o `par` engatou?
// ============================================================

/// Roda uma vez, na primeira chamada, e imprime o veredito. Sem PSTL compilado
/// o contendor 4 e simplesmente omitido; com PSTL compilado mas serial de fato,
/// o bench aborta -- publicar um numero contra um `par` que nao paralelizou
/// seria pior que nao publicar nada.
fn pstl_disponivel() -> bool {
    static PORTAO: OnceLock<bool> = OnceLock::new();
    *PORTAO.get_or_init(|| {
        trava_tbb();
        if !mm::has_pstl() {
            eprintln!(
                "\n[PSTL] compilado SEM std::execution::par -- contendor 4 omitido.\n\
                 [PSTL] para inclui-lo: MULTIMERGE_PSTL=1 com o g++ do UCRT64 e -ltbb12.\n"
            );
            return false;
        }

        const N: usize = 5_000_000;
        let mut r = rng(0xBEEF_0001);
        let base: Vec<u64> = (0..N).map(|_| r.gen()).collect();

        // Duas rodadas: a primeira aquece o alocador e a pool de threads da TBB
        // e e descartada. Uma medicao unica nao vale nada nesta carga.
        let mut seq = 0.0;
        let mut par = 0.0;
        for _ in 0..2 {
            let mut d = base.clone();
            let t = Instant::now();
            mm::seq_sort_u64(&mut d);
            seq = t.elapsed().as_secs_f64() * 1e3;

            let mut d = base.clone();
            let t = Instant::now();
            mm::pstl_sort_u64(&mut d);
            par = t.elapsed().as_secs_f64() * 1e3;
        }

        let ganho = seq / par;
        eprintln!(
            "\n[PSTL] 5M aleatorio: seq={seq:.1} ms  par={par:.1} ms  ganho={ganho:.2}x  \
             ({} threads Rayon)",
            rayon::current_num_threads()
        );
        if ganho < 2.0 {
            panic!(
                "\n[PSTL] o `par` NAO engatou ({ganho:.2}x, esperado >= 2x).\n\
                 Os overloads existem mas rodaram serial. No MSYS2 isso e o\n\
                 MINGW64: medido 0,95x la contra 3,62x no UCRT64. Compile com\n\
                 CXX=/c/msys64/ucrt64/bin/g++.exe, ou rode sem MULTIMERGE_PSTL\n\
                 para omitir o contendor 4.\n"
            );
        }
        eprintln!("[PSTL] contendor 4 habilitado.\n");
        true
    })
}

// ============================================================
// CENARIOS u64
// ============================================================

fn ordenado(n: usize) -> Vec<u64> {
    (0..n as u64).collect()
}

fn invertido(n: usize) -> Vec<u64> {
    (0..n as u64).rev().collect()
}

fn aleatorio(n: usize) -> Vec<u64> {
    let mut r = rng(0xA11E_A701);
    (0..n).map(|_| r.gen()).collect()
}

/// 1% dos elementos trocados de lugar. E o cenario mais duro para um motor
/// adaptativo: ha ordem suficiente para a deteccao gastar tempo procurando, e
/// nao o bastante para ela pagar.
fn quase_ordenado(n: usize) -> Vec<u64> {
    let mut d: Vec<u64> = (0..n as u64).collect();
    let mut r = rng(0x9A5E_0001);
    for _ in 0..(n / 100) {
        let a = r.gen_range(0..n);
        let b = r.gen_range(0..n);
        d.swap(a, b);
    }
    d
}

fn dente_de_serra(n: usize) -> Vec<u64> {
    let mut r = rng(0x5A57_0074);
    let dentes = 1000;
    let por_dente = (n / dentes).max(1);
    let mut d = Vec::with_capacity(n);
    for i in 0..dentes {
        let ini = (i * por_dente) as u64;
        let fim = ini + por_dente as u64;
        if r.gen_bool(0.5) {
            d.extend(ini..fim);
        } else {
            d.extend((ini..fim).rev());
        }
    }
    let feitos = dentes * por_dente;
    if feitos < n {
        d.extend((feitos as u64)..(n as u64));
    }
    d.truncate(n);
    d
}

fn baixa_cardinalidade(n: usize) -> Vec<u64> {
    let mut r = rng(0x10CA_4D1A);
    (0..n).map(|_| r.gen_range(0..64u64)).collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ChaveLarga {
    chave: u64,
    carga: u64,
}
impl PartialOrd for ChaveLarga {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for ChaveLarga {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.chave.cmp(&o.chave)
    }
}
fn larga_de(v: Vec<u64>) -> Vec<ChaveLarga> {
    v.into_iter()
        .enumerate()
        .map(|(i, c)| ChaveLarga { chave: c, carga: i as u64 })
        .collect()
}

// ============================================================
// ARENA DE TEXTO
//
// Uma unica arena de CString vazada ('static) alimenta OS DOIS lados: o Rust
// recebe `&'static str` e o C++ recebe `*const c_char` para exatamente os
// mesmos bytes, na mesma ordem. Sem isso os dois lados ordenariam datasets
// diferentes e a comparacao nao diria nada.
//
// O texto e ASCII sem NUL interno, entao a ordem lexicografica por memcmp do
// Rust e a ordem por strcmp do C++ coincidem elemento a elemento.
// ============================================================

struct Arena {
    strs: Vec<&'static str>,
    ptrs: Vec<*const c_char>,
}

// Os ponteiros apontam para memoria vazada, imutavel e viva pelo resto do
// processo. Nada aqui e escrito depois da construcao.
unsafe impl Sync for Arena {}
unsafe impl Send for Arena {}

impl Arena {
    fn nova(valores: Vec<String>) -> &'static Arena {
        let cstrings: Vec<CString> = valores
            .into_iter()
            .map(|s| CString::new(s).expect("texto gerado nao contem NUL"))
            .collect();
        let cstrings: &'static [CString] = Box::leak(cstrings.into_boxed_slice());
        let strs = cstrings.iter().map(|c| c.to_str().unwrap()).collect();
        let ptrs = cstrings.iter().map(|c| c.as_ptr()).collect();
        Box::leak(Box::new(Arena { strs, ptrs }))
    }
}

fn arena_cromossomos() -> &'static Arena {
    static A: OnceLock<&'static Arena> = OnceLock::new();
    A.get_or_init(|| {
        let v = [
            "chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10",
            "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19",
            "chr20", "chr21", "chr22", "chrX", "chrY", "chrM",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        Arena::nova(v)
    })
}

fn arena_sequencias() -> &'static Arena {
    static A: OnceLock<&'static Arena> = OnceLock::new();
    A.get_or_init(|| {
        let mut r = rng(0x5EED_0001);
        const ALFABETO: [u8; 4] = *b"ACGT";
        let v: Vec<String> = (0..262_144)
            .map(|_| {
                let len = r.gen_range(5..200);
                (0..len).map(|_| ALFABETO[r.gen_range(0..4)] as char).collect()
            })
            .collect();
        Arena::nova(v)
    })
}

/// Indices sorteados sobre a arena. Os dois lados consomem esta MESMA lista.
fn indices(n: usize, cardinalidade: usize, semente: u64) -> Vec<usize> {
    let mut r = rng(semente);
    (0..n).map(|_| r.gen_range(0..cardinalidade)).collect()
}

// ============================================================
// MEDICAO
// ============================================================

fn grupo<'a>(c: &'a mut Criterion, cenario: &str, n: usize) -> BenchmarkGroup<'a, WallTime> {
    let mut g = c.benchmark_group(cenario);
    g.sample_size(amostras(n));
    g.measurement_time(Duration::from_secs(SEG_MEDICAO));
    g.throughput(Throughput::Elements(n as u64));
    g
}

fn mede_u64(c: &mut Criterion, cenario: &str, n: usize, base: &[u64]) {
    let mut g = grupo(c, cenario, n);
    g.bench_with_input(BenchmarkId::new("1_Rayon_par_sort", n), &n, |b, _| {
        b.iter_batched(|| base.to_vec(), |mut d| d.par_sort(), lote(n))
    });
    g.bench_with_input(BenchmarkId::new("2_MultiMerge_Stable_Rust", n), &n, |b, _| {
        b.iter_batched(|| base.to_vec(), |mut d| mm::sort_stable(black_box(&mut d)), lote(n))
    });
    g.bench_with_input(BenchmarkId::new("3_MultiMerge_Cpp", n), &n, |b, _| {
        b.iter_batched(|| base.to_vec(), |mut d| mm::cpp_sort_u64(black_box(&mut d)), lote(n))
    });
    if pstl_disponivel() {
        g.bench_with_input(BenchmarkId::new("4_PSTL_TBB_Cpp", n), &n, |b, _| {
            b.iter_batched(|| base.to_vec(), |mut d| mm::pstl_sort_u64(black_box(&mut d)), lote(n))
        });
    }
    g.bench_with_input(BenchmarkId::new("5_Rayon_par_sort_unstable", n), &n, |b, _| {
        b.iter_batched(|| base.to_vec(), |mut d| d.par_sort_unstable(), lote(n))
    });
    g.bench_with_input(BenchmarkId::new("6_MultiMerge_Rust", n), &n, |b, _| {
        b.iter_batched(|| base.to_vec(), |mut d| mm::sort(black_box(&mut d)), lote(n))
    });
    // As entradas com chave radix: mesma saida das de cima, e o dado caotico
    // deixa de passar por comparacao. Nos cenarios estruturados fazem o mesmo
    // caminho que 6, 2 e 3 -- servem de linha de controle.
    g.bench_with_input(BenchmarkId::new("7_MultiMerge_Rust_Radix", n), &n, |b, _| {
        b.iter_batched(|| base.to_vec(), |mut d| mm::sort_radix(black_box(&mut d)), lote(n))
    });
    g.bench_with_input(BenchmarkId::new("8_MultiMerge_Stable_Rust_Radix", n), &n, |b, _| {
        b.iter_batched(|| base.to_vec(), |mut d| mm::sort_stable_radix(black_box(&mut d)), lote(n))
    });
    g.bench_with_input(BenchmarkId::new("9_MultiMerge_Cpp_Radix", n), &n, |b, _| {
        b.iter_batched(|| base.to_vec(), |mut d| mm::cpp_sort_radix_u64(black_box(&mut d)), lote(n))
    });
    g.finish();
}

/// So os dois contendores Rust: a chave de 16 bytes nao tem ponte FFI.
fn mede_rust<T: Ord + Copy + Send + Sync + 'static>(
    c: &mut Criterion,
    cenario: &str,
    n: usize,
    base: &[T],
) {
    let mut g = grupo(c, cenario, n);
    g.bench_with_input(BenchmarkId::new("1_Rayon_par_sort", n), &n, |b, _| {
        b.iter_batched(|| base.to_vec(), |mut d| d.par_sort(), lote(n))
    });
    g.bench_with_input(BenchmarkId::new("2_MultiMerge_Stable_Rust", n), &n, |b, _| {
        b.iter_batched(|| base.to_vec(), |mut d| mm::sort_stable(black_box(&mut d)), lote(n))
    });
    g.bench_with_input(BenchmarkId::new("5_Rayon_par_sort_unstable", n), &n, |b, _| {
        b.iter_batched(|| base.to_vec(), |mut d| d.par_sort_unstable(), lote(n))
    });
    g.bench_with_input(BenchmarkId::new("6_MultiMerge_Rust", n), &n, |b, _| {
        b.iter_batched(|| base.to_vec(), |mut d| mm::sort(black_box(&mut d)), lote(n))
    });
    g.finish();
}

fn mede_texto(c: &mut Criterion, cenario: &str, n: usize, arena: &'static Arena, idx: &[usize]) {
    let rust_base: Vec<&'static str> = idx.iter().map(|&i| arena.strs[i]).collect();
    let cpp_base: Vec<*const c_char> = idx.iter().map(|&i| arena.ptrs[i]).collect();

    let mut g = grupo(c, cenario, n);
    g.bench_with_input(BenchmarkId::new("1_Rayon_par_sort", n), &n, |b, _| {
        b.iter_batched(|| rust_base.to_vec(), |mut d| d.par_sort(), lote(n))
    });
    g.bench_with_input(BenchmarkId::new("2_MultiMerge_Stable_Rust", n), &n, |b, _| {
        b.iter_batched(|| rust_base.to_vec(), |mut d| mm::sort_stable(black_box(&mut d)), lote(n))
    });
    g.bench_with_input(BenchmarkId::new("3_MultiMerge_Cpp", n), &n, |b, _| {
        b.iter_batched(
            || cpp_base.to_vec(),
            |mut d| unsafe { mm::cpp_sort_str(black_box(&mut d)) },
            lote(n),
        )
    });
    if pstl_disponivel() {
        g.bench_with_input(BenchmarkId::new("4_PSTL_TBB_Cpp", n), &n, |b, _| {
            b.iter_batched(
                || cpp_base.to_vec(),
                |mut d| unsafe { mm::pstl_sort_str(black_box(&mut d)) },
                lote(n),
            )
        });
    }
    g.bench_with_input(BenchmarkId::new("5_Rayon_par_sort_unstable", n), &n, |b, _| {
        b.iter_batched(|| rust_base.to_vec(), |mut d| d.par_sort_unstable(), lote(n))
    });
    g.bench_with_input(BenchmarkId::new("6_MultiMerge_Rust", n), &n, |b, _| {
        b.iter_batched(|| rust_base.to_vec(), |mut d| mm::sort(black_box(&mut d)), lote(n))
    });
    g.finish();
}

// ============================================================
// SUITES
// ============================================================

fn bench_u64(c: &mut Criterion) {
    trava_tbb();
    let cenarios: &[(&str, fn(usize) -> Vec<u64>)] = &[
        ("u64/Aleatorio", aleatorio),
        ("u64/Invertido", invertido),
        ("u64/Ordenado", ordenado),
        ("u64/QuaseOrdenado", quase_ordenado),
        ("u64/DenteDeSerra", dente_de_serra),
        ("u64/BaixaCardinalidade", baixa_cardinalidade),
    ];
    for &(nome, gera) in cenarios {
        for n in tamanhos() {
            let base = gera(n);
            mede_u64(c, nome, n, &base);
        }
    }
}

fn bench_texto(c: &mut Criterion) {
    trava_tbb();
    for n in tamanhos() {
        // Um cenario de cada vez: em 30M cada base de texto ja e meio giga do
        // lado Rust, e segurar as duas ao mesmo tempo nao cabe.
        let a = arena_sequencias();
        let idx = indices(n, a.strs.len(), 0x5EED_0002);
        mede_texto(c, "texto/Sequencia", n, a, &idx);
        drop(idx);

        let a = arena_cromossomos();
        let idx = indices(n, a.strs.len(), 0xC420_0000);
        mede_texto(c, "texto/Cromossomo", n, a, &idx);
    }
}

fn bench_chave_larga(c: &mut Criterion) {
    trava_tbb();
    if !ligado("MM_LARGA") {
        return;
    }
    let cenarios: &[(&str, fn(usize) -> Vec<u64>)] = &[
        ("larga16/DenteDeSerra", dente_de_serra),
        ("larga16/BaixaCardinalidade", baixa_cardinalidade),
    ];
    for &(nome, gera) in cenarios {
        for n in tamanhos() {
            let base = larga_de(gera(n));
            mede_rust::<ChaveLarga>(c, nome, n, &base);
        }
    }
}

criterion_group!(benches, bench_u64, bench_texto, bench_chave_larga);
criterion_main!(benches);
