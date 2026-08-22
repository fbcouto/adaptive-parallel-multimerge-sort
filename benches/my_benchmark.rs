use adaptive_parallel_multimerge_sort::cpp_sort_u64;
use adaptive_parallel_multimerge_sort::sort as multi_merge;
use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::time::Duration;

const AMOSTRAS: usize = 20;
const SEG_MEDICAO: u64 = 10;

fn tamanhos() -> Vec<usize> {
    std::env::var("MM_SIZES")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect::<Vec<_>>())
        .filter(|v: &Vec<usize>| !v.is_empty())
        .unwrap_or_else(|| vec![1_000_000, 5_000_000, 10_000_000])
}

fn rng(semente: u64) -> StdRng { StdRng::seed_from_u64(semente) }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ChaveLarga { chave: u64, carga: u64 }
impl PartialOrd for ChaveLarga {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) }
}
impl Ord for ChaveLarga {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering { self.chave.cmp(&o.chave) }
}

fn ordenado(n: usize) -> Vec<u64> { (0..n as u64).collect() }
fn invertido(n: usize) -> Vec<u64> { (0..n as u64).rev().collect() }

fn aleatorio(n: usize) -> Vec<u64> {
    let mut r = rng(0xA11E_A701);
    (0..n).map(|_| r.gen()).collect()
}

fn dente_de_serra(n: usize) -> Vec<u64> {
    let mut r = rng(0x5A57_0074);
    let dentes = 1000;
    let por_dente = (n / dentes).max(1);
    let mut d = Vec::with_capacity(n);
    for i in 0..dentes {
        let ini = (i * por_dente) as u64;
        let fim = ini + por_dente as u64;
        if r.gen_bool(0.5) { d.extend(ini..fim); } else { d.extend((ini..fim).rev()); }
    }
    let feitos = dentes * por_dente;
    if feitos < n { d.extend((feitos as u64)..(n as u64)); }
    d.truncate(n);
    d
}

fn baixa_cardinalidade(n: usize) -> Vec<u64> {
    let mut r = rng(0x10CA_4D1A);
    (0..n).map(|_| r.gen_range(0..64u64)).collect()
}

fn larga_de(v: Vec<u64>) -> Vec<ChaveLarga> {
    v.into_iter().enumerate().map(|(i, c)| ChaveLarga { chave: c, carga: i as u64 }).collect()
}

const CROMOSSOMOS: &[&str] = &[
    "chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11",
    "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20", "chr21",
    "chr22", "chrX", "chrY", "chrM",
];

fn arena_sequencias() -> &'static [String] {
    let mut r = rng(0x5EED_0001);
    const ALFABETO: [u8; 4] = *b"ACGT";
    let v: Vec<String> = (0..262_144)
        .map(|_| {
            let len = r.gen_range(5..200);
            (0..len).map(|_| ALFABETO[r.gen_range(0..4)] as char).collect()
        })
        .collect();
    Box::leak(v.into_boxed_slice())
}

fn tipo_cromossomo(n: usize) -> Vec<&'static str> {
    let mut r = rng(0xC420_0000);
    (0..n).map(|_| CROMOSSOMOS[r.gen_range(0..CROMOSSOMOS.len())]).collect()
}

fn tipo_sequencia(n: usize, arena: &'static [String]) -> Vec<&'static str> {
    let mut r = rng(0x5EED_0002);
    (0..n).map(|_| arena[r.gen_range(0..arena.len())].as_str()).collect()
}

fn mede<T: Ord + Copy + Send + Sync + 'static>(
    c: &mut Criterion, cenario: &str, n: usize, base: &[T], cpp: Option<&[u64]>,
) {
    let mut g = c.benchmark_group(cenario);
    g.sample_size(AMOSTRAS);
    g.measurement_time(Duration::from_secs(SEG_MEDICAO));
    g.throughput(Throughput::Elements(n as u64));

    g.bench_with_input(BenchmarkId::new("1_Rayon_estavel", n), &n, |b, _| {
        b.iter_batched(|| base.to_vec(), |mut d| d.par_sort(), BatchSize::LargeInput)
    });
    g.bench_with_input(BenchmarkId::new("2_MultiMerge_Rust", n), &n, |b, _| {
        b.iter_batched(|| base.to_vec(), |mut d| multi_merge(black_box(&mut d)), BatchSize::LargeInput)
    });
    if let Some(bu) = cpp {
        g.bench_with_input(BenchmarkId::new("3_MultiMerge_Cpp", n), &n, |b, _| {
            b.iter_batched(|| bu.to_vec(), |mut d| cpp_sort_u64(black_box(&mut d)), BatchSize::LargeInput)
        });
    }
    g.finish();
}

fn bench_u64(c: &mut Criterion) {
    let cenarios: &[(&str, fn(usize) -> Vec<u64>)] = &[
        ("u64/Ordenado", ordenado),
        ("u64/Invertido", invertido),
        ("u64/Aleatorio", aleatorio),
        ("u64/DenteDeSerra", dente_de_serra),
        ("u64/BaixaCardinalidade", baixa_cardinalidade),
    ];
    for &(nome, gera) in cenarios {
        for n in tamanhos() {
            let base = gera(n);
            let copia = base.clone();
            mede(c, nome, n, &base, Some(&copia));
        }
    }
}

fn bench_chave_larga(c: &mut Criterion) {
    let cenarios: &[(&str, fn(usize) -> Vec<u64>)] = &[
        ("larga16/DenteDeSerra", dente_de_serra),
        ("larga16/BaixaCardinalidade", baixa_cardinalidade),
    ];
    for &(nome, gera) in cenarios {
        for n in tamanhos() {
            let base = larga_de(gera(n));
            mede::<ChaveLarga>(c, nome, n, &base, None);
        }
    }
}

fn bench_texto(c: &mut Criterion) {
    let arena = arena_sequencias();
    let escalas: Vec<usize> = tamanhos().into_iter().map(|n| n / 10).filter(|&n| n >= 100_000).collect();
    for n in escalas {
        let base = tipo_cromossomo(n);
        mede::<&'static str>(c, "texto/Cromossomo", n, &base, None);
        let base = tipo_sequencia(n, arena);
        mede::<&'static str>(c, "texto/Sequencia", n, &base, None);
    }
}

criterion_group!(benches, bench_u64, bench_chave_larga, bench_texto);
criterion_main!(benches);
