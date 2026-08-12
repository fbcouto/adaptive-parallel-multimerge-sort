use adaptive_parallel_multimerge_sort::sort as multi_merge;
use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use std::time::Duration;

// ==========================================
// GERADORES DE CENARIO
// ==========================================

fn generate_sorted(size: usize) -> Vec<u64> {
    (0..size as u64).collect()
}

fn generate_reversed(size: usize) -> Vec<u64> {
    (0..size as u64).rev().collect()
}

fn generate_random(size: usize) -> Vec<u64> {
    let mut rng = thread_rng();
    (0..size).map(|_| rng.gen()).collect()
}

fn generate_sawtooth(size: usize) -> Vec<u64> {
    let mut rng = thread_rng();
    let mut data = Vec::with_capacity(size);
    let num_teeth = 1000;
    let tooth_size = size / num_teeth;
    for i in 0..num_teeth {
        let start = (i * tooth_size) as u64;
        let end = start + tooth_size as u64;
        if rng.gen_bool(0.5) {
            data.extend(start..end);
        } else {
            data.extend((start..end).rev());
        }
    }
    let iterations_done = num_teeth * tooth_size;
    if iterations_done < size {
        data.extend((iterations_done as u64)..(size as u64));
    }
    data
}

/// Poucos valores possiveis (0..64) -> muitas duplicatas garantidas pra
/// qualquer tamanho >> 64. Esse e' o cenario que mais importa pra validar
/// a promessa de estabilidade -- e' exatamente o "banco de dados com
/// chaves secundarias repetidas" que o README usa como motivacao, e nenhum
/// dos outros geradores produz duplicatas de verdade (Sorted/Reversed/
/// Sawtooth usam ranges, Random sorteia sobre o u64 inteiro).
fn generate_low_cardinality(size: usize) -> Vec<u64> {
    let mut rng = thread_rng();
    (0..size).map(|_| rng.gen_range(0..64)).collect()
}

// ==========================================
// MOTOR DE BENCHMARK
// ==========================================

fn bench_final_arena(c: &mut Criterion) {
    let sizes = [
       1_000_000,  //, 10_000_000
       100_000_000
    // 300_000_000
    ];

    let scenarios = [
        ("Scenario_Sorted", generate_sorted as fn(usize) -> Vec<u64>),
        ("Scenario_Reversed", generate_reversed as fn(usize) -> Vec<u64>),
        ("Scenario_Random", generate_random as fn(usize) -> Vec<u64>),
        ("Scenario_Sawtooth_1000", generate_sawtooth as fn(usize) -> Vec<u64>),
        ("Scenario_LowCardinality", generate_low_cardinality as fn(usize) -> Vec<u64>),
    ];

    for (scenario_name, generator) in scenarios.iter() {
        let mut group = c.benchmark_group(*scenario_name);

        // 20 amostras (o minimo do Criterion e' 10). Em 300M elementos, o
        // tempo de parede por iteracao ja e' alto o bastante pra que o
        // numero real de amostras coletadas fique bem abaixo disso mesmo
        // assim -- isso e' esperado, nao e' bug. Se quiser intervalos de
        // confianca mais apertados justamente no topo, suba
        // measurement_time so' pra esse tamanho.
        group.sample_size(20);
        group.measurement_time(Duration::from_secs(20));

        for &size in sizes.iter() {
            group.throughput(Throughput::Elements(size as u64));
            let base_data = generator(size);

            // 1. Rayon estavel: o concorrente que importa de verdade -- mesma
            // garantia de estabilidade que o MultiMerge promete. (Rayon e'
            // uma crate externa; nao e' "standard library" do Rust.)
            group.bench_with_input(BenchmarkId::new("1_Rayon_Stable_Baseline", size), &size, |b, _| {
                b.iter_batched(
                    || base_data.clone(),
                    |mut d| d.par_sort(),
                    BatchSize::LargeInput,
                )
            });

            // 2. Sua engine -- sempre estavel, em qualquer cenario, incluindo
            // Random e LowCardinality.
            group.bench_with_input(BenchmarkId::new("2_MultiMerge_Stable", size), &size, |b, _| {
                b.iter_batched(
                    || base_data.clone(),
                    |mut d| multi_merge(black_box(&mut d)),
                    BatchSize::LargeInput,
                )
            });
        }
        group.finish();
    }
}

// ==========================================
// CENARIOS COM STRING (genomica: nome de cromossomo, sequencia/alelo)
//
// multi_merge_sort e' generico (T: Ord+Clone+Send+Sync), entao String
// funciona sem mudar nada no multimerge.rs -- mas o CUSTO por elemento e'
// bem diferente de u64: comparacao deixa de ser O(1) (String::cmp percorre
// byte a byte), e .clone() aloca no heap em vez de so' copiar bits. Os
// dois cenarios abaixo cobrem as duas pontas do uso real em VCF: campo
// CHROM (curto, poucos valores possiveis) e alelo/sequencia (mais longo,
// tamanho variavel).
// ==========================================
/*
const CHROM_NAMES: &[&str] = &[
    "chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10",
    "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20",
    "chr21", "chr22", "chrX", "chrY", "chrM",
];

/// Poucos valores possiveis (25 nomes de cromossomo), ordem aleatoria --
/// e' literalmente o campo CHROM de um VCF real, curto e baixa cardinalidade.
fn generate_chrom_like(size: usize) -> Vec<String> {
    let mut rng = thread_rng();
    (0..size)
        .map(|_| CHROM_NAMES[rng.gen_range(0..CHROM_NAMES.len())].to_string())
        .collect()
}

/// Strings de tamanho variavel (5 a 200 caracteres) sobre o alfabeto ACGT --
/// simula alelo/sequencia curta, faixa tipica de SNPs a indels pequenos.
/// Alta cardinalidade (tamanho E conteudo variam), ao contrario do ChromLike.
fn generate_sequence_like(size: usize) -> Vec<String> {
    let mut rng = thread_rng();
    const ALPHABET: [u8; 4] = *b"ACGT";
    (0..size)
        .map(|_| {
            let len = rng.gen_range(5..200);
            (0..len)
                .map(|_| ALPHABET[rng.gen_range(0..4)] as char)
                .collect::<String>()
        })
        .collect()
}

fn bench_string_scenarios(c: &mut Criterion) {
    // tamanhos menores que os de u64 de proposito: cada iteracao clona o
    // Vec<String> inteiro pro setup (BatchSize::LargeInput), e clonar
    // String aloca -- 10M Strings clonadas por amostra seria uma escala
    // de tempo bem diferente da dos cenarios de u64. Tambem e' mais perto
    // da escala real de um VCF (milhares a poucos milhoes de variantes,
    // raramente centenas de milhoes).
    let sizes = [100_000, 1_000_000, 5_000_000];

    let scenarios: [(&str, fn(usize) -> Vec<String>); 2] = [
        ("Scenario_ChromLike", generate_chrom_like as fn(usize) -> Vec<String>),
        ("Scenario_SequenceLike", generate_sequence_like as fn(usize) -> Vec<String>),
    ];

    for (scenario_name, generator) in scenarios.iter() {
        let mut group = c.benchmark_group(*scenario_name);
        group.sample_size(20);
        group.measurement_time(Duration::from_secs(20));

        for &size in sizes.iter() {
            group.throughput(Throughput::Elements(size as u64));
            let base_data = generator(size);

            group.bench_with_input(BenchmarkId::new("1_Rayon_Stable_Baseline", size), &size, |b, _| {
                b.iter_batched(
                    || base_data.clone(),
                    |mut d| d.par_sort(),
                    BatchSize::LargeInput,
                )
            });

            group.bench_with_input(BenchmarkId::new("2_MultiMerge_Stable", size), &size, |b, _| {
                b.iter_batched(
                    || base_data.clone(),
                    |mut d| multi_merge(black_box(&mut d)),
                    BatchSize::LargeInput,
                )
            });
        }
        group.finish();
    }
}

// ==========================================
// COMPARACAO DIRETA: OS MESMOS K-MERS COMO STRING vs CODIFICADOS EM U64
//
// Mesma ideia do seu Kmer128/encode_kmer: 2 bits por base (A=00,C=01,G=10,
// T=11), empacotados por deslocamento a esquerda -- cabe folgado num u64
// (ate' 32 bases; K=21 aqui e' um tamanho real e comum em bioinformatica).
// As DUAS representacoes vem do MESMO pool de sequencias aleatorias, pra
// comparacao ser justa -- so' muda o tipo que chega no sort.
// ==========================================

const KMER_LEN: usize = 21;

fn random_dna_bases(rng: &mut impl Rng, len: usize) -> Vec<u8> {
    const BASES: [u8; 4] = *b"ACGT";
    (0..len).map(|_| BASES[rng.gen_range(0..4)]).collect()
}

fn encode_kmer_u64(seq: &[u8]) -> u64 {
    let mut bits: u64 = 0;
    for &b in seq {
        bits <<= 2;
        bits |= match b {
            b'C' => 0b01,
            b'G' => 0b10,
            b'T' => 0b11,
            _ => 0b00, // 'A' (e qualquer coisa fora do alfabeto, por seguranca)
        };
    }
    bits
}

fn generate_kmer_pool(size: usize) -> Vec<Vec<u8>> {
    let mut rng = thread_rng();
    (0..size).map(|_| random_dna_bases(&mut rng, KMER_LEN)).collect()
}

fn bench_kmer_encoding_comparison(c: &mut Criterion) {
    let sizes = [100_000, 1_000_000, 5_000_000];
    let mut group = c.benchmark_group("Scenario_Kmer_String_vs_U64");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(20));

    for &size in sizes.iter() {
        group.throughput(Throughput::Elements(size as u64));

        let pool = generate_kmer_pool(size);
        let as_strings: Vec<String> = pool.iter().map(|s| String::from_utf8(s.clone()).unwrap()).collect();
        let as_u64: Vec<u64> = pool.iter().map(|s| encode_kmer_u64(s)).collect();

        group.bench_with_input(BenchmarkId::new("1_String_Rayon", size), &size, |b, _| {
            b.iter_batched(|| as_strings.clone(), |mut d| d.par_sort(), BatchSize::LargeInput)
        });
        group.bench_with_input(BenchmarkId::new("2_String_MultiMerge", size), &size, |b, _| {
            b.iter_batched(|| as_strings.clone(), |mut d| multi_merge(black_box(&mut d)), BatchSize::LargeInput)
        });
        group.bench_with_input(BenchmarkId::new("3_U64_Rayon", size), &size, |b, _| {
            b.iter_batched(|| as_u64.clone(), |mut d| d.par_sort(), BatchSize::LargeInput)
        });
        group.bench_with_input(BenchmarkId::new("4_U64_MultiMerge", size), &size, |b, _| {
            b.iter_batched(|| as_u64.clone(), |mut d| multi_merge(black_box(&mut d)), BatchSize::LargeInput)
        });
    }
    group.finish();
}

// ==========================================
// MICRO-BENCHMARK: insertion sort vs .sort() padrao, em pedacos bem pequenos
//
// Testa so' a premissa, isolada da arquitetura toda: pra que faixa de N (se
// alguma) um insertion sort na mao bate o .sort() do Rust? So' depois de
// responder isso vale decidir SE e ONDE encaixar -- hoje as folhas do
// parallel_recursive_sort ficam na casa dos milhares (leaf_size), bem
// acima de onde insertion sort costuma compensar (dezenas). Tamanhos aqui
// vao de 4 a 256 elementos de proposito, pra cobrir a faixa onde a duvida
// realmente existe.
// ==========================================

fn insertion_sort<T: Ord>(arr: &mut [T]) {
    for i in 1..arr.len() {
        let mut j = i;
        while j > 0 && arr[j - 1] > arr[j] {
            arr.swap(j - 1, j);
            j -= 1;
        }
    }
}

fn bench_insertion_vs_std_sort(c: &mut Criterion) {
    let sizes = [4usize, 8, 16, 32, 64, 128, 256];
    let mut group = c.benchmark_group("Scenario_Insertion_vs_Std");
    group.sample_size(100);

    for &size in sizes.iter() {
        let mut rng = thread_rng();
        let base_data: Vec<u64> = (0..size).map(|_| rng.gen()).collect();
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("1_Std_Sort", size), &size, |b, _| {
            b.iter_batched(|| base_data.clone(), |mut d| d.sort(), BatchSize::SmallInput)
        });

        group.bench_with_input(BenchmarkId::new("2_Insertion_Sort", size), &size, |b, _| {
            b.iter_batched(
                || base_data.clone(),
                |mut d| insertion_sort(&mut d),
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

// ==========================================
// MERGE_SEQ vs INSERTION SORT NA CONCATENACAO, pra runs pequenos ja
// ORDENADOS (2x2, 2x3, 3x3) -- a pergunta especifica que veio depois do
// benchmark anterior. Diferente do de cima: la' testava insertion sort
// contra .sort() em dados NAO ordenados; aqui as duas listas de entrada
// JA estao ordenadas (como saem da Fase 1), entao merge_seq tem a
// vantagem teorica de ja saber disso. Reimplementa merge_seq aqui porque
// e' privada no multimerge.rs -- mesma logica exata, so' pra comparar
// isolada.
// ==========================================

fn merge_seq_bench<T: Ord + Clone>(a: &[T], b: &[T], dest: &mut [T]) {
    let (mut i, mut j, mut k) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        if a[i] <= b[j] {
            dest[k] = a[i].clone();
            i += 1;
        } else {
            dest[k] = b[j].clone();
            j += 1;
        }
        k += 1;
    }
    if i < a.len() {
        dest[k..].clone_from_slice(&a[i..]);
    }
    if j < b.len() {
        dest[k..].clone_from_slice(&b[j..]);
    }
}

fn insertion_sort_concat<T: Ord + Clone>(a: &[T], b: &[T], dest: &mut [T]) {
    let na = a.len();
    dest[..na].clone_from_slice(a);
    dest[na..].clone_from_slice(b);
    insertion_sort(dest);
}

fn generate_tiny_sorted_pair(rng: &mut impl Rng, len_a: usize, len_b: usize) -> (Vec<u64>, Vec<u64>) {
    let mut a: Vec<u64> = (0..len_a).map(|_| rng.gen_range(0..1000)).collect();
    let mut b: Vec<u64> = (0..len_b).map(|_| rng.gen_range(0..1000)).collect();
    a.sort();
    b.sort();
    (a, b)
}

fn bench_tiny_merge_vs_insertion(c: &mut Criterion) {
    // pares aleatorios com o range de valores compartilhado entre A e B, pra
    // ter uma mistura realista de quanto os dois se entrelacam -- nem
    // sempre A inteiro < B inteiro, nem sempre entrelacado ao maximo.
    let pair_sizes = [(2usize, 2usize), (2, 3), (3, 3)];
    let mut group = c.benchmark_group("Scenario_TinyMerge_vs_Insertion");
    group.sample_size(100);

    for &(la, lb) in pair_sizes.iter() {
        let mut rng = thread_rng();
        let pairs: Vec<(Vec<u64>, Vec<u64>)> =
            (0..2000).map(|_| generate_tiny_sorted_pair(&mut rng, la, lb)).collect();
        let total = la + lb;
        let label = format!("{la}x{lb}");

        group.bench_with_input(BenchmarkId::new("1_MergeSeq", label.clone()), &label, |bch, _| {
            bch.iter_batched(
                || (pairs.clone(), vec![0u64; total]),
                |(ps, mut dest)| {
                    for (a, b) in &ps {
                        merge_seq_bench(a, b, &mut dest);
                    }
                },
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("2_InsertionOnConcat", label.clone()), &label, |bch, _| {
            bch.iter_batched(
                || (pairs.clone(), vec![0u64; total]),
                |(ps, mut dest)| {
                    for (a, b) in &ps {
                        insertion_sort_concat(a, b, &mut dest);
                    }
                },
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}
*/
criterion_group!(
    benches,
    bench_final_arena   //,
  //  bench_string_scenarios,
  //  bench_kmer_encoding_comparison,
  //  bench_insertion_vs_std_sort,
  //  bench_tiny_merge_vs_insertion
);
criterion_main!(benches);
