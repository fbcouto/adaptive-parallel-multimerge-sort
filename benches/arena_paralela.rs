use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, BatchSize};
use adaptive_parallel_multimerge_sort::sort as multi_merge; // O alias que definimos
use rayon::prelude::*;
use rand::{thread_rng, Rng};

fn generate_random_u64(size: usize) -> Vec<u64> {
    let mut rng = thread_rng();
    (0..size).map(|_| rng.gen()).collect()
}

fn bench_arena_paralela(c: &mut Criterion) {
    let mut group = c.benchmark_group("Performance_Sort");
    group.sample_size(10); 

    let sizes = [1_000_000, 5_000_000, 10_000_000, 50_000_000, 100_000_000]; 

    for size in sizes.iter() {
        let data = generate_random_u64(*size);

        // 1. Rayon Parallel Sort
        group.bench_with_input(BenchmarkId::new("Rayon par_sort", size), size, |b, _| {
            // O iter_batched é vital: ele prepara os dados FORA do cronómetro
            b.iter_batched(
                || data.clone(), // Setup: prepara a cópia
                |mut d| d.par_sort(), // Medição: apenas o sort
                BatchSize::SmallInput,
            )
        });

        // 2. MultiMerge Adaptativo
        group.bench_with_input(BenchmarkId::new("MultiMerge (Seu Algoritmo)", size), size, |b, _| {
            b.iter_batched(
                || data.clone(), 
                |mut d| multi_merge(&mut d), 
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

criterion_group!(benches, bench_arena_paralela);
criterion_main!(benches);