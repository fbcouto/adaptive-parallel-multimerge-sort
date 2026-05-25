use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, BatchSize, Throughput};
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use adaptive_parallel_multimerge_sort::sort as multi_merge;
use std::time::Duration;

// ==========================================
// SCENARIO GENERATORS
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
        
        // Randomly decides if this "tooth" goes ascending or descending
        if rng.gen_bool(0.5) {
            data.extend(start..end);
        } else {
            data.extend((start..end).rev());
        }
    }
    
    // Fills any remainder if the size is not perfectly divisible
    let iterations_done = num_teeth * tooth_size;
    if iterations_done < size {
        data.extend((iterations_done as u64)..(size as u64));
    }
    data
}

// ==========================================
// BENCHMARK ENGINE
// ==========================================

fn bench_final_arena(c: &mut Criterion) {
    let sizes = [
        1_000_000, 
        10_000_000,
        100_000_000,
        300_000_000];

    // Define the scenarios and their respective generators
    let scenarios = [
       // ("Scenario_Sorted", generate_sorted as fn(usize) -> Vec<u64>),
       // ("Scenario_Reversed", generate_reversed as fn(usize) -> Vec<u64>),
        ("Scenario_Random", generate_random as fn(usize) -> Vec<u64>),
       // ("Scenario_Sawtooth_1000", generate_sawtooth as fn(usize) -> Vec<u64>),
    ];

    for (scenario_name, generator) in scenarios.iter() {
        let mut group = c.benchmark_group(*scenario_name);
        
        // Aggressive settings to support the 100M load
        group.sample_size(10); 
        group.measurement_time(Duration::from_secs(15));

        for &size in sizes.iter() {
            group.throughput(Throughput::Elements(size as u64));
            
            // Generate base data for this specific scenario and size
            let base_data = generator(size);

            // 1. Rayon Unstable (PDQSort in-place)
        //    group.bench_with_input(BenchmarkId::new("1_Rayon_Unstable", size), &size, |b, _| {
        //        b.iter_batched(
         //           || base_data.clone(),
          //         |mut d| d.par_sort_unstable(),
          //          BatchSize::LargeInput,
        //        )
       //     });

            // 2. Rayon Stable (Standard Rust Parallel Merge)
            group.bench_with_input(BenchmarkId::new("1_Rayon_Stable", size), &size, |b, _| {
                b.iter_batched(
                    || base_data.clone(),
                    |mut d| d.par_sort(),
                    BatchSize::LargeInput,
                )
            });

            // 3. MultiMerge (Your Adaptive Engine)
            group.bench_with_input(BenchmarkId::new("2_MultiMerge", size), &size, |b, _| {
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

criterion_group!(benches, bench_final_arena);
criterion_main!(benches);