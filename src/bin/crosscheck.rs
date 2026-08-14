//! CROSS-CHECK DAS TRES ENGINES
//!
//! Gera arquivos de entrada, manda as tres engines ordenarem os MESMOS dados,
//! grava as tres saidas em disco e compara tudo.
//!
//!   cargo +stable-x86_64-pc-windows-gnu run --release --bin crosscheck
//!
//! Arquivos gerados em .\crosscheck\ (formato: u64 cru, little-endian):
//!   <padrao>_input.bin      entrada
//!   <padrao>_rayon.bin      saida do Rayon par_sort  (baseline)
//!   <padrao>_rust.bin       saida do MultiMerge Rust
//!   <padrao>_cpp.bin        saida do MultiMerge C++
//!
//! POR QUE VARIOS PADROES E NAO SO ALEATORIO:
//! com dados 100% aleatorios o escudo de entropia dispara e as duas engines
//! MultiMerge desviam para o fallback. Deteccao de runs, merge bidirecional e
//! k-vias nao chegam a ser executados. O aleatorio testa o fallback; os outros
//! padroes e que exercitam o motor.
//!
//! O QUE ESTE TESTE PEGA: ordem errada, elemento perdido, elemento duplicado.
//! O QUE NAO PEGA: instabilidade. Dois u64 iguais sao indistinguiveis, entao
//! nenhuma comparacao de arquivo consegue ver troca entre eles. Para isso
//! existem stability_test.cpp e tests/stability.rs, que carregam indice junto
//! da chave.

use adaptive_parallel_multimerge_sort::{cpp_sort_u64, sort as multimerge_rust};
use rayon::prelude::*;
use std::fs;
use std::io::{BufWriter, Write};
use std::time::Instant;

const N: usize = 20_000_000;
const DIR: &str = "crosscheck";

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s >> 11
}

fn write_bin(path: &str, v: &[u64]) {
    let f = fs::File::create(path).expect("criar arquivo");
    let mut w = BufWriter::new(f);
    for x in v {
        w.write_all(&x.to_le_bytes()).expect("escrever");
    }
    w.flush().expect("flush");
}

/// Primeiro indice em que dois vetores diferem.
fn first_diff(a: &[u64], b: &[u64]) -> Option<usize> {
    if a.len() != b.len() {
        return Some(a.len().min(b.len()));
    }
    a.iter().zip(b.iter()).position(|(x, y)| x != y)
}

fn first_unsorted(v: &[u64]) -> Option<usize> {
    v.windows(2).position(|w| w[0] > w[1]).map(|i| i + 1)
}

fn patterns() -> Vec<(&'static str, Vec<u64>)> {
    let mut s = 20260814u64;
    let mut out: Vec<(&'static str, Vec<u64>)> = Vec::new();

    // 1. aleatorio puro - dispara o escudo, testa o FALLBACK
    out.push((
        "random",
        (0..N).map(|_| lcg(&mut s)).collect(),
    ));

    // 2. sawtooth - muitos runs sobrepostos: deteccao + k-vias + bidirecional
    out.push(("sawtooth", (0..N as u64).map(|i| i % 1000).collect()));

    // 3. estruturado + trechos caoticos - foi o padrao que pegou o bug do
    //    overlap contract, em que o C++ ordenava errado em 28 de 40 casos
    let mut v: Vec<u64> = (0..N as u64).collect();
    for _ in 0..25 {
        let start = (lcg(&mut s) as usize) % (N - 5000);
        for i in start..start + 5000 {
            v[i] = lcg(&mut s) % 100;
        }
    }
    out.push(("estruturado_caotico", v));

    // 4. baixa cardinalidade - massa de empates
    out.push((
        "baixa_cardinalidade",
        (0..N).map(|_| lcg(&mut s) % 16).collect(),
    ));

    // 5. quase ordenado - exercita os atalhos
    let mut v: Vec<u64> = (0..N as u64).collect();
    for _ in 0..(N / 1000) {
        let i = (lcg(&mut s) as usize) % N;
        v[i] = lcg(&mut s) % (N as u64);
    }
    out.push(("quase_ordenado", v));

    // 6. invertido - caminho do parallel_reverse
    out.push(("invertido", (0..N as u64).rev().collect()));

    out
}

fn main() {
    fs::create_dir_all(DIR).expect("criar diretorio");
    println!("n = {} elementos por padrao ({:.0} MB por arquivo)\n", N, N as f64 * 8.0 / 1e6);

    let mut falhas = 0usize;

    for (nome, entrada) in patterns() {
        write_bin(&format!("{}/{}_input.bin", DIR, nome), &entrada);

        // referencia independente: sort estavel sequencial da std.
        // Se as tres engines concordarem mas todas estiverem erradas, e ele
        // que denuncia.
        let mut referencia = entrada.clone();
        referencia.sort();

        let mut a = entrada.clone();
        let t = Instant::now();
        a.par_sort();
        let t_rayon = t.elapsed().as_secs_f64();

        let mut b = entrada.clone();
        let t = Instant::now();
        multimerge_rust(&mut b);
        let t_rust = t.elapsed().as_secs_f64();

        let mut c = entrada.clone();
        let t = Instant::now();
        cpp_sort_u64(&mut c);
        let t_cpp = t.elapsed().as_secs_f64();

        write_bin(&format!("{}/{}_rayon.bin", DIR, nome), &a);
        write_bin(&format!("{}/{}_rust.bin", DIR, nome), &b);
        write_bin(&format!("{}/{}_cpp.bin", DIR, nome), &c);

        println!("=== {} ===", nome);
        println!(
            "  tempos:  rayon {:.1} ms | rust {:.1} ms | cpp {:.1} ms",
            t_rayon * 1e3,
            t_rust * 1e3,
            t_cpp * 1e3
        );

        let mut ok_padrao = true;
        for (rotulo, saida) in [("rayon", &a), ("rust", &b), ("cpp", &c)] {
            let ord = first_unsorted(saida);
            let vs_ref = first_diff(saida, &referencia);
            let bom = ord.is_none() && vs_ref.is_none();
            if !bom {
                ok_padrao = false;
                falhas += 1;
            }
            print!("  {:<6} ordenado: {:<5}", rotulo, if ord.is_none() { "sim" } else { "NAO" });
            match vs_ref {
                None => println!("  identico a referencia: sim"),
                Some(i) => println!(
                    "  identico a referencia: NAO  (indice {}: obtido {} / esperado {})",
                    i, saida[i], referencia[i]
                ),
            }
            if let Some(i) = ord {
                println!("      primeira inversao no indice {}: {} > {}", i, saida[i - 1], saida[i]);
            }
        }

        // comparacao par a par das tres saidas
        let pares = [
            ("rayon", "rust", first_diff(&a, &b)),
            ("rayon", "cpp", first_diff(&a, &c)),
            ("rust", "cpp", first_diff(&b, &c)),
        ];
        for (x, y, d) in pares {
            match d {
                None => println!("  {} == {}", x, y),
                Some(i) => {
                    ok_padrao = false;
                    falhas += 1;
                    println!("  {} != {}  DIVERGEM no indice {}", x, y, i);
                }
            }
        }
        println!("  {}\n", if ok_padrao { "OK" } else { "<<<<<< PROBLEMA" });
    }

    println!("=======================================");
    if falhas == 0 {
        println!("TODAS AS ENGINES CONCORDAM E BATEM COM A REFERENCIA");
    } else {
        println!("{} verificacao(oes) FALHARAM", falhas);
    }
    println!("=======================================");
    println!("arquivos em .\\{}\\", DIR);
    std::process::exit(if falhas == 0 { 0 } else { 1 });
}
