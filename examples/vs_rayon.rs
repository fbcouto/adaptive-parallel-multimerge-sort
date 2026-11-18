// Os dois motores Rust contra o Rayon: os programas que medem as tabelas das
// secoes 6 e 7 do README (menos a do radix em inteiros, que e `vs_rayon_radix`).
// Usam a API publica do crate.
//
//   cargo run --release --example vs_rayon -- u64    [n] [rodadas]   padrao: 10000000 9
//   cargo run --release --example vs_rayon -- text   [n] [rodadas]   padrao:  2000000 9
//   cargo run --release --example vs_rayon -- prefix [n] [rodadas]   padrao:  5000000 9
//
//   u64     10M de `u64`, onze formatos de dado            (secao 6, 1a tabela)
//   text    2M de `&str`, seis formatos                    (secao 6, 2a tabela)
//   prefix  5M de `&str` pela entrada `sort_by_prefix_key` (secao 7)
//
// As tabelas publicadas: `u64 10000000 15`, `text 2000000 9`, `prefix 5000000 7`,
// cada uma rodada duas vezes, ficando em cada celula a menor das duas medianas.
//
// Metodo do projeto: 1 rodada de aquecimento descartada (nela a saida de cada
// competidor e conferida contra a referencia) + `rodadas` medidas; copia nova
// do dado fora do cronometro; mediana e minimo.
//
// A ORDEM dos competidores e SORTEADA a cada rodada (semente fixa), nao girada:
// a rotacao ciclica deixa cada competidor sempre atras do mesmo vizinho, e em
// linhas de poucos milissegundos isso vira 1 a 2 ms de diferenca falsa.
//
// Um competidor aparece duas vezes (A/A): a diferenca entre as duas medianas e
// o piso de ruido da linha. A coluna "buffer" e o `sort_stable` com o buffer do
// chamador reaproveitado entre chamadas.
use adaptive_parallel_multimerge_sort::{
    sort, sort_by_prefix_key, sort_stable, sort_stable_by_prefix_key, sort_stable_by_with_buffer,
};
use rayon::prelude::*;
use std::cell::RefCell;
use std::time::Instant;

struct Rng(u64);
impl Rng {
    fn prox(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn abaixo(&mut self, n: usize) -> usize {
        (self.prox() % n as u64) as usize
    }
}

type Competidor<'a, T> = (&'a str, Box<dyn Fn(&mut [T]) + 'a>);

fn mede<T: Copy + Send + Sync + PartialEq>(cenario: &str, master: &[T], esperado: &[T], rodadas: usize, competidores: &[Competidor<T>]) {
    let k = competidores.len();
    let mut work = master.to_vec();
    let mut tempos: Vec<Vec<f64>> = vec![Vec::new(); k];
    let mut sorteio = Rng(0x0DE5_0000 ^ master.len() as u64);
    let mut ordem: Vec<usize> = (0..k).collect();
    for r in 0..=rodadas {
        for i in (1..k).rev() {
            ordem.swap(i, sorteio.abaixo(i + 1));
        }
        for &c in &ordem {
            work.par_chunks_mut(1 << 16).zip(master.par_chunks(1 << 16)).for_each(|(d, s)| d.copy_from_slice(s));
            let ini = Instant::now();
            (competidores[c].1)(&mut work);
            let ms = ini.elapsed().as_secs_f64() * 1000.0;
            if r == 0 {
                assert!(work == esperado, "{cenario}: {} devolveu resultado errado", competidores[c].0);
            } else {
                tempos[c].push(ms);
            }
        }
    }
    print!("{cenario:<34}");
    let mut medianas = Vec::new();
    for (c, t) in tempos.iter_mut().enumerate() {
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        medianas.push(t[t.len() / 2]);
        print!(" | {}: {:7.1} ({:7.1})", competidores[c].0, t[t.len() / 2], t[0]);
    }
    // Piso de ruido: os dois competidores cujo nome comeca igual e termina em "(A/A)".
    if let Some(aa) = competidores.iter().position(|(n, _)| n.ends_with("(A/A)")) {
        let nome = competidores[aa].0.trim_end_matches("(A/A)");
        if let Some(orig) = competidores.iter().position(|(n, _)| *n == nome) {
            let (a, b) = (medianas[orig], medianas[aa]);
            print!(" | ruido: {:.1}%", (a - b).abs() / a.min(b) * 100.0);
        }
    }
    println!();
}

// ------------------------------------------------------------------ u64
fn k_runs(n: usize, k: usize) -> Vec<u64> {
    let per = n / k;
    let mut d = Vec::with_capacity(n);
    for r in 0..k {
        let len = if r + 1 == k { n - per * (k - 1) } else { per };
        d.extend((0..len as u64).map(|i| i * k as u64 + r as u64));
    }
    d
}
fn aleatorio(n: usize, semente: u64) -> Vec<u64> {
    let mut r = Rng(semente);
    (0..n).map(|_| r.prox()).collect()
}
fn poucas_chaves(n: usize, k: u64) -> Vec<u64> {
    let mut r = Rng(0x10CA ^ k);
    (0..n).map(|_| r.prox() % k).collect()
}
fn quase_ordenado(n: usize) -> Vec<u64> {
    let mut q: Vec<u64> = (0..n as u64).collect();
    let mut r = Rng(0x9A5E_0001);
    for _ in 0..(n / 100) {
        let (a, b) = (r.abaixo(n), r.abaixo(n));
        q.swap(a, b);
    }
    q
}
/// 1000 dentes em faixas de valor disjuntas, direcao de cada dente sorteada.
fn dentes_disjuntos(n: usize) -> Vec<u64> {
    let mut r = Rng(0x5A57_0074);
    let por = n / 1000;
    let mut s: Vec<u64> = Vec::with_capacity(n);
    for i in 0..1000 {
        let ini = (i * por) as u64;
        let fim = ini + por as u64;
        if r.prox() & 1 == 0 {
            s.extend(ini..fim);
        } else {
            s.extend((ini..fim).rev());
        }
    }
    s.extend((s.len() as u64)..(n as u64));
    s
}

fn tabela_u64(n: usize, rodadas: usize) {
    let mid = n / 2;
    let mut com_ilha = aleatorio(n, 0xA11E);
    com_ilha[mid - 100..mid + 300].sort();
    let mut r = Rng(7);
    let mut com_caos: Vec<u64> = (0..n as u64).collect();
    for x in &mut com_caos[n / 4 - 100..n / 4 + 300] {
        *x = r.prox() % n as u64;
    }
    let cenarios: Vec<(&str, Vec<u64>)> = vec![
        ("ja ordenado", (0..n as u64).collect()),
        ("invertido", (0..n as u64).rev().collect()),
        ("invertido, platos de 500", (0..n as u64).map(|i| (n as u64 - i) / 500).collect()),
        ("dentes em faixas disjuntas", dentes_disjuntos(n)),
        ("2 runs intercalados", k_runs(n, 2)),
        ("64 runs intercalados", k_runs(n, 64)),
        ("ordenado + 400 caoticos", com_caos),
        ("quase ordenado (1% trocas)", quase_ordenado(n)),
        ("aleatorio, 64 chaves", poucas_chaves(n, 64)),
        ("aleatorio, alta cardinalidade", aleatorio(n, 0xA11E)),
        ("aleatorio + 400 ordenados no meio", com_ilha),
    ];
    let buffer: RefCell<Vec<u64>> = RefCell::new(Vec::new());
    for (nome, master) in &cenarios {
        let mut esperado = master.clone();
        esperado.par_sort_unstable();
        let competidores: Vec<Competidor<u64>> = vec![
            ("par_sort", Box::new(|a: &mut [u64]| a.par_sort())),
            ("sort_stable", Box::new(|a: &mut [u64]| sort_stable(a))),
            ("sort_stable(A/A)", Box::new(|a: &mut [u64]| sort_stable(a))),
            ("buffer", Box::new(|a: &mut [u64]| sort_stable_by_with_buffer(a, &mut buffer.borrow_mut(), u64::cmp))),
            ("par_sort_unstable", Box::new(|a: &mut [u64]| a.par_sort_unstable())),
            ("sort", Box::new(|a: &mut [u64]| sort(a))),
        ];
        mede(nome, master, &esperado, rodadas, &competidores);
    }
}

// ------------------------------------------------------------------ &str
fn arena(n: usize, semente: u64) -> Vec<&'static str> {
    let mut r = Rng(semente);
    let mut txt = String::with_capacity(n * 21);
    let mut cortes = Vec::with_capacity(n + 1);
    cortes.push(0usize);
    for _ in 0..n {
        let len = 12 + r.abaixo(16);
        for _ in 0..len {
            txt.push((b'a' + r.abaixo(26) as u8) as char);
        }
        cortes.push(txt.len());
    }
    let txt: &'static str = Box::leak(txt.into_boxed_str());
    cortes.windows(2).map(|w| &txt[w[0]..w[1]]).collect()
}
fn k_runs_de<T: Copy>(ordenado: &[T], k: usize) -> Vec<T> {
    let mut d = Vec::with_capacity(ordenado.len());
    for r in 0..k {
        d.extend(ordenado.iter().skip(r).step_by(k).copied());
    }
    d
}
fn quase_de<T: Copy>(ord: &[T]) -> Vec<T> {
    let n = ord.len();
    let mut q = ord.to_vec();
    let mut r = Rng(0x9A5E_0001);
    for _ in 0..(n / 100) {
        let (a, b) = (r.abaixo(n), r.abaixo(n));
        q.swap(a, b);
    }
    q
}

fn tabela_texto(n: usize, rodadas: usize) {
    type S = &'static str;
    let brutas = arena(n, 0x5EED);
    let mut ord = brutas.clone();
    ord.par_sort_unstable();
    let mut r = Rng(0xCA05);
    let poucas: Vec<S> = (0..n).map(|_| brutas[r.abaixo(64)]).collect();
    let cenarios: Vec<(&str, Vec<S>)> = vec![
        ("&str 2 runs intercalados", k_runs_de(&ord, 2)),
        ("&str 8 runs intercalados", k_runs_de(&ord, 8)),
        ("&str 64 runs intercalados", k_runs_de(&ord, 64)),
        ("&str quase ordenado (1% trocas)", quase_de(&ord)),
        ("&str aleatorio, 64 distintas", poucas),
        ("&str aleatorio", brutas.clone()),
    ];
    for (nome, master) in &cenarios {
        let mut esperado = master.clone();
        esperado.par_sort_unstable();
        let competidores: Vec<Competidor<S>> = vec![
            ("par_sort", Box::new(|a: &mut [S]| a.par_sort())),
            ("sort_stable", Box::new(|a: &mut [S]| sort_stable(a))),
            ("sort_stable(A/A)", Box::new(|a: &mut [S]| sort_stable(a))),
            ("par_sort_unstable", Box::new(|a: &mut [S]| a.par_sort_unstable())),
            ("sort", Box::new(|a: &mut [S]| sort(a))),
        ];
        mede(nome, master, &esperado, rodadas, &competidores);
    }
}

// ------------------------------------------------------------------ &str por prefixo
/// Os 8 primeiros bytes, em big-endian, com zeros a direita se faltar.
fn prefixo8(s: &str) -> u64 {
    let b = s.as_bytes();
    let mut p = [0u8; 8];
    let k = b.len().min(8);
    p[..k].copy_from_slice(&b[..k]);
    u64::from_be_bytes(p)
}

/// Como um wrapper deve usar a entrada de prefixo com texto: o prefixo e
/// calculado UMA vez e viaja junto da referencia, de modo que quase toda
/// comparacao se resolve sem tocar a memoria da string. O tempo de montar o
/// vetor de pares e de devolver as referencias ordenadas esta DENTRO da medida.
fn com_prefixo_carregado(a: &mut [&'static str], estavel: bool) {
    let mut pares: Vec<(u64, &'static str)> = a.par_iter().map(|s| (prefixo8(s), *s)).collect();
    let cmp = |x: &(u64, &'static str), y: &(u64, &'static str)| x.0.cmp(&y.0).then_with(|| x.1.cmp(y.1));
    if estavel {
        sort_stable_by_prefix_key(&mut pares, |p| p.0, cmp);
    } else {
        sort_by_prefix_key(&mut pares, |p| p.0, cmp);
    }
    a.par_iter_mut().zip(pares.par_iter()).for_each(|(d, p)| *d = p.1);
}

fn hex32(n: usize, semente: u64, prefixo: &str) -> Vec<&'static str> {
    const HEX: &[u8] = b"0123456789abcdef";
    let mut r = Rng(semente);
    let largura = prefixo.len() + 32;
    let mut txt = String::with_capacity(n * largura);
    for _ in 0..n {
        txt.push_str(prefixo);
        let (a, b) = (r.prox(), r.prox());
        for i in 0..16 {
            txt.push(HEX[((a >> (i * 4)) & 15) as usize] as char);
        }
        for i in 0..16 {
            txt.push(HEX[((b >> (i * 4)) & 15) as usize] as char);
        }
    }
    let txt: &'static str = Box::leak(txt.into_boxed_str());
    (0..n).map(|i| &txt[i * largura..(i + 1) * largura]).collect()
}

/// Texto pela entrada de prefixo: `sort_by_prefix_key` contra as entradas por
/// comparacao e contra o Rayon. O caso de uso e a coluna de texto de um banco
/// (o `md5` de 32 caracteres do benchmark do multimerge_pg).
fn tabela_prefixo(n: usize, rodadas: usize) {
    type S = &'static str;
    let md5 = hex32(n, 0x3D5, "");
    let mut r = Rng(0xCA05);
    let poucas: Vec<S> = (0..n).map(|_| md5[r.abaixo(64)]).collect();
    let mut ordenado = md5.clone();
    ordenado.par_sort_unstable();
    let cenarios: Vec<(&str, Vec<S>)> = vec![
        ("hex32 aleatorio", md5),
        ("letras 12-27 aleatorio", arena(n, 0x5EED)),
        ("hex32 aleatorio, 64 distintas", poucas),
        ("hex32, 10 bytes iguais na frente", hex32(n, 0x3D5, "ABCDEFGHIJ")),
        ("hex32 ja ordenado (CONTROLE)", ordenado),
    ];
    for (nome, master) in &cenarios {
        let mut esperado = master.clone();
        esperado.par_sort_unstable();
        let competidores: Vec<Competidor<S>> = vec![
            ("par_sort_unstable", Box::new(|a: &mut [S]| a.par_sort_unstable())),
            ("sort", Box::new(|a: &mut [S]| sort(a))),
            ("prefixo_direto", Box::new(|a: &mut [S]| sort_by_prefix_key(a, |s| prefixo8(s), |x, y| x.cmp(y)))),
            ("prefixo", Box::new(|a: &mut [S]| com_prefixo_carregado(a, false))),
            ("prefixo(A/A)", Box::new(|a: &mut [S]| com_prefixo_carregado(a, false))),
            ("par_sort", Box::new(|a: &mut [S]| a.par_sort())),
            ("sort_stable", Box::new(|a: &mut [S]| sort_stable(a))),
            ("prefixo_estavel", Box::new(|a: &mut [S]| com_prefixo_carregado(a, true))),
        ];
        mede(nome, master, &esperado, rodadas, &competidores);
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let qual = a.get(1).map(|s| s.as_str()).unwrap_or("u64");
    let padrao = match qual {
        "text" | "texto" => 2_000_000,
        "prefix" | "prefixo" => 5_000_000,
        _ => 10_000_000,
    };
    let n: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(padrao);
    let rodadas: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(9);
    println!(
        "{qual}: n = {n}; threads = {}; {rodadas} rodadas + 1 de aquecimento, ordem sorteada; ms: mediana (minimo)",
        rayon::current_num_threads()
    );
    match qual {
        "text" | "texto" => tabela_texto(n, rodadas),
        "prefix" | "prefixo" => tabela_prefixo(n, rodadas),
        _ => tabela_u64(n, rodadas),
    }
}
