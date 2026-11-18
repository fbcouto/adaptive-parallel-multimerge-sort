// As entradas com chave radix contra as entradas por comparacao e contra o
// Rayon: o programa que mede a tabela de 30M da secao 7 do README. Usa a API
// publica do crate.
//
//   cargo run --release --example vs_rayon_radix -- [n] [rodadas] [i32|u64|linhas]
//                                                    padrao: 30000000 5 (tudo)
//
// A tabela publicada: `30000000 5`, rodada duas vezes, ficando em cada celula
// o menor dos dois minimos.
//
// Metodo do projeto: 1 rodada de aquecimento descartada + `rodadas` medidas;
// a ORDEM dos competidores e sorteada a cada rodada; copia nova do dado fora
// do cronometro; minimo e mediana. Cada cenario tem um competidor repetido (A/A),
// que da o piso de ruido, e o cenario "ja ordenado" e a LINHA DE CONTROLE: ali
// o radix nem e chamado e os tempos tem de ser iguais com e sem ele.
use adaptive_parallel_multimerge_sort::{
    sort, sort_by_key, sort_by_radix_key, sort_radix, sort_stable, sort_stable_by_key,
    sort_stable_by_radix_key, sort_stable_radix, RadixKey,
};
use rayon::prelude::*;
use std::time::Instant;

fn splitmix(n: usize, seed: u64) -> impl Iterator<Item = u64> {
    let mut s = seed;
    (0..n).map(move |_| {
        s = s.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    })
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Linha {
    chave: u64,
    idx: u64,
}

type Competidor<'a, T> = (&'a str, Box<dyn Fn(&mut [T]) + 'a>);

fn mede<T: Copy + Send + Sync>(
    cenario: &str,
    master: &[T],
    rodadas: usize,
    competidores: &[Competidor<T>],
    confere: &dyn Fn(&[T]) -> bool,
) {
    let k = competidores.len();
    let mut work = master.to_vec();
    let mut tempos: Vec<Vec<f64>> = vec![Vec::new(); k];
    // Ordem SORTEADA a cada rodada (semente fixa), nao girada: na rotacao
    // ciclica cada competidor roda sempre atras do mesmo vizinho.
    let mut sorteio = splitmix(usize::MAX, 0x0DE5 ^ master.len() as u64);
    let mut ordem: Vec<usize> = (0..k).collect();
    for r in 0..=rodadas {
        for i in (1..k).rev() {
            ordem.swap(i, (sorteio.next().unwrap() % (i as u64 + 1)) as usize);
        }
        for j in 0..k {
            let c = ordem[j];
            work.par_chunks_mut(1 << 16).zip(master.par_chunks(1 << 16)).for_each(|(d, s)| d.copy_from_slice(s));
            let ini = Instant::now();
            (competidores[c].1)(&mut work);
            let ms = ini.elapsed().as_secs_f64() * 1000.0;
            if r == 0 {
                assert!(confere(&work), "{cenario}: {} devolveu resultado errado", competidores[c].0);
            } else {
                tempos[c].push(ms);
            }
        }
    }
    print!("{cenario:<34}");
    for (c, t) in tempos.iter_mut().enumerate() {
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        print!(" | {}: {:7.1} ({:7.1})", competidores[c].0, t[0], t[t.len() / 2]);
    }
    println!();
}

fn inteiros<T>(tipo: &str, n: usize, rodadas: usize, de: impl Fn(u64) -> T + Copy)
where
    T: RadixKey + std::fmt::Debug,
{
    let cenarios: Vec<(&str, Vec<T>)> = vec![
        ("aleatorio", splitmix(n, 42).map(de).collect()),
        ("64 valores", splitmix(n, 42).map(|z| de(z % 64)).collect()),
        ("enviesado (90% em 0..1000)", splitmix(n, 42).map(|z| if z % 10 != 0 { de((z >> 8) % 1000) } else { de(z) }).collect()),
        ("ja ordenado (CONTROLE)", {
            let mut v: Vec<T> = splitmix(n, 42).map(de).collect();
            v.par_sort_unstable();
            v
        }),
    ];
    for (nome, master) in &cenarios {
        let mut esperado = master.clone();
        esperado.par_sort_unstable();
        let confere = |w: &[T]| w == &esperado[..];
        let competidores: Vec<Competidor<T>> = vec![
            ("rayon_inst", Box::new(|a: &mut [T]| a.par_sort_unstable())),
            ("sort", Box::new(|a: &mut [T]| sort(a))),
            ("sort_radix", Box::new(|a: &mut [T]| sort_radix(a))),
            ("sort_radix(A/A)", Box::new(|a: &mut [T]| sort_radix(a))),
            ("rayon_est", Box::new(|a: &mut [T]| a.par_sort())),
            ("sort_stable", Box::new(|a: &mut [T]| sort_stable(a))),
            ("sort_stable_radix", Box::new(|a: &mut [T]| sort_stable_radix(a))),
        ];
        mede(&format!("{tipo} {nome}"), master, rodadas, &competidores, &confere);
    }
}

fn linhas(n: usize, rodadas: usize) {
    let faz = |f: &dyn Fn(u64) -> u64| -> Vec<Linha> {
        splitmix(n, 42).enumerate().map(|(i, z)| Linha { chave: f(z), idx: i as u64 }).collect()
    };
    let cenarios: Vec<(&str, Vec<Linha>)> = vec![
        ("aleatorio", faz(&|z| z)),
        ("64 valores", faz(&|z| z % 64)),
        ("ja ordenado (CONTROLE)", {
            let mut v = faz(&|z| z);
            v.par_sort_unstable_by_key(|l| l.chave);
            v
        }),
    ];
    for (nome, master) in &cenarios {
        // Ordenado pela chave e, entre chaves iguais, com idx crescente quando
        // o competidor e estavel (a ordem do dado e a do idx).
        let ordenado = |w: &[Linha]| w.len() == master.len() && w.par_windows(2).all(|p| p[0].chave <= p[1].chave);
        let estavel = |w: &[Linha]| ordenado(w) && w.par_windows(2).all(|p| p[0].chave != p[1].chave || p[0].idx < p[1].idx);
        let competidores: Vec<Competidor<Linha>> = vec![
            ("rayon_inst", Box::new(|a: &mut [Linha]| a.par_sort_unstable_by_key(|l| l.chave))),
            ("sort_by_key", Box::new(|a: &mut [Linha]| sort_by_key(a, |l| l.chave))),
            ("sort_by_radix_key", Box::new(|a: &mut [Linha]| sort_by_radix_key(a, |l| l.chave))),
        ];
        mede(&format!("linha16 inst {nome}"), master, rodadas, &competidores, &ordenado);
        let competidores: Vec<Competidor<Linha>> = vec![
            ("rayon_est", Box::new(|a: &mut [Linha]| a.par_sort_by_key(|l| l.chave))),
            ("sort_stable_by_key", Box::new(|a: &mut [Linha]| sort_stable_by_key(a, |l| l.chave))),
            ("sort_stable_by_radix_key", Box::new(|a: &mut [Linha]| sort_stable_by_radix_key(a, |l| l.chave))),
            ("idem(A/A)", Box::new(|a: &mut [Linha]| sort_stable_by_radix_key(a, |l| l.chave))),
        ];
        mede(&format!("linha16 est  {nome}"), master, rodadas, &competidores, &estavel);
    }
}

fn main() {
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(30_000_000);
    let rodadas: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    let quais = std::env::args().nth(3).unwrap_or_else(|| "tudo".into());
    println!("n = {n}; threads = {}; {rodadas} rodadas + 1 de aquecimento; ms: minimo (mediana)", rayon::current_num_threads());
    if quais == "tudo" || quais == "i32" {
        inteiros("i32", n, rodadas, |z| z as i32);
    }
    if quais == "tudo" || quais == "u64" {
        inteiros("u64", n, rodadas, |z| z);
    }
    if quais == "tudo" || quais == "linhas" {
        linhas(n, rodadas);
    }
}
