// As entradas com chave radix, pela API publica.
//
// O resultado tem de ser o mesmo das entradas por comparacao: so o caminho do
// dado aleatorio muda. Os tamanhos ficam ACIMA dos limiares do motor (131 mil
// para elemento de ate 4 bytes, 524 mil acima; 32 mil no estavel), senao o
// radix nem entra e o teste nao testa nada. A cobertura fina da rotina -- cada
// folha, a reentrada, a contagem -- esta nos testes de unidade do motor, que
// forcam os limiares em miniatura.
use adaptive_parallel_multimerge_sort::{
    sort_by_prefix_key, sort_by_radix_key, sort_radix, sort_stable_by_prefix_key,
    sort_stable_by_radix_key, sort_stable_radix, RadixKey,
};

fn lcg(s: &mut u64) -> u64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *s >> 11
}

/// Formatos de chave que mandam a rotina por caminhos diferentes: faixa
/// inteira, faixa pequena (contagem), poucos valores, enviesada (a rotina
/// desiste e o quicksort assume), e os tres ja estruturados (o radix nem e
/// chamado: runs).
fn formatos(n: usize, semente: u64) -> Vec<(&'static str, Vec<u64>)> {
    let mut s = semente;
    let mut gera = |f: &mut dyn FnMut(u64, usize) -> u64| -> Vec<u64> {
        (0..n).map(|i| { let z = lcg(&mut s); f(z, i) }).collect()
    };
    vec![
        ("faixa inteira", gera(&mut |z, _| z.wrapping_mul(0x9E3779B97F4A7C15))),
        ("faixa de 50 mil", gera(&mut |z, _| z % 50_000)),
        ("64 valores", gera(&mut |z, _| (z % 64) * 1_000_003)),
        ("enviesada", gera(&mut |z, _| if z % 10 == 0 { z.wrapping_mul(0x9E3779B97F4A7C15) } else { z % 1000 })),
        ("ordenada", gera(&mut |_, i| i as u64)),
        ("invertida", gera(&mut |_, i| (n - i) as u64)),
        ("dente de serra", gera(&mut |_, i| (i % 1000) as u64)),
    ]
}

fn confere_inteiros<T: RadixKey + std::fmt::Debug>(tipo: &str, n: usize, converte: impl Fn(u64) -> T) {
    for (nome, chaves) in formatos(n, 17) {
        let v: Vec<T> = chaves.iter().map(|&k| converte(k)).collect();
        let mut referencia = v.clone();
        referencia.sort();

        let mut w = v.clone();
        sort_radix(&mut w);
        assert!(w == referencia, "{tipo}, {nome}: sort_radix difere do sort da std");

        let mut w = v.clone();
        sort_stable_radix(&mut w);
        assert!(w == referencia, "{tipo}, {nome}: sort_stable_radix difere do sort da std");
    }
}

#[test]
fn inteiros_nos_dois_motores() {
    confere_inteiros("u64", 600_000, |k| k);
    // Com sinal: metade negativa, para o bit de sinal entrar na conta.
    confere_inteiros("i64", 600_000, |k| k as i64);
    confere_inteiros("i32", 300_000, |k| k as i32);
    confere_inteiros("u32", 300_000, |k| k as u32);
    confere_inteiros("i16", 300_000, |k| k as i16);
    confere_inteiros("u8", 300_000, |k| k as u8);
}

#[test]
fn bordas_e_tamanhos_em_volta_dos_limiares() {
    let mut s = 5u64;
    for n in [0usize, 1, 2, 3, 4095, 4096, 4097, 32_767, 32_768, 32_769, 131_071, 131_072, 131_073] {
        let v: Vec<i32> = (0..n).map(|_| lcg(&mut s) as i32).collect();
        let mut referencia = v.clone();
        referencia.sort();
        let mut w = v.clone();
        sort_radix(&mut w);
        assert!(w == referencia, "sort_radix, n={n}");
        let mut w = v.clone();
        sort_stable_radix(&mut w);
        assert!(w == referencia, "sort_stable_radix, n={n}");
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Linha {
    chave: u64,
    idx: u64,
}

// Chave u64 com dado junto: o estavel tem de bater, par a par, com o sort
// estavel da std (pega ordem errada, perda, duplicacao e troca entre iguais);
// o instavel tem de sair ordenado e com a mesma colecao.
#[test]
fn chave_exata_com_dado_junto() {
    for (nome, chaves) in formatos(600_000, 23) {
        let v: Vec<Linha> = chaves.iter().enumerate().map(|(i, &k)| Linha { chave: k, idx: i as u64 }).collect();
        let mut referencia = v.clone();
        referencia.sort_by_key(|l| l.chave);

        let mut w = v.clone();
        sort_stable_by_radix_key(&mut w, |l| l.chave);
        assert!(w == referencia, "{nome}: sort_stable_by_radix_key difere do sort estavel da std");

        let mut w = v.clone();
        sort_by_radix_key(&mut w, |l| l.chave);
        assert!(w.windows(2).all(|p| p[0].chave <= p[1].chave), "{nome}: sort_by_radix_key nao ordenou");
        w.sort_unstable_by_key(|l| (l.chave, l.idx));
        assert!(w == referencia, "{nome}: sort_by_radix_key perdeu ou duplicou elemento");
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Registro {
    texto: [u8; 12],
    idx: u32,
}

fn prefixo(r: &Registro) -> u64 {
    u64::from_be_bytes(r.texto[..8].try_into().unwrap())
}

// Chave maior que 64 bits: texto de 12 bytes. O prefixo sao os 8 primeiros, em
// big-endian; os 4 ultimos so o comparador ve. Tres casos: prefixo que
// distingue quase tudo, prefixo de poucos valores (o desempate faz o trabalho)
// e prefixo igual em todos (o radix nao distingue nada).
#[test]
fn prefixo_com_desempate_pelo_comparador() {
    const LETRAS: &[u8; 36] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let n = 600_000usize;
    for (nome, livres_no_prefixo) in [("prefixo largo", 8usize), ("prefixo de poucos valores", 1), ("prefixo constante", 0)] {
        let mut s = 31u64;
        let v: Vec<Registro> = (0..n)
            .map(|i| {
                let mut texto = [b'A'; 12];
                for (j, b) in texto.iter_mut().enumerate() {
                    // So os ultimos `livres_no_prefixo` bytes do prefixo e os 4
                    // de fora dele variam.
                    if j >= 8 - livres_no_prefixo {
                        *b = LETRAS[(lcg(&mut s) % 36) as usize];
                    }
                }
                Registro { texto, idx: i as u32 }
            })
            .collect();
        let mut referencia = v.clone();
        referencia.sort_by(|a, b| a.texto.cmp(&b.texto));

        let mut w = v.clone();
        sort_stable_by_prefix_key(&mut w, prefixo, |a, b| a.texto.cmp(&b.texto));
        assert!(w == referencia, "{nome}: sort_stable_by_prefix_key difere do sort estavel da std");

        let mut w = v.clone();
        sort_by_prefix_key(&mut w, prefixo, |a, b| a.texto.cmp(&b.texto));
        assert!(w.windows(2).all(|p| p[0].texto <= p[1].texto), "{nome}: sort_by_prefix_key nao ordenou");
        w.sort_unstable_by(|a, b| (a.texto, a.idx).cmp(&(b.texto, b.idx)));
        assert!(w == referencia, "{nome}: sort_by_prefix_key perdeu ou duplicou elemento");
    }
}

// O exemplo de uso do README, para ele nao apodrecer.
#[test]
fn exemplo_do_readme() {
    // Integers: same result as `sort`, radix on random input.
    let mut ids: Vec<i32> = vec![42, -7, 0, 19, -7, 3];
    sort_radix(&mut ids);
    assert_eq!(ids, [-7, -7, 0, 3, 19, 42]);

    // Rows ordered by an integer key; the stable engine keeps equal keys in order.
    let mut rows: Vec<(u32, &str)> = vec![(7, "a"), (3, "b"), (7, "c"), (3, "d")];
    sort_stable_by_radix_key(&mut rows, |r| r.0 as u64);
    assert_eq!(rows, [(3, "b"), (3, "d"), (7, "a"), (7, "c")]);

    // Keys longer than 64 bits: the first 8 bytes as the radix prefix, the
    // full comparison to break ties.
    let mut names: Vec<[u8; 12]> = vec![*b"multimerge-b", *b"multimerge-a", *b"adaptive----"];
    sort_by_prefix_key(&mut names, |n| u64::from_be_bytes(n[..8].try_into().unwrap()), |a, b| a.cmp(b));
    assert_eq!(names, [*b"adaptive----", *b"multimerge-a", *b"multimerge-b"]);
}
