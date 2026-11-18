// O motor C++ pelas entradas novas da ponte FFI: radix em u64, prefixo em
// texto, e o dente de serra em faixas disjuntas pelo `cpp_sort_u64`.
//
// O resultado tem de ser o do sort de referencia, elemento por elemento. A
// estabilidade do C++ com dado junto da chave nao e visivel por um array de
// u64: quem a confere, par a par contra o std::stable_sort, e o
// `src/stability_test.cpp`, que tambem roda o motor com os limiares do radix em
// miniatura.
use adaptive_parallel_multimerge_sort as mm;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

fn lcg(s: &mut u64) -> u64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *s >> 11
}

fn confere_u64(nome: &str, v: Vec<u64>) {
    let mut esperado = v.clone();
    esperado.sort_unstable();
    for (motor, f) in [("cpp_sort_radix_u64", mm::cpp_sort_radix_u64 as fn(&mut [u64])), ("cpp_sort_u64", mm::cpp_sort_u64)] {
        let mut w = v.clone();
        f(&mut w);
        assert!(w == esperado, "{motor}, {nome}: difere do sort de referencia");
    }
}

// Tamanho acima do piso do radix no C++ (32 mil), em formatos que mandam a
// rotina por caminhos diferentes.
#[test]
fn radix_u64_no_motor_cpp() {
    let n = 600_000usize;
    let mut s = 41u64;
    let mut gera = |f: &mut dyn FnMut(u64, usize) -> u64| -> Vec<u64> {
        (0..n).map(|i| { let z = lcg(&mut s); f(z, i) }).collect()
    };
    confere_u64("faixa inteira", gera(&mut |z, _| z.wrapping_mul(0x9E3779B97F4A7C15)));
    confere_u64("faixa de 50 mil (contagem)", gera(&mut |z, _| z % 50_000));
    confere_u64("64 valores", gera(&mut |z, _| (z % 64) * 1_000_003));
    confere_u64("enviesada", gera(&mut |z, _| if z % 10 == 0 { z.wrapping_mul(0x9E3779B97F4A7C15) } else { z % 1000 }));
    confere_u64("40 bits com prefixo comum", gera(&mut |z, _| 0xABCD_0000_0000_0000 | (z & 0xFF_FFFF_FFFF)));
    confere_u64("ordenada", gera(&mut |_, i| i as u64));
    confere_u64("invertida com repetidos", gera(&mut |_, i| ((n - i) / 3) as u64));

    let mut s = 7u64;
    for m in [0usize, 1, 2, 8192, 8193, 32_767, 32_768, 32_769, 65_537, 262_145] {
        confere_u64(&format!("n={m}"), (0..m).map(|_| lcg(&mut s)).collect());
    }
}

// Ordenado, com cada bloco impar invertido: dentes em faixas de valor
// disjuntas, direcao alternada. `plato` = quantas vezes o maior de cada dente
// aparece -- com 2 ou mais, o topo do dente descendente tem chaves repetidas,
// que e o caso em que devolver um elemento so nao bastava.
fn serra_disjunta(n: usize, dente: usize, plato: usize) -> Vec<u64> {
    (0..n)
        .map(|i| {
            let d = i / dente;
            let len = dente.min(n - d * dente);
            let mut k = i % dente;
            if d % 2 == 1 {
                k = len - 1 - k;
            }
            (d * dente + k.min(len.saturating_sub(plato))) as u64
        })
        .collect()
}

#[test]
fn dente_de_serra_em_faixas_disjuntas_no_motor_cpp() {
    for dente in [100usize, 1000, 4096, 4097, 33_000] {
        for plato in [1usize, 2, 3, 17] {
            confere_u64(&format!("dente {dente}, plato {plato}"), serra_disjunta(600_000, dente, plato));
        }
    }
    let mut v = serra_disjunta(900_000, 1000, 1);
    v[451_001] = v[451_000]; // uma unica duplicata, no topo de um dente descendente
    confere_u64("uma duplicata no topo de um dente", v);
}

/// Strings ASCII sem NUL interno. `prefixo` fixa os primeiros bytes de todas,
/// para o prefixo de 8 bytes distinguir pouco ou nada.
fn arena(distintas: usize, prefixo: &[u8], semente: u64) -> Vec<CString> {
    const ALFABETO: &[u8] = b"ACGT";
    let mut s = semente;
    (0..distintas)
        .map(|_| {
            let len = (lcg(&mut s) % 30) as usize;
            let mut bytes = prefixo.to_vec();
            bytes.extend((0..len).map(|_| ALFABETO[(lcg(&mut s) % 4) as usize]));
            CString::new(bytes).unwrap()
        })
        .collect()
}

#[test]
fn prefixo_em_texto_no_motor_cpp() {
    // (nome, prefixo comum): sem prefixo comum o radix distingue quase tudo
    // (e ha strings de 0 a 7 bytes, mais curtas que o prefixo); com 5 bytes
    // comuns distingue pouco; com 9, nada -- tudo fica para o strcmp.
    for (nome, prefixo) in [("sem prefixo comum", &b""[..]), ("5 bytes comuns", &b"GATTA"[..]), ("9 bytes comuns", &b"GATTACAGA"[..])] {
        let a = arena(50_000, prefixo, 0xC0FFEE);
        let mut s = 0xABCDEFu64;
        let idx: Vec<usize> = (0..400_000).map(|_| (lcg(&mut s) % a.len() as u64) as usize).collect();

        let mut ptrs: Vec<*const c_char> = idx.iter().map(|&i| a[i].as_ptr()).collect();
        let mut esperado: Vec<&CStr> = idx.iter().map(|&i| a[i].as_c_str()).collect();
        esperado.sort();
        unsafe { mm::cpp_sort_radix_str(&mut ptrs) };
        let obtido: Vec<&CStr> = ptrs.iter().map(|&p| unsafe { CStr::from_ptr(p) }).collect();
        assert!(obtido == esperado, "{nome}: cpp_sort_radix_str difere do sort de referencia");
    }
}
