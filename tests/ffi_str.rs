// Correcao da ponte FFI de texto.
//
// O bench de strings entrega ao C++ um array de `*const c_char` reinterpretado
// como array de um wrapper com `operator<=>` por strcmp. Isso e a unica parte
// do caminho novo onde um erro de layout ou de comparador produziria um
// resultado plausivel e errado -- entao a verificacao e EXATA contra o sort da
// biblioteca padrao, o que pega perda e duplicacao de elemento, nao so ordem.
use adaptive_parallel_multimerge_sort as mm;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

fn lcg(s: &mut u64) -> u64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *s >> 11
}

/// Arena de strings ASCII sem NUL interno, com repeticao suficiente para
/// exercitar chaves iguais.
fn arena(distintas: usize, semente: u64) -> Vec<CString> {
    const ALFABETO: &[u8] = b"ACGT";
    let mut s = semente;
    (0..distintas)
        .map(|_| {
            let len = 1 + (lcg(&mut s) % 40) as usize;
            let bytes: Vec<u8> =
                (0..len).map(|_| ALFABETO[(lcg(&mut s) % 4) as usize]).collect();
            CString::new(bytes).unwrap()
        })
        .collect()
}

fn confere(nome: &str, arena: &[CString], indices: &[usize], sorter: unsafe fn(&mut [*const c_char])) {
    let mut ptrs: Vec<*const c_char> = indices.iter().map(|&i| arena[i].as_ptr()).collect();

    let mut esperado: Vec<&CStr> = indices.iter().map(|&i| arena[i].as_c_str()).collect();
    esperado.sort();

    unsafe { sorter(&mut ptrs) };

    let obtido: Vec<&CStr> = ptrs.iter().map(|&p| unsafe { CStr::from_ptr(p) }).collect();

    assert_eq!(obtido.len(), esperado.len(), "{nome}: mudou o numero de elementos");
    assert!(
        obtido.windows(2).all(|w| w[0] <= w[1]),
        "{nome}: NAO ORDENADO"
    );
    assert_eq!(obtido, esperado, "{nome}: difere do sort de referencia");
}

fn casos() -> Vec<(&'static str, Vec<CString>, Vec<usize>)> {
    let mut out = Vec::new();

    // Grande o bastante para passar do leaf_size e acionar a arvore de merge.
    let a = arena(4096, 0xC0FFEE);
    let mut s = 0xABCDEFu64;
    let idx: Vec<usize> = (0..400_000).map(|_| (lcg(&mut s) % 4096) as usize).collect();
    out.push(("aleatorio", a, idx));

    // Ja ordenado: o motor deve detectar o run e retornar sem ordenar.
    let mut a = arena(200_000, 0x5EED);
    a.sort();
    let idx: Vec<usize> = (0..a.len()).collect();
    out.push(("ordenado", a, idx));

    // Invertido: uma passada de reversao.
    let mut a = arena(200_000, 0x5EED);
    a.sort();
    a.reverse();
    let idx: Vec<usize> = (0..a.len()).collect();
    out.push(("invertido", a, idx));

    // Cardinalidade baixissima: quase tudo empatado.
    let a = arena(8, 0xBA1);
    let mut s = 0x1234u64;
    let idx: Vec<usize> = (0..300_000).map(|_| (lcg(&mut s) % 8) as usize).collect();
    out.push(("baixa_cardinalidade", a, idx));

    // Caminho curto: n abaixo do leaf_size.
    let a = arena(64, 0x77);
    let mut s = 0x99u64;
    let idx: Vec<usize> = (0..1000).map(|_| (lcg(&mut s) % 64) as usize).collect();
    out.push(("pequeno", a, idx));

    out
}

#[test]
fn multimerge_cpp_ordena_strings() {
    for (nome, a, idx) in casos() {
        confere(&format!("multimerge/{nome}"), &a, &idx, mm::cpp_sort_str);
    }
}

#[test]
fn pstl_ordena_strings() {
    for (nome, a, idx) in casos() {
        confere(&format!("pstl/{nome}"), &a, &idx, mm::pstl_sort_str);
    }
}

#[test]
fn multimerge_rust_ordena_strings() {
    // Controle: o mesmo dado pelo caminho Rust. Se este falhar, o problema nao
    // esta na ponte.
    for (nome, a, idx) in casos() {
        let strs: Vec<&str> = idx.iter().map(|&i| a[i].to_str().unwrap()).collect();
        let mut obtido = strs.clone();
        mm::sort(&mut obtido);
        let mut esperado = strs;
        esperado.sort();
        assert_eq!(obtido, esperado, "rust/{nome}: difere do sort de referencia");
    }
}
