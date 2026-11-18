// Correcao dos dois motores, pela API publica.
//
//   sort_stable  comparado EXATAMENTE, par a par, contra o sort estavel da std.
//                Pega ordem errada, elemento perdido, elemento duplicado e
//                troca entre chaves iguais.
//   sort         (instavel) tem que sair ordenado e com exatamente a mesma
//                colecao de pares {chave, idx}. A ordem entre iguais e livre.
use adaptive_parallel_multimerge_sort::{sort, sort_stable};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Keyed { key: u64, idx: u64 }
impl PartialOrd for Keyed { fn partial_cmp(&self,o:&Self)->Option<std::cmp::Ordering>{Some(self.cmp(o))} }
impl Ord for Keyed { fn cmp(&self,o:&Self)->std::cmp::Ordering{ self.key.cmp(&o.key) } }

fn lcg(s:&mut u64)->u64{ *s=s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); *s>>11 }

fn check(name:&str, mut v:Vec<Keyed>) {
    for (i,e) in v.iter_mut().enumerate(){ e.idx = i as u64; }
    let mut reference = v.clone();
    reference.sort_by(|a,b| a.key.cmp(&b.key));      // sort() do Rust e estavel

    // ---- estavel
    let mut work = v.clone();
    sort_stable(&mut work);
    let ordered = work.windows(2).all(|w| w[0].key <= w[1].key);
    let stable  = work.windows(2).all(|w| w[0].key != w[1].key || w[0].idx < w[1].idx);
    let exact   = work == reference;
    assert!(ordered, "{}: sort_stable NAO ORDENOU", name);
    assert!(stable,  "{}: sort_stable NAO E ESTAVEL", name);
    assert!(exact,   "{}: sort_stable difere do sort estavel de referencia", name);

    // ---- instavel
    let mut work = v.clone();
    sort(&mut work);
    assert!(work.windows(2).all(|w| w[0].key <= w[1].key), "{}: sort NAO ORDENOU", name);
    // Mesma colecao: com (chave, idx) como ordem total, as duas listas tem que
    // coincidir depois de ordenadas -- pega perda e duplicacao.
    work.sort_unstable_by_key(|e| (e.key, e.idx));
    assert!(work == reference, "{}: sort perdeu ou duplicou elemento", name);
}

// O exemplo de uso do README, para ele nao apodrecer.
#[test] fn exemplo_do_readme(){
    use adaptive_parallel_multimerge_sort::{sort_by, sort_stable_by_key, sort_stable_by_with_buffer};
    let mut keys: Vec<u64> = vec![9, 5, 1, 3, 7, 8, 2, 6, 4, 0, 15, 12];
    sort(&mut keys);
    assert!(keys.windows(2).all(|w| w[0] <= w[1]));
    sort_by(&mut keys, |a, b| b.cmp(a));
    assert!(keys.windows(2).all(|w| w[0] >= w[1]));

    let mut rows: Vec<(u32, u64)> = vec![(7, 0), (3, 1), (7, 2), (3, 3)];
    sort_stable(&mut rows);
    sort_stable_by_key(&mut rows, |r| r.0);
    assert_eq!(rows, [(3, 1), (3, 3), (7, 0), (7, 2)]);

    let mut scratch: Vec<(u32, u64)> = Vec::new();
    sort_stable_by_with_buffer(&mut rows, &mut scratch, |a, b| a.0.cmp(&b.0));
    assert_eq!(rows, [(3, 1), (3, 3), (7, 0), (7, 2)]);
}

#[test] fn todos_iguais(){ check("todos iguais", vec![Keyed{key:42,idx:0}; 200_000]); }

#[test] fn baixa_cardinalidade(){
    let mut s=1u64; let v:Vec<_>=(0..200_000).map(|_| Keyed{key:lcg(&mut s)%8, idx:0}).collect();
    check("8 chaves distintas", v);
}

#[test] fn descendente_com_platos(){
    let n=200_000u64;
    let v:Vec<_>=(0..n).map(|i| Keyed{key:(n-i)/500, idx:0}).collect();
    check("descendente com platos de 500", v);
}

// Invertido com chaves repetidas, em varios tamanhos de plato -- inclusive
// platos maiores que o bloco de deteccao e platos de 2, que o escudo antigo
// classificava como caos.
#[test] fn invertido_com_chaves_repetidas(){
    for plato in [2u64, 3, 17, 500, 40_000, 100_000] {
        let n=300_000u64;
        let v:Vec<_>=(0..n).map(|i| Keyed{key:(n-i)/plato, idx:0}).collect();
        check(&format!("invertido, platos de {}", plato), v);
    }
}

#[test] fn sawtooth(){
    let v:Vec<_>=(0..300_000u64).map(|i| Keyed{key:i%1000, idx:0}).collect();
    check("sawtooth", v);
}

// Dentes em faixas de valor disjuntas, direcao sorteada: depois de reverter os
// descendentes o array ja esta ordenado, e nada precisa ser fundido.
#[test] fn dentes_em_faixas_disjuntas(){
    let mut s=5u64;
    for dente in [40u64, 700, 9_000] {
        let n=400_000u64;
        let v:Vec<_>=(0..n).map(|i| {
            let d=i/dente; let k=i%dente;
            let sobe = { let mut t=d.wrapping_mul(0x9E3779B97F4A7C15)^s; lcg(&mut t)&1==0 };
            Keyed{key: d*dente/2 + if sobe { k/2 } else { (dente-1-k)/2 }, idx:0}
        }).collect();
        s = lcg(&mut s);
        check(&format!("dentes disjuntos de {}", dente), v);
    }
}

#[test] fn estruturado_mais_caos(){
    let mut s=7u64;
    let mut v:Vec<_>=(0..300_000u64).map(|i| Keyed{key:i/1000, idx:0}).collect();
    for i in 34_700..34_950 { v[i].key = lcg(&mut s)%4; }
    check("estruturado + caos", v);
}

// O ponto cego do escudo antigo: caos com um trecho ordenado bem no meio, e
// ordem com um trecho caotico bem no meio.
#[test] fn remendo_no_meio(){
    let mut s=11u64;
    let n=400_000usize;
    let mut v:Vec<_>=(0..n).map(|_| Keyed{key:lcg(&mut s)%100_000, idx:0}).collect();
    v[n/2-200..n/2+300].sort();
    check("caos + remendo ordenado no meio", v);

    let mut v:Vec<_>=(0..n as u64).map(|i| Keyed{key:i/7, idx:0}).collect();
    for i in n/2-200..n/2+300 { v[i].key = lcg(&mut s)%100_000; }
    check("ordem + remendo caotico no meio", v);
}

#[test] fn bordas(){
    let mut s=3u64;
    for n in [0usize,1,2,3,119,120,4095,4096,4097,8190,8191,32767,32768,32769,65534] {
        let v:Vec<_>=(0..n).map(|_| Keyed{key:lcg(&mut s)%16, idx:0}).collect();
        check(&format!("n={}",n), v);
    }
}

#[test] fn fuzz_estruturado_caotico(){
    let mut s=99u64;
    for t in 0..150 {
        let n = 5_000 + (lcg(&mut s) as usize % 250_000);
        let card = 1 + lcg(&mut s)%500;
        let mut v:Vec<_>=(0..n).map(|i| Keyed{key:(i as u64/13)%card, idx:0}).collect();
        let patches = 1 + lcg(&mut s)%6;
        for _ in 0..patches {
            let st = lcg(&mut s) as usize % n;
            let len = 200 + lcg(&mut s) as usize % 6000;
            for i in st..(st+len).min(n) { v[i].key = lcg(&mut s)%card; }
        }
        check(&format!("fuzz #{}",t), v);
    }
}
