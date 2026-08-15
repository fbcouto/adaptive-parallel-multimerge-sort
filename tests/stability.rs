use adaptive_parallel_multimerge_sort::sort as multi_merge_sort;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Keyed { key: u64, idx: u64 }
impl PartialOrd for Keyed { fn partial_cmp(&self,o:&Self)->Option<std::cmp::Ordering>{Some(self.cmp(o))} }
impl Ord for Keyed { fn cmp(&self,o:&Self)->std::cmp::Ordering{ self.key.cmp(&o.key) } }

fn lcg(s:&mut u64)->u64{ *s=s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); *s>>11 }

fn check(name:&str, mut v:Vec<Keyed>) {
    for (i,e) in v.iter_mut().enumerate(){ e.idx = i as u64; }
    let mut reference = v.clone();
    reference.sort_by(|a,b| a.key.cmp(&b.key));      // sort() do Rust e estavel
    let mut work = v.clone();
    multi_merge_sort(&mut work);
    let ordered = work.windows(2).all(|w| w[0].key <= w[1].key);
    let stable  = work.windows(2).all(|w| w[0].key != w[1].key || w[0].idx < w[1].idx);
    let exact   = work == reference;
    assert!(ordered, "{}: NAO ORDENADO", name);
    assert!(stable,  "{}: NAO ESTAVEL", name);
    assert!(exact,   "{}: difere do sort estavel de referencia", name);
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

#[test] fn sawtooth(){
    let v:Vec<_>=(0..300_000u64).map(|i| Keyed{key:i%1000, idx:0}).collect();
    check("sawtooth", v);
}

#[test] fn estruturado_mais_caos(){
    let mut s=7u64;
    let mut v:Vec<_>=(0..300_000u64).map(|i| Keyed{key:i/1000, idx:0}).collect();
    for i in 34_700..34_950 { v[i].key = lcg(&mut s)%4; }
    check("estruturado + caos", v);
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
