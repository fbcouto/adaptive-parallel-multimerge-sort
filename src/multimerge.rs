// =====================================================================
//  ADAPTIVE PARALLEL MULTIMERGE SORT -- motor Rust
//
//  Dois motores sobre a mesma deteccao de runs e a mesma arvore de merge:
//
//    multi_merge_sort          NAO estavel. E o padrao.
//    multi_merge_sort_stable   estavel. Para quando a ordem relativa de chaves
//                              iguais importa (SGBD, ordenacao em cascata).
//
//  Cada um tem as variantes `_by` (comparador) e `_by_key` (extrator de chave).
//
//  sort(arr)
//    |- n pequeno .......................... sort da std
//    |- ESCUDO (janelas espalhadas) ........ caotico -> caminho caotico
//    '- DETECCAO de runs (monoide, paralela)
//         |- runs curtos consecutivos ...... viram UM bloco, ordenado pela std
//         |- runs descendentes ............. revertidos no lugar
//         |- runs ja em ordem global ....... fundidos sem mover dado
//         '- o que sobrar .................. arvore de merge k-vias
//
//  ONDE OS DOIS MOTORES DIFEREM
//
//  1. Run descendente com chaves repetidas. Um trecho 9 9 7 7 3 e descendente
//     (nao-crescente), mas inverte-lo troca a ordem dos iguais. O instavel
//     inverte e pronto. O estavel inverte e depois desinverte cada bloco de
//     iguais (`restore_ties`), o que devolve a ordem original entre eles. Os
//     dois resolvem o caso em O(n); antes o motor quebrava o trecho em um run
//     por plato e pagava a arvore inteira.
//  2. Bloco sem estrutura. O estavel usa o `sort` da std (driftsort), o
//     instavel usa o `sort_unstable` (ipnsort).
//  3. Dado caotico. O estavel ordena em blocos e funde pela arvore; o instavel
//     delega ao quicksort paralelo, que trabalha no lugar e nao aloca buffer.
//
//  CHAVE RADIX
//
//  As entradas `_radix`, `_by_radix_key` e `_by_prefix_key` dizem ao motor
//  quais sao os BITS da chave. Com isso o dado aleatorio deixa de passar por
//  comparacao: uma passada de radix pelos bits altos e cada balde resolvido
//  dentro do cache (ver "CAMINHO CAOTICO COM CHAVE"). Vale para os dois
//  motores; o resto do caminho -- escudo, runs, arvore -- nao muda.
//
//  O motor nao tem `unsafe`. `T: Copy` e o que permite ao borrow checker
//  provar o merge bidirecional: as duas threads leem as fontes por referencia
//  compartilhada e escrevem em metades disjuntas do destino.
// =====================================================================

use rayon::prelude::*;
use std::cmp::Ordering::{self, Equal, Greater, Less};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

/// Fan-out of the k-way merge.
pub const KWAY_FANOUT: usize = 8;

// =====================================================================
//  API PUBLICA
// =====================================================================

/// Sorts the slice. **Not stable**: equal elements may be reordered.
///
/// This is the default engine. It detects pre-existing order (ascending runs,
/// descending runs, blocks already in global order) and only pays for the
/// disorder that is actually there. Use [`multi_merge_sort_stable`] when the
/// relative order of equal keys must be preserved.
///
/// Allocates a buffer of `arr.len()` elements only when runs have to be
/// merged. Chaotic input is sorted in place.
pub fn multi_merge_sort<T: Ord + Copy + Send + Sync>(arr: &mut [T]) {
    sort_with(arr, &T::cmp, false, &Tuning::for_type::<T>(), None);
}

/// [`multi_merge_sort`] with a comparator.
pub fn multi_merge_sort_by<T, F>(arr: &mut [T], compare: F)
where
    T: Copy + Send + Sync,
    F: Fn(&T, &T) -> Ordering + Sync,
{
    sort_with(arr, &compare, false, &Tuning::for_type::<T>(), None);
}

/// [`multi_merge_sort`] with a key extractor.
pub fn multi_merge_sort_by_key<T, K, F>(arr: &mut [T], key: F)
where
    T: Copy + Send + Sync,
    K: Ord,
    F: Fn(&T) -> K + Sync,
{
    sort_with(arr, &|a: &T, b: &T| key(a).cmp(&key(b)), false, &Tuning::for_type::<T>(), None);
}

/// Sorts the slice, **preserving the relative order of equal elements**.
///
/// Same adaptive machinery as [`multi_merge_sort`]. Always allocates a buffer
/// of `arr.len()` elements unless the input turns out to be a single run.
pub fn multi_merge_sort_stable<T: Ord + Copy + Send + Sync>(arr: &mut [T]) {
    sort_with(arr, &T::cmp, true, &Tuning::for_type::<T>(), None);
}

/// [`multi_merge_sort_stable`] with a comparator.
pub fn multi_merge_sort_stable_by<T, F>(arr: &mut [T], compare: F)
where
    T: Copy + Send + Sync,
    F: Fn(&T, &T) -> Ordering + Sync,
{
    sort_with(arr, &compare, true, &Tuning::for_type::<T>(), None);
}

/// [`multi_merge_sort_stable`] with a key extractor.
pub fn multi_merge_sort_stable_by_key<T, K, F>(arr: &mut [T], key: F)
where
    T: Copy + Send + Sync,
    K: Ord,
    F: Fn(&T) -> K + Sync,
{
    sort_with(arr, &|a: &T, b: &T| key(a).cmp(&key(b)), true, &Tuning::for_type::<T>(), None);
}

/// [`multi_merge_sort_by`] reusing the caller's working memory.
///
/// `scratch` is grown to `arr.len()` on first use and kept between calls. A
/// server that sorts repeatedly pays for the buffer once: no allocation and no
/// fresh pages to fault in on later calls. Its contents are unspecified
/// afterwards. Pass `T::cmp` as the comparator for the natural order.
pub fn multi_merge_sort_by_with_buffer<T, F>(arr: &mut [T], scratch: &mut Vec<T>, compare: F)
where
    T: Copy + Send + Sync,
    F: Fn(&T, &T) -> Ordering + Sync,
{
    sort_with(arr, &compare, false, &Tuning::for_type::<T>(), Some(scratch));
}

/// [`multi_merge_sort_stable_by`] reusing the caller's working memory. See
/// [`multi_merge_sort_by_with_buffer`].
pub fn multi_merge_sort_stable_by_with_buffer<T, F>(
    arr: &mut [T],
    scratch: &mut Vec<T>,
    compare: F,
) where
    T: Copy + Send + Sync,
    F: Fn(&T, &T) -> Ordering + Sync,
{
    sort_with(arr, &compare, true, &Tuning::for_type::<T>(), Some(scratch));
}

/// Integer types whose order is the order of an unsigned integer of up to 64
/// bits. Implemented for the primitive integers.
///
/// A type that implements it can use [`multi_merge_sort_radix`] and
/// [`multi_merge_sort_stable_radix`], where chaotic input is sorted by radix
/// instead of by comparison.
pub trait RadixKey: Ord + Copy + Send + Sync {
    /// Order-preserving map to `u64`: `a < b` exactly when
    /// `a.to_radix() < b.to_radix()`.
    fn to_radix(self) -> u64;
    /// Inverse of [`RadixKey::to_radix`].
    fn from_radix(bits: u64) -> Self;
}

macro_rules! radix_key_unsigned {
    ($($t:ty),*) => {$(
        impl RadixKey for $t {
            #[inline(always)]
            fn to_radix(self) -> u64 {
                self as u64
            }
            #[inline(always)]
            fn from_radix(bits: u64) -> Self {
                bits as $t
            }
        }
    )*};
}
radix_key_unsigned!(u8, u16, u32, u64, usize);

macro_rules! radix_key_signed {
    ($($t:ty => $u:ty),*) => {$(
        impl RadixKey for $t {
            #[inline(always)]
            fn to_radix(self) -> u64 {
                ((self as $u) ^ (1 << (<$u>::BITS - 1))) as u64
            }
            #[inline(always)]
            fn from_radix(bits: u64) -> Self {
                ((bits as $u) ^ (1 << (<$u>::BITS - 1))) as $t
            }
        }
    )*};
}
radix_key_signed!(i8 => u8, i16 => u16, i32 => u32, i64 => u64, isize => usize);

/// [`multi_merge_sort`] for integer keys. **Not stable.**
///
/// Same engine and same result. The difference is the chaotic path: random
/// input is sorted by a parallel radix pass plus in-cache bucket sorts instead
/// of the parallel quicksort. That path allocates a buffer of `arr.len()`
/// elements, except when the keys span 16 bits or less (counting sort, no
/// buffer). If the buffer cannot be allocated, or the keys are too skewed for
/// radix to pay off, the quicksort takes over.
pub fn multi_merge_sort_radix<T: RadixKey>(arr: &mut [T]) {
    sort_with_key(arr, &T::cmp, false, &Tuning::for_type::<T>(), None, &Bare);
}

/// [`multi_merge_sort_stable`] for integer keys. See
/// [`multi_merge_sort_radix`].
pub fn multi_merge_sort_stable_radix<T: RadixKey>(arr: &mut [T]) {
    sort_with_key(arr, &T::cmp, true, &Tuning::for_type::<T>(), None, &Bare);
}

/// Sorts by a `u64` key that decides the order completely: elements with equal
/// keys are equal. **Not stable.** Chaotic input is sorted by radix on the key.
pub fn multi_merge_sort_by_radix_key<T, K>(arr: &mut [T], key: K)
where
    T: Copy + Send + Sync,
    K: Fn(&T) -> u64 + Sync,
{
    let cmp = |a: &T, b: &T| key(a).cmp(&key(b));
    sort_with_key(arr, &cmp, false, &Tuning::for_type::<T>(), None, &Exact(&key));
}

/// [`multi_merge_sort_by_radix_key`], preserving the relative order of
/// elements with equal keys.
pub fn multi_merge_sort_stable_by_radix_key<T, K>(arr: &mut [T], key: K)
where
    T: Copy + Send + Sync,
    K: Fn(&T) -> u64 + Sync,
{
    let cmp = |a: &T, b: &T| key(a).cmp(&key(b));
    sort_with_key(arr, &cmp, true, &Tuning::for_type::<T>(), None, &Exact(&key));
}

/// Sorts with `compare`, using `prefix` as a radix key for chaotic input.
/// **Not stable.**
///
/// `prefix` must agree with `compare` wherever it can tell two elements apart:
/// `prefix(a) < prefix(b)` implies `compare(a, b) == Less`. Elements with the
/// same prefix are ordered by `compare`. This is the entry for keys longer
/// than 64 bits, such as strings: the prefix is the first 8 bytes, big-endian.
pub fn multi_merge_sort_by_prefix_key<T, K, F>(arr: &mut [T], prefix: K, compare: F)
where
    T: Copy + Send + Sync,
    K: Fn(&T) -> u64 + Sync,
    F: Fn(&T, &T) -> Ordering + Sync,
{
    sort_with_key(arr, &compare, false, &Tuning::for_type::<T>(), None, &Prefix(&prefix));
}

/// [`multi_merge_sort_by_prefix_key`], preserving the relative order of equal
/// elements.
pub fn multi_merge_sort_stable_by_prefix_key<T, K, F>(arr: &mut [T], prefix: K, compare: F)
where
    T: Copy + Send + Sync,
    K: Fn(&T) -> u64 + Sync,
    F: Fn(&T, &T) -> Ordering + Sync,
{
    sort_with_key(arr, &compare, true, &Tuning::for_type::<T>(), None, &Prefix(&prefix));
}

/// Run decomposition of `arr` as signed lengths: positive for a non-decreasing
/// run, negative for a non-increasing one.
///
/// A negative run may contain repeated keys. Reversing it sorts it, but only
/// [`multi_merge_sort_stable`] restores the original order among the equals.
pub fn detect_global_trend<T: Ord + Sync>(arr: &[T]) -> Vec<i64> {
    let t = Tuning::for_type::<T>();
    detect_runs(arr, &T::cmp, t.detect_block, t.min_run, usize::MAX)
        .unwrap_or_default()
        .iter()
        .map(|r| if r.is_desc() { -(r.len() as i64) } else { r.len() as i64 })
        .collect()
}

// =====================================================================
//  AJUSTES
//
//  Um unico lugar para os limiares. `pub(crate)` para que os testes forcem
//  recursao profunda com dados pequenos, e para que uma sonda de medicao
//  varra valores sem recompilar o motor.
// =====================================================================

// `Structured` e `Chaotic` so sao construidos pelos testes e pela sonda de
// medicao: forcam um caminho para exercita-lo ou cronometra-lo isolado.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Route {
    /// O escudo decide.
    Auto,
    /// Forca deteccao de runs + arvore.
    Structured,
    /// Forca o caminho caotico.
    Chaotic,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Tuning {
    /// Ate aqui o sort da std resolve sozinho.
    pub small_sort: usize,
    /// Bloco da deteccao de runs (com 1 elemento de sobreposicao).
    pub detect_block: usize,
    /// Run mais curto que isto nao vale um no da arvore: runs curtos
    /// consecutivos sao agrupados e ordenados como um bloco.
    pub min_run: usize,
    /// A arvore so entra quando, depois de encadear, os runs tem em media
    /// pelo menos este comprimento.
    ///
    /// Em u64 a arvore sobre runs naturais e o caminho de blocos empatam de
    /// 256 ate 16384 elementos por run, e os blocos ganham de 8% a 15% entre
    /// 32 e 64. Com chave cara (`&str`) a arvore ganha com folga mesmo em run
    /// curto, porque reordenar o bloco repete comparacoes que os runs ja
    /// tinham resolvido: ordenado com 1% de trocas (run medio ~50), 367 ms
    /// contra 520; 64 runs de 31 mil strings, 175 contra 262. O valor fica
    /// baixo para proteger a chave cara, que e onde o erro custa mais.
    pub struct_min_avg_run: usize,
    /// O mesmo limiar quando a comparacao e barata. Ali o custo e mover
    /// memoria, nao comparar, e os blocos empatam com a arvore de 256 para
    /// cima e ganham abaixo: ordenado com 1% de trocas em 10M u64, 128 ms nos
    /// blocos contra 146 na arvore.
    pub struct_min_avg_run_cheap: usize,
    /// Piso do numero de segmentos aceitos, para arrays pequenos.
    pub min_segs: usize,
    /// Folga do aborto antecipado da deteccao.
    pub abort_slack: usize,
    /// Tamanho do bloco ordenado pela std (caminho caotico estavel e teto do
    /// agrupamento de runs curtos).
    pub chunk: usize,
    /// Grao paralelo do merge: acima disto corta por co-rank / multisequencia.
    pub par_grain: usize,
    /// Acima disto um pedaco de merge binario roda bidirecional (2 tarefas).
    pub bidir_min: usize,
    /// No da arvore binaria abaixo disto recursa sem `rayon::join`.
    pub seq_node: usize,
    /// Subarvore ate este tamanho usa merge binario; acima, k-vias.
    pub kway_min_elems: usize,
    /// Tile do merge k-vias, em elementos.
    pub tile: usize,
    /// Escudo: acima deste percentual de inversoes de direcao os runs sao
    /// curtos demais para a deteccao valer a passada.
    pub chaos_percent: usize,
    /// Escudo: acima deste percentual o dado e aleatorio de fato, nao apenas
    /// de runs curtos. So interessa ao motor instavel.
    pub random_percent: usize,
    /// Comparacao "barata": ate este custo medio, em nanossegundos, medido na
    /// amostra do escudo.
    pub cheap_ns: usize,
    /// Fixa o veredito de custo em vez de medir. Para os testes.
    pub force_cheap: Option<bool>,
    /// Radix: abaixo deste tamanho o caminho caotico do INSTAVEL segue por
    /// comparacao. Medido com o motor (minimo de 60, 8 threads): em i32 o
    /// radix perde ate 65 mil (0,87 contra 0,64 ms) e ganha de 131 mil em
    /// diante (1,04 contra 1,30); em u64 empata em 262 mil e ganha em 1M.
    pub radix_min: usize,
    /// O mesmo piso para o ESTAVEL, que e mais baixo: ali o concorrente e o
    /// sort estavel em blocos, e o radix ja ganha em 32 mil (0,59 contra 0,79
    /// ms em i32; 0,54 contra 0,80 em u64) e empata em 16 mil.
    pub radix_min_stable: usize,
    /// Radix: bits do digito da passada MSD. 8 venceu 10 (mais baldes custam
    /// mais no espalhamento do que economizam nas folhas).
    pub radix_digit: u32,
    /// Radix: chave "nua" cuja faixa cabe nestes bits e resolvida por
    /// contagem, sem buffer.
    pub radix_count_bits: u32,
    /// Radix: o balde e ordenado por LSD quando sobram ate estes bits e a
    /// comparacao e barata; acima, pelo sort da std. Em i32 (sobram 24) o LSD
    /// vence; em u64 de faixa inteira (sobram 56) perde.
    pub radix_lsd_cheap: u32,
    /// O mesmo limiar quando a comparacao e cara: ali o LSD compensa por mais
    /// passadas (registro de 6 bytes, sobram 39: 411 ms contra 741 em 30M).
    pub radix_lsd_costly: u32,
    /// Radix: elementos por pedaco do histograma e do espalhamento paralelos.
    pub radix_piece: usize,
    /// Radix: balde acima disto, e de 1/`radix_big_div` do trecho, passa pela
    /// rotina de novo em vez de ir para uma folha sequencial.
    pub radix_big: usize,
    pub radix_big_div: usize,
    /// Radix: quando a folha NAO e um LSD curto (sobram mais bits que
    /// `radix_lsd_cheap`: chave larga, texto), o balde acima deste tamanho
    /// leva um segundo corte MSD antes da folha. Medido com o motor, 8
    /// threads, corte em 64 mil contra sem corte: texto de 6 bytes em u64,
    /// 30M, 356 -> 320 ms; u64 aleatorio, 100M, 1190 -> 1073 (estavel 1796 ->
    /// 1211). Com folha LSD curta o corte ATRAPALHA (i32, 100M: 562 -> 606),
    /// por isso a regra olha a folha.
    pub radix_recut: usize,
    /// Radix: um balde com mais de 1/`radix_skew` do dado denuncia chave
    /// enviesada; nada e movido e o caminho por comparacao assume.
    pub radix_skew: usize,
    /// Radix: balde menor que isto vai direto para o sort da std.
    pub radix_leaf_min: usize,
    pub route: Route,
}

impl Tuning {
    pub(crate) fn for_type<T>() -> Self {
        let e = std::mem::size_of::<T>().max(1);
        Tuning {
            small_sort: 4096,
            detect_block: 32_768,
            min_run: 16,
            struct_min_avg_run: 48,
            struct_min_avg_run_cheap: 256,
            min_segs: 8,
            abort_slack: 64,
            chunk: 65_536,
            par_grain: 65_536,
            bidir_min: 4096,
            seq_node: 2048,
            kway_min_elems: ((2usize << 20) / e).max(2),
            tile: (32_768 / (8 * e)).clamp(128, 2048),
            chaos_percent: 9,
            random_percent: 35,
            cheap_ns: 10,
            force_cheap: None,
            radix_min: if e <= 4 { 131_072 } else { 524_288 },
            radix_min_stable: 32_768,
            radix_digit: 8,
            radix_count_bits: 16,
            radix_lsd_cheap: 24,
            radix_lsd_costly: 40,
            radix_piece: 65_536,
            radix_big: 2 << 20,
            radix_big_div: 16,
            radix_recut: 65_536,
            radix_skew: 4,
            radix_leaf_min: 128,
            route: Route::Auto,
        }
    }

    /// Quantos segmentos um array de `n` elementos pode ter para a arvore
    /// ainda compensar.
    fn allowed_segs(&self, n: usize, cheap: bool) -> usize {
        let avg = if cheap { self.struct_min_avg_run_cheap } else { self.struct_min_avg_run };
        (n / avg.max(1)).max(self.min_segs)
    }
}

// =====================================================================
//  ENTRADA
// =====================================================================

pub(crate) fn sort_with<T, F>(
    arr: &mut [T],
    cmp: &F,
    stable: bool,
    t: &Tuning,
    scratch: Option<&mut Vec<T>>,
) where
    T: Copy + Send + Sync,
    F: Fn(&T, &T) -> Ordering + Sync,
{
    sort_with_key(arr, cmp, stable, t, scratch, &NoKey);
}

/// `sort_with` com a estrategia de chave do caminho caotico: `NoKey` (so
/// comparacao), `Bare`, `Exact` ou `Prefix`.
pub(crate) fn sort_with_key<T, F, K>(
    arr: &mut [T],
    cmp: &F,
    stable: bool,
    t: &Tuning,
    scratch: Option<&mut Vec<T>>,
    key: &K,
) where
    T: Copy + Send + Sync,
    F: Fn(&T, &T) -> Ordering + Sync,
    K: Keyer<T>,
{
    let n = arr.len();
    if n < 2 {
        return;
    }
    if n <= t.small_sort {
        leaf_sort(arr, cmp, stable);
        return;
    }
    let probe = match t.route {
        Route::Chaotic => Probe { rate: 100, cheap: false },
        Route::Structured => Probe { rate: 0, cheap: false },
        Route::Auto => sample(arr, cmp, t.cheap_ns),
    };
    let rate = probe.rate;
    let cheap = t.force_cheap.unwrap_or(probe.cheap);
    if rate > t.chaos_percent {
        let kind = if rate > t.random_percent { Chaos::Random } else { Chaos::ShortRuns };
        sort_chaotic(arr, cmp, stable, t, scratch, kind, cheap, key);
        return;
    }
    // ATALHO DO INSTAVEL: dado globalmente ascendente, com desordem espalhada
    // e comparacao barata, e o melhor caso do quicksort paralelo -- ordenado
    // com 1% de trocas em 10M u64 sai em 55 ms, contra 113-128 de qualquer
    // caminho de merge. Exige que haja desordem na amostra (senao a deteccao
    // resolve em uma passada) e que a comparacao seja barata (com `&str` o
    // mesmo dado leva 733 ms nele contra 367 da arvore).
    if !stable
        && t.route == Route::Auto
        && cheap
        && rate >= 1
        && globally_ascending(arr, cmp)
    {
        arr.par_sort_unstable_by(|a, b| cmp(a, b));
        return;
    }
    // A deteccao desiste sozinha quando o dado se mostra fragmentado demais
    // para a arvore: o custo de ter perguntado fica limitado a uma fracao da
    // varredura, em vez da passada inteira.
    let limit = match t.route {
        Route::Auto => 2 * t.allowed_segs(n, cheap) + t.abort_slack,
        _ => usize::MAX,
    };
    match detect_runs(arr, cmp, t.detect_block, t.min_run, limit) {
        Some(runs) => sort_structured(arr, &runs, cmp, stable, t, scratch, cheap, key),
        None => sort_chaotic(arr, cmp, stable, t, scratch, Chaos::Fragmented, cheap, key),
    }
}

#[inline]
fn leaf_sort<T, F>(s: &mut [T], cmp: &F, stable: bool)
where
    F: Fn(&T, &T) -> Ordering,
{
    if stable {
        s.sort_by(|a, b| cmp(a, b));
    } else {
        s.sort_unstable_by(|a, b| cmp(a, b));
    }
}

// =====================================================================
//  FASE 0: ESCUDO
//
//  Oito janelas espalhadas pelo array inteiro, nao uma janela no meio: um
//  ponto unico deixava 0,004% do dado decidir o caminho dos outros 99,996%.
//
//  Empates NAO contam como direcao. 4 4 3 3 2 2 e um run descendente perfeito,
//  e contar `<=` como "subindo" o classificava como caos puro.
//
//  O veredito deixou de ser critico: errar para "estruturado" custa uma
//  passada de deteccao, porque runs curtos sao agrupados e ordenados em bloco
//  de qualquer forma; errar para "caotico" custa a mesma passada, porque os
//  blocos ja ordenados sao fundidos sem mover dado.
// =====================================================================

/// O que a amostra do escudo diz sobre o dado.
struct Probe {
    /// Percentual de comparacoes em que a direcao inverteu. Aleatorio de alta
    /// cardinalidade da ~66; de duas chaves, ~50; um dente de serra de
    /// comprimento L da 200/L. Ordenado, invertido e platos dao zero.
    rate: usize,
    /// A comparacao e barata? Medido no relogio, sobre as mesmas 512
    /// comparacoes: um inteiro custa 1 a 3 ns, uma string fora de cache passa
    /// de 50. O erro so tem uma direcao possivel -- uma preempcao no meio da
    /// amostra faz o barato parecer caro -- e essa e a direcao conservadora:
    /// o motor deixa de usar um atalho, nunca usa um que nao devia.
    cheap: bool,
}

fn sample<T, F>(arr: &[T], cmp: &F, cheap_ns: usize) -> Probe
where
    F: Fn(&T, &T) -> Ordering,
{
    const WINDOWS: usize = 8;
    const SPAN: usize = 64;
    let n = arr.len();
    if n < 3 {
        return Probe { rate: 0, cheap: false };
    }
    let clock = std::time::Instant::now();
    let span = SPAN.min(n - 1);
    let last_start = n - 1 - span;
    let (mut seen, mut changes) = (0usize, 0usize);
    for w in 0..WINDOWS {
        let start = last_start / (WINDOWS - 1) * w;
        let mut dir = Equal;
        for i in start..start + span {
            let o = cmp(&arr[i], &arr[i + 1]);
            if o != Equal {
                if dir != Equal && o != dir {
                    changes += 1;
                }
                dir = o;
            }
        }
        seen += span;
        if last_start == 0 {
            break;
        }
    }
    let nanos = clock.elapsed().as_nanos() as usize;
    Probe {
        rate: changes * 100 / seen.max(1),
        cheap: nanos <= cheap_ns * seen,
    }
}

/// O array e globalmente ascendente, a menos de elementos fora do lugar?
///
/// 256 amostras igualmente espacadas, em 16 grupos; a MEDIANA de cada grupo
/// tem que nao decrescer. A mediana ignora elementos isolados fora do lugar
/// (ordenado com 1% de trocas passa), e runs intercalados reprovam: a cada
/// troca de run o nivel cai e fica la embaixo.
///
/// Serve para uma decisao so: se o motor instavel pode entregar um dado
/// fragmentado ao quicksort. Ver `sort_chaotic`.
fn globally_ascending<T, F>(arr: &[T], cmp: &F) -> bool
where
    T: Copy,
    F: Fn(&T, &T) -> Ordering,
{
    const GROUPS: usize = 16;
    const PER: usize = 16;
    let n = arr.len();
    if n < GROUPS * PER * 4 {
        return false;
    }
    let step = n / (GROUPS * PER);
    let mut prev: Option<T> = None;
    for g in 0..GROUPS {
        let mut s = [arr[0]; PER];
        for (k, slot) in s.iter_mut().enumerate() {
            *slot = arr[(g * PER + k) * step];
        }
        s.sort_unstable_by(|a, b| cmp(a, b));
        let med = s[PER / 2];
        if let Some(p) = prev {
            if cmp(&p, &med) == Greater {
                return false;
            }
        }
        prev = Some(med);
    }
    true
}

// =====================================================================
//  FASE 1: DETECCAO DE RUNS
//
//  Um run e descrito por (tipo, comprimento) empacotado em 64 bits.
//
//    ASC        nao-decrescente
//    FLAT       todos iguais -- neutro: serve tanto a um vizinho ascendente
//               quanto a um descendente
//    DESC       estritamente decrescente
//    DESC_TIES  nao-crescente COM chaves repetidas
//
//  A direcao de um run e decidida pelo PRIMEIRO par estritamente diferente,
//  nao pelo primeiro par. Comecar por `<=` classificava todo plato como
//  ascendente, e um array invertido com chaves repetidas nunca era visto como
//  invertido: virava um run por plato.
// =====================================================================

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Run(u64);

const ASC: u64 = 0;
const FLAT: u64 = 1;
const DESC: u64 = 2;
const DESC_TIES: u64 = 3;

impl Run {
    #[inline]
    fn new(kind: u64, len: usize) -> Run {
        Run((kind << 62) | len as u64)
    }
    #[inline]
    fn kind(self) -> u64 {
        self.0 >> 62
    }
    #[inline]
    fn len(self) -> usize {
        (self.0 & ((1u64 << 62) - 1)) as usize
    }
    #[inline]
    fn is_desc(self) -> bool {
        self.kind() >= DESC
    }
}

/// Decompoe `arr` em runs, acrescentando a `out`.
///
/// Devolve uma estimativa de quantos SEGMENTOS este trecho custaria a arvore:
/// um trecho de runs curtos conta uma vez (vira um bloco ordenado), e um run
/// longo que encadeia com o anterior nao conta (vai ser fundido sem mover
/// dado). E essa contagem que alimenta o aborto antecipado da deteccao.
fn scan_runs<T, F>(arr: &[T], cmp: &F, out: &mut Vec<Run>, min_run: usize) -> usize
where
    F: Fn(&T, &T) -> Ordering,
{
    let n = arr.len();
    let first = out.len();
    let mut head = 0usize;
    let mut count = SegCount { segs: 0, have_prev: false, prev_long: false, prev_max: 0, in_short: false };
    while head < n {
        // Plato inicial: anda ate o primeiro par estritamente diferente.
        let mut tail = head + 1;
        let mut ord = Equal;
        while tail < n {
            ord = cmp(&arr[tail - 1], &arr[tail]);
            if ord != Equal {
                break;
            }
            tail += 1;
        }
        if tail == n {
            out.push(Run::new(FLAT, n - head));
            count.add(arr, cmp, head, n, false, min_run);
            break;
        }
        let plateau = tail - head > 1;
        tail += 1;
        if ord == Less {
            while tail < n && cmp(&arr[tail - 1], &arr[tail]) != Greater {
                tail += 1;
            }
            out.push(Run::new(ASC, tail - head));
            count.add(arr, cmp, head, tail, false, min_run);
        } else {
            let mut ties = plateau;
            while tail < n {
                match cmp(&arr[tail - 1], &arr[tail]) {
                    Less => break,
                    Equal => ties = true,
                    Greater => {}
                }
                tail += 1;
            }
            // ELEMENTO ROUBADO. A varredura gulosa entrega ao run ascendente
            // anterior o primeiro elemento deste run descendente -- que e o
            // MAXIMO dele. Em ... 3 9 | 8 7 6 5 4 o 9 fica a esquerda, e ai o
            // run ascendente termina em 9 e nunca encadeia com o descendente,
            // cujo minimo e 4. Devolver o 9 deixa 3 <= 4 e os dois viram um
            // run so. Sem isto o dente de serra em faixas disjuntas ficava com
            // um run a cada troca de direcao (medido: 1,25x de ganho contra
            // 11x com a devolucao).
            //
            // O roubo pode levar MAIS de um elemento: a varredura ascendente
            // aceita empate, entao em ... 3 9 9 | 8 7 6 os dois 9 ficam a
            // esquerda. Devolver so um deixava o ascendente terminando em 9 e
            // o array inteiro ia para um merge (medido em int32, 30M, 1000
            // dentes: 13,6 ms sem a duplicata, 66 ms com UMA duplicata no
            // topo de um dente). Devolve-se o plato final inteiro; com dois ou
            // mais iguais o descendente passa a ter chave repetida.
            let mut start = head;
            if out.len() > first {
                let last = out.len() - 1;
                let prev = out[last];
                if prev.kind() == ASC && prev.len() >= 2 {
                    let mut k = 1;
                    while k < prev.len() - 1 && cmp(&arr[head - 1 - k], &arr[head - 1]) == Equal {
                        k += 1;
                    }
                    if cmp(&arr[head - 1 - k], &arr[tail - 1]) != Greater {
                        let l = prev.len() - k;
                        out[last] = Run::new(if l == 1 { FLAT } else { ASC }, l);
                        start = head - k;
                        count.prev_max = head - 1 - k;
                        if k > 1 {
                            ties = true;
                        }
                    }
                }
            }
            out.push(Run::new(if ties { DESC_TIES } else { DESC }, tail - start));
            count.add(arr, cmp, start, tail, true, min_run);
        }
        head = tail;
    }
    count.segs
}

/// Contabilidade de segmentos de um bloco, para o aborto antecipado.
struct SegCount {
    segs: usize,
    have_prev: bool,
    /// O run anterior era longo?
    prev_long: bool,
    /// Indice do MAIOR elemento do run anterior.
    prev_max: usize,
    /// Estamos dentro de um trecho de runs curtos (ja contado)?
    in_short: bool,
}

impl SegCount {
    /// Registra o run `arr[start..end]`.
    fn add<T, F>(&mut self, arr: &[T], cmp: &F, start: usize, end: usize, desc: bool, min_run: usize)
    where
        F: Fn(&T, &T) -> Ordering,
    {
        let (min_i, max_i) = if desc { (end - 1, start) } else { (start, end - 1) };
        if end - start >= min_run {
            // O primeiro run do bloco nao conta: nao da para saber daqui se
            // ele encadeia com o bloco anterior.
            let chained = !self.have_prev
                || (self.prev_long && cmp(&arr[self.prev_max], &arr[min_i]) != Greater);
            if !chained {
                self.segs += 1;
            }
            self.prev_long = true;
            self.in_short = false;
        } else {
            if !self.in_short {
                self.segs += 1;
                self.in_short = true;
            }
            self.prev_long = false;
        }
        self.have_prev = true;
        self.prev_max = max_i;
    }
}

/// Costura dois runs que COMPARTILHAM um elemento: o ultimo de `l` e o
/// primeiro de `r`. Opera so sobre o metadata, sem ler o array -- e a operacao
/// do monoide. Devolve o run da esquerda e, se sobrar, o da direita.
///
/// A associatividade vale a menos de decomposicoes equivalentes: agrupamentos
/// diferentes podem cortar um elemento isolado em lugares diferentes, e todas
/// as respostas descrevem o mesmo array corretamente.
fn join_runs(l: Run, r: Run) -> (Run, Option<Run>) {
    let (lk, ll, rk, rl) = (l.kind(), l.len(), r.kind(), r.len());
    let kind = match (lk, rk) {
        (FLAT, FLAT) => Some(FLAT),
        (ASC, ASC) | (ASC, FLAT) | (FLAT, ASC) => Some(ASC),
        // FLAT ao lado de um descendente: entra no run. Se o plato tem mais
        // de um elemento, o run passa a ter chaves repetidas.
        (FLAT, _) => Some(if ll > 1 { DESC_TIES } else { rk }),
        (_, FLAT) => Some(if rl > 1 { DESC_TIES } else { lk }),
        // Direcoes opostas: nao ha como unir.
        (ASC, _) | (_, ASC) => None,
        _ => Some(if lk == DESC_TIES || rk == DESC_TIES { DESC_TIES } else { DESC }),
    };
    match kind {
        Some(k) => (Run::new(k, ll + rl - 1), None),
        None => {
            // O elemento compartilhado fica com a esquerda.
            let rest = match rl - 1 {
                0 => None,
                1 => Some(Run::new(FLAT, 1)),
                m => Some(Run::new(rk, m)),
            };
            (l, rest)
        }
    }
}

/// Costura, no lugar, o run em `mark - 1` com o run em `mark`.
fn stitch_at(v: &mut Vec<Run>, mark: usize) {
    if mark == 0 || mark >= v.len() {
        return;
    }
    let (l, r) = join_runs(v[mark - 1], v[mark]);
    v[mark - 1] = l;
    match r {
        Some(r) => v[mark] = r,
        None => {
            v.remove(mark);
        }
    }
}

fn stitch(mut left: Vec<Run>, right: Vec<Run>) -> Vec<Run> {
    if left.is_empty() {
        return right;
    }
    if right.is_empty() {
        return left;
    }
    let last = left.len() - 1;
    let (l, r) = join_runs(left[last], right[0]);
    left[last] = l;
    if let Some(r) = r {
        left.push(r);
    }
    left.extend_from_slice(&right[1..]);
    left
}

/// Deteccao paralela. Blocos com 1 elemento de sobreposicao, reduzidos pelo
/// monoide. O `fold` do Rayon da a cada tarefa UM vetor que acumula os blocos
/// consecutivos dela, em vez de um vetor por bloco.
///
/// A versao anterior tinha dois niveis (macro 32768 / micro 512). Medido em
/// dado ordenado, um nivel e dois niveis empatam, e ambos empatam com um
/// `is_sorted` paralelo cru: a varredura e sequencial e a largura de banda
/// manda. O nivel micro so custava -- um `Vec` a cada 512 elementos.
///
/// `seg_limit` e o aborto antecipado: quando a soma dos segmentos estimados
/// passa do limite, os blocos restantes nao sao varridos e a funcao devolve
/// `None`. Dado quase ordenado com 1% de trocas, por exemplo, estoura o
/// limite depois de varrer uma fracao minima do array.
fn detect_runs<T, F>(
    arr: &[T],
    cmp: &F,
    block: usize,
    min_run: usize,
    seg_limit: usize,
) -> Option<Vec<Run>>
where
    T: Sync,
    F: Fn(&T, &T) -> Ordering + Sync,
{
    let n = arr.len();
    if n == 0 {
        return Some(Vec::new());
    }
    if n == 1 {
        return Some(vec![Run::new(FLAT, 1)]);
    }
    let block = block.max(2);
    let step = block - 1;
    let blocks = (n - 2) / step + 1;
    let segs = AtomicUsize::new(0);
    let runs = (0..blocks)
        .into_par_iter()
        .fold(Vec::new, |mut acc: Vec<Run>, i| {
            if segs.load(Relaxed) > seg_limit {
                return acc;
            }
            let start = i * step;
            let end = (start + block).min(n);
            let mark = acc.len();
            let s = scan_runs(&arr[start..end], cmp, &mut acc, min_run);
            segs.fetch_add(s, Relaxed);
            stitch_at(&mut acc, mark);
            acc
        })
        .reduce(Vec::new, stitch);
    if segs.load(Relaxed) > seg_limit {
        None
    } else {
        Some(runs)
    }
}

// =====================================================================
//  FASE 2: NORMALIZACAO
//
//  Transforma a lista de runs em segmentos ASCENDENTES, no lugar:
//
//    Keep      ja ascendente
//    Reverse   descendente -> inverte (e desfaz os empates, se estavel)
//    Sort      varios runs curtos seguidos -> ordena o trecho de uma vez
//
//  Depois disso a arvore so enxerga runs ascendentes, e a direcao deixa de
//  ser assunto dela.
// =====================================================================

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Op {
    Keep,
    Reverse { ties: bool },
    Sort,
}

struct Plan {
    lens: Vec<usize>,
    ops: Vec<Op>,
    /// Elementos que cairam em segmentos `Sort`.
    sorted_elems: usize,
}

struct Pending {
    len: usize,
    count: usize,
    first: Run,
}

impl Plan {
    fn natural(&mut self, r: Run) {
        self.lens.push(r.len());
        self.ops.push(if r.is_desc() {
            Op::Reverse { ties: r.kind() == DESC_TIES }
        } else {
            Op::Keep
        });
    }

    fn flush(&mut self, p: &mut Pending) {
        match p.count {
            0 => {}
            // Um run curto isolado ja e monotono: nao precisa de sort.
            1 => self.natural(p.first),
            _ => {
                self.lens.push(p.len);
                self.ops.push(Op::Sort);
                self.sorted_elems += p.len;
            }
        }
        p.len = 0;
        p.count = 0;
    }
}

/// Agrupa runs curtos consecutivos em blocos a ordenar.
///
/// Fundir run a run so compensa quando o run e longo. Medido em dente de
/// serra de comprimento L: com L=16 ordenar em bloco ganha da arvore, com L=32
/// empata, com L>=64 a arvore ganha. Um trecho de runs curtos vira um bloco de
/// ate `chunk` elementos -- e um array aleatorio que escapou do escudo vira
/// simplesmente uma sequencia de blocos, sem penhasco.
fn plan_segments(runs: &[Run], min_run: usize, chunk: usize) -> Plan {
    let mut plan = Plan {
        lens: Vec::with_capacity(runs.len()),
        ops: Vec::with_capacity(runs.len()),
        sorted_elems: 0,
    };
    let mut pend = Pending { len: 0, count: 0, first: Run::new(FLAT, 0) };
    for &r in runs {
        let len = r.len();
        if len < min_run {
            if pend.count > 0 && pend.len + len > chunk {
                plan.flush(&mut pend);
            }
            if pend.count == 0 {
                pend.first = r;
            }
            pend.len += len;
            pend.count += 1;
        } else {
            plan.flush(&mut pend);
            plan.natural(r);
        }
    }
    plan.flush(&mut pend);
    plan
}

/// Aplica as operacoes em paralelo, cada uma na sua fatia.
fn apply_ops<T, F>(arr: &mut [T], plan: &Plan, cmp: &F, stable: bool)
where
    T: Send,
    F: Fn(&T, &T) -> Ordering + Sync,
{
    let mut jobs: Vec<(&mut [T], Op)> = Vec::new();
    let mut rest: &mut [T] = arr;
    for (&len, &op) in plan.lens.iter().zip(&plan.ops) {
        let (head, tail) = std::mem::take(&mut rest).split_at_mut(len);
        if op != Op::Keep {
            jobs.push((head, op));
        }
        rest = tail;
    }
    jobs.into_par_iter().for_each(|(s, op)| match op {
        Op::Reverse { ties } => {
            parallel_reverse(s);
            if ties && stable {
                restore_ties(s, cmp);
            }
        }
        Op::Sort => leaf_sort(s, cmp, stable),
        Op::Keep => {}
    });
}

fn parallel_reverse<T: Send>(arr: &mut [T]) {
    let n = arr.len();
    if n <= 100_000 {
        arr.reverse();
        return;
    }
    let (left, right) = arr.split_at_mut(n / 2);
    left.par_iter_mut()
        .zip(right.par_iter_mut().rev())
        .for_each(|(a, b)| std::mem::swap(a, b));
}

/// `s` esta ordenado, mas cada bloco de chaves iguais ficou na ordem INVERSA
/// da original (acabou de sair de um `reverse`). Desinverte cada bloco.
///
/// Como `s` ja esta ordenado, as bordas do bloco que contem o meio saem por
/// busca binaria, e as duas metades seguem em paralelo.
fn restore_ties<T, F>(s: &mut [T], cmp: &F)
where
    T: Send,
    F: Fn(&T, &T) -> Ordering + Sync,
{
    let n = s.len();
    if n < 2 {
        return;
    }
    if n <= 16_384 {
        let mut i = 0usize;
        while i < n {
            let mut j = i + 1;
            while j < n && cmp(&s[i], &s[j]) == Equal {
                j += 1;
            }
            if j - i > 1 {
                s[i..j].reverse();
            }
            i = j;
        }
        return;
    }
    let mid = n / 2;
    let lo = s[..mid].partition_point(|x| cmp(x, &s[mid]) == Less);
    let hi = mid + s[mid..].partition_point(|x| cmp(x, &s[mid]) != Greater);
    let (left, rest) = s.split_at_mut(lo);
    let (group, right) = rest.split_at_mut(hi - lo);
    rayon::join(
        || restore_ties(left, cmp),
        || {
            rayon::join(
                || {
                    if group.len() > 1 {
                        parallel_reverse(group)
                    }
                },
                || restore_ties(right, cmp),
            )
        },
    );
}

/// Funde, so nos offsets, segmentos ascendentes adjacentes que ja estao em
/// ordem global (ultimo de um <= primeiro do seguinte). Uma comparacao por
/// fronteira, nenhum elemento movido.
///
/// Antes, o atalho "ja em ordem" vivia DENTRO da arvore e era um memcpy por
/// nivel: mil dentes em faixas disjuntas pagavam log(1000) passadas de copia.
/// Medido em 10M u64: 112 ms -> 9,9 ms.
fn coalesce<T, F>(arr: &[T], offsets: Vec<usize>, cmp: &F) -> Vec<usize>
where
    T: Sync,
    F: Fn(&T, &T) -> Ordering + Sync,
{
    let s = offsets.len() - 1;
    if s <= 1 {
        return offsets;
    }
    let is_break = |i: usize| cmp(&arr[offsets[i] - 1], &arr[offsets[i]]) == Greater;
    let breaks: Vec<bool> = if s > 4096 {
        (1..s).into_par_iter().map(is_break).collect()
    } else {
        (1..s).map(is_break).collect()
    };
    let mut out = Vec::with_capacity(s + 1);
    out.push(offsets[0]);
    for i in 1..s {
        if breaks[i - 1] {
            out.push(offsets[i]);
        }
    }
    out.push(offsets[s]);
    out
}

fn prefix_sums(lens: &[usize]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(lens.len() + 1);
    let mut off = 0usize;
    offsets.push(0);
    for &l in lens {
        off += l;
        offsets.push(off);
    }
    offsets
}

#[allow(clippy::too_many_arguments)]
fn sort_structured<T, F, K>(
    arr: &mut [T],
    runs: &[Run],
    cmp: &F,
    stable: bool,
    t: &Tuning,
    scratch: Option<&mut Vec<T>>,
    cheap: bool,
    key: &K,
) where
    T: Copy + Send + Sync,
    F: Fn(&T, &T) -> Ordering + Sync,
    K: Keyer<T>,
{
    let n = arr.len();
    let plan = plan_segments(runs, t.min_run, t.chunk);
    let offsets = prefix_sums(&plan.lens);

    if t.route == Route::Auto {
        // A decisao exata, com o metadata inteiro na mao: quantos runs sobram
        // DEPOIS de encadear? Mil dentes em faixas disjuntas sao mil runs e
        // um unico run encadeado; mil dentes intercalados sao mil runs mesmo.
        let allowed = t.allowed_segs(n, cheap);
        let fragmented = plan.lens.len() > allowed
            && runs_after_coalesce(arr, &plan, &offsets, cmp) > allowed;
        if fragmented {
            sort_chaotic(arr, cmp, stable, t, scratch, Chaos::Fragmented, cheap, key);
            return;
        }
        // O escudo foi enganado e a maior parte do dado nao tem run que se
        // aproveite. O estavel segue: os blocos `Sort` do plano ja sao o
        // caminho caotico dele. O instavel escolhe pelo comprimento medio do
        // run bruto -- aleatorio da ~2,4, dente de serra curto da o proprio
        // comprimento do dente.
        if !stable && plan.sorted_elems * 2 > n {
            let kind = if runs.len() * 6 > n { Chaos::Random } else { Chaos::ShortRuns };
            sort_chaotic(arr, cmp, stable, t, scratch, kind, cheap, key);
            return;
        }
    }

    apply_ops(arr, &plan, cmp, stable);
    let offsets = coalesce(arr, offsets, cmp);
    merge_segments(arr, &offsets, cmp, t, scratch);
}

/// Quantos runs sobrariam depois de normalizar e encadear -- calculado ANTES
/// de mover qualquer elemento, so com os extremos de cada segmento. Um
/// segmento `Sort` nao tem extremos conhecidos e conta como quebra.
fn runs_after_coalesce<T, F>(arr: &[T], plan: &Plan, offsets: &[usize], cmp: &F) -> usize
where
    T: Sync,
    F: Fn(&T, &T) -> Ordering + Sync,
{
    let s = plan.lens.len();
    if s <= 1 {
        return s;
    }
    let is_break = |i: usize| {
        let (prev, next) = (plan.ops[i - 1], plan.ops[i]);
        if prev == Op::Sort || next == Op::Sort {
            return true;
        }
        let max_prev = match prev {
            Op::Reverse { .. } => &arr[offsets[i - 1]],
            _ => &arr[offsets[i] - 1],
        };
        let min_next = match next {
            Op::Reverse { .. } => &arr[offsets[i + 1] - 1],
            _ => &arr[offsets[i]],
        };
        cmp(max_prev, min_next) == Greater
    };
    let breaks = if s > 4096 {
        (1..s).into_par_iter().filter(|&i| is_break(i)).count()
    } else {
        (1..s).filter(|&i| is_break(i)).count()
    };
    breaks + 1
}

// =====================================================================
//  CAMINHO CAOTICO
// =====================================================================

/// Por que o dado veio parar no caminho caotico.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Chaos {
    /// Aleatorio de fato: a direcao inverte em mais de um terco da amostra.
    Random,
    /// Runs curtos, mas com ordem local (dente de serra de 16 a 60).
    ShortRuns,
    /// Runs de verdade, so que em numero grande demais para a arvore.
    Fragmented,
}

#[allow(clippy::too_many_arguments)]
fn sort_chaotic<T, F, K>(
    arr: &mut [T],
    cmp: &F,
    stable: bool,
    t: &Tuning,
    mut scratch: Option<&mut Vec<T>>,
    kind: Chaos,
    cheap: bool,
    key: &K,
) where
    T: Copy + Send + Sync,
    F: Fn(&T, &T) -> Ordering + Sync,
    K: Keyer<T>,
{
    // RADIX: dado aleatorio de fato e chave conhecida. Vale para os dois
    // motores. Se a rotina desistir (chave enviesada, sem memoria para o
    // buffer) nada foi movido e o caminho por comparacao abaixo segue igual.
    // Runs curtos e dado fragmentado continuam por comparacao: ali ha ordem
    // local que o radix nao aproveita.
    let radix_min = if stable { t.radix_min_stable } else { t.radix_min };
    if K::ENABLED && kind == Chaos::Random && arr.len() >= radix_min {
        // No estavel a folha por comparacao e o `sort` estavel, mais caro que
        // o `sort_unstable`: o LSD compensa por mais passadas mesmo com
        // comparacao barata (texto de 6 bytes em u64, 30M: 384 ms com folha
        // LSD contra 554 com folha por comparacao).
        let rx = Radix {
            cmp,
            key,
            stable,
            t,
            lsd_max: if cheap && !stable { t.radix_lsd_cheap } else { t.radix_lsd_costly },
        };
        if radix_sort(arr, &rx, scratch.as_mut().map(|v| &mut **v)) {
            return;
        }
    }
    // INSTAVEL: o quicksort paralelo trabalha no lugar e sem buffer, mas so
    // em dois casos.
    //
    //  - Dado aleatorio. Contra ordenar blocos e fundir, em 10M u64: 124 ms
    //    contra 138-154; 52 contra 64-82 (64 chaves); 18 contra 50-59
    //    (2 chaves).
    //  - Dado globalmente ascendente com elementos fora do lugar E comparacao
    //    barata (ver o atalho em `sort_with`).
    //
    // Fora disso ele NAO e seguro. Runs intercalados sao um padrao ruim para
    // a escolha de pivo, e com chave cara o estrago e de ordem de grandeza:
    // 8 runs de 250 mil strings levam 1942 ms nele contra 109 ms da arvore;
    // 64 runs, 1121 ms contra 175. E em dente de serra curto de u64 os blocos
    // ja ganham (L=32: 116 ms contra 148).
    if !stable {
        let quick = match kind {
            Chaos::Random => true,
            Chaos::ShortRuns => false,
            Chaos::Fragmented => cheap && globally_ascending(arr, cmp),
        };
        if quick {
            arr.par_sort_unstable_by(|a, b| cmp(a, b));
            return;
        }
    }
    // BLOCOS: ordenados pelo `sort` da std (driftsort, que aproveita os runs
    // de dentro do bloco) e fundidos pela arvore. E o caminho do motor
    // estavel, e o do instavel quando o quicksort nao serve.
    //
    // Contra o `par_sort` do Rayon, em 10M u64: 175 ms contra 191 (aleatorio),
    // 102 contra 185 (1024 chaves), 89 contra 190 (64 chaves), 78 contra 154
    // (2 chaves). O bloco de 65536 venceu ou empatou em toda a varredura
    // (16K, 64K, 256K, n/8, n/4) -- o mesmo valor que o motor C++ ja usava.
    let chunk = t.chunk.max(1);
    let n = arr.len();
    if n <= chunk {
        leaf_sort(arr, cmp, true);
        return;
    }
    arr.par_chunks_mut(chunk).for_each(|c| leaf_sort(c, cmp, true));
    let mut offsets: Vec<usize> = (0..n).step_by(chunk).collect();
    offsets.push(n);
    let offsets = coalesce(arr, offsets, cmp);
    merge_segments(arr, &offsets, cmp, t, scratch);
}

// =====================================================================
//  CAMINHO CAOTICO COM CHAVE: RADIX
//
//  Quando o chamador diz quais sao os bits da chave, o dado aleatorio nao
//  precisa de comparacao:
//
//    1. AMOSTRA (de graca): 1024 elementos espalhados. Se um balde dos bits
//       altos ficaria com mais de 1/4 deles, a chave e enviesada e a rotina
//       desiste antes de varrer o array.
//    2. FAIXA: uma varredura paralela (AND e OR das chaves) diz quais bits
//       variam. Chave "nua" com ate 16 bits variando: CONTAGEM -- histograma
//       paralelo e preenchimento paralelo, sem buffer.
//    3. HISTOGRAMA dos 8 bits mais altos que variam. Se um balde passaria de
//       1/4 do dado (a amostra errou), desiste: nada foi movido ainda.
//    4. So entao o buffer e alocado e o array e ESPALHADO em 256 baldes, em
//       paralelo: o destino e fatiado de antemao em faixas disjuntas
//       [balde][pedaco], cada pedaco escreve nas suas.
//    5. Cada balde, em paralelo e dentro do cache: radix LSD sequencial se
//       sobram poucos bits, sort da std se sobram muitos.
//    6. Balde grande demais passa pela rotina de novo.
//
//  Medido em 30M / 100M aleatorios, 8 threads, buffer alocado no cronometro,
//  contra o quicksort paralelo (protótipo de 2026-11-17, ver CLAUDE.md):
//  i32 344 -> 152 ms / 1274 -> 528; u64 406 -> 278 / 1463 -> 1084; 64 valores
//  em i32 132 -> 23; enviesado (90% em 0..1000) empata.
//
//  Por que MSD + folha e nao LSD puro: o LSD varre o array inteiro duas vezes
//  por byte da chave (16 varreduras em u64, onde perde do quicksort: 540
//  contra 372 ms). A passada MSD unica deixa o resto no cache e nao depende
//  da largura da chave.
//
//  ESTABILIDADE. O espalhamento percorre os pedacos em ordem e cada balde
//  recebe primeiro o que veio do pedaco 0, depois do 1...: preserva a ordem
//  dos iguais. O LSD e estavel por construcao. A folha por comparacao usa o
//  `sort` estavel quando o motor e o estavel. A contagem so existe para chave
//  nua, em que iguais sao indistinguiveis.
//
//  A rotina devolve `false` SEM TER MOVIDO NADA quando nao serve; o chamador
//  segue pelo caminho por comparacao.
// =====================================================================

/// Como o caminho caotico enxerga a chave de um elemento.
pub(crate) trait Keyer<T>: Sync {
    /// Ha chave? `NoKey` desliga o radix em tempo de compilacao.
    const ENABLED: bool;
    /// Chaves iguais implicam elementos iguais na comparacao. Libera a folha
    /// LSD. `false` para prefixo: ali o desempate e sempre por comparacao.
    const EXACT: bool;
    /// O elemento e a propria chave e se reconstroi dos bits. Libera a
    /// contagem.
    const BARE: bool;
    fn bits(&self, x: &T) -> u64;
    fn rebuild(&self, bits: u64) -> T;
}

/// Sem chave: so comparacao. E o que as entradas sem `radix` no nome usam.
pub(crate) struct NoKey;
impl<T> Keyer<T> for NoKey {
    const ENABLED: bool = false;
    const EXACT: bool = false;
    const BARE: bool = false;
    #[inline(always)]
    fn bits(&self, _: &T) -> u64 {
        0
    }
    fn rebuild(&self, _: u64) -> T {
        unreachable!("NoKey nao reconstroi elemento")
    }
}

/// O elemento e um inteiro: ele e a chave.
pub(crate) struct Bare;
impl<T: RadixKey> Keyer<T> for Bare {
    const ENABLED: bool = true;
    const EXACT: bool = true;
    const BARE: bool = true;
    #[inline(always)]
    fn bits(&self, x: &T) -> u64 {
        x.to_radix()
    }
    #[inline(always)]
    fn rebuild(&self, bits: u64) -> T {
        T::from_radix(bits)
    }
}

/// A chave extraida decide a ordem inteira; o elemento pode levar mais dado.
pub(crate) struct Exact<K>(pub K);
impl<T, K: Fn(&T) -> u64 + Sync> Keyer<T> for Exact<K> {
    const ENABLED: bool = true;
    const EXACT: bool = true;
    const BARE: bool = false;
    #[inline(always)]
    fn bits(&self, x: &T) -> u64 {
        (self.0)(x)
    }
    fn rebuild(&self, _: u64) -> T {
        unreachable!("chave extraida nao reconstroi elemento")
    }
}

/// A chave extraida e so o comeco da ordem; o comparador desempata.
pub(crate) struct Prefix<K>(pub K);
impl<T, K: Fn(&T) -> u64 + Sync> Keyer<T> for Prefix<K> {
    const ENABLED: bool = true;
    const EXACT: bool = false;
    const BARE: bool = false;
    #[inline(always)]
    fn bits(&self, x: &T) -> u64 {
        (self.0)(x)
    }
    fn rebuild(&self, _: u64) -> T {
        unreachable!("prefixo nao reconstroi elemento")
    }
}

/// O que a rotina carrega de um nivel para o outro.
struct Radix<'a, F, K> {
    cmp: &'a F,
    key: &'a K,
    stable: bool,
    t: &'a Tuning,
    /// Folha LSD: ate quantos bits restantes.
    lsd_max: u32,
}

/// Sort por comparacao de um trecho grande, quando o radix nao serve nele.
fn radix_fallback<T, F>(a: &mut [T], cmp: &F, stable: bool)
where
    T: Send,
    F: Fn(&T, &T) -> Ordering + Sync,
{
    if stable {
        a.par_sort_by(|x, y| cmp(x, y));
    } else {
        a.par_sort_unstable_by(|x, y| cmp(x, y));
    }
}

/// Ordena `arr` por radix. Devolve `false`, sem ter movido nenhum elemento,
/// quando a chave e enviesada, quando o prefixo nao distingue nada ou quando
/// nao ha memoria para o buffer.
fn radix_sort<T, F, K>(arr: &mut [T], rx: &Radix<F, K>, scratch: Option<&mut Vec<T>>) -> bool
where
    T: Copy + Send + Sync,
    F: Fn(&T, &T) -> Ordering + Sync,
    K: Keyer<T>,
{
    let n = arr.len();
    if n < 2 {
        return true;
    }
    let t = rx.t;
    if radix_skewed(arr, rx.key, t) {
        return false;
    }
    let (and, or) = radix_range(arr, rx.key);
    let diff = and ^ or;
    if diff == 0 {
        // Todas as chaves iguais. Com chave exata nao ha o que ordenar; com
        // prefixo, ele nao distingue nada e a comparacao assume.
        return K::EXACT;
    }
    let hi = 64 - diff.leading_zeros();
    if K::BARE && hi <= t.radix_count_bits {
        radix_count(arr, rx.key, or, hi);
        return true;
    }
    // O histograma vem ANTES do buffer: se a chave for enviesada, desistir
    // nao custou nem a alocacao.
    let plan = match radix_plan(arr, rx.key, t, hi) {
        Some(p) => p,
        None => return false,
    };
    let mut owned: Vec<T> = Vec::new();
    let buf: &mut [T] = match scratch {
        Some(v) if v.len() >= n => &mut v[..n],
        Some(v) => {
            v.clear();
            if v.try_reserve_exact(n).is_err() {
                return false;
            }
            arr.par_iter().copied().collect_into_vec(v);
            &mut v[..]
        }
        None => {
            if owned.try_reserve_exact(n).is_err() {
                return false;
            }
            arr.par_iter().copied().collect_into_vec(&mut owned);
            &mut owned[..]
        }
    };
    radix_scatter(arr, buf, &plan, rx.key);
    radix_buckets(arr, buf, &plan, rx, 0);
    true
}

/// A chave e enviesada? Olha 1024 elementos espalhados: se a faixa deles e
/// larga e um balde dos bits altos fica com mais de 1/`radix_skew` da amostra,
/// o espalhamento nao dividiria o trabalho.
fn radix_skewed<T, K>(a: &[T], key: &K, t: &Tuning) -> bool
where
    K: Keyer<T>,
{
    let n = a.len();
    let m = 1024.min(n);
    let step = n / m;
    let (mut and, mut or) = (u64::MAX, 0u64);
    for i in 0..m {
        let b = key.bits(&a[i * step]);
        and &= b;
        or |= b;
    }
    let hi = 64 - (and ^ or).leading_zeros();
    if K::BARE && hi <= t.radix_count_bits {
        return false; // candidato a contagem; a varredura exata decide
    }
    if hi == 0 {
        return true; // a amostra inteira tem a mesma chave
    }
    let b = t.radix_digit.clamp(1, hi.min(16));
    let shift = hi - b;
    let mask = (1u64 << b) - 1;
    let mut h = vec![0usize; 1usize << b];
    for i in 0..m {
        h[((key.bits(&a[i * step]) >> shift) & mask) as usize] += 1;
    }
    h.iter().any(|&c| c * t.radix_skew > m)
}

/// AND e OR de todas as chaves: os bits em que diferem sao os que variam.
fn radix_range<T, K>(a: &[T], key: &K) -> (u64, u64)
where
    T: Sync,
    K: Keyer<T>,
{
    a.par_chunks(1 << 16)
        .map(|c| {
            c.iter().fold((u64::MAX, 0u64), |(and, or), x| {
                let b = key.bits(x);
                (and & b, or | b)
            })
        })
        .reduce(|| (u64::MAX, 0u64), |l, r| (l.0 & r.0, l.1 | r.1))
}

/// Chave nua de faixa pequena: conta e preenche, sem buffer. `or` e o OU de
/// todas as chaves e `hi` quantos bits de baixo variam.
fn radix_count<T, K>(a: &mut [T], key: &K, or: u64, hi: u32)
where
    T: Copy + Send + Sync,
    K: Keyer<T>,
{
    let n = a.len();
    let nb = 1usize << hi;
    let low = (nb - 1) as u64;
    let base = or & !low; // os bits de cima sao iguais em todas as chaves
    let pieces = (n >> 20).clamp(1, 16);
    let chunk = (n + pieces - 1) / pieces;
    let hists: Vec<Vec<usize>> = a
        .par_chunks(chunk)
        .map(|c| {
            let mut h = vec![0usize; nb];
            for x in c {
                h[(key.bits(x) & low) as usize] += 1;
            }
            h
        })
        .collect();
    let mut parts: Vec<(&mut [T], T)> = Vec::new();
    let mut rest: &mut [T] = a;
    for d in 0..nb {
        let c: usize = hists.iter().map(|h| h[d]).sum();
        if c > 0 {
            let (s, r) = rest.split_at_mut(c);
            parts.push((s, key.rebuild(base | d as u64)));
            rest = r;
        }
    }
    parts.into_par_iter().for_each(|(s, v)| {
        if s.len() > (1 << 17) {
            s.par_chunks_mut(1 << 16).for_each(|c| c.fill(v));
        } else {
            s.fill(v);
        }
    });
}

/// Histograma da passada MSD de um trecho.
struct RadixPlan {
    /// Bits abaixo do digito: o que sobra para a folha.
    shift: u32,
    mask: u64,
    nb: usize,
    /// Elementos por pedaco.
    chunk: usize,
    /// Um histograma por pedaco.
    hists: Vec<Vec<usize>>,
    /// Limites dos baldes no destino (`nb + 1` valores).
    bounds: Vec<usize>,
}

/// Conta o digito MSD -- os `radix_digit` bits mais altos entre os `hi` que
/// variam. `None` se um balde ficaria com mais de 1/`radix_skew` do trecho.
fn radix_plan<T, K>(a: &[T], key: &K, t: &Tuning, hi: u32) -> Option<RadixPlan>
where
    T: Sync,
    K: Keyer<T>,
{
    let n = a.len();
    let b = t.radix_digit.clamp(1, hi.min(16));
    let shift = hi - b;
    let nb = 1usize << b;
    let mask = (nb - 1) as u64;
    let pieces = (n / t.radix_piece.max(1)).clamp(1, 128);
    let chunk = (n + pieces - 1) / pieces;
    let hists: Vec<Vec<usize>> = a
        .par_chunks(chunk)
        .map(|c| {
            let mut h = vec![0usize; nb];
            for x in c {
                h[((key.bits(x) >> shift) & mask) as usize] += 1;
            }
            h
        })
        .collect();
    let mut bounds = vec![0usize; nb + 1];
    for d in 0..nb {
        bounds[d + 1] = bounds[d] + hists.iter().map(|h| h[d]).sum::<usize>();
    }
    let limit = n / t.radix_skew.max(1);
    if (0..nb).any(|d| bounds[d + 1] - bounds[d] > limit) {
        return None;
    }
    Some(RadixPlan { shift, mask, nb, chunk, hists, bounds })
}

/// Espalha `src` em `dst` conforme o plano, em paralelo. O destino e fatiado
/// antes: balde por balde e, dentro do balde, pedaco por pedaco. Cada pedaco
/// recebe as suas `nb` faixas e escreve por elas -- sem `unsafe`, e na ordem
/// que mantem os iguais como vieram.
fn radix_scatter<T, K>(src: &[T], dst: &mut [T], p: &RadixPlan, key: &K)
where
    T: Copy + Send + Sync,
    K: Keyer<T>,
{
    let mut slots: Vec<Vec<&mut [T]>> =
        (0..p.hists.len()).map(|_| Vec::with_capacity(p.nb)).collect();
    let mut rest: &mut [T] = dst;
    for d in 0..p.nb {
        for (piece, h) in p.hists.iter().enumerate() {
            let (a, r) = rest.split_at_mut(h[d]);
            slots[piece].push(a);
            rest = r;
        }
    }
    src.par_chunks(p.chunk).zip(slots.into_par_iter()).for_each(|(c, mut cur)| {
        for x in c {
            let d = ((key.bits(x) >> p.shift) & p.mask) as usize;
            let s = std::mem::take(&mut cur[d]);
            if let Some((head, tail)) = s.split_first_mut() {
                *head = *x;
                cur[d] = tail;
            }
        }
    });
}

/// Ordena cada balde. O dado esta em `buf`; o resultado termina em `arr`.
fn radix_buckets<T, F, K>(arr: &mut [T], buf: &mut [T], p: &RadixPlan, rx: &Radix<F, K>, depth: u32)
where
    T: Copy + Send + Sync,
    F: Fn(&T, &T) -> Ordering + Sync,
    K: Keyer<T>,
{
    let t = rx.t;
    let big = (arr.len() / t.radix_big_div.max(1)).max(t.radix_big);
    let shift = p.shift;
    // Folha estreita = LSD de poucas passadas: ali so reentra o balde grande
    // de verdade. Folha larga (muitas passadas ou comparacao): um segundo
    // corte MSD sai mais barato que ordenar um balde que nao cabe no cache.
    let narrow = K::EXACT && shift <= t.radix_lsd_cheap;
    let recut = if narrow { big } else { t.radix_recut.min(big) };
    let mut pairs: Vec<(&mut [T], &mut [T])> = Vec::with_capacity(p.nb);
    let (mut ra, mut rb): (&mut [T], &mut [T]) = (arr, buf);
    for w in p.bounds.windows(2) {
        let len = w[1] - w[0];
        let (a, r) = ra.split_at_mut(len);
        ra = r;
        let (s, r) = rb.split_at_mut(len);
        rb = r;
        if len > 0 {
            pairs.push((a, s));
        }
    }
    pairs.into_par_iter().for_each(|(a, s)| {
        let len = a.len();
        if K::EXACT && shift == 0 {
            // O digito cobriu todos os bits que variam: chaves identicas.
            a.copy_from_slice(s);
        } else if len > recut && depth < 6 {
            // Balde grande (dado com concentracao, ou folha larga): de volta
            // para `arr` e a rotina de novo, paralela por dentro, com o
            // proprio pedaco do buffer como rascunho.
            par_copy(a, s, t.par_grain);
            radix_bucket(a, s, rx, depth + 1);
        } else if K::EXACT && len >= t.radix_leaf_min && shift <= rx.lsd_max {
            radix_lsd(s, a, rx.key, shift);
        } else {
            a.copy_from_slice(s);
            if len > big {
                radix_fallback(a, rx.cmp, rx.stable);
            } else {
                leaf_sort(a, rx.cmp, rx.stable);
            }
        }
    });
}

/// A rotina de novo em um balde grande. O dado esta em `a`; `s` e rascunho.
fn radix_bucket<T, F, K>(a: &mut [T], s: &mut [T], rx: &Radix<F, K>, depth: u32)
where
    T: Copy + Send + Sync,
    F: Fn(&T, &T) -> Ordering + Sync,
    K: Keyer<T>,
{
    let (and, or) = radix_range(a, rx.key);
    let diff = and ^ or;
    if diff == 0 {
        if !K::EXACT {
            radix_fallback(a, rx.cmp, rx.stable);
        }
        return;
    }
    let hi = 64 - diff.leading_zeros();
    if K::BARE && hi <= rx.t.radix_count_bits {
        radix_count(a, rx.key, or, hi);
        return;
    }
    match radix_plan(a, rx.key, rx.t, hi) {
        Some(p) => {
            radix_scatter(a, s, &p, rx.key);
            radix_buckets(a, s, &p, rx, depth);
        }
        None => radix_fallback(a, rx.cmp, rx.stable),
    }
}

/// Radix LSD sequencial de um balde, sobre os `low_bits` de baixo da chave
/// (os de cima sao iguais no balde inteiro). O dado esta em `src`; o
/// resultado termina em `dst`. Passada de digito constante e pulada.
fn radix_lsd<T, K>(src: &mut [T], dst: &mut [T], key: &K, low_bits: u32)
where
    T: Copy,
    K: Keyer<T>,
{
    let passes = (low_bits + 7) / 8;
    let mut in_src = true;
    for p in 0..passes {
        let sh = p * 8;
        let (a, b): (&[T], &mut [T]) =
            if in_src { (&*src, &mut *dst) } else { (&*dst, &mut *src) };
        let mut h = [0usize; 256];
        for x in a {
            h[((key.bits(x) >> sh) & 0xFF) as usize] += 1;
        }
        if h.iter().any(|&c| c == a.len()) {
            continue;
        }
        let mut pos = [0usize; 256];
        let mut acc = 0usize;
        for d in 0..256 {
            pos[d] = acc;
            acc += h[d];
        }
        for x in a {
            let d = ((key.bits(x) >> sh) & 0xFF) as usize;
            b[pos[d]] = *x;
            pos[d] += 1;
        }
        in_src = !in_src;
    }
    if in_src {
        dst.copy_from_slice(src);
    }
}

// =====================================================================
//  FASE 3: ARVORE DE MERGE
//
//  Entrada: `offsets` delimita runs ASCENDENTES em `arr`.
//
//  O BUFFER
//
//  Sem buffer do chamador, ele nasce como COPIA PARALELA do array. Isso troca
//  duas coisas da versao anterior -- o preenchimento sequencial
//  (`vec![arr[0]; n]`, uma thread escrevendo n elementos que seriam
//  sobrescritos) e a copia das folhas de paridade impar -- por uma passada
//  paralela so. Com os dois lados iguais na largada, toda folha ja esta onde
//  precisa estar.
//
//  A medicao que tinha concluido que inicializar o buffer "nao custa nada"
//  (457,4 contra 458,5 ms) estava confundida: quando `arr[0]` e o inteiro
//  zero, `vec![arr[0]; n]` vira `calloc` e o sistema entrega paginas zeradas
//  sob demanda. Com `arr[0] != 0` o preenchimento custa ~14 ms por 80 MB
//  (64 runs em 10M u64: 64,0 ms com zero na frente, 77,5 sem). Dado real
//  quase nunca comeca com zero.
//
//  Com buffer do chamador (`*_with_buffer`), ja inicializado e do tamanho
//  certo, nao ha nada a criar nem pagina nova a tocar: so as folhas de
//  paridade impar copiam o proprio trecho.
// =====================================================================

fn merge_segments<T, F>(
    arr: &mut [T],
    offsets: &[usize],
    cmp: &F,
    t: &Tuning,
    scratch: Option<&mut Vec<T>>,
) where
    T: Copy + Send + Sync,
    F: Fn(&T, &T) -> Ordering + Sync,
{
    if offsets.len() <= 2 {
        return;
    }
    let n = arr.len();
    match scratch {
        Some(buf) if buf.len() >= n => {
            tree_kway(arr, &mut buf[..n], offsets, cmp, t, false, true);
        }
        Some(buf) => {
            arr.par_iter().copied().collect_into_vec(buf);
            tree_kway(arr, &mut buf[..], offsets, cmp, t, false, false);
        }
        None => {
            let mut buffer: Vec<T> = Vec::with_capacity(n);
            arr.par_iter().copied().collect_into_vec(&mut buffer);
            tree_kway(arr, &mut buffer, offsets, cmp, t, false, false);
        }
    }
}

/// Recursao K-aria sobre os runs. Hibrida: k-vias nos niveis que vao a DRAM,
/// binaria abaixo de `kway_min_elems`, onde o trecho ja mora em cache.
///
/// Contrato: o dado de entrada esta em `v`. Na saida o trecho ordenado esta
/// em `buf` se `into_buf`, senao em `v`. Se `copy_leaves` e falso, `buf` ja
/// nasceu como copia de `v` e a folha nao tem nada a fazer; se e verdadeiro,
/// `buf` tem lixo valido e a folha de paridade impar copia o proprio trecho.
fn tree_kway<T, F>(v: &mut [T], buf: &mut [T], offsets: &[usize], cmp: &F, t: &Tuning, into_buf: bool, copy_leaves: bool)
where
    T: Copy + Send + Sync,
    F: Fn(&T, &T) -> Ordering + Sync,
{
    let runs = offsets.len() - 1;
    if runs <= 1 {
        if into_buf && copy_leaves {
            par_copy(buf, v, t.par_grain);
        }
        return;
    }
    let base = offsets[0];
    let total = offsets[runs] - base;
    if runs == 2 || total <= t.kway_min_elems {
        tree_binary(v, buf, offsets, cmp, t, into_buf, copy_leaves);
        return;
    }

    // Cortes balanceados por ELEMENTO, nao por contagem de runs.
    let g = KWAY_FANOUT.min(runs);
    let mut cm = [0usize; KWAY_FANOUT + 1];
    cm[g] = runs;
    for i in 1..g {
        let target = base + total / g * i + total % g * i / g;
        let idx = offsets.partition_point(|&o| o < target);
        cm[i] = idx.clamp(cm[i - 1] + 1, runs - (g - i));
    }

    {
        let mut parts: Vec<(&mut [T], &mut [T], usize, usize)> = Vec::with_capacity(g);
        let mut vv: &mut [T] = &mut *v;
        let mut bb: &mut [T] = &mut *buf;
        let mut prev = 0usize;
        for i in 0..g {
            let e = offsets[cm[i + 1]] - base;
            let (vl, vr) = std::mem::take(&mut vv).split_at_mut(e - prev);
            let (bl, br) = std::mem::take(&mut bb).split_at_mut(e - prev);
            parts.push((vl, bl, cm[i], cm[i + 1]));
            vv = vr;
            bb = br;
            prev = e;
        }
        parts.into_par_iter().for_each(|(sv, sb, lo, hi)| {
            tree_kway(sv, sb, &offsets[lo..=hi], cmp, t, !into_buf, copy_leaves);
        });
    }

    let (src, dst): (&[T], &mut [T]) = if into_buf { (&*v, buf) } else { (&*buf, v) };

    // Grupos adjacentes ja em ordem viram UM stream.
    let mut streams: [&[T]; KWAY_FANOUT] = [&[]; KWAY_FANOUT];
    let mut ns = 0usize;
    let mut gs = 0usize;
    for i in 0..g {
        let e = offsets[cm[i + 1]] - base;
        let join_next = i + 1 < g && cmp(&src[e - 1], &src[e]) != Greater;
        if !join_next {
            streams[ns] = &src[gs..e];
            ns += 1;
            gs = e;
        }
    }
    kway_merge(&streams[..ns], dst, cmp, t);
}

/// Arvore binaria, com split balanceado por elementos.
fn tree_binary<T, F>(v: &mut [T], buf: &mut [T], offsets: &[usize], cmp: &F, t: &Tuning, into_buf: bool, copy_leaves: bool)
where
    T: Copy + Send + Sync,
    F: Fn(&T, &T) -> Ordering + Sync,
{
    let runs = offsets.len() - 1;
    if runs <= 1 {
        if into_buf && copy_leaves {
            par_copy(buf, v, t.par_grain);
        }
        return;
    }
    let base = offsets[0];
    let total = offsets[runs] - base;
    let target = base + total / 2;
    let split = offsets.partition_point(|&o| o < target).clamp(1, runs - 1);
    let mid = offsets[split] - base;
    {
        let (vl, vr) = v.split_at_mut(mid);
        let (bl, br) = buf.split_at_mut(mid);
        let (lo, ro) = (&offsets[..=split], &offsets[split..]);
        if total > t.seq_node {
            rayon::join(
                || tree_binary(vl, bl, lo, cmp, t, !into_buf, copy_leaves),
                || tree_binary(vr, br, ro, cmp, t, !into_buf, copy_leaves),
            );
        } else {
            tree_binary(vl, bl, lo, cmp, t, !into_buf, copy_leaves);
            tree_binary(vr, br, ro, cmp, t, !into_buf, copy_leaves);
        }
    }
    let (src, dst): (&[T], &mut [T]) = if into_buf { (&*v, buf) } else { (&*buf, v) };
    let (a, b) = src.split_at(mid);
    merge_two(a, b, dst, cmp, t);
}

// =====================================================================
//  MERGE DE DOIS: CO-RANK + BIDIRECIONAL
// =====================================================================

/// Maior `i` tal que `a[..i]` e `b[..k - i]` sao os `k` menores, com empate
/// para `a`. E o corte que mantem o merge estavel.
fn co_rank<T, F>(k: usize, a: &[T], b: &[T], cmp: &F) -> (usize, usize)
where
    F: Fn(&T, &T) -> Ordering,
{
    let (m, n) = (a.len(), b.len());
    let mut lo = k.saturating_sub(n);
    let mut hi = k.min(m);
    while lo < hi {
        let i = lo + (hi - lo + 1) / 2;
        let j = k - i;
        if j == n || cmp(&a[i - 1], &b[j]) != Greater {
            lo = i;
        } else {
            hi = i - 1;
        }
    }
    (lo, k - lo)
}

/// Merge para FRENTE: emite os `dest.len()` menores. Empate -> A.
#[inline]
fn merge_front<T, F>(a: &[T], b: &[T], dest: &mut [T], cmp: &F)
where
    T: Copy,
    F: Fn(&T, &T) -> Ordering,
{
    let n = dest.len();
    let (mut i, mut j, mut k) = (0usize, 0usize, 0usize);
    while k < n && i < a.len() && j < b.len() {
        if cmp(&b[j], &a[i]) == Less {
            dest[k] = b[j];
            j += 1;
        } else {
            dest[k] = a[i];
            i += 1;
        }
        k += 1;
    }
    if k < n {
        let rest = n - k;
        if i < a.len() {
            dest[k..].copy_from_slice(&a[i..i + rest]);
        } else {
            dest[k..].copy_from_slice(&b[j..j + rest]);
        }
    }
}

/// Merge para TRAS: emite os `dest.len()` maiores. Empate -> B.
///
/// A regra de desempate e a COMPLEMENTAR da de `merge_front`, e e isso que faz
/// o teorema valer: os k menores (empate -> A) e os total-k maiores
/// (empate -> B) particionam o multiconjunto para QUALQUER k. Nenhum ponto de
/// corte precisa ser calculado -- cada lado so conta as proprias saidas.
#[inline]
fn merge_back<T, F>(a: &[T], b: &[T], dest: &mut [T], cmp: &F)
where
    T: Copy,
    F: Fn(&T, &T) -> Ordering,
{
    let (mut qa, mut qb, mut k) = (a.len(), b.len(), dest.len());
    while k > 0 && qa > 0 && qb > 0 {
        k -= 1;
        if cmp(&b[qb - 1], &a[qa - 1]) == Less {
            qa -= 1;
            dest[k] = a[qa];
        } else {
            qb -= 1;
            dest[k] = b[qb];
        }
    }
    if k > 0 {
        if qa > 0 {
            dest[..k].copy_from_slice(&a[qa - k..qa]);
        } else {
            dest[..k].copy_from_slice(&b[qb - k..qb]);
        }
    }
}

/// Merge paralelo de dois runs, de fontes imutaveis. Corta por co-rank ate o
/// grao; em cada pedaco tenta primeiro os atalhos de faixas disjuntas e so
/// entao funde -- com uma thread vindo da frente e outra do fim.
fn merge_two<T, F>(a: &[T], b: &[T], dest: &mut [T], cmp: &F, t: &Tuning)
where
    T: Copy + Send + Sync,
    F: Fn(&T, &T) -> Ordering + Sync,
{
    let total = dest.len();
    if total > t.par_grain && total >= 2 {
        let k = total / 2;
        let (i, j) = co_rank(k, a, b, cmp);
        let (al, ar) = a.split_at(i);
        let (bl, br) = b.split_at(j);
        let (dl, dr) = dest.split_at_mut(k);
        rayon::join(
            || merge_two(al, bl, dl, cmp, t),
            || merge_two(ar, br, dr, cmp, t),
        );
        return;
    }
    if a.is_empty() {
        dest.copy_from_slice(b);
        return;
    }
    if b.is_empty() {
        dest.copy_from_slice(a);
        return;
    }
    // Faixas disjuntas: e uma concatenacao.
    if cmp(&a[a.len() - 1], &b[0]) != Greater {
        let (d1, d2) = dest.split_at_mut(a.len());
        d1.copy_from_slice(a);
        d2.copy_from_slice(b);
        return;
    }
    if cmp(&b[b.len() - 1], &a[0]) == Less {
        let (d1, d2) = dest.split_at_mut(b.len());
        d1.copy_from_slice(b);
        d2.copy_from_slice(a);
        return;
    }
    if total > t.bidir_min {
        let (df, db) = dest.split_at_mut(total / 2);
        rayon::join(
            || merge_front(a, b, df, cmp),
            || merge_back(a, b, db, cmp),
        );
    } else {
        merge_front(a, b, dest, cmp);
    }
}

fn par_copy<T: Copy + Send + Sync>(dest: &mut [T], src: &[T], grain: usize) {
    if dest.len() <= grain.saturating_mul(4) {
        dest.copy_from_slice(src);
        return;
    }
    let g = grain.max(1);
    dest.par_chunks_mut(g)
        .zip(src.par_chunks(g))
        .for_each(|(d, s)| d.copy_from_slice(s));
}

// =====================================================================
//  MERGE K-VIAS COM BLOCAGEM DE CACHE
//
//  O merge binario paga log2(runs) passadas sobre a memoria, e a maquina
//  satura a banda: o custo e o NUMERO DE PASSADAS. Um merge de K vias paga
//  logK(runs).
//
//  NAO usa loser tree: medido, custa ~12x mais por elemento que o merge
//  simples, porque `head[w]` com `w` em tempo de execucao nao e promovido a
//  registrador. Cada tile e cortado por `multiseq_partition` e fundido por uma
//  arvore BINARIA que cabe em L1 -- o kernel rapido continua o mesmo.
// =====================================================================

/// Acha `out[i]` com soma igual a `rank`, tal que os prefixos sao exatamente
/// os `rank` menores elementos. Empates sao distribuidos em ordem de stream,
/// o que mantem a estabilidade e concorda com o desempate de `merge_front`.
///
/// As buscas binarias rodam so dentro de `[lo, hi)`. Todo pivo sai de dentro
/// do intervalo vivo, entao fica estritamente entre os pivos anteriores, e
/// nenhum elemento fora do intervalo pode mudar de lado. O resultado e
/// identico ao da busca no run inteiro; contado em 8 runs, as comparacoes por
/// elemento do merge caem de 6,24 para 5,22 (o ideal e 4,00).
fn multiseq_partition<T, F>(seq: &[&[T]], rank: usize, out: &mut [usize], cmp: &F)
where
    F: Fn(&T, &T) -> Ordering,
{
    let ns = seq.len();
    let mut lo = [0usize; KWAY_FANOUT];
    let mut hi = [0usize; KWAY_FANOUT];
    let mut plt = [0usize; KWAY_FANOUT];
    let mut ple = [0usize; KWAY_FANOUT];
    for i in 0..ns {
        hi[i] = seq[i].len();
    }

    loop {
        // Pivo: o meio do intervalo mais largo.
        let mut m = usize::MAX;
        let mut w = 0usize;
        for i in 0..ns {
            if hi[i] - lo[i] > w {
                w = hi[i] - lo[i];
                m = i;
            }
        }
        if m == usize::MAX {
            out[..ns].copy_from_slice(&lo[..ns]);
            return;
        }
        let p = &seq[m][lo[m] + w / 2];

        let (mut lt, mut le) = (0usize, 0usize);
        for i in 0..ns {
            let s = seq[i];
            plt[i] = lo[i] + s[lo[i]..hi[i]].partition_point(|x| cmp(x, p) == Less);
            ple[i] = plt[i] + s[plt[i]..hi[i]].partition_point(|x| cmp(x, p) != Greater);
            lt += plt[i];
            le += ple[i];
        }

        if lt <= rank && rank <= le {
            let mut need = rank - lt;
            for i in 0..ns {
                let take = (ple[i] - plt[i]).min(need);
                out[i] = plt[i] + take;
                need -= take;
            }
            return;
        }
        if lt > rank {
            hi[..ns].copy_from_slice(&plt[..ns]);
        } else {
            lo[..ns].copy_from_slice(&ple[..ns]);
        }
    }
}

/// Um nivel da arvore binaria do tile: funde pares de segmentos contiguos de
/// `src` em `dst`. Devolve os novos segmentos (offset, len) e quantos sao.
fn merge_level<T, F>(
    src: &[T],
    segs: &[(usize, usize)],
    dst: &mut [T],
    cmp: &F,
) -> ([(usize, usize); KWAY_FANOUT], usize)
where
    T: Copy,
    F: Fn(&T, &T) -> Ordering,
{
    let mut out = [(0usize, 0usize); KWAY_FANOUT];
    let mut cnt = 0usize;
    let mut off = 0usize;
    let mut i = 0usize;
    while i < segs.len() {
        let (o1, l1) = segs[i];
        let l = if i + 1 == segs.len() {
            dst[off..off + l1].copy_from_slice(&src[o1..o1 + l1]);
            l1
        } else {
            let (_, l2) = segs[i + 1];
            let (left, right) = src[o1..o1 + l1 + l2].split_at(l1);
            merge_front(left, right, &mut dst[off..off + l1 + l2], cmp);
            l1 + l2
        };
        out[cnt] = (off, l);
        off += l;
        cnt += 1;
        i += 2;
    }
    (out, cnt)
}

/// Merge de 3..=K runs com blocagem de cache: a DRAM ve uma leitura e uma
/// escrita, em vez de log2(K) passadas.
fn kway_tiled<T, F>(runs: &[&[T]], dest: &mut [T], cmp: &F, tile: usize)
where
    T: Copy,
    F: Fn(&T, &T) -> Ordering,
{
    let ns = runs.len();
    let total = dest.len();
    if ns == 1 {
        dest.copy_from_slice(&runs[0][..total]);
        return;
    }
    if ns == 2 {
        merge_front(runs[0], runs[1], dest, cmp);
        return;
    }
    let filler = match runs.iter().find(|r| !r.is_empty()) {
        Some(r) => r[0],
        None => return,
    };
    let tile = tile.max(1);
    let mut scratch = vec![filler; 2 * tile];
    let (s1, s2) = scratch.split_at_mut(tile);

    let mut cur = [0usize; KWAY_FANOUT];
    let mut done = 0usize;

    while done < total {
        let want = tile.min(total - done);

        // Janela restrita: um tile consome no maximo `want` de qualquer run.
        let mut win: [&[T]; KWAY_FANOUT] = [&[]; KWAY_FANOUT];
        for i in 0..ns {
            let rem = &runs[i][cur[i]..];
            win[i] = &rem[..rem.len().min(want)];
        }
        let mut cut = [0usize; KWAY_FANOUT];
        multiseq_partition(&win[..ns], want, &mut cut[..ns], cmp);

        // Nivel 1: das janelas para s1.
        let mut segs = [(0usize, 0usize); KWAY_FANOUT];
        let mut cnt = 0usize;
        let mut off = 0usize;
        let mut i = 0usize;
        while i < ns {
            let l = if i + 1 == ns {
                let l = cut[i];
                s1[off..off + l].copy_from_slice(&win[i][..l]);
                l
            } else {
                let l = cut[i] + cut[i + 1];
                merge_front(
                    &win[i][..cut[i]],
                    &win[i + 1][..cut[i + 1]],
                    &mut s1[off..off + l],
                    cmp,
                );
                l
            };
            segs[cnt] = (off, l);
            off += l;
            cnt += 1;
            i += 2;
        }

        // Niveis intermediarios em s1/s2 (residentes em L1); o ULTIMO nivel
        // escreve direto no destino -- a unica escrita que chega na DRAM.
        let out = &mut dest[done..done + want];
        if cnt > 2 {
            let (segs2, _) = merge_level(&s1[..want], &segs[..cnt], s2, cmp);
            let (o1, l1) = segs2[0];
            let (_, l2) = segs2[1];
            let (left, right) = s2[o1..o1 + l1 + l2].split_at(l1);
            merge_front(left, right, out, cmp);
        } else {
            let (o1, l1) = segs[0];
            let (_, l2) = segs[1];
            let (left, right) = s1[o1..o1 + l1 + l2].split_at(l1);
            merge_front(left, right, out, cmp);
        }

        for i in 0..ns {
            cur[i] += cut[i];
        }
        done += want;
    }
}

/// Merge k-vias paralelo: corta por rank e recursa.
///
/// Streams vazios sao descartados a cada nivel. Depois de um corte e comum
/// sobrar um ou dois streams com dado num pedaco; sem o descarte eles
/// atravessavam os tres niveis do tile como se fossem oito. Com UM stream e
/// copia, com DOIS e o merge binario por co-rank -- que escala com as threads,
/// onde antes o caso de dois runs ficava preso em duas.
fn kway_merge<T, F>(runs: &[&[T]], dest: &mut [T], cmp: &F, t: &Tuning)
where
    T: Copy + Send + Sync,
    F: Fn(&T, &T) -> Ordering + Sync,
{
    let mut live: [&[T]; KWAY_FANOUT] = [&[]; KWAY_FANOUT];
    let mut ns = 0usize;
    for r in runs {
        if !r.is_empty() {
            live[ns] = r;
            ns += 1;
        }
    }
    let runs = &live[..ns];
    let total = dest.len();
    match ns {
        0 => return,
        1 => {
            par_copy(dest, runs[0], t.par_grain);
            return;
        }
        2 => {
            merge_two(runs[0], runs[1], dest, cmp, t);
            return;
        }
        _ => {}
    }
    if total <= t.par_grain || total < 2 {
        kway_tiled(runs, dest, cmp, t.tile);
        return;
    }

    let mut cut = [0usize; KWAY_FANOUT];
    multiseq_partition(runs, total / 2, &mut cut[..ns], cmp);

    let mut left: [&[T]; KWAY_FANOUT] = [&[]; KWAY_FANOUT];
    let mut right: [&[T]; KWAY_FANOUT] = [&[]; KWAY_FANOUT];
    let mut ltot = 0usize;
    for i in 0..ns {
        let (l, r) = runs[i].split_at(cut[i]);
        left[i] = l;
        right[i] = r;
        ltot += cut[i];
    }
    let (dl, dr) = dest.split_at_mut(ltot);
    rayon::join(
        || kway_merge(&left[..ns], dl, cmp, t),
        || kway_merge(&right[..ns], dr, cmp, t),
    );
}

// =====================================================================
//  TESTES
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[derive(Clone, Copy, Debug)]
    struct Keyed {
        key: u32,
        idx: u32,
    }
    impl PartialEq for Keyed {
        fn eq(&self, o: &Self) -> bool {
            self.key == o.key
        }
    }
    impl Eq for Keyed {}
    impl PartialOrd for Keyed {
        fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
            Some(self.cmp(o))
        }
    }
    impl Ord for Keyed {
        fn cmp(&self, o: &Self) -> Ordering {
            self.key.cmp(&o.key)
        }
    }

    fn pairs(v: &[Keyed]) -> Vec<(u32, u32)> {
        v.iter().map(|x| (x.key, x.idx)).collect()
    }

    fn keyed(keys: impl IntoIterator<Item = u32>) -> Vec<Keyed> {
        keys.into_iter()
            .enumerate()
            .map(|(i, key)| Keyed { key, idx: i as u32 })
            .collect()
    }

    /// Ajuste minusculo: forca co-rank, bidirecional, k-vias e tiles a
    /// recursar fundo com poucas centenas de elementos.
    fn tiny(route: Route) -> Tuning {
        Tuning {
            small_sort: 1,
            detect_block: 7,
            min_run: 3,
            struct_min_avg_run: 20,
            struct_min_avg_run_cheap: 30,
            min_segs: 2,
            abort_slack: 1,
            chunk: 16,
            par_grain: 6,
            bidir_min: 3,
            seq_node: 8,
            kway_min_elems: 12,
            tile: 5,
            chaos_percent: 9,
            random_percent: 35,
            cheap_ns: 10,
            force_cheap: None,
            // Radix em miniatura: digito de 2 bits, contagem so ate 2 bits,
            // baldes "grandes" com 24 elementos, para a rotina reentrar e
            // passar por todas as folhas com poucas centenas de elementos.
            radix_min: 8,
            radix_min_stable: 8,
            radix_digit: 2,
            radix_count_bits: 2,
            radix_lsd_cheap: 8,
            radix_lsd_costly: 16,
            radix_piece: 16,
            radix_big: 24,
            radix_big_div: 16,
            radix_recut: 12,
            radix_skew: 2,
            radix_leaf_min: 4,
            route,
        }
    }

    /// Estavel: identico, par a par, ao sort estavel da std.
    fn check_stable(v: &[Keyed], t: &Tuning, what: &str) {
        let mut reference = v.to_vec();
        reference.sort();
        let mut work = v.to_vec();
        sort_with(&mut work, &Keyed::cmp, true, t, None);
        assert_eq!(pairs(&work), pairs(&reference), "estavel: {what}");
    }

    /// Instavel: ordenado e com exatamente os mesmos pares (nada perdido,
    /// nada duplicado).
    fn check_unstable(v: &[Keyed], t: &Tuning, what: &str) {
        let mut work = v.to_vec();
        sort_with(&mut work, &Keyed::cmp, false, t, None);
        assert!(work.windows(2).all(|w| w[0].key <= w[1].key), "instavel nao ordenou: {what}");
        let mut got = pairs(&work);
        let mut want = pairs(v);
        got.sort_unstable();
        want.sort_unstable();
        assert_eq!(got, want, "instavel perdeu ou duplicou: {what}");
    }

    /// Gerador de padroes: dentes asc/desc com platos, faixas disjuntas ou
    /// intercaladas, trechos de ruido.
    fn pattern(rng: &mut StdRng, n: usize) -> Vec<Keyed> {
        let mut keys: Vec<u32> = Vec::with_capacity(n);
        let card = [1u32, 2, 3, 8, 50, 1000][rng.gen_range(0..6)];
        let disjoint = rng.gen_bool(0.5);
        let mut base = 0u32;
        while keys.len() < n {
            let len = rng.gen_range(1..=(n / 3).max(2)).min(n - keys.len());
            let plateau = rng.gen_range(1..=4u32);
            match rng.gen_range(0..4) {
                0 => keys.extend((0..len as u32).map(|k| base + k / plateau)),
                1 => keys.extend((0..len as u32).map(|k| base + (len as u32 - 1 - k) / plateau)),
                2 => keys.extend((0..len).map(|_| base + rng.gen_range(0..card))),
                _ => keys.extend(std::iter::repeat(base).take(len)),
            }
            if disjoint {
                base += len as u32 / plateau + rng.gen_range(0..2);
            } else if rng.gen_bool(0.3) {
                base = rng.gen_range(0..card);
            }
        }
        keyed(keys)
    }

    #[test]
    fn run_decomposition_is_valid_for_every_block_size() {
        let mut rng = StdRng::seed_from_u64(11);
        for trial in 0..600 {
            let n = rng.gen_range(1..400);
            let v = pattern(&mut rng, n);
            let block = [2usize, 3, 5, 7, 64, 32_768][trial % 6];
            let runs = detect_runs(&v, &Keyed::cmp, block, 3, usize::MAX).unwrap();
            let mut pos = 0usize;
            for r in &runs {
                let s = &v[pos..pos + r.len()];
                let non_dec = s.windows(2).all(|w| w[0].key <= w[1].key);
                let non_inc = s.windows(2).all(|w| w[0].key >= w[1].key);
                let strict_dec = s.windows(2).all(|w| w[0].key > w[1].key);
                match r.kind() {
                    ASC => assert!(non_dec, "ASC invalido, ensaio {trial}"),
                    FLAT => assert!(non_dec && non_inc, "FLAT invalido, ensaio {trial}"),
                    DESC => assert!(strict_dec, "DESC com empate, ensaio {trial}"),
                    _ => assert!(non_inc, "DESC_TIES invalido, ensaio {trial}"),
                }
                assert!(r.len() >= 1);
                pos += r.len();
            }
            assert_eq!(pos, n, "runs nao cobrem o array, ensaio {trial}");
        }
    }

    #[test]
    fn reversed_with_repeated_keys_is_one_run() {
        // O caso que motivou DESC_TIES: invertido com platos.
        let v = keyed((0..50_000u32).map(|i| (50_000 - i) / 500));
        for block in [7usize, 512, 32_768] {
            let runs = detect_runs(&v, &Keyed::cmp, block, 64, usize::MAX).unwrap();
            assert_eq!(runs.len(), 1, "bloco {block}: {} runs", runs.len());
            assert_eq!(runs[0].kind(), DESC_TIES);
        }
        check_stable(&v, &Tuning::for_type::<Keyed>(), "invertido com platos");
        check_unstable(&v, &Tuning::for_type::<Keyed>(), "invertido com platos");
    }

    #[test]
    fn stolen_element_is_returned_so_disjoint_teeth_chain() {
        // 1000 dentes em faixas disjuntas, direcao alternada: tem que virar
        // um run so depois de normalizar.
        let tooth = 300u32;
        let keys = (0..1000u32).flat_map(|d| {
            (0..tooth).map(move |k| d * tooth + if d % 2 == 0 { k } else { tooth - 1 - k })
        });
        let mut v = keyed(keys);
        let t = Tuning::for_type::<Keyed>();
        // A deteccao NAO pode abortar aqui: mil runs que encadeiam contam
        // como um punhado de segmentos.
        let limit = 2 * t.allowed_segs(v.len(), true) + t.abort_slack;
        let runs = detect_runs(&v, &Keyed::cmp, t.detect_block, t.min_run, limit)
            .expect("dentes encadeaveis nao podem disparar o aborto");
        let plan = plan_segments(&runs, t.min_run, t.chunk);
        let offsets = prefix_sums(&plan.lens);
        assert!(runs_after_coalesce(&v, &plan, &offsets, &Keyed::cmp) <= 12);
        apply_ops(&mut v, &plan, &Keyed::cmp, true);
        let offsets = coalesce(&v, prefix_sums(&plan.lens), &Keyed::cmp);
        // Fronteiras de bloco de deteccao podem deixar um punhado de cortes.
        assert!(offsets.len() - 1 <= 12, "{} runs depois de coalescer", offsets.len() - 1);
    }

    #[test]
    fn stolen_plateau_is_returned_whole_when_a_tooth_top_has_ties() {
        // Como o teste anterior, mas o topo de cada dente descendente tem o
        // maximo repetido (2 ou 3 vezes). A varredura ascendente engole o
        // plato inteiro; se devolver so um elemento, cada dente vira uma
        // quebra e o array todo vai para um merge. Chaves repetidas tambem
        // nos dentes ascendentes, para a devolucao nao confundir platos.
        let tooth = 300u32;
        for reps in [2u32, 3] {
            let keys = (0..1000u32).flat_map(move |d| {
                (0..tooth).map(move |k| {
                    let v = if d % 2 == 0 { k } else { tooth - 1 - k };
                    // topo do dente (v == tooth-1) repetido `reps` vezes:
                    // os `reps` maiores valores colapsam no maximo.
                    let v = if v + reps >= tooth { tooth - 1 } else { v };
                    d * tooth + v
                })
            });
            let mut v = keyed(keys);
            let t = Tuning::for_type::<Keyed>();
            let limit = 2 * t.allowed_segs(v.len(), true) + t.abort_slack;
            let runs = detect_runs(&v, &Keyed::cmp, t.detect_block, t.min_run, limit)
                .expect("dentes encadeaveis nao podem disparar o aborto");
            let plan = plan_segments(&runs, t.min_run, t.chunk);
            let offsets = prefix_sums(&plan.lens);
            assert!(
                runs_after_coalesce(&v, &plan, &offsets, &Keyed::cmp) <= 12,
                "reps={reps}: {} runs previstos depois de encadear",
                runs_after_coalesce(&v, &plan, &offsets, &Keyed::cmp)
            );
            let original = v.clone();
            apply_ops(&mut v, &plan, &Keyed::cmp, true);
            let offsets = coalesce(&v, prefix_sums(&plan.lens), &Keyed::cmp);
            assert!(
                offsets.len() - 1 <= 12,
                "reps={reps}: {} runs depois de coalescer",
                offsets.len() - 1
            );
            // E o resultado final continua exato, estavel e instavel.
            check_stable(&original, &t, &format!("dentes com topo repetido x{reps}"));
            check_unstable(&original, &t, &format!("dentes com topo repetido x{reps}"));
            // Tambem com graos minusculos, para o plato cruzar fronteiras de bloco.
            check_stable(&original, &tiny(Route::Auto), &format!("tiny, topo repetido x{reps}"));
            check_unstable(&original, &tiny(Route::Auto), &format!("tiny, topo repetido x{reps}"));
        }
    }

    #[test]
    fn restore_ties_undoes_the_reversal_of_equal_blocks() {
        let mut rng = StdRng::seed_from_u64(5);
        for _ in 0..200 {
            let n = rng.gen_range(1..40_000);
            let plateau = rng.gen_range(1..2000u32);
            let v = keyed((0..n as u32).map(|i| (n as u32 - i) / plateau));
            let mut reference = v.clone();
            reference.sort();
            let mut work = v.clone();
            work.reverse();
            restore_ties(&mut work, &Keyed::cmp);
            assert_eq!(pairs(&work), pairs(&reference));
        }
    }

    #[test]
    fn co_rank_boundary_is_valid() {
        let mut rng = StdRng::seed_from_u64(4242);
        for _ in 0..2000 {
            let m: usize = rng.gen_range(0..30);
            let n: usize = rng.gen_range(0..30);
            let mut a: Vec<i32> = (0..m).map(|_| rng.gen_range(0..8)).collect();
            let mut b: Vec<i32> = (0..n).map(|_| rng.gen_range(0..8)).collect();
            a.sort();
            b.sort();
            for k in 0..=(m + n) {
                let (i, j) = co_rank(k, &a, &b, &i32::cmp);
                assert_eq!(i + j, k);
                assert!(i <= a.len() && j <= b.len());
                if i > 0 && j < b.len() {
                    assert!(a[i - 1] <= b[j]);
                }
                if j > 0 && i < a.len() {
                    assert!(b[j - 1] < a[i]);
                }
            }
        }
    }

    #[test]
    fn front_and_back_merges_partition_the_output_for_every_cut() {
        let mut rng = StdRng::seed_from_u64(77);
        for _ in 0..300 {
            let m: usize = rng.gen_range(0..40);
            let n: usize = rng.gen_range(0..40);
            let mut idx = 0u32;
            let mut mk = |len: usize, rng: &mut StdRng| -> Vec<Keyed> {
                let mut v: Vec<Keyed> = (0..len)
                    .map(|_| {
                        idx += 1;
                        Keyed { key: rng.gen_range(0..6), idx }
                    })
                    .collect();
                v.sort();
                v
            };
            let a = mk(m, &mut rng);
            let b = mk(n, &mut rng);
            let mut reference = a.clone();
            reference.extend_from_slice(&b);
            reference.sort();
            for k in 0..=(m + n) {
                let mut dest = vec![Keyed { key: 0, idx: 0 }; m + n];
                let (df, db) = dest.split_at_mut(k);
                merge_front(&a, &b, df, &Keyed::cmp);
                merge_back(&a, &b, db, &Keyed::cmp);
                assert_eq!(pairs(&dest), pairs(&reference), "m={m} n={n} k={k}");
            }
        }
    }

    #[test]
    fn merge_two_matches_reference_with_deep_corank_subdivision() {
        let t = tiny(Route::Auto);
        let mut rng = StdRng::seed_from_u64(31416);
        for trial in 0..500 {
            let m: usize = rng.gen_range(0..150);
            let n: usize = rng.gen_range(0..150);
            let card = [2u32, 10, 1000][trial % 3];
            let mut all = keyed((0..m + n).map(|_| rng.gen_range(0..card)));
            let (a, b) = all.split_at_mut(m);
            a.sort();
            b.sort();
            let mut reference = all.clone();
            reference.sort();
            let mut dest = vec![Keyed { key: 0, idx: 0 }; m + n];
            merge_two(&all[..m], &all[m..], &mut dest, &Keyed::cmp, &t);
            assert_eq!(pairs(&dest), pairs(&reference), "ensaio {trial}: m={m} n={n}");
        }
    }

    #[test]
    fn multiseq_partition_selects_the_stable_prefix() {
        let mut rng = StdRng::seed_from_u64(909);
        for trial in 0..800 {
            let ns = rng.gen_range(1..=KWAY_FANOUT);
            let card = [2u32, 5, 100][trial % 3];
            let mut idx = 0u32;
            let seqs: Vec<Vec<Keyed>> = (0..ns)
                .map(|_| {
                    let len = rng.gen_range(0..25);
                    let mut s: Vec<Keyed> = (0..len)
                        .map(|_| {
                            idx += 1;
                            Keyed { key: rng.gen_range(0..card), idx }
                        })
                        .collect();
                    s.sort();
                    s
                })
                .collect();
            let refs: Vec<&[Keyed]> = seqs.iter().map(|s| s.as_slice()).collect();
            let total: usize = seqs.iter().map(|s| s.len()).sum();
            // Referencia: concatena em ordem de stream e ordena estavel.
            let mut reference: Vec<Keyed> = seqs.iter().flatten().copied().collect();
            reference.sort();
            for rank in 0..=total {
                let mut out = [0usize; KWAY_FANOUT];
                multiseq_partition(&refs, rank, &mut out[..ns], &Keyed::cmp);
                assert_eq!(out[..ns].iter().sum::<usize>(), rank);
                let mut got: Vec<(u32, u32)> = (0..ns)
                    .flat_map(|i| seqs[i][..out[i]].iter().map(|x| (x.key, x.idx)))
                    .collect();
                let mut want = pairs(&reference[..rank]);
                got.sort_unstable();
                want.sort_unstable();
                assert_eq!(got, want, "ensaio {trial}: ns={ns} rank={rank}");
            }
        }
    }

    #[test]
    fn every_route_matches_reference_with_tiny_grains() {
        let mut rng = StdRng::seed_from_u64(2024);
        for trial in 0..1500 {
            let n = rng.gen_range(0..700);
            let v = pattern(&mut rng, n);
            for route in [Route::Auto, Route::Structured, Route::Chaotic] {
                let mut t = tiny(route);
                if trial % 2 == 1 {
                    // Varia a regra de decisao: aceita mais fragmentacao.
                    t.struct_min_avg_run = 4;
                    t.struct_min_avg_run_cheap = 6;
                    t.abort_slack = 0;
                }
                // Os dois vereditos de custo, sem depender do relogio.
                t.force_cheap = Some(trial % 3 == 0);
                let what = format!("ensaio {trial}, n={n}, {route:?}");
                check_stable(&v, &t, &what);
                check_unstable(&v, &t, &what);
            }
        }
    }

    #[test]
    fn default_tuning_matches_reference_across_patterns() {
        let mut rng = StdRng::seed_from_u64(7);
        let t = Tuning::for_type::<Keyed>();
        for trial in 0..60 {
            let n = rng.gen_range(0..400_000);
            let v = pattern(&mut rng, n);
            let what = format!("ensaio {trial}, n={n}");
            check_stable(&v, &t, &what);
            check_unstable(&v, &t, &what);
        }
    }

    #[test]
    fn default_tuning_forced_routes_reach_the_kway_tree() {
        // 600 mil pares de 8 bytes passam de `kway_min_elems`: a arvore k-vias
        // e os tiles rodam com os graos de producao.
        let mut rng = StdRng::seed_from_u64(99);
        for trial in 0..6 {
            let n = 600_000 + rng.gen_range(0..50_000);
            let card = [3u32, 1000, u32::MAX][trial % 3];
            let v = keyed((0..n).map(|_| rng.gen_range(0..card)));
            for route in [Route::Structured, Route::Chaotic] {
                let mut t = Tuning::for_type::<Keyed>();
                t.route = route;
                let what = format!("ensaio {trial}, {route:?}");
                check_stable(&v, &t, &what);
                check_unstable(&v, &t, &what);
            }
        }
    }

    #[test]
    fn sorted_and_reversed_inputs_are_single_runs() {
        let mut v: Vec<i32> = (0..50_000).collect();
        multi_merge_sort(&mut v);
        assert!(v.windows(2).all(|w| w[0] <= w[1]));
        let mut v: Vec<i32> = (0..50_000).rev().collect();
        multi_merge_sort_stable(&mut v);
        assert!(v.windows(2).all(|w| w[0] <= w[1]));
        assert_eq!(detect_global_trend(&(0..50_000).collect::<Vec<i32>>()), vec![50_000]);
        assert_eq!(detect_global_trend(&(0..50_000).rev().collect::<Vec<i32>>()), vec![-50_000]);
    }

    #[test]
    fn comparator_and_key_variants() {
        let mut rng = StdRng::seed_from_u64(1);
        let base: Vec<(u32, u32)> = (0..200_000u32).map(|i| (rng.gen_range(0..500), i)).collect();

        // Comparador invertido, estavel: confere contra a std.
        let mut reference = base.clone();
        reference.sort_by(|a, b| b.0.cmp(&a.0));
        let mut work = base.clone();
        multi_merge_sort_stable_by(&mut work, |a, b| b.0.cmp(&a.0));
        assert_eq!(work, reference);

        // Chave, estavel.
        let mut reference = base.clone();
        reference.sort_by_key(|x| x.0);
        let mut work = base.clone();
        multi_merge_sort_stable_by_key(&mut work, |x| x.0);
        assert_eq!(work, reference);

        // Instavel: ordenado pela chave e mesma colecao.
        let mut work = base.clone();
        multi_merge_sort_by_key(&mut work, |x| x.0);
        assert!(work.windows(2).all(|w| w[0].0 <= w[1].0));
        work.sort_unstable();
        let mut want = base.clone();
        want.sort_unstable();
        assert_eq!(work, want);

        let mut work = base.clone();
        multi_merge_sort_by(&mut work, |a, b| b.cmp(a));
        assert!(work.windows(2).all(|w| w[0] >= w[1]));
    }

    #[test]
    fn caller_buffer_is_reused_across_calls_of_any_size() {
        let mut rng = StdRng::seed_from_u64(55);
        let mut scratch: Vec<Keyed> = Vec::new();
        // Tamanhos fora de ordem: o buffer cresce, sobra, cresce de novo.
        for (trial, n) in [300_000usize, 40_000, 700_000, 5, 150_000, 700_000, 0].into_iter().enumerate() {
            for route in [Route::Auto, Route::Structured, Route::Chaotic] {
                let v = pattern(&mut rng, n);
                let mut reference = v.clone();
                reference.sort();
                let mut t = Tuning::for_type::<Keyed>();
                t.route = route;

                let mut work = v.clone();
                sort_with(&mut work, &Keyed::cmp, true, &t, Some(&mut scratch));
                assert_eq!(pairs(&work), pairs(&reference), "estavel, ensaio {trial}, {route:?}");

                let mut work = v.clone();
                sort_with(&mut work, &Keyed::cmp, false, &t, Some(&mut scratch));
                assert!(work.windows(2).all(|w| w[0].key <= w[1].key));
                let mut got = pairs(&work);
                let mut want = pairs(&v);
                got.sort_unstable();
                want.sort_unstable();
                assert_eq!(got, want, "instavel, ensaio {trial}, {route:?}");
            }
        }
        // Graos minusculos, para a copia de folha rodar em todas as paridades.
        let mut scratch: Vec<Keyed> = vec![Keyed { key: 0, idx: 0 }; 900];
        for trial in 0..600 {
            let n = rng.gen_range(0..700);
            let v = pattern(&mut rng, n);
            let mut reference = v.clone();
            reference.sort();
            let route = [Route::Auto, Route::Structured, Route::Chaotic][trial % 3];
            let mut work = v.clone();
            sort_with(&mut work, &Keyed::cmp, true, &tiny(route), Some(&mut scratch));
            assert_eq!(pairs(&work), pairs(&reference), "tiny, ensaio {trial}, {route:?}");
        }

        // API publica.
        let mut scratch: Vec<u64> = Vec::new();
        for n in [200_000u64, 50_000, 400_000] {
            let mut a: Vec<u64> = (0..n).map(|i| (i * 7919) % 1000).collect();
            let mut b = a.clone();
            let mut want = a.clone();
            want.sort();
            multi_merge_sort_by_with_buffer(&mut a, &mut scratch, u64::cmp);
            multi_merge_sort_stable_by_with_buffer(&mut b, &mut scratch, u64::cmp);
            assert_eq!(a, want);
            assert_eq!(b, want);
        }
    }

    // ------------------------------------------------------------- radix

    /// Chaves u64 nos formatos que mudam o caminho do radix.
    fn radix_keys(rng: &mut StdRng, n: usize) -> Vec<u64> {
        let kind = rng.gen_range(0..7);
        let base: u64 = rng.gen_range(0..1u64 << 40);
        (0..n)
            .map(|_| match kind {
                0 => rng.gen_range(0..=u64::MAX), // faixa inteira
                1 => rng.gen_range(0..1000u64),   // faixa pequena
                2 => rng.gen_range(0..3u64),      // poucos valores
                3 => {
                    // enviesado: quase tudo num canto, o resto espalhado
                    if rng.gen_range(0..10) != 0 {
                        rng.gen_range(0..16u64)
                    } else {
                        rng.gen_range(0..=u64::MAX)
                    }
                }
                4 => 42,                               // tudo igual
                5 => base + rng.gen_range(0..200u64),  // faixa pequena deslocada
                _ => rng.gen_range(0..=u64::MAX) >> 20, // 44 bits
            })
            .collect()
    }

    #[test]
    fn radix_bare_integers_match_reference_on_both_engines() {
        let mut rng = StdRng::seed_from_u64(77);
        for trial in 0..500 {
            let n = rng.gen_range(0..2500);
            let v = radix_keys(&mut rng, n);
            let mut want = v.clone();
            want.sort();
            // O mesmo dado truncado para i32: sinal e largura menor.
            let vi: Vec<i32> = v.iter().map(|&x| x as i32).collect();
            let mut want_i = vi.clone();
            want_i.sort();
            let route = [Route::Chaotic, Route::Auto][trial % 2];
            let mut t = tiny(route);
            t.force_cheap = Some(trial % 4 < 2);
            for stable in [false, true] {
                let mut w = v.clone();
                sort_with_key(&mut w, &u64::cmp, stable, &t, None, &Bare);
                assert_eq!(w, want, "u64, ensaio {trial}, estavel={stable}, {route:?}");
                let mut w = vi.clone();
                sort_with_key(&mut w, &i32::cmp, stable, &t, None, &Bare);
                assert_eq!(w, want_i, "i32, ensaio {trial}, estavel={stable}, {route:?}");
            }
        }
        // Tamanho real, ajuste padrao, API publica: o aleatorio cai no radix.
        for n in [70_000usize, 300_000] {
            for modulo in [0u64, 1 << 40, 50_000, 3] {
                let v: Vec<u64> = (0..n)
                    .map(|_| {
                        let x = rng.gen_range(0..=u64::MAX);
                        if modulo == 0 {
                            x
                        } else {
                            x % modulo
                        }
                    })
                    .collect();
                let mut want = v.clone();
                want.sort();
                let mut a = v.clone();
                multi_merge_sort_radix(&mut a);
                assert_eq!(a, want, "u64 n={n} modulo={modulo}");
                let mut b = v.clone();
                multi_merge_sort_stable_radix(&mut b);
                assert_eq!(b, want, "u64 estavel n={n} modulo={modulo}");
                let s: Vec<i16> = v.iter().map(|&x| x as i16).collect();
                let mut want_s = s.clone();
                want_s.sort();
                let mut a = s.clone();
                multi_merge_sort_radix(&mut a);
                assert_eq!(a, want_s, "i16 n={n} modulo={modulo}");
                let mut b = s;
                multi_merge_sort_stable_radix(&mut b);
                assert_eq!(b, want_s, "i16 estavel n={n} modulo={modulo}");
                let wi: Vec<i32> = v.iter().map(|&x| x as i32).collect();
                let mut want_w = wi.clone();
                want_w.sort();
                let mut a = wi;
                multi_merge_sort_radix(&mut a);
                assert_eq!(a, want_w, "i32 n={n} modulo={modulo}");
            }
        }
        // O mapa da chave preserva a ordem e tem inverso, inclusive nos extremos.
        for x in [i32::MIN, -1, 0, 1, i32::MAX] {
            assert_eq!(i32::from_radix(x.to_radix()), x);
        }
        for x in [i64::MIN, -1, 0, 1, i64::MAX] {
            assert_eq!(i64::from_radix(x.to_radix()), x);
        }
        for x in [i8::MIN, -1, 0, 1, i8::MAX] {
            assert_eq!(i8::from_radix(x.to_radix()), x);
        }
        assert!(i32::MIN.to_radix() < (-1i32).to_radix());
        assert!((-1i32).to_radix() < 0i32.to_radix());
        assert!(0i32.to_radix() < i32::MAX.to_radix());
        assert_eq!(u64::from_radix(u64::MAX.to_radix()), u64::MAX);
    }

    #[test]
    fn radix_exact_key_keeps_payload_and_stability() {
        let mut rng = StdRng::seed_from_u64(78);
        let key = Exact(|k: &Keyed| k.key as u64);
        for trial in 0..400 {
            let n = rng.gen_range(0..2500);
            let card = [1u32, 2, 7, 300, 100_000, u32::MAX][rng.gen_range(0..6)];
            let v = keyed((0..n).map(|_| rng.gen_range(0..card)));
            let t = tiny([Route::Chaotic, Route::Auto][trial % 2]);
            let mut reference = v.clone();
            reference.sort();
            let mut w = v.clone();
            sort_with_key(&mut w, &Keyed::cmp, true, &t, None, &key);
            assert_eq!(pairs(&w), pairs(&reference), "estavel, ensaio {trial}, card {card}");
            let mut w = v.clone();
            sort_with_key(&mut w, &Keyed::cmp, false, &t, None, &key);
            assert!(w.windows(2).all(|p| p[0].key <= p[1].key), "instavel nao ordenou, ensaio {trial}");
            let (mut got, mut exp) = (pairs(&w), pairs(&v));
            got.sort_unstable();
            exp.sort_unstable();
            assert_eq!(got, exp, "instavel perdeu ou duplicou, ensaio {trial}");
        }
        // API publica, tamanho real. Muitas chaves repetidas: e onde a
        // estabilidade aparece.
        for card in [5000u32, u32::MAX] {
            let v = keyed((0..250_000).map(|_| rng.gen_range(0..card)));
            let mut reference = v.clone();
            reference.sort();
            let mut a = v.clone();
            multi_merge_sort_stable_by_radix_key(&mut a, |k| k.key as u64);
            assert_eq!(pairs(&a), pairs(&reference), "estavel, card {card}");
            let mut b = v.clone();
            multi_merge_sort_by_radix_key(&mut b, |k| k.key as u64);
            assert!(b.windows(2).all(|p| p[0].key <= p[1].key), "instavel, card {card}");
            let (mut got, mut exp) = (pairs(&b), pairs(&v));
            got.sort_unstable();
            exp.sort_unstable();
            assert_eq!(got, exp, "instavel perdeu ou duplicou, card {card}");
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    struct Wide {
        a: u16,
        b: u16,
        idx: u32,
    }

    /// A ordem de `Wide` nos testes: por `a`, depois por `b`. O prefixo
    /// radix e so o `a`.
    fn wide_cmp(x: &Wide, y: &Wide) -> Ordering {
        (x.a, x.b).cmp(&(y.a, y.b))
    }

    #[test]
    fn radix_prefix_key_orders_ties_with_the_comparator() {
        let mut rng = StdRng::seed_from_u64(79);
        let key = Prefix(|w: &Wide| w.a as u64);
        let make = |rng: &mut StdRng, n: usize, ca: u16, cb: u16| -> Vec<Wide> {
            (0..n)
                .map(|i| Wide { a: rng.gen_range(0..ca), b: rng.gen_range(0..cb), idx: i as u32 })
                .collect()
        };
        let check_unstable_wide = |w: &[Wide], v: &[Wide], what: &str| {
            assert!(w.windows(2).all(|p| wide_cmp(&p[0], &p[1]) != Greater), "nao ordenou: {what}");
            let (mut got, mut exp) = (w.to_vec(), v.to_vec());
            got.sort();
            exp.sort();
            assert_eq!(got, exp, "perdeu ou duplicou: {what}");
        };
        for trial in 0..400 {
            let n = rng.gen_range(0..2500);
            let ca = [1u16, 2, 9, 500, u16::MAX][rng.gen_range(0..5)];
            let cb = [1u16, 3, 1000][rng.gen_range(0..3)];
            let v = make(&mut rng, n, ca, cb);
            let t = tiny([Route::Chaotic, Route::Auto][trial % 2]);
            let mut reference = v.clone();
            reference.sort_by(wide_cmp);
            let mut w = v.clone();
            sort_with_key(&mut w, &wide_cmp, true, &t, None, &key);
            assert_eq!(w, reference, "estavel, ensaio {trial}, ca {ca}, cb {cb}");
            let mut w = v.clone();
            sort_with_key(&mut w, &wide_cmp, false, &t, None, &key);
            check_unstable_wide(&w, &v, &format!("ensaio {trial}, ca {ca}, cb {cb}"));
        }
        // API publica, tamanho real: prefixo largo, prefixo de poucos valores
        // e prefixo que nao distingue nada.
        for (ca, cb) in [(u16::MAX, 1000u16), (40, 1000), (1, 1000)] {
            let v = make(&mut rng, 250_000, ca, cb);
            let mut reference = v.clone();
            reference.sort_by(wide_cmp);
            let mut a = v.clone();
            multi_merge_sort_stable_by_prefix_key(&mut a, |w| w.a as u64, wide_cmp);
            assert_eq!(a, reference, "estavel, ca {ca}");
            let mut b = v.clone();
            multi_merge_sort_by_prefix_key(&mut b, |w| w.a as u64, wide_cmp);
            check_unstable_wide(&b, &v, &format!("publica, ca {ca}"));
        }
    }

    #[test]
    fn radix_reuses_the_caller_buffer() {
        let mut rng = StdRng::seed_from_u64(80);
        let mut scratch: Vec<u64> = Vec::new();
        for n in [100_000usize, 30_000, 250_000, 250_000] {
            let v: Vec<u64> = (0..n).map(|_| rng.gen_range(0..=u64::MAX)).collect();
            let mut want = v.clone();
            want.sort();
            for stable in [false, true] {
                let mut w = v.clone();
                let t = Tuning::for_type::<u64>();
                sort_with_key(&mut w, &u64::cmp, stable, &t, Some(&mut scratch), &Bare);
                assert_eq!(w, want, "n={n}, estavel={stable}");
            }
        }
        let mut scratch: Vec<u64> = vec![0; 100];
        for trial in 0..300 {
            let n = rng.gen_range(0..700);
            let v = radix_keys(&mut rng, n);
            let mut want = v.clone();
            want.sort();
            let mut w = v.clone();
            sort_with_key(&mut w, &u64::cmp, trial % 2 == 0, &tiny(Route::Chaotic), Some(&mut scratch), &Bare);
            assert_eq!(w, want, "tiny, ensaio {trial}");
        }
    }

    #[test]
    fn edge_lengths() {
        let mut rng = StdRng::seed_from_u64(3);
        let t = Tuning::for_type::<Keyed>();
        for n in [0usize, 1, 2, 3, 4095, 4096, 4097, 8191, 32_767, 32_768, 32_769, 65_534, 65_535, 65_536, 65_537] {
            let v = keyed((0..n).map(|_| rng.gen_range(0..16)));
            check_stable(&v, &t, &format!("n={n}"));
            check_unstable(&v, &t, &format!("n={n}"));
            let v = keyed((0..n as u32).map(|i| i / 3));
            check_stable(&v, &t, &format!("ordenado n={n}"));
            let v = keyed((0..n as u32).map(|i| (n as u32 - i) / 3));
            check_stable(&v, &t, &format!("invertido n={n}"));
            check_unstable(&v, &t, &format!("invertido n={n}"));
        }
    }
}
