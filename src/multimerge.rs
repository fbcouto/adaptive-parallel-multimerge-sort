use rayon::prelude::*;

// ==========================================
// 1. UTILITÁRIOS GENÉRICOS
// ==========================================

fn insertion_sort<T: Ord>(arr: &mut [T]) {
    for i in 1..arr.len() {
        let mut j = i;
        while j > 0 && arr[j - 1] > arr[j] {
            arr.swap(j - 1, j);
            j -= 1;
        }
    }
}

fn detectar_tendencia_global<T: Ord>(arr: &mut [T]) -> bool {
    let n = arr.len();
    if n <= 1 { return true; }

    // Ordenado?
    if arr.windows(2).all(|w| w[0] <= w[1]) { return true; }
    
    // Invertido?
    if arr.windows(2).all(|w| w[0] >= w[1]) {
        arr.reverse();
        return true;
    }
    false
}

// ==========================================
// 2. MOTOR ADAPTATIVO (INTERFACE PÚBLICA)
// ==========================================

pub fn ordenar_multi_merge<T: Ord + Clone + Send>(arr: &mut [T]) {
    let n = arr.len();
    if n < 1024 {
        insertion_sort(arr);
        return;
    }
    
    if detectar_tendencia_global(arr) { return; }

    // Heurística de Caos
    let mut e_caos_puro = false;
    if n > 120 {
        let mid = n / 2;
        let mut mudancas_direcao = 0;
        let mut subindo = arr[mid] <= arr[mid + 1];
        for i in (mid + 1)..(mid + 100).min(n - 1) {
            let direcao_atual = arr[i] <= arr[i + 1];
            if direcao_atual != subindo {
                mudancas_direcao += 1;
                subindo = direcao_atual;
            }
        }
        if mudancas_direcao > 15 { e_caos_puro = true; }
    }

    if !e_caos_puro {
        // Rota Merge Paralelo
        let mut buffer = vec![arr[0].clone(); n];
        let num_threads = rayon::current_num_threads();
        let threshold = (n / num_threads).max(1_000_000); 
        sort_recursivo_paralelo(arr, &mut buffer, threshold);
    } else {
        // Rota Quicksort Paralelo
        arr.par_sort_unstable();
    }
}

// ==========================================
// 3. LÓGICA DE FUSÃO (REDUZIDA)
// ==========================================

fn sort_recursivo_paralelo<T: Ord + Clone + Send>(arr: &mut [T], buffer: &mut [T], threshold: usize) {
    let n = arr.len();
    if n <= threshold {
        arr.sort_unstable(); // Usamos o sorte padrão estável aqui para simplificar
        return;
    }
    let mid = n / 2;
    let (left, right) = arr.split_at_mut(mid);
    let (buf_left, buf_right) = buffer.split_at_mut(mid);
    rayon::join(
        || sort_recursivo_paralelo(left, buf_left, threshold),
        || sort_recursivo_paralelo(right, buf_right, threshold)
    );
    mesclar_estavel(arr, mid, buffer);
}

fn mesclar_estavel<T: Ord + Clone>(arr: &mut [T], mid: usize, buffer: &mut [T]) {
    if arr[mid - 1] <= arr[mid] { return; }
    buffer[..arr.len()].clone_from_slice(arr);
    let (mut i, mut j, mut k) = (0, mid, 0);
    let n = arr.len();
    while i < mid && j < n {
        if buffer[i] <= buffer[j] { arr[k] = buffer[i].clone(); i += 1; }
        else { arr[k] = buffer[j].clone(); j += 1; }
        k += 1;
    }
    if i < mid { arr[k..n].clone_from_slice(&buffer[i..mid]); }
}