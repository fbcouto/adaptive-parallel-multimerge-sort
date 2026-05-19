use rayon::prelude::*;

fn calcular_minrun(mut n: usize) -> usize {
    let mut r = 0;
    while n >= 64 {
        r |= n & 1;
        n >>= 1;
    }
    n + r
}

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

    let mut ordenado = true;
    for i in 0..n - 1 {
        if arr[i] > arr[i + 1] {
            ordenado = false;
            break;
        }
    }
    if ordenado { return true; }

    let mut invertido = true;
    for i in 0..n - 1 {
        if arr[i] < arr[i + 1] {
            invertido = false;
            break;
        }
    }
    if invertido {
        arr.reverse();
        return true;
    }

    false
}

pub fn ordenar_multi_merge<T: Ord + Clone + Send>(arr: &mut [T]) {
    let n = arr.len();
    if n < 1024 {
        insertion_sort(arr);
        return;
    }
    
    if detectar_tendencia_global(arr) { return; }

    // HEURÍSTICA DE OSCILAÇÃO
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
        
        if mudancas_direcao > 15 {
            e_caos_puro = true;
        }
    }

    if !e_caos_puro {
        let mut buffer = vec![arr[0].clone(); n];
        let num_threads = rayon::current_num_threads();
        
        let threshold = (n / num_threads).max(32_768); 
        
        sort_recursivo_paralelo(arr, &mut buffer, threshold);
    } else {
        arr.par_sort_unstable();
    }
}

fn ordenar_sequencial_timsort_style<T: Ord + Clone>(arr: &mut [T], buffer: &mut [T]) {
    let n = arr.len();
    let minrun = calcular_minrun(n);

    for i in (0..n).step_by(minrun) {
        let end = (i + minrun).min(n);
        insertion_sort(&mut arr[i..end]);
    }

    let mut tamanho_bloco = minrun;
    while tamanho_bloco < n {
        for esq in (0..n).step_by(tamanho_bloco * 2) {
            let meio = (esq + tamanho_bloco).min(n);
            let dir = (esq + tamanho_bloco * 2).min(n);
            if meio < dir {
                mesclar_estavel(&mut arr[esq..dir], buffer, meio - esq);
            }
        }
        tamanho_bloco *= 2;
    }
}

fn sort_recursivo_paralelo<T: Ord + Clone + Send>(arr: &mut [T], buffer: &mut [T], threshold: usize) {
    let n = arr.len();
    if n <= threshold {
        ordenar_sequencial_timsort_style(arr, buffer);
        return;
    }

    let meio = n / 2;
    let (arr_esq, arr_dir) = arr.split_at_mut(meio);
    let (buf_esq, buf_dir) = buffer.split_at_mut(meio);

    rayon::join(
        || sort_recursivo_paralelo(arr_esq, buf_esq, threshold),
        || sort_recursivo_paralelo(arr_dir, buf_dir, threshold),
    );

    mesclar_estavel(arr, buffer, meio);
}

fn mesclar_estavel<T: Ord + Clone>(arr: &mut [T], buffer: &mut [T], meio: usize) {
    let n = arr.len();
    buffer[..n].clone_from_slice(&arr[..n]);

    let mut i = 0;
    let mut j = meio;
    let mut k = 0;

    while i < meio && j < n {
        if buffer[i] <= buffer[j] {
            arr[k] = buffer[i].clone();
            i += 1;
        } else {
            arr[k] = buffer[j].clone();
            j += 1;
        }
        k += 1;
    }

    if i < meio {
        arr[k..k + (meio - i)].clone_from_slice(&buffer[i..meio]);
    } else if j < n {
        arr[k..k + (n - j)].clone_from_slice(&buffer[j..n]);
    }
}

pub fn ordenar_multi_merge_generic<T: Ord>(arr: &mut [T]) {
    let name = std::any::type_name::<T>();

    if name == std::any::type_name::<i32>() {
        let safe_arr = unsafe { &mut *(arr as *mut [T] as *mut [i32]) };
        ordenar_multi_merge(safe_arr);
    } else if name == std::any::type_name::<u64>() {
        let safe_arr = unsafe { &mut *(arr as *mut [T] as *mut [u64]) };
        ordenar_multi_merge(safe_arr);
    } else if name == std::any::type_name::<u32>() {
        let safe_arr = unsafe { &mut *(arr as *mut [T] as *mut [u32]) };
        ordenar_multi_merge(safe_arr);
    } else if name == std::any::type_name::<u8>() {
        let safe_arr = unsafe { &mut *(arr as *mut [T] as *mut [u8]) };
        ordenar_multi_merge(safe_arr);
    } else if name == std::any::type_name::<u16>() {
        let safe_arr = unsafe { &mut *(arr as *mut [T] as *mut [u16]) };
        ordenar_multi_merge(safe_arr);
    } else if name == std::any::type_name::<u128>() {
        let safe_arr = unsafe { &mut *(arr as *mut [T] as *mut [u128]) };
        ordenar_multi_merge(safe_arr);
    } else {
        arr.sort_unstable();
    }
}