use adaptive_parallel_multimerge_sort::sort; // Seu alias
use rand::{thread_rng, Rng}; // Certifique-se de que rand está no seu Cargo.toml

// Definimos o gerador aqui para que o teste o possa ver
fn generate_random_u64(size: usize) -> Vec<u64> {
    let mut rng = thread_rng();
    (0..size).map(|_| rng.gen()).collect()
}

#[test]
fn testar_integridade_final() {
    let mut data = generate_random_u64(1_000_000);
    sort(&mut data); // Use o alias configurado
    assert!(data.windows(2).all(|w| w[0] <= w[1]), "O ARRAY NÃO ESTÁ ORDENADO!");
}

#[test]
fn testar_integridade_do_motor() {
    let mut dados = vec![5, 8, 2, 9, 1, 3, 7, 4, 6];
    sort(&mut dados);
    assert!(dados.windows(2).all(|w| w[0] <= w[1]), "Erro em dados aleatórios!");
}

#[test]
fn testar_inversamente_ordenados() {
    let mut dados = vec![10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
    sort(&mut dados);
    assert!(dados.windows(2).all(|w| w[0] <= w[1]), "Erro em dados inversos!");
}