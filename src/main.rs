// O nome do crate é o que está no seu Cargo.toml
use adaptive_parallel_multimerge_sort::sort;

fn main() {
    // Exemplo de teste simples
    let mut data = vec![9, 5, 1, 3, 7, 8, 2, 6, 4, 0, 15, 12];
    
    println!("Antes da ordenação: {:?}", data);
    
    // Chama a sua função exportada no lib.rs
    sort(&mut data);
    
    println!("Após a ordenação:   {:?}", data);
    
    // Verificação simples
    let is_sorted = data.windows(2).all(|w| w[0] <= w[1]);
    println!("Ordenação correta?  {}", is_sorted);
}