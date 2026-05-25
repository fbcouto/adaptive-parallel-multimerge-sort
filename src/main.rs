// The crate name matches what is in your Cargo.toml
use adaptive_parallel_multimerge_sort::sort;

fn main() {
    // Simple test example
    let mut data = vec![9, 5, 1, 3, 7, 8, 2, 6, 4, 0, 15, 12];
    
    println!("Before sorting: {:?}", data);
    
    // Calls your exported function from lib.rs
    sort(&mut data);
    
    println!("After sorting:  {:?}", data);
    
    // Simple verification
    let is_sorted = data.windows(2).all(|w| w[0] <= w[1]);
    println!("Is sorted? {}", is_sorted);
}