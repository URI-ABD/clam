use std::time::Instant;
extern crate distances;
use distances::cosine;

	
fn main() {
	let n = 100000000;
    let x: Vec<f64> = vec![0.2; n];
    let y: Vec<f64> = vec![0.3; n];
    let start = Instant::now();
    let prod = cosine(&x,&y);
    println!("{}", start.elapsed().as_secs());
    println!("{}", prod);
    // assert!(approx_eq!(f64, prod, 6000000.0, ulps=2));
}