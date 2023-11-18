//! Tests to generate random numbers.

use distances::{number::Bool, Number};
use test_case::test_case;

#[test_case(0)]
#[test_case(1)]
#[test_case(10)]
#[test_case(100)]
#[test_case(1000)]
fn test_rand_gen(len: usize) {
    let mut rng = rand::thread_rng();
    test_vec::<f32, _>(len, &mut rng);
    test_vec::<f64, _>(len, &mut rng);
    test_vec::<i8, _>(len, &mut rng);
    test_vec::<i16, _>(len, &mut rng);
    test_vec::<i32, _>(len, &mut rng);
    test_vec::<i64, _>(len, &mut rng);
    test_vec::<i128, _>(len, &mut rng);
    test_vec::<u8, _>(len, &mut rng);
    test_vec::<u16, _>(len, &mut rng);
    test_vec::<u32, _>(len, &mut rng);
    test_vec::<u64, _>(len, &mut rng);
    test_vec::<u128, _>(len, &mut rng);
    test_vec::<Bool, _>(len, &mut rng);
}

fn test_vec<T: Number, R: rand::Rng>(len: usize, rng: &mut R) {
    let vec = (0..len).map(|_| T::next_random(rng)).collect::<Vec<_>>();
    assert_eq!(vec.len(), len);
}
