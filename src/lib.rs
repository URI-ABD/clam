// use lazy_static::lazy_static;
// use std::collections::HashMap;

pub mod categorical;
pub mod numeric;
pub mod string;

// lazy_static! {
//     static ref FUNCS: HashMap<&'static str, u8> = {
//         let mut funcs = HashMap::new();
//         funcs.insert("euclidean", 1);
//         funcs.insert("euclideansq", 1);
//         funcs.insert("cosine", 1);
//         funcs.insert("manhattan", 1);
//         funcs.insert("hamming", 1);
//         funcs.insert("levenshtein", 1);
//         funcs
//     };
// }

#[cfg(test)]
mod tests {
    use float_cmp::approx_eq;

    use super::*;

    #[test]
    fn test_hamming() {
        let x = [1, 2, 3];
        let y = [1, 1, 1];
        assert_eq!(categorical::hamming(&x, &y), 2);
    }
    #[test]
    fn test_manhattan() {
        let x: [f64; 3] = [1., 2., 3.];
        let y: [f64; 3] = [1., 1., 1.];
        assert_eq!(numeric::manhattan(&x, &y), 3.);
    }
    #[test]
    fn test_cosine() {
        let x = [1.0, 1.0, 1.0];
        let y = [1.0, 1.0, 1.0];
        assert!(approx_eq!(f64, numeric::cosine(&x, &y), 1.0, ulps = 2));
    }
    #[test]
    fn test_euclidean() {
        let x = [0.0, 3.0];
        let y = [4.0, 0.0];
        assert_eq!(numeric::euclidean(&x, &y), 5.0);
    }
    #[test]
    fn test_euclideansq() {
        let x = [0.0, 3.0];
        let y = [4.0, 0.0];
        assert_eq!(numeric::euclideansq(&x, &y), 25.0);
    }
}
