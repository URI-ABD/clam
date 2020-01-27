pub use crate::distance::*;

pub mod linalg {
	extern crate rayon;
	extern crate num;
	use num::Num;
	use std::iter::Sum;
	use rayon::prelude::*;
	pub fn dot<T: Num + Send + Sync + Copy + Sum>(x: &[T], y: &[T]) -> T {
		x.par_iter()
					.zip(y.par_iter())
					.map(|(a, b)| (*a) * (*b))
					.sum()
	}

	pub fn sub<T: Num + Send + Sync + Copy + Sum>(x: &[T], y: &[T]) -> Vec<T> {
		x.par_iter()
					.zip(y.par_iter())
					.map(|(a, b)| (*a) - (*b))
					.collect()
	}
	pub fn mul<T: Num + Send + Sync + Copy + Sum>(x: &[T], y: &[T]) -> Vec<T> {
		x.par_iter()
					.zip(y.par_iter())
					.map(|(a, b)| (*a) * (*b))
					.collect()
	}

}

pub mod distance {
	extern crate rayon;
	use std::cmp::PartialEq;
	use crate::linalg::dot;
	use rayon::prelude::*;
	pub fn euclidean(x: &[f64], y: &[f64]) -> f64 {
		euclideansq(x,y).sqrt()
	}

	pub fn euclideansq(x: &[f64], y: &[f64]) -> f64 {
		x.par_iter()
 					.zip(y.par_iter())
 					.map(|(a,b)| (a-b)*(a-b))
 					.sum()
	}

	pub fn cosine(x: &[f64], y: &[f64]) -> f64 {
		let num = dot(x,y);
		let dem = dot(x,x).sqrt() * dot(y,y).sqrt();
		num/dem
	}

	pub fn manhattan(x: &[i64], y: &[i64]) -> u64 {
		x.par_iter()
				   .zip(y.par_iter())
				   .map(|(a,b)| i64::abs(a-b) as u64)
				   .sum()
	}

	pub fn hamming<T: PartialEq + Sync>(x: &[T], y: &[T]) -> u64 {
		x.par_iter()
				   .zip(y.par_iter())
				   .map(|(a,b)| if a == b {0} else {1})
				   .sum()
	}

	pub fn levenshtein(a: &str, b: &str) -> usize {
        let len_a = a.chars().count();
        let len_b = b.chars().count();
        if len_a < len_b{
            return levenshtein(b, a)
        }
        // handle special case of 0 length
        if len_a == 0 {
            return len_b
        } else if len_b == 0 {
            return len_a
        }

        let len_b = len_b + 1;

        let mut pre;
        let mut tmp;
        let mut cur = vec![0; len_b];

        // initialize DP table for string b
        for i in 1..len_b {
            cur[i] = i;
        }

        // calculate edit distance
        for (i,ca) in a.chars().enumerate() {
            // get first column for this row
            pre = cur[0];
            cur[0] = i + 1;
            for (j, cb) in b.chars().enumerate() {
                tmp = cur[j + 1];
                cur[j + 1] = std::cmp::min(
                    // deletion
                    tmp + 1, std::cmp::min(
                    // insertion
                    cur[j] + 1,
                    // match or substitution
                    pre + if ca == cb { 0 } else { 1 }));
                pre = tmp;
            }
        }
        cur[len_b - 1]
    }

}

#[cfg(test)]
mod tests {
	use crate::distance;
	use float_cmp::approx_eq;
    #[test]
    fn test_hamming() {
    	let x = [1, 2, 3];
    	let y = [1, 1, 1];
        assert_eq!(distance::hamming(&x,&y), 2);
    }
    #[test]
    fn test_manhattan() {
    	let x = [1, 2, 3];
    	let y = [1, 1, 1];
        assert_eq!(distance::manhattan(&x,&y), 3);
    }
    #[test]
    fn test_cosine() {
    	let x = [1.0, 1.0, 1.0];
    	let y = [1.0, 1.0, 1.0];
        assert!(approx_eq!(f64, distance::cosine(&x,&y), 1.0, ulps=2));
    }
    #[test]
    fn test_euclidean() {
    	let x = [0.0,3.0];
    	let y = [4.0,0.0];
        assert_eq!(distance::euclidean(&x,&y), 5.0);
    }
    #[test]
    fn test_euclideansq() {
    	let x = [0.0,3.0];
    	let y = [4.0,0.0];
        assert_eq!(distance::euclideansq(&x,&y), 25.0);
    }
}