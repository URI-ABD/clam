//! 1D Wasserstein distance
use distances::{number::Float, Number};

/// Compute the Wasserstein distance between two 1D distributions.
///
/// Uses Euclidean distance as the ground metric.
///
/// See the [SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html) for more information.
pub fn wasserstein<T: Number, U: Float>(u_values: &Vec<T>, v_values: &Vec<T>) -> U {
    // Both vecs are assumed to be sorted. This simplifies the calculations to simply be
    // the summation of the differences of each index as the vecs can be considered to be
    // cumulative distribution functions. Therefore, we can use the Wasserstein definition:
    // W(u, v) = integral_-∞^∞(|U - V|)
    
    let u_values = u_values.iter().map(|f| U::from(*f)).collect::<Vec<U>>();
    let v_values = v_values.iter().map(|f| U::from(*f)).collect::<Vec<U>>();
    
    u_values.iter()
        .zip(v_values.iter())
        // find the absolute difference
        .map(|(&l, &r)| l.abs_diff(r))
        // find the sum
        .sum::<U>()
}

#[cfg(test)]
mod wasserstein_tests{
    use rand::{thread_rng, Rng};

    use crate::data::wasserstein::wasserstein;

    const K: usize = 100000;

    #[test]
    fn wasserstein_test(){
        let mut dirt: Vec<f32> = vec![0.; K];
        let mut holes: Vec<f32> = vec![0.; K];

        dirt = dirt.iter().map(|_| thread_rng().r#gen::<f32>()).collect();
        holes = holes.iter().map(|_| thread_rng().r#gen::<f32>()).collect();

        let t = std::time::Instant::now();

        dirt.sort_by(|a, b| a.partial_cmp(b).unwrap());
        holes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let res: f32 = wasserstein(&dirt, &holes);

        let time = t.elapsed().as_secs_f64();

        println!("Time: {}", time);

        println!("{}", res);
        }

        #[test]
        fn deep_tests(){
        for _ in 0..100{
            identity_test();
            symmetry_test();
            triangle_inequality_test();
        }
    }

    #[test]
    fn identity_test(){
        let mut dirt = vec![0.; K];
        dirt = dirt.iter().map(|_| thread_rng().r#gen::<f32>()).collect();

        let res: f32 = wasserstein(&dirt, &dirt);

        assert_eq!(res, 0.);
    }

    #[test]
    fn symmetry_test(){
        let mut dirt = vec![0.; K];
        let mut holes = vec![0.; K];

        dirt = dirt.iter().map(|_| thread_rng().r#gen::<f32>()).collect();
        holes = holes.iter().map(|_| thread_rng().r#gen::<f32>()).collect();

        let res1: f32 = wasserstein(&dirt, &holes);
        let res2: f32 = wasserstein(&holes, &dirt);

        assert_eq!(res1, res2);
    }

    #[test]
    fn triangle_inequality_test(){
            
        let mut v1 = vec![0.; K];
        let mut v2 = vec![0.; K];
        let mut v3 = vec![0.; K];

        v1 = v1.iter().map(|_| thread_rng().r#gen::<f32>()).collect();
        v2 = v2.iter().map(|_| thread_rng().r#gen::<f32>()).collect();
        v3 = v3.iter().map(|_| thread_rng().r#gen::<f32>()).collect();

        let d_v1_v3: f32 = wasserstein(&v1, &v3);
        let d_v1_v2: f32 = wasserstein(&v1, &v2);
        let d_v2_v3: f32 = wasserstein(&v2, &v3);

        assert!(d_v1_v3 <= d_v1_v2 + d_v2_v3);
    }
}