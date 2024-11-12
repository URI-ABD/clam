

mod data;
mod utils;

#[test]
fn test_neighborhood_aware() {
    use utils::configure_logger;
    
    let _log_guard = configure_logger("./main_neighborhood_aware_test.log");
    
    use abd_clam::{Ball, Cluster, Partition};
    use data::NeighborhoodAware;
    
    let k = 10;
    
    let n = 20000;
    let dim = 2;
    let inlier_mean = 0.2;
    let outlier_mean = 0.25;
    let inlier_std = 0.1;
    let outlier_std = 0.13;
    
    let data = data::read_or_generate(
        None,
        &data::VecMetric::Euclidean,
        Some(n),
        Some(dim),
        Some(inlier_mean),
        Some(inlier_std),
        None,
    ).unwrap();
    
    let criteria = |c: &Ball<_, _, _>| c.cardinality() > 1;
    let root = Ball::new_tree(&data, &criteria, None);

    let data = NeighborhoodAware::par_new(&data, &root, k);
    
    let root = root.with_dataset_type::<NeighborhoodAware>();

    let test_cardinality = 10;
    let outliers = data::gen_random(outlier_mean, outlier_std, test_cardinality, dim, None);
    let inliers = data::gen_random(inlier_mean, inlier_std, test_cardinality, dim, None);

    let outlier_results: Vec<_> = outliers
            .iter()
            .map(|outlier| data.is_outlier(&root, outlier))
            .collect();
    
    let outlier_results = outlier_results.into_iter().enumerate().collect::<Vec<_>>();
    
    let inlier_results: Vec<_> = inliers
            .iter()
            .map(|inlier| data.is_outlier(&root, inlier))
            .collect();

    let inlier_results = inlier_results.into_iter().enumerate().collect::<Vec<_>>();
    
    print!("Outlier Results:\n {outlier_results:?}\n");
    print!("Inlier Results:\n {inlier_results:?}");
}
