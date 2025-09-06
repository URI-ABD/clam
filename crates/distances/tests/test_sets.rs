#![allow(missing_docs)]

use rand::RngExt;

use distances::{
    Number,
    number::Addition,
    sets::{dice, hausdorff, jaccard, kulsinski},
    vectors::{euclidean, manhattan},
};

#[test]
#[allow(clippy::float_equality_without_abs)]
fn test_dice() {
    let x = vec![1, 2, 3];
    let y = vec![1, 2, 3];

    let distance: f32 = dice(&x, &y);
    assert!(distance < f32::EPSILON);

    let x = vec![1, 2, 3, 4, 5];
    let y = vec![6, 7, 8, 9, 10];

    let distance: f32 = dice(&x, &y);
    assert!((distance - 1.0) < f32::EPSILON);

    let x = vec![1, 2, 3, 4, 5];
    let y = vec![4, 5, 6, 7, 8];

    let distance: f32 = dice(&x, &y);
    assert!((distance - 0.6) < f32::EPSILON);

    let x = vec![1, 2, 3, 4, 5];
    let y = vec![5, 6, 7, 8, 9];

    let distance: f32 = dice(&x, &y);
    assert!((distance - 0.8) < f32::EPSILON);
}

/// Generates a length 1000 set with each bit flipped randomly on or off
fn gen_set() -> Vec<u16> {
    let mut vec = Vec::new();
    for i in 0..1000 {
        if rand::random() {
            vec.push(i);
        }
    }
    vec
}

/// Random exhaustive testing of set distances, manually creating union and intersection values
#[test]
fn sets_test() {
    for _ in 0..10000 {
        let x: Vec<u16> = gen_set();
        let y: Vec<u16> = gen_set();
        let mut union: usize = 0;
        let mut intersection: usize = 0;
        let mut size: usize = 0;
        for i in 0_u16..1000 {
            if x.contains(&i) || y.contains(&i) {
                union += 1;
                size += 1;
            }
            if x.contains(&i) && y.contains(&i) {
                intersection += 1;
                size += 1;
            }
        }
        let mut distance: f32;
        let mut real_distance: f32;

        distance = jaccard(&x, &y);
        if union == 0 {
            real_distance = 0.0;
        } else {
            real_distance = 1_f32 - (intersection as f32) / (union as f32);
        }
        assert!((distance - real_distance).abs() < f32::EPSILON);

        distance = dice(&x, &y);
        if union == 0 {
            real_distance = 0.0;
        } else {
            real_distance = 1_f32 - (2_f32 * ((intersection as f32) / (size as f32)));
        }
        assert!((distance - real_distance).abs() < f32::EPSILON);

        distance = kulsinski(&x, &y);
        if union == 0 {
            real_distance = 0.0;
        } else {
            real_distance = 1_f32 - (intersection as f32) / ((union + union - intersection) as f32);
        }
        assert!((distance - real_distance).abs() < f32::EPSILON);
    }
}

/// Boundary testing for set distances, equal sets or one zero set
#[test]
fn bounds_test() {
    let x: Vec<u16> = gen_set();
    let y: Vec<u16> = Vec::new();

    let mut distance: f32;

    distance = jaccard(&x, &x);
    assert!(distance < f32::EPSILON);
    distance = jaccard(&x, &y);
    assert!((distance - 1.0).abs() < f32::EPSILON);
    distance = jaccard(&y, &y);
    assert!((distance - 1.0).abs() < f32::EPSILON);

    distance = dice(&x, &x);
    assert!(distance < f32::EPSILON);
    distance = dice(&x, &y);
    assert!((distance - 1.0).abs() < f32::EPSILON);
    distance = dice(&y, &y);
    assert!((distance - 1.0).abs() < f32::EPSILON);

    distance = kulsinski(&x, &x);
    assert!(distance < f32::EPSILON);
    distance = kulsinski(&x, &y);
    assert!((distance - 1.0).abs() < f32::EPSILON);
    distance = kulsinski(&y, &y);
    assert!((distance - 1.0).abs() < f32::EPSILON);
}

#[test]
fn hausdorff_test() {
    // Helper function to generate a vector of random points with given count and dimension
    fn gen_points(count: usize, dim: usize) -> Vec<Vec<u16>> {
        let mut rng = rand::rng();
        (0..count).map(|_| (0..dim).map(|_| rng.random_range(0..100)).collect()).collect()
    }

    // Generate random sets of points for Hausdorff distance testing
    let x: Vec<Vec<u16>> = gen_points(5, 2); // 5 points, 2D
    let y: Vec<Vec<u16>> = gen_points(5, 3); // 5 points, 3D
    let z: Vec<Vec<u16>> = gen_points(5, 4); // 5 points, 4D

    // euclidean testing
    let euc_ground_dist = |a: &Vec<u16>, b: &Vec<u16>| euclidean::<_, f32>(a, b);
    let distance_xx = hausdorff(&x, &x, euc_ground_dist);
    assert!(
        distance_xx < f32::EPSILON,
        "Expected `distance_xx` to be less than `f32::EPSILON`, but got {distance_xx:.2e}"
    ); // identity test
    let distance_yy = hausdorff(&y, &y, euc_ground_dist);
    assert!(
        distance_yy < f32::EPSILON,
        "Expected `distance_yy` to be less than `f32::EPSILON`, but got {distance_yy:.2e}"
    ); // identity test
    let distance_xy = hausdorff(&x, &y, euc_ground_dist);
    let distance_yx = hausdorff(&y, &x, euc_ground_dist);
    let diff = distance_xy.abs_diff(distance_yx);
    assert!(
        diff < f32::EPSILON,
        "Expected `distance_xy` and `distance_yx` to be equal, but got {distance_xy:.2e} and {distance_yx:.2e} with a difference of {diff:.2e}"
    ); // symmetry test

    // triangle inequality test for euclidean
    let distance_xz = hausdorff(&x, &z, euc_ground_dist);
    let distance_yz = hausdorff(&y, &z, euc_ground_dist);
    let longest_side = distance_xy.max(distance_xz).max(distance_yz);
    let sum_of_others = distance_xy + distance_xz + distance_yz - longest_side;
    assert!(
        longest_side <= sum_of_others,
        "Expected `longest_side` to be less than or equal to `sum_of_others`, but got {longest_side:.2e} and {sum_of_others:.2e} among `xy`: {distance_xy:.2e}, `xz`: {distance_xz:.2e}, `yz`: {distance_yz:.2e}"
    ); // triangle inequality test

    // manhattan testing
    let man_ground_dist = |a: &Vec<u16>, b: &Vec<u16>| manhattan(a, b).as_f32();
    let distance_xx = hausdorff(&x, &x, man_ground_dist);
    assert!(
        distance_xx < f32::EPSILON,
        "Expected `distance_xx` to be less than `f32::EPSILON`, but got {distance_xx:.2e}"
    ); // identity test
    let distance_yy = hausdorff(&y, &y, man_ground_dist);
    assert!(
        distance_yy < f32::EPSILON,
        "Expected `distance_yy` to be less than `f32::EPSILON`, but got {distance_yy:.2e}"
    ); // identity test
    let distance_xy = hausdorff(&x, &y, man_ground_dist);
    let distance_yx = hausdorff(&y, &x, man_ground_dist);
    let diff = distance_xy.abs_diff(distance_yx);
    assert!(
        diff < f32::EPSILON,
        "Expected `distance_xy` and `distance_yx` to be equal, but got {distance_xy:.2e} and {distance_yx:.2e} with a difference of {diff:.2e}"
    ); // symmetry test

    // triangle inequality test for manhattan
    let distance_xz = hausdorff(&x, &z, man_ground_dist);
    let distance_yz = hausdorff(&y, &z, man_ground_dist);
    let longest_side = distance_xy.max(distance_xz).max(distance_yz);
    let sum_of_others = distance_xy + distance_xz + distance_yz - longest_side;
    assert!(
        longest_side <= sum_of_others,
        "Expected `longest_side` to be less than or equal to `sum_of_others`, but got {longest_side:.2e} and {sum_of_others:.2e} among `xy`: {distance_xy:.2e}, `xz`: {distance_xz:.2e}, `yz`: {distance_yz:.2e}"
    ); // triangle inequality test

    // jaccard testing
    let jac_ground_dist = |a: &Vec<u16>, b: &Vec<u16>| jaccard::<_, f32>(a, b);
    let distance_xx = hausdorff(&x, &x, jac_ground_dist);
    assert!(
        distance_xx < f32::EPSILON,
        "Expected `distance_xx` to be less than `f32::EPSILON`, but got {distance_xx:.2e}"
    ); // identity test
    let distance_yy = hausdorff(&y, &y, jac_ground_dist);
    assert!(
        distance_yy < f32::EPSILON,
        "Expected `distance_yy` to be less than `f32::EPSILON`, but got {distance_yy:.2e}"
    ); // identity test
    let distance_xy = hausdorff(&x, &y, jac_ground_dist);
    let distance_yx = hausdorff(&y, &x, jac_ground_dist);
    let diff = distance_xy.abs_diff(distance_yx);
    assert!(
        diff < f32::EPSILON,
        "Expected `distance_xy` and `distance_yx` to be equal, but got {distance_xy:.2e} and {distance_yx:.2e} with a difference of {diff:.2e}"
    ); // symmetry test

    // triangle inequality test for jaccard
    let distance_xz = hausdorff(&x, &z, jac_ground_dist);
    let distance_yz = hausdorff(&y, &z, jac_ground_dist);
    let longest_side = distance_xy.max(distance_xz).max(distance_yz);
    let sum_of_others = distance_xy + distance_xz + distance_yz - longest_side;
    assert!(
        longest_side <= sum_of_others,
        "Expected `longest_side` to be less than or equal to `sum_of_others`, but got {longest_side:.2e} and {sum_of_others:.2e} among `xy`: {distance_xy:.2e}, `xz`: {distance_xz:.2e}, `yz`: {distance_yz:.2e}"
    );
    // triangle inequality test
}
