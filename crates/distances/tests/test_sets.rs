use distances::sets::dice;
use distances::sets::jaccard;
use distances::sets::kulsinski;

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


/// Kai's tests for hausdorff distance
#[test]
fn hausdorff_test() {

    // TODO: property-based testing - equality, symmetry, triangle inequality

    // random sets I made up for testing
    let x: Vec<Vec<u16>> = vec![vec![1, 2], vec![2, 3], vec![3, 4], vec![5, 4], vec![5, 1]];
    let y: Vec<Vec<u16>> = vec![vec![5, 2], vec![3, 2], vec![2, 4], vec![7, 3], vec![8, 9]];
    let z: Vec<Vec<u16>> = vec![vec![2, 1], vec![6, 3], vec![1, 4], vec![3, 3], vec![5, 1]]; // for triange inequality

    // euclidean testing
    let distance_xx: f32 = distances::sets::hausdorff(&x, &x, compare, euclidean);
    assert!(distance_xx < f32::EPSILON); // identity test
    let distance_yy: f32 = distances::sets::hausdorff(&y, &y, compare, euclidean);
    assert!(distance_yy < f32::EPSILON); // identity test
    let distance_xy: f32 = distances::sets::hausdorff(&x, &y, compare, euclidean);
    assert!(distance_xy - 1.0 < f32::EPSILON); // another test
    let distance_yx: f32 = distances::sets::hausdorff(&y, &x, compare, euclidean);
    // ensure two distances are equal for symmetry
    assert!(distance_xy - distance_yx < f32::EPSILON);

    // triangle inequality test for euclidian
    let distance_xz: f32 = distances::sets::hausdorff(&x, &z, compare, euclidean);
    let distance_yz: f32 = distances::sets::hausdorff(&y, &z, compare, euclidean);
    // of the three sides xy, xz, and yz, the longest side should be less than or equal to the sum of the other two sides
    // first, find the longest side
    let longest_side = distance_xy.max(distance_xz).max(distance_yz);
    // now, find the sum of the other two sides
    let sum_of_others = distance_xy + distance_xz + distance_yz - longest_side;
    // finally, check that the longest side is less than or equal to the sum of the other two sides
    assert!(longest_side <= sum_of_others);


    // manhattan testing
    let distance_xx: f32 = distances::sets::hausdorff(&x, &x, compare, manhattan);
    assert!(distance_xx < f32::EPSILON); // identity test
    let distance_yy: f32 = distances::sets::hausdorff(&y, &y, compare, manhattan);
    assert!(distance_yy < f32::EPSILON); // identity test
    let distance_xy: f32 = distances::sets::hausdorff(&x, &y, compare, manhattan);
    assert!(distance_xy - 1.0 < f32::EPSILON); // another test
    let distance_yx: f32 = distances::sets::hausdorff(&y, &x, compare, manhattan);
    // ensure two distances are equal for symmetry
    assert!(distance_xy - distance_yx < f32::EPSILON);

    // triangle inequality test for manhattan
    let distance_xz: f32 = distances::sets::hausdorff(&x, &z, compare, manhattan);
    let distance_yz: f32 = distances::sets::hausdorff(&y, &z, compare, manhattan);
    // of the three sides xy, xz, and yz, the longest side should be less than or equal to the sum of the other two sides
    // first, find the longest side
    let longest_side = distance_xy.max(distance_xz).max(distance_yz);
    // now, find the sum of the other two sides
    let sum_of_others = distance_xy + distance_xz + distance_yz - longest_side;
    // finally, check that the longest side is less than or equal to the sum of the other two sides
    assert!(longest_side <= sum_of_others);


    // jaccard testing
    let distance_xx: f32 = distances::sets::hausdorff(&x, &x, compare, jaccard);
    assert!(distance_xx < f32::EPSILON); // identity test
    let distance_yy: f32 = distances::sets::hausdorff(&y, &y, compare, jaccard);
    assert!(distance_yy < f32::EPSILON); // identity test
    let distance_xy: f32 = distances::sets::hausdorff(&x, &y, compare, jaccard);
    assert!(distance_xy - 1.0 < f32::EPSILON); // another test
    let distance_yx: f32 = distances::sets::hausdorff(&y, &x, compare, jaccard);
    // ensure two distances are equal for symmetry
    assert!(distance_xy - distance_yx < f32::EPSILON);


    // triangle inequality test for jaccard
    let distance_xz: f32 = distances::sets::hausdorff(&x, &z, compare, jaccard);
    let distance_yz: f32 = distances::sets::hausdorff(&y, &z, compare, jaccard);

    // print the 3 distances for debugging
    println!("distance_xy: {}", distance_xy);
    println!("distance_xz: {}", distance_xz);
    println!("distance_yz: {}", distance_yz);

    // of the three sides xy, xz, and yz, the longest side should be less than or equal to the sum of the other two sides
    // first, find the longest side
    let longest_side = distance_xy.max(distance_xz).max(distance_yz);
    // now, find the sum of the other two sides
    let sum_of_others = distance_xy + distance_xz + distance_yz - longest_side;
    // finally, check that the longest side is less than or equal to the sum of the other two sides
    assert!(longest_side <= sum_of_others);
}

// Kai's functions for hausdorff distance testing - if you'd like me to make them lambdas or something, let me know

// compare function for hausdorff distance
fn compare(a: f32, b: f32) -> bool {
    a < b
}

// euclidean distance function between two points
fn euclidean(a: &[u16], b: &[u16]) -> f32 {
    let mut sum: f32 = 0.0;
    for i in 0..a.len() {
        sum += (a[i] as f32 - b[i] as f32).powi(2);
    }
    sum.sqrt()
}

// manhattan distance function between two points
fn manhattan(a: &[u16], b: &[u16]) -> f32 {
    let mut sum: f32 = 0.0;
    for i in 0..a.len() {
        sum += (a[i] as f32 - b[i] as f32).abs();
    }
    sum
}