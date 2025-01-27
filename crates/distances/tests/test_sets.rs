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

        // Kai's tests for hausdorff distance: hopefully they're right!
        // 1D test: a = {-5, 0, 1, 4}, b = {-2, 3, 7}, h = 3.0
        let a = vec![-5, 0, 1, 4];
        let b = vec![-2, 3, 7];
        assert_eq!(hausdorff(&a, &b), 3);
        
        // 2D test: a = {(0, 2), (1, -1), (3, 5)}, b = {(-2, 4), (3, 6), (2, 3)}, h = sqrt(17)
        let a = vec![(0, 2), (1, -1), (3, 5)];
        let b = vec![(-2, 4), (3, 6), (2, 3)];
        assert_eq!(hausdorff(&a, &b), 17f64.sqrt());

        // 3D test: a = {(0, 1, 2), (-2, 1, 3), (1, -4, 2)}, b = {(-1, 0, 1), (3, 2, 1), (5, 3, 0)}, h = sqrt(21)
        let a = vec![(0, 1, 2), (-2, 1, 3), (1, -4, 2)];
        let b = vec![(-1, 0, 1), (3, 2, 1), (5, 3, 0)];
        assert_eq!(hausdorff(&a, &b), 21f64.sqrt());

        // 4D test: a = {(1, 2, 3, 4), (5, 2, 1, -3), (-2, -1, 0, 1)}, b = {(3, 4, -4, 1), (0, 2, 1, 0)}, h = sqrt(34)
        let a = vec![(1, 2, 3, 4), (5, 2, 1, -3), (-2, -1, 0, 1)];
        let b = vec![(3, 4, -4, 1), (0, 2, 1, 0)];
        assert_eq!(hausdorff(&a, &b), 34f64.sqrt());
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
