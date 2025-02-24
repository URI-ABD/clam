//! Tests for the `mbed` module.

use abd_clam::{
    cluster::Partition,
    mbed::{MassSpringSystem, Spring, Vector},
    metric::Euclidean,
    Ball, Cluster, Dataset, FlatVec,
};
use distances::{number::Multiplication, Number};

#[test]
fn test_spring() -> Result<(), String> {
    let items = vec![Vector::new([0.0, 0.0]), Vector::new([0.0, 1.0])];
    let data = FlatVec::new(items)?;
    let metric = Euclidean;
    let criteria = |_: &Ball<_>| true;
    let seed = Some(42);
    let root = Ball::new_tree(&data, &metric, &criteria, seed);
    let (a, b) = {
        let children = root.children();
        if children.len() != 2 {
            return Err("expected 2 children".to_string());
        }
        (children[0], children[1])
    };

    let l0 = data.one_to_one(a.arg_center(), b.arg_center(), &metric);
    let l = l0.as_f32().double();
    let mut ab = Spring::new([a, b], l0, l, 0, 0.5);

    assert_eq!(ab.k(), 1.0);
    assert_eq!(ab.dx(), 1.0);
    assert_eq!(ab.f_mag(), -1.0);
    assert_eq!(ab.ratio(), 1.0);
    assert_eq!(ab.potential_energy(), 0.5);

    ab.update_length(l0.as_f32().half());

    assert_eq!(ab.k(), 1.0);
    assert_eq!(ab.dx(), -0.5);
    assert_eq!(ab.f_mag(), 0.5);
    assert_eq!(ab.ratio(), 0.5);
    assert_eq!(ab.potential_energy(), 0.125);

    let state = vec![
        [Vector::new([0.0, 0.0]), Vector::new([0.0, 0.0])],
        [Vector::new([0.0, 1.0]), Vector::new([0.0, 0.0])],
    ];
    let system = MassSpringSystem::<2, _, f32, Ball<_>>::new(data.cardinality())?.with_initial_state(&state)?;

    let uv = ab.unit_vector(&system);
    assert_eq!(uv, Vector::new([0.0, 1.0]));
    assert_eq!(-uv, Vector::new([0.0, -1.0]));

    Ok(())
}
