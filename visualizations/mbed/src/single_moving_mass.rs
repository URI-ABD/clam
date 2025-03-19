//! Simulating for a single moving mass.

use abd_clam::{
    Ball, Cluster, Dataset, FlatVec, Metric,
    cluster::Partition,
    mbed::{
        Vector,
        mss::{Mass, Spring},
    },
    metric::Manhattan,
};
use distances::{
    Number,
    number::{Addition, Float},
};
use generational_arena::{Arena, Index};

pub fn one_moving_mass<F: Float>(drag: F, dt: F, l: F, n: usize) -> [Vec<F>; 6] {
    let data = create_data::<F, 10>();
    let metric = Manhattan;
    let [a, b] = create_cluster_pair(&data, &metric);

    let k = F::ONE;
    let l0 = a.distance_to(&b, &data, &metric);
    let l = l0 * l;
    let dx = l0 - l;

    let (mut masses, [i, j]) = create_arena::<_, _, _, 2>(&a, &b, l);

    let mut s = Spring::new(i, j, &masses, k, l0, 0, true);

    // Check the properties of the spring.
    assert_eq!(s.dx(), dx);
    assert_eq!(s.ratio(), dx.abs() / l0);
    assert_eq!(s.pe(), k * dx.square().half());

    let (x1s, x2s, kes, pes, fs) = (0..n)
        .map(|_| {
            masses[i].add_f(s.f());
            masses[j].sub_f(s.f());

            masses[i].apply_f(dt, drag);
            masses[j].apply_f(dt, drag);

            let ke = masses[i].ke() + masses[j].ke();
            let pe = s.pe();
            let f = s.f().magnitude() * Addition::neg(s.dx().signum());

            s.recalculate(&masses);

            (masses[i].x()[0], masses[j].x()[0], ke, pe, f)
        })
        .collect();

    let ts = (0..n).map(|t| F::from(t) * dt).collect();

    [ts, x1s, x2s, kes, pes, fs]
}

/// Creates a `FlatVec` with two `[F; DIM]`s.
fn create_data<F: Float, const DIM: usize>() -> FlatVec<[F; DIM], usize> {
    let a = [F::ZERO; DIM];
    let b = {
        let mut b = a;
        b[0] = F::ONE;
        b
    };
    FlatVec::new(vec![a, b]).unwrap_or_else(|_| unreachable!("We added two items."))
}

/// Creates a tree from the data.
fn create_cluster_pair<I, T: Number, M: Metric<I, T>, D: Dataset<I>>(data: &D, metric: &M) -> [Ball<T>; 2] {
    let criteria = |_: &Ball<T>| true;
    let mut root = Ball::new_tree_iterative(data, metric, &criteria, None, 128);
    let children = root.take_children();
    let [a, b] = [children[0].clone(), children[1].clone()];
    [*a, *b]
}

/// Creates an `Arena` with two `Mass`es.
fn create_arena<'a, T: Number, C: Cluster<T>, F: Float, const DIM: usize>(
    a: &'a C,
    b: &'a C,
    l: F,
) -> (Arena<Mass<'a, T, C, F, DIM>>, [Index; 2]) {
    let mut arena = Arena::new();

    let ax = Vector::new([F::ZERO; DIM]);
    let bx = {
        let mut bx = ax;
        bx[0] = l;
        bx
    };
    let v = Vector::zero();
    let a = Mass::new(a, ax, v);
    let b = Mass::new(b, bx, v);
    let a = arena.insert(a);
    let b = arena.insert(b);
    (arena, [a, b])
}
