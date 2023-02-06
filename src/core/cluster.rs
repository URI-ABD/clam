use std::f64::consts::SQRT_2;

use bitvec::prelude::*;

use super::partition_criteria::PartitionCriteria;
use crate::geometry::tetrahedron::*;
use crate::geometry::triangle::*;
use crate::prelude::*;
use crate::utils::helpers;

pub type Ratios = [f64; 6];
pub type History = BitVec;

/// Some values to be stored at build time that will be useful for partition.
type BuildCache<'a> = (Vec<f64>, Vec<f64>);

#[derive(Debug)]
enum ClusterVariant {
    Singleton([usize; 1]),                  // [a]
    Dipole(f64, f64, f64, [usize; 2]),      // diameter, radius, lfd, [a, b]
    Trigon(f64, f64, [usize; 3], Triangle), // radius, lfd, [a, b, c], abc
}

impl ClusterVariant {
    fn radius(&self) -> f64 {
        match self {
            ClusterVariant::Singleton(_) => 0.,
            ClusterVariant::Dipole(_, r, ..) => *r,
            ClusterVariant::Trigon(r, ..) => *r,
        }
    }

    fn lfd(&self) -> f64 {
        match self {
            ClusterVariant::Singleton(_) => 1.,
            ClusterVariant::Dipole(.., l, _) => *l,
            ClusterVariant::Trigon(_, l, ..) => *l,
        }
    }

    fn extrema(&self) -> Vec<usize> {
        match self {
            ClusterVariant::Singleton([a]) => vec![*a],
            ClusterVariant::Dipole(.., [a, b]) => vec![*a, *b],
            ClusterVariant::Trigon(.., [a, b, c], _) => vec![*a, *b, *c],
        }
    }

    fn name(&self) -> &str {
        match self {
            ClusterVariant::Singleton(_) => "Singleton",
            ClusterVariant::Dipole(..) => "Dipole",
            ClusterVariant::Trigon(..) => "Trigon",
        }
    }
}

#[derive(Debug)]
pub enum ClusterContents<'a, T, S>
where
    T: Number + 'a,
    S: Space<'a, T> + 'a,
{
    Indices(Vec<usize>),
    Children([Box<Cluster<'a, T, S>>; 2]),
}

#[derive(Debug)]
pub struct Cluster<'a, T, S>
where
    T: Number + 'a,
    S: Space<'a, T> + 'a,
{
    space: &'a S,
    history: History,
    cardinality: usize,
    variant: ClusterVariant,
    contents: ClusterContents<'a, T, S>,
    ratios: Option<Ratios>,
    t: std::marker::PhantomData<T>,
    naive_radius: f64,
    scaled_radius: f64,
    build_cache: Option<BuildCache<'a>>,
}

impl<'a, T, S> Cluster<'a, T, S>
where
    T: Number + 'a,
    S: Space<'a, T> + 'a,
{
    pub fn new_root(space: &'a S) -> Self {
        let indices = space.data().indices();
        assert!(!indices.is_empty(), "`space.data().indices()` must not be empty.");
        Self::new(space, bitvec::bitvec![1], indices)
    }

    #[inline(always)]
    fn new_singleton(space: &'a S, history: History, indices: Vec<usize>) -> Self {
        Self {
            space,
            history,
            cardinality: indices.len(),
            variant: ClusterVariant::Singleton([indices[0]]),
            contents: ClusterContents::Indices(indices),
            ratios: None,
            t: Default::default(),
            naive_radius: 0.,
            scaled_radius: 0.,
            build_cache: None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn new_dipole(
        space: &'a S,
        history: History,
        indices: Vec<usize>,
        diameter: f64,
        lfd: f64,
        [a, b]: [usize; 2],
        naive_radius: f64,
        scaled_radius: f64,
        build_cache: Option<BuildCache>,
    ) -> Self {
        let radius = diameter / 2.;
        assert!(radius <= naive_radius, "radii: {radius:.12} vs {naive_radius:.12}");
        Self {
            space,
            history,
            cardinality: indices.len(),
            variant: ClusterVariant::Dipole(diameter, diameter / 2., lfd, [a, b]),
            contents: ClusterContents::Indices(indices),
            ratios: None,
            t: Default::default(),
            naive_radius,
            scaled_radius,
            build_cache,
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn new_trigon(
        space: &'a S,
        history: History,
        indices: Vec<usize>,
        radius: f64,
        lfd: f64,
        [a, b, c]: [usize; 3],
        abc: Triangle,
        naive_radius: f64,
        scaled_radius: f64,
        build_cache: Option<BuildCache>,
    ) -> Self {
        // assert!(radius <= naive_radius, "radii: {radius:.12} vs {naive_radius:.12}");
        Self {
            space,
            history,
            cardinality: indices.len(),
            variant: ClusterVariant::Trigon(radius, lfd, [a, b, c], abc),
            contents: ClusterContents::Indices(indices),
            ratios: None,
            t: Default::default(),
            naive_radius,
            scaled_radius,
            build_cache,
        }
    }

    fn new(space: &'a S, history: History, indices: Vec<usize>) -> Self {
        match indices.len() {
            0 => panic!("Not allowed with no indices! {history:?}"),
            1 => Self::new_singleton(space, history, indices),
            2 => {
                let [a, b] = [indices[0], indices[1]];
                let diameter = space.one_to_one(a, b);
                Self::new_dipole(
                    space,
                    history,
                    vec![a, b],
                    diameter,
                    1.,
                    [a, b],
                    diameter / 2.,
                    diameter / 2.,
                    Some((vec![0., diameter], vec![diameter, 0.])),
                )
            }
            _ => {
                // The geometric median
                let m = {
                    let arg_samples = if indices.len() < 100 {
                        indices.clone()
                    } else {
                        let n = ((indices.len() as f64).sqrt()) as usize;
                        space.choose_unique(n, &indices)
                    };
                    let sample_distances = space
                        .pairwise(&arg_samples)
                        .into_iter()
                        .map(|v| v.into_iter().sum::<f64>())
                        .collect::<Vec<_>>();
                    arg_samples[helpers::arg_min(&sample_distances).0]
                };

                // the instance farthest from `m`
                let (a, naive_radius) = {
                    let m_distances = space.one_to_many(m, &indices);
                    let (a, naive_radius) = helpers::arg_max(&m_distances);
                    (indices[a], naive_radius)
                };
                if naive_radius < EPSILON {
                    return Self::new_singleton(space, history, indices);
                }
                let a_distances = space.one_to_many(a, &indices);

                // the instance farthest from `a`
                let (b, ab) = {
                    let (b, ab) = helpers::arg_max(&a_distances);
                    (indices[b], ab)
                };
                let b_distances = space.one_to_many(b, &indices);

                // Make triangles to find `c`, the instance which makes the maximal cosine-angle with `a` and `b`.
                let triangles = indices
                    .iter()
                    .zip(a_distances.iter())
                    .zip(b_distances.iter())
                    .filter(|((&i, &ac), &bc)| {
                        i != a && i != b && ac > EPSILON && ab > EPSILON && makes_triangle([ab, ac, bc])
                    })
                    .map(|((&i, &ac), &bc)| (i, Triangle::with_edges_unchecked([ac, bc, ab])))
                    .collect::<Vec<_>>();

                if triangles.is_empty() {
                    // either there are only two unique instances or all instances are colinear
                    let radius = ab / 2.;
                    let radial_distances = a_distances.iter().map(|&d| (d - radius).abs()).collect::<Vec<_>>();
                    let lfd = helpers::get_lfd(radius, &radial_distances);
                    let build_cache = (a_distances, b_distances);
                    return Self::new_dipole(
                        space,
                        history,
                        indices,
                        ab,
                        lfd,
                        [a, b],
                        radius,
                        radius,
                        Some(build_cache),
                    );
                }

                // find `c` and the triangle `abc` which produced the maximal cosine
                let (c, cab) = triangles
                    .into_iter()
                    .max_by(|(_, l), (_, r)| l.cos_a().partial_cmp(&r.cos_a()).unwrap())
                    .unwrap();
                if cab.cos_a() <= 0. {
                    // No acute angle was possible so we have an ellipsoid-shaped Dipole. This should be rare.
                    let radius = ab / 2.;
                    let radial_distances = a_distances.iter().map(|&d| (d - radius).abs()).collect::<Vec<_>>();
                    let lfd = helpers::get_lfd(radius, &radial_distances);
                    let build_cache = (a_distances, b_distances);
                    return Self::new_dipole(
                        space,
                        history,
                        indices,
                        ab,
                        lfd,
                        [a, b],
                        radius,
                        radius,
                        Some(build_cache),
                    );
                }

                let [ac, bc, _] = cab.edge_lengths();
                let abc = Triangle::with_edges_unchecked([ab, ac, bc]);
                let triangle_radius = abc.r_sq().sqrt();
                let scaled_radius = triangle_radius * SQRT_2;
                let c_distances = space.one_to_many(c, &indices);

                // make tetrahedra to find the maximal radius for the cluster.
                let radial_distances = indices
                    .iter()
                    .zip(a_distances.iter())
                    .zip(b_distances.iter())
                    .zip(c_distances.iter())
                    .filter(|(((&i, &ad), &bd), &cd)| {
                        i != a
                            && i != b
                            && i != c
                            && ad > EPSILON
                            && bd > EPSILON
                            && cd > EPSILON
                            && makes_triangle([ab, ad, bd])
                            && makes_triangle([ac, ad, cd])
                            && makes_triangle([bc, bd, cd])
                    })
                    .map(|(((_, &ad), &bd), &cd)| {
                        Tetrahedron::with_edges_unchecked([ab, ac, bc, ad, bd, cd])
                            .od_sq()
                            .sqrt()
                    })
                    .collect::<Vec<_>>();

                if radial_distances.is_empty() {
                    let lfd = 2.; // TODO: have a think about this case
                    let build_cache = (a_distances, b_distances);
                    Self::new_trigon(
                        space,
                        history,
                        indices,
                        triangle_radius,
                        lfd,
                        [a, b, c],
                        abc,
                        naive_radius,
                        scaled_radius,
                        Some(build_cache),
                    )
                } else {
                    let radius = {
                        let radius = helpers::arg_max(&radial_distances).1;
                        if radius > triangle_radius {
                            radius
                        } else {
                            triangle_radius
                        }
                    };
                    let lfd = helpers::get_lfd(radius, &radial_distances);
                    let build_cache = (a_distances, b_distances);
                    Self::new_trigon(
                        space,
                        history,
                        indices,
                        radius,
                        lfd,
                        [a, b, c],
                        abc,
                        naive_radius,
                        scaled_radius,
                        Some(build_cache),
                    )
                }
            }
        }
    }

    fn partition_once(&self) -> [Self; 2] {
        let indices = match &self.contents {
            ClusterContents::Indices(indices) => indices,
            _ => panic!("Impossible!"),
        };

        let extrema = self.extrema();
        let [a, b] = [extrema[0], extrema[1]];

        let (a_distances, b_distances) = self.build_cache.as_ref().unwrap();
        let lefties = indices
            .iter()
            .zip(a_distances.iter())
            .zip(b_distances.iter())
            .filter(|((&i, _), _)| i != a && i != b)
            .map(|((_, &l), &r)| l <= r)
            .collect::<Vec<_>>();

        let mut right_indices = indices
            .iter()
            .filter(|&&i| i != a && i != b)
            .zip(lefties.iter())
            .filter(|(_, &b)| !b)
            .map(|(&i, _)| i)
            .collect::<Vec<_>>();
        right_indices.push(b);

        let mut left_indices = indices
            .iter()
            .filter(|&&i| i != a && i != b)
            .zip(lefties.iter())
            .filter(|(_, &b)| b)
            .map(|(&i, _)| i)
            .collect::<Vec<_>>();
        left_indices.push(a);

        let (left_indices, right_indices) = if left_indices.len() < right_indices.len() {
            (right_indices, left_indices)
        } else {
            (left_indices, right_indices)
        };

        let left_history = {
            let mut history = self.history.clone();
            history.push(false);
            history
        };
        let right_history = {
            let mut history = self.history.clone();
            history.push(true);
            history
        };

        let left = Self::new(self.space, left_history, left_indices);
        let right = Self::new(self.space, right_history, right_indices);

        [left, right]
    }

    pub fn partition(mut self, partition_criteria: &PartitionCriteria<'a, T, S>, recursive: bool) -> Self {
        if partition_criteria.check(&self) {
            let [left, right] = self.partition_once();

            let (left, right) = if recursive {
                (
                    left.partition(partition_criteria, recursive),
                    right.partition(partition_criteria, recursive),
                )
                // rayon::join(
                //     || left.partition(partition_criteria, recursive),
                //     || right.partition(partition_criteria, recursive),
                // )
            } else {
                (left, right)
            };
            self.contents = ClusterContents::Children([Box::new(left), Box::new(right)]);
        }
        self.build_cache = None;
        self
    }

    #[allow(unused_mut, unused_variables)]
    pub fn with_ratios(mut self, normalized: bool) -> Self {
        todo!()
    }

    pub fn space(&self) -> &'a S {
        self.space
    }

    pub fn cardinality(&self) -> usize {
        self.cardinality
    }

    pub fn indices(&self) -> Vec<usize> {
        match &self.contents {
            ClusterContents::Indices(indices) => indices.clone(),
            ClusterContents::Children([left, right]) => {
                left.indices().into_iter().chain(right.indices().into_iter()).collect()
            }
        }
    }

    pub fn history(&self) -> &History {
        &self.history
    }

    pub fn name(&self) -> String {
        let d = self.history().len();
        let padding = if d % 4 == 0 { 0 } else { 4 - d % 4 };
        let bin_name = (0..padding)
            .map(|_| "0")
            .chain(self.history.iter().map(|b| if *b { "1" } else { "0" }))
            .collect::<Vec<_>>();
        bin_name
            .chunks_exact(4)
            .map(|s| {
                let [a, b, c, d] = [s[0], s[1], s[2], s[3]];
                let s = format!("{a}{b}{c}{d}");
                let s = u8::from_str_radix(&s, 2).unwrap();
                format!("{s:01x}")
            })
            .collect::<Vec<_>>()
            .join("")
    }

    pub fn variant_name(&self) -> &str {
        self.variant.name()
    }

    pub fn depth(&self) -> usize {
        self.history.len() - 1
    }

    pub fn is_singleton(&self) -> bool {
        matches!(self.variant, ClusterVariant::Singleton(_))
    }

    pub fn is_leaf(&self) -> bool {
        matches!(self.contents, ClusterContents::Indices(_))
    }

    pub fn radius(&self) -> f64 {
        self.variant.radius()
    }

    pub fn naive_radius(&self) -> f64 {
        self.naive_radius
    }

    pub fn scaled_radius(&self) -> f64 {
        self.scaled_radius
    }

    pub fn lfd(&self) -> f64 {
        self.variant.lfd()
    }

    pub fn contents(&self) -> &ClusterContents<'a, T, S> {
        &self.contents
    }

    pub fn children(&self) -> [&Self; 2] {
        match &self.contents {
            ClusterContents::Indices(_) => panic!("Please don't do this to me!!!"),
            ClusterContents::Children([left, right]) => [left.as_ref(), right.as_ref()],
        }
    }

    pub fn ratios(&self) -> Ratios {
        self.ratios
            .expect("Please call `with_ratios` after `build` before using this method.")
    }

    pub fn extrema(&self) -> Vec<usize> {
        self.variant.extrema()
    }

    pub fn subtree(&self) -> Vec<&Self> {
        let subtree = vec![self];
        match &self.contents {
            ClusterContents::Indices(_) => subtree,
            ClusterContents::Children([left, right]) => subtree
                .into_iter()
                .chain(left.subtree().into_iter())
                .chain(right.subtree().into_iter())
                .collect(),
        }
    }

    pub fn num_descendants(&self) -> usize {
        self.subtree().len() - 1
    }

    pub fn max_leaf_depth(&self) -> usize {
        self.subtree().into_iter().map(|c| c.depth()).max().unwrap()
    }

    pub fn distance_to_indexed(&self, index: usize) -> f64 {
        self.distance_to_query(self.space.data().get(index))
    }

    pub fn distance_to_query(&self, query: &[T]) -> f64 {
        match &self.variant {
            ClusterVariant::Singleton([a]) => {
                // let center = self.space.data().get(self.indices()[0]);
                self.space.query_to_one(query, *a)
            }
            ClusterVariant::Dipole(diameter, radius, _, [a, b]) => {
                let distances = self.space.query_to_many(query, &[*a, *b]);
                let [ac, bc] = [distances[0], distances[1]];
                if ac < EPSILON || bc < EPSILON {
                    *radius
                } else {
                    let abc = Triangle::with_edges_unchecked([*diameter, ac, bc]);
                    abc.cm_sq().sqrt()
                }
            }
            ClusterVariant::Trigon(.., [a, b, c], abc) => {
                let distances = self.space.query_to_many(query, &[*a, *b, *c]);
                let [ab, ac, bc] = abc.edge_lengths();
                let [ad, bd, cd] = [distances[0], distances[1], distances[2]];
                let distance =
                    if let Ok(mut abcd) = Tetrahedron::with_edges(['a', 'b', 'c', 'd'], [ab, ac, bc, ad, bd, cd]) {
                        abcd.od_sq()
                    } else {
                        abc.r_sq()
                    };
                distance.sqrt()
            }
        }
    }

    pub fn distance_to_other(&self, other: &Self) -> f64 {
        match &self.variant {
            ClusterVariant::Singleton([a]) => {
                let center = self.space.data().get(*a);
                other.distance_to_query(center)
            }
            ClusterVariant::Dipole(diameter, radius, _, [a, b]) => {
                let [ac, bc] = [other.distance_to_indexed(*a), other.distance_to_indexed(*b)];
                if ac < EPSILON || bc < EPSILON {
                    *radius
                } else {
                    Triangle::with_edges_unchecked([*diameter, ac, bc]).cm_sq().sqrt()
                }
            }
            ClusterVariant::Trigon(.., [a, b, c], abc) => {
                let [ad, bd, cd] = [
                    other.distance_to_indexed(*a),
                    other.distance_to_indexed(*b),
                    other.distance_to_indexed(*c),
                ];
                let distance_sq = if ad < EPSILON || bd < EPSILON || cd < EPSILON {
                    abc.r_sq()
                } else {
                    Tetrahedron::with_triangle_unchecked(abc.clone(), [ad, bd, cd]).od_sq()
                };
                distance_sq.sqrt()
            }
        }
    }
}
