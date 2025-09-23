//! Traits and types for `Cluster`s that have been compressed and can be used
//! for decompression.

use rayon::prelude::*;

use crate::{pancakes::squishy_ball::SquishyBall, Cluster, Dataset, DistanceValue, ParCluster, ParDataset, Permutable};

use super::{CodecContents, CodecItem, Decoder, Encoder, ParDecoder, ParEncoder};

/// A `Cluster` that has been compressed using `PanCAKES`.
#[must_use]
pub struct SquishedBall<I, T: DistanceValue, Enc: Encoder<I, Dec>, Dec: Decoder<I, Enc>> {
    /// The depth of this `SquishedBall` in the tree.
    depth: usize,
    /// The offset of this `SquishedBall` in the dataset.
    offset: usize,
    /// The number of items in this `SquishedBall`.
    cardinality: usize,
    /// The radius of this `SquishedBall`.
    radius: T,
    /// The local fractal dimension of this `SquishedBall`.
    lfd: f32,
    /// The index of the item that is the center of this `SquishedBall`.
    arg_center: usize,
    /// The index of the item that is farthest from the center.
    arg_radial: usize,
    /// The center of this `SquishedBall`, either encoded or decoded.
    pub(crate) center: CodecItem<I, Enc, Dec>,
    /// The contents of this `SquishedBall`, either child `SquishedBall`s or
    /// encoded items.
    pub(crate) contents: CodecContents<I, T, Enc, Dec>,
}

impl<I, T, Enc, Dec> Clone for SquishedBall<I, T, Enc, Dec>
where
    I: Clone,
    T: DistanceValue + Clone,
    Enc: Encoder<I, Dec>,
    Dec: Decoder<I, Enc>,
    Enc::Bytes: Clone,
{
    fn clone(&self) -> Self {
        Self {
            depth: self.depth,
            offset: self.offset,
            cardinality: self.cardinality,
            radius: self.radius,
            lfd: self.lfd,
            arg_center: self.arg_center,
            arg_radial: self.arg_radial,
            center: self.center.clone(),
            contents: self.contents.clone(),
        }
    }
}

impl<I, T, Enc, Dec> SquishedBall<I, T, Enc, Dec>
where
    T: DistanceValue,
    Enc: Encoder<I, Dec>,
    Dec: Decoder<I, Enc>,
{
    /// Create a new `SquishedBall`.
    pub fn from_cluster_tree<D, S, M>(
        root: S,
        data: &mut D,
        metric: &M,
        encoder: &Enc,
        trim_min_depth: usize,
    ) -> (Self, Vec<usize>)
    where
        I: Clone,
        D: Permutable<I>,
        S: Cluster<T>,
        M: Fn(&I, &I) -> T,
    {
        let (root, permutation) = SquishyBall::from_cluster_tree(root, data, metric, encoder);
        let root = Self::adapt_tree_recursive(root.trim(trim_min_depth), data, encoder);
        (root, permutation)
    }

    /// Recursive helper function for [`from_cluster_tree`](Self::from_cluster_tree).
    fn adapt_tree_recursive<D, S>(mut source: SquishyBall<T, S>, data: &D, encoder: &Enc) -> Self
    where
        I: Clone,
        D: Dataset<I>,
        S: Cluster<T>,
    {
        let depth = source.depth();
        let offset = source.offset();
        let cardinality = source.cardinality();
        let radius = source.radius();
        let lfd = source.lfd();
        let arg_center = source.arg_center();
        let arg_radial = source.arg_radial();

        let center = data.get(source.arg_center());

        let contents = if source.is_leaf() {
            CodecContents::Leaf(
                source
                    .indices()
                    .into_iter()
                    .map(|i| data.get(i))
                    .map(|item| encoder.encode(item, center))
                    .map(CodecItem::new_encoded)
                    .collect(),
            )
        } else {
            CodecContents::Recursive(
                source
                    .take_children()
                    .into_iter()
                    .map(|child| Self::adapt_tree_recursive(*child, data, encoder))
                    .map(|mut child| {
                        child.center = child.center.encode(encoder, center);
                        child
                    })
                    .map(Box::new)
                    .collect(),
            )
        };

        Self {
            depth,
            offset,
            cardinality,
            radius,
            lfd,
            arg_center,
            arg_radial,
            center: CodecItem::new_decoded(center.clone()),
            contents,
        }
    }

    /// Decode all items in the tree, keeping the tree structure intact.
    pub fn decode_tree(&mut self, decoder: &Dec) {
        match &self.center {
            CodecItem::Encoded(encoded) => {
                let center = decoder.decode_raw(encoded);
                self.contents.decode_subtree(decoder, &center);
                self.center = CodecItem::new_decoded(center);
            }
            CodecItem::Decoded(center) => {
                self.contents.decode_subtree(decoder, center);
            }
        }
    }

    /// Decodes and returns all items in the tree.
    pub fn decode_all(self, decoder: &Dec) -> Vec<I> {
        match &self.center {
            CodecItem::Encoded(encoded) => {
                let center = decoder.decode_raw(encoded);
                self.contents.decode_all(decoder, &center)
            }
            CodecItem::Decoded(center) => self.contents.decode_all(decoder, center),
        }
    }
}

impl<I, T, Enc, Dec> SquishedBall<I, T, Enc, Dec>
where
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    Enc: ParEncoder<I, Dec>,
    Dec: ParDecoder<I, Enc>,
    Enc::Bytes: Send + Sync,
{
    /// Parallel version of [`from_cluster_tree`](SquishedBall::from_cluster_tree).
    pub fn par_from_cluster_tree<D, S, M>(
        root: S,
        data: &mut D,
        metric: &M,
        encoder: &Enc,
        trim_min_depth: usize,
    ) -> (Self, Vec<usize>)
    where
        I: Clone,
        D: ParDataset<I> + Permutable<I>,
        S: ParCluster<T>,
        M: (Fn(&I, &I) -> T) + Send + Sync,
    {
        let (root, permutation) = SquishyBall::par_from_cluster_tree(root, data, metric, encoder);
        let root = Self::par_adapt_tree_recursive(root.par_trim(trim_min_depth), data, encoder);
        (root, permutation)
    }

    /// Parallel version of [`adapt_tree_recursive`](SquishedBall::adapt_tree_recursive).
    fn par_adapt_tree_recursive<D, S>(mut source: SquishyBall<T, S>, data: &D, encoder: &Enc) -> Self
    where
        I: Clone,
        D: ParDataset<I>,
        S: ParCluster<T>,
    {
        let depth = source.depth();
        let offset = source.offset();
        let cardinality = source.cardinality();
        let radius = source.radius();
        let lfd = source.lfd();
        let arg_center = source.arg_center();
        let arg_radial = source.arg_radial();

        let center = data.get(source.arg_center());

        let contents = if source.is_leaf() {
            CodecContents::Leaf(
                source
                    .indices()
                    .into_par_iter()
                    .map(|i| data.get(i))
                    .map(|item| encoder.encode(item, center))
                    .map(CodecItem::new_encoded)
                    .collect(),
            )
        } else {
            CodecContents::Recursive(
                source
                    .take_children()
                    .into_par_iter()
                    .map(|child| Self::par_adapt_tree_recursive(*child, data, encoder))
                    .map(|mut child| {
                        child.center = child.center.par_encode(encoder, center);
                        child
                    })
                    .map(Box::new)
                    .collect(),
            )
        };

        Self {
            depth,
            offset,
            cardinality,
            radius,
            lfd,
            arg_center,
            arg_radial,
            center: CodecItem::new_decoded(center.clone()),
            contents,
        }
    }

    /// Parallel version of [`decode_tree`](Self::decode_tree).
    pub fn par_decode_tree(&mut self, decoder: &Dec) {
        match &self.center {
            CodecItem::Encoded(encoded) => {
                let center = decoder.par_decode_raw(encoded);
                self.contents.par_decode_subtree(decoder, &center);
                self.center = CodecItem::new_decoded(center);
            }
            CodecItem::Decoded(center) => {
                self.contents.par_decode_subtree(decoder, center);
            }
        }
    }

    /// Parallel version of [`decode_all`](Self::decode_all).
    pub fn par_decode_all(self, decoder: &Dec) -> Vec<I> {
        match &self.center {
            CodecItem::Encoded(encoded) => {
                let center = decoder.par_decode_raw(encoded);
                self.contents.par_decode_all(decoder, &center)
            }
            CodecItem::Decoded(center) => self.contents.par_decode_all(decoder, center),
        }
    }
}

impl<I, T, Enc, Dec> std::fmt::Debug for SquishedBall<I, T, Enc, Dec>
where
    T: DistanceValue + std::fmt::Debug,
    Enc: Encoder<I, Dec>,
    Dec: Decoder<I, Enc>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SquishedBall")
            .field("depth", &self.depth)
            .field("offset", &self.offset)
            .field("cardinality", &self.cardinality)
            .field("radius", &self.radius)
            .field("lfd", &self.lfd)
            .field("arg_center", &self.arg_center)
            .field("arg_radial", &self.arg_radial)
            .field(
                "center_state",
                &format_args!(
                    "{}",
                    match &self.center {
                        CodecItem::Decoded(_) => "Decoded",
                        CodecItem::Encoded(_) => "Encoded",
                    }
                ),
            )
            .field(
                "contents",
                &format_args!(
                    "{}",
                    match &self.contents {
                        CodecContents::Recursive(children) => format!("Recursive({} children)", children.len()),
                        CodecContents::Leaf(items) => format!("Leaf({} items)", items.len()),
                    }
                ),
            )
            .finish()
    }
}

impl<I, T, Enc, Dec> PartialEq for SquishedBall<I, T, Enc, Dec>
where
    T: DistanceValue,
    Enc: Encoder<I, Dec>,
    Dec: Decoder<I, Enc>,
{
    fn eq(&self, other: &Self) -> bool {
        self.offset == other.offset && self.cardinality == other.cardinality
    }
}

impl<I, T, Enc, Dec> Eq for SquishedBall<I, T, Enc, Dec>
where
    T: DistanceValue,
    Enc: Encoder<I, Dec>,
    Dec: Decoder<I, Enc>,
{
}

impl<I, T, Enc, Dec> PartialOrd for SquishedBall<I, T, Enc, Dec>
where
    T: DistanceValue,
    Enc: Encoder<I, Dec>,
    Dec: Decoder<I, Enc>,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<I, T, Enc, Dec> Ord for SquishedBall<I, T, Enc, Dec>
where
    T: DistanceValue,
    Enc: Encoder<I, Dec>,
    Dec: Decoder<I, Enc>,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.offset
            .cmp(&other.offset)
            .then_with(|| other.cardinality.cmp(&self.cardinality))
    }
}

impl<I, T, Enc, Dec> Cluster<T> for SquishedBall<I, T, Enc, Dec>
where
    T: DistanceValue,
    Enc: Encoder<I, Dec>,
    Dec: Decoder<I, Enc>,
{
    fn depth(&self) -> usize {
        self.depth
    }

    fn cardinality(&self) -> usize {
        self.cardinality
    }

    fn arg_center(&self) -> usize {
        self.arg_center
    }

    fn set_arg_center(&mut self, arg_center: usize) {
        self.arg_center = arg_center;
    }

    fn radius(&self) -> T {
        self.radius
    }

    fn arg_radial(&self) -> usize {
        self.arg_radial
    }

    fn set_arg_radial(&mut self, arg_radial: usize) {
        self.arg_radial = arg_radial;
    }

    fn lfd(&self) -> f32 {
        self.lfd
    }

    fn contains(&self, idx: usize) -> bool {
        (self.offset..self.offset + self.cardinality).contains(&idx)
    }

    fn indices(&self) -> Vec<usize> {
        (self.offset..self.offset + self.cardinality).collect()
    }

    fn set_indices(&mut self, _: &[usize]) {}

    fn take_indices(&mut self) -> Vec<usize> {
        self.indices()
    }

    fn children(&self) -> Vec<&Self> {
        match &self.contents {
            CodecContents::Leaf(_) => vec![],
            CodecContents::Recursive(children) => children.iter().map(AsRef::as_ref).collect(),
        }
    }

    fn children_mut(&mut self) -> Vec<&mut Self> {
        match &mut self.contents {
            CodecContents::Leaf(_) => vec![],
            CodecContents::Recursive(children) => children.iter_mut().map(AsMut::as_mut).collect(),
        }
    }

    fn set_children(&mut self, children: Vec<Box<Self>>) {
        self.contents = CodecContents::Recursive(children);
    }

    fn take_children(&mut self) -> Vec<Box<Self>> {
        match &mut self.contents {
            CodecContents::Leaf(_) => vec![],
            CodecContents::Recursive(children) => core::mem::take(children),
        }
    }

    fn is_descendant_of(&self, other: &Self) -> bool {
        other.contains(self.offset) && self.cardinality < other.cardinality
    }
}

impl<I, T, Enc, Dec> ParCluster<T> for SquishedBall<I, T, Enc, Dec>
where
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    Enc: ParEncoder<I, Dec>,
    Enc::Bytes: Send + Sync,
    Dec: ParDecoder<I, Enc>,
{
    fn par_indices(&self) -> impl ParallelIterator<Item = usize> {
        (self.offset..self.offset + self.cardinality).into_par_iter()
    }
}
