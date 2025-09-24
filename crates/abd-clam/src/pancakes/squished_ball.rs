//! Traits and types for `Cluster`s that have been compressed and can be used
//! for decompression.

use rayon::prelude::*;

use crate::{Cluster, Dataset, DatasetMut, DistanceValue, ParCluster};

use super::{CodecContents, CodecItem, Decoder, Encoder, SquishyBall};

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

impl<I, T, Enc, Dec> SquishedBall<I, T, Enc, Dec>
where
    T: DistanceValue,
    Enc: Encoder<I, Dec>,
    Dec: Decoder<I, Enc>,
{
    /// Create a new `SquishedBall`.
    pub fn from_cluster_tree<D, S>(root: S, data: &mut D, encoder: &Enc, trim_min_depth: usize) -> (Self, Vec<usize>)
    where
        D: DatasetMut<I>,
        S: Cluster<T>,
    {
        let (root, permutation) = SquishyBall::from_cluster_tree(root, data, encoder);
        let root = Self::from_squishy_ball(root.trim(trim_min_depth), data, encoder);
        (root, permutation)
    }

    /// Create a new `SquishedBall` from a `SquishyBall`.
    ///
    /// This assumes that the `SquishyBall` has already been trimmed to the
    /// desired depth.
    pub fn from_squishy_ball<D, S>(root: SquishyBall<T, S>, data: &D, encoder: &Enc) -> Self
    where
        D: Dataset<I>,
        S: Cluster<T>,
    {
        // SAFETY: We immediately replace the placeholder center value with the
        // correctly encoded center value after the recursion completes.
        #[allow(unsafe_code)]
        unsafe {
            let (mut root, center) = Self::adapt_tree_recursive(root, data, encoder);
            root.center = CodecItem::Encoded(encoder.encode_raw(center));
            root
        }
    }

    /// Recursive helper function for [`from_squishy_ball`](Self::from_squishy_ball).
    ///
    /// SAFETY: The returned `SquishedBall` has its center set to a placeholder
    /// value that must be replaced by the caller before use. The function
    /// returns a reference to the decoded center item that the caller can use
    /// for this purpose.
    ///
    /// This invariant ensures that all child nodes can encode their contents in
    /// terms of their own _decoded_ center. After recursion completes, the
    /// caller can then encode the child's center in terms of the parent's
    /// _decoded_ center.
    ///
    /// This invariant also allows us to avoid requiring `I: Clone`, which could
    /// be expensive for large item types, or `Enc::Bytes: Default`, which would
    /// likely be non-sensical for many encoders.
    ///
    /// # Returns
    ///
    /// The adapted `SquishedBall` and a reference to the decoded center item
    /// for that `SquishedBall`.
    #[allow(unsafe_code)]
    unsafe fn adapt_tree_recursive<'a, D, S>(mut source: SquishyBall<T, S>, data: &'a D, encoder: &Enc) -> (Self, &'a I)
    where
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
                    .map(CodecItem::Encoded)
                    .collect(),
            )
        } else {
            CodecContents::Recursive(
                source
                    .take_children()
                    .into_iter()
                    .map(|child| {
                        // SAFETY: The placeholder value is replaced immediately
                        // after this call returns.
                        #[allow(unsafe_code)]
                        unsafe {
                            let (mut child, child_center) = Self::adapt_tree_recursive(*child, data, encoder);
                            child.center = CodecItem::Encoded(encoder.encode(child_center, center));
                            child
                        }
                    })
                    .map(Box::new)
                    .collect(),
            )
        };

        let root = Self {
            depth,
            offset,
            cardinality,
            radius,
            lfd,
            arg_center,
            arg_radial,
            // SAFETY: This is a placeholder value that will be replaced by
            // the caller.
            center: CodecItem::Encoded(unsafe { core::mem::zeroed() }),
            contents,
        };

        (root, center)
    }

    /// Encode all items in the tree, keeping the tree structure intact.
    pub fn encode_from_root(&mut self, encoder: &Enc) {
        match &self.center {
            CodecItem::Encoded(_) => (), // Already encoded, nothing to do.
            CodecItem::Decoded(center) => {
                self.contents.encode_subtree(encoder, center);
                self.center = CodecItem::Encoded(encoder.encode_raw(center));
            }
        }
    }

    /// Encode the subtree rooted at this node, using the provided encoder and
    /// the center of the parent node as the reference item.
    pub(crate) fn encode_subtree(&mut self, encoder: &Enc, parent_center: &I) {
        // INVARIANT: The center is encoded only after all its children have
        // been encoded.
        match &self.center {
            CodecItem::Encoded(_) => (), // Already encoded, nothing to do.
            CodecItem::Decoded(center) => {
                self.contents.encode_subtree(encoder, center);
                self.center = CodecItem::Encoded(encoder.encode(center, parent_center));
            }
        }
    }

    /// Decode all items in the tree, keeping the tree structure intact.
    pub fn decode_from_root(&mut self, decoder: &Dec) {
        match &self.center {
            CodecItem::Encoded(encoded) => {
                let center = decoder.decode_raw(encoded);
                self.contents.decode_subtree(decoder, &center);
                self.center = CodecItem::Decoded(center);
            }
            CodecItem::Decoded(center) => {
                self.contents.decode_subtree(decoder, center);
            }
        }
    }

    /// Decode the subtree rooted at this node, using the provided decoder and
    /// the center of the parent node as the reference item.
    pub(crate) fn decode_subtree(&mut self, decoder: &Dec, parent_center: &I) {
        match &self.center {
            CodecItem::Encoded(encoded) => {
                let center = decoder.decode(parent_center, encoded);
                self.contents.decode_subtree(decoder, &center);
                self.center = CodecItem::Decoded(center);
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
    Enc: super::ParEncoder<I, Dec>,
    Enc::Output: Send + Sync,
    Dec: super::ParDecoder<I, Enc>,
{
    fn par_indices(&self) -> impl ParallelIterator<Item = usize> {
        (self.offset..self.offset + self.cardinality).into_par_iter()
    }
}
