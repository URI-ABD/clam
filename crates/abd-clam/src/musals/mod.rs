//! Multiple Sequence Alignment At Scale (`MuSAlS`) with CLAM.

use crate::{DistanceValue, Tree};

mod alignment;
pub mod quality;
mod sequence;
mod tree;

pub use alignment::{CostMatrix, Direction, Edit, Edits, Substitutions};
pub use quality::{MeasurableAlignmentQuality, SumOfPairs};
pub use sequence::AlignedSequence;

/// A tree of aligned sequences, constructed using the [`Tree::align`] and [`Tree::par_align`] methods.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "pancakes", derive(deepsize::DeepSizeOf))]
pub struct MusalsTree<Id, T, A, M> {
    /// The inner tree.
    tree: Tree<Id, AlignedSequence, T, A, M>,
    /// The cost matrix used for alignment.
    cost_matrix: CostMatrix<T>,
}

impl<Id, T, A, M> core::ops::Deref for MusalsTree<Id, T, A, M> {
    type Target = Tree<Id, AlignedSequence, T, A, M>;

    fn deref(&self) -> &Self::Target {
        &self.tree
    }
}

impl<Id, T, A, M> core::ops::DerefMut for MusalsTree<Id, T, A, M> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.tree
    }
}

impl<Id, T, A, M> MusalsTree<Id, T, A, M> {
    /// Returns a reference to the cost matrix used for alignment.
    pub const fn cost_matrix(&self) -> &CostMatrix<T> {
        &self.cost_matrix
    }

    /// Takes ownership of the inner tree and the cost matrix, returning them as a tuple.
    pub fn into_inner(self) -> (Tree<Id, AlignedSequence, T, A, M>, CostMatrix<T>) {
        (self.tree, self.cost_matrix)
    }

    /// Recovers and returns the tree of unaligned sequences.
    ///
    /// This method transforms the tree of aligned sequences back into a tree of unaligned sequences by extracting the original sequences from the aligned ones.
    /// The provided `metric` is the distance metric to be used for the unaligned sequences in the resulting tree.
    pub fn into_unaligned<NewM>(self, metric: NewM) -> (Tree<Id, String, T, A, NewM>, CostMatrix<T>)
    where
        NewM: Fn(&String, &String) -> T,
    {
        let Self { tree, cost_matrix } = self;
        let items = tree.items.into_iter().map(|(id, aligned_seq, loc)| (id, aligned_seq.original(), loc)).collect();
        let tree = Tree { items, metric };
        (tree, cost_matrix)
    }

    /// Changes the distance metric of the tree to the provided `metric`, returning a new `MusalsTree` with the same aligned sequences and cost matrix but the
    /// new metric.
    pub fn with_metric<NewM>(self, metric: NewM) -> MusalsTree<Id, T, A, NewM> {
        let Self { tree, cost_matrix } = self;
        let Tree { items, .. } = tree;
        let tree = Tree { items, metric };
        MusalsTree { tree, cost_matrix }
    }
}

impl<Id, T, A, M> Tree<Id, String, T, A, M>
where
    T: DistanceValue,
    M: Fn(&String, &String) -> T,
{
    /// Aligns the sequences in the tree using `MuSAlS`.
    pub fn align<NewM>(self, cost_matrix: CostMatrix<T>, metric: NewM) -> MusalsTree<Id, T, A, NewM>
    where
        NewM: Fn(&AlignedSequence, &AlignedSequence) -> T,
    {
        let items = self.items.into_iter().map(|(id, seq, loc)| (id, AlignedSequence::from(seq), loc)).collect();
        let mut tree = Tree { items, metric: self.metric };
        tree.align_bottom_up(&cost_matrix);
        let tree = tree.with_metric(metric);
        MusalsTree { tree, cost_matrix }
    }
}

impl<Id, T, A, M> Tree<Id, String, T, A, M>
where
    Id: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Fn(&String, &String) -> T + Send + Sync,
{
    /// Parallel version of [`align`](Self::align).
    pub fn par_align<NewM>(self, cost_matrix: CostMatrix<T>, metric: NewM) -> MusalsTree<Id, T, A, NewM>
    where
        NewM: Fn(&AlignedSequence, &AlignedSequence) -> T + Send + Sync,
    {
        let items = self.items.into_iter().map(|(id, seq, loc)| (id, AlignedSequence::from(seq), loc)).collect();
        let mut tree = Tree { items, metric: self.metric };
        tree.par_align_bottom_up(&cost_matrix);
        let tree = tree.with_metric(metric);
        MusalsTree { tree, cost_matrix }
    }
}

#[cfg(feature = "serde")]
impl<Id, T, A, M> serde::Serialize for MusalsTree<Id, T, A, M>
where
    Id: serde::Serialize,
    T: serde::Serialize,
    A: serde::Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        (&self.tree, &self.cost_matrix).serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, Id, T, A> serde::Deserialize<'de> for MusalsTree<Id, T, A, ()>
where
    Id: serde::Deserialize<'de>,
    T: serde::Deserialize<'de>,
    A: serde::Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let (tree, cost_matrix) = <(Tree<Id, AlignedSequence, T, A, ()>, CostMatrix<T>)>::deserialize(deserializer)?;
        Ok(Self { tree, cost_matrix })
    }
}

#[cfg(feature = "serde")]
impl<Id, T, A, M> databuf::Encode for MusalsTree<Id, T, A, M>
where
    Id: databuf::Encode,
    T: databuf::Encode,
    A: databuf::Encode,
{
    fn encode<const CONFIG: u16>(&self, buffer: &mut (impl std::io::Write + ?Sized)) -> std::io::Result<()> {
        self.tree.encode::<CONFIG>(buffer)?;
        self.cost_matrix.encode::<CONFIG>(buffer)
    }
}

#[cfg(feature = "serde")]
impl<'de, Id, T, A> databuf::Decode<'de> for MusalsTree<Id, T, A, ()>
where
    Id: databuf::Decode<'de>,
    T: databuf::Decode<'de>,
    A: databuf::Decode<'de>,
{
    fn decode<const CONFIG: u16>(buffer: &mut &'de [u8]) -> databuf::Result<Self> {
        let tree = Tree::decode::<CONFIG>(buffer)?;
        let cost_matrix = CostMatrix::decode::<CONFIG>(buffer)?;
        Ok(Self { tree, cost_matrix })
    }
}
