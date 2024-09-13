//! Writing `Cluster` trees to CSV files.

use std::io::Write;

use distances::Number;

use crate::Dataset;

use super::Cluster;

/// Write a `Cluster` to a CSV file.
#[allow(clippy::module_name_repetitions)]
pub trait WriteCsv<I, U, D, const N: usize>: Cluster<I, U, D>
where
    U: Number,
    D: Dataset<I, U>,
{
    /// Returns the names of the columns in the CSV file.
    fn header(&self) -> [String; N];

    /// Returns a row, corresponding to the `Cluster`, for the CSV file.
    fn row(&self) -> [String; N];

    /// Write to a CSV file, all the clusters in the tree.
    ///
    /// # Errors
    ///
    /// - If the file cannot be created.
    /// - If the file cannot be written to.
    /// - If the header cannot be written to the file.
    /// - If any row cannot be written to the file.
    fn write_to_csv<P: AsRef<std::path::Path>>(&self, path: &P) -> Result<(), String> {
        let line = |items: [String; N]| {
            let mut line = items.join(",");
            line.push('\n');
            line
        };

        // Create the file and write the header.
        let mut file = std::fs::File::create(path).map_err(|e| e.to_string())?;
        file.write_all(line(self.header()).as_bytes())
            .map_err(|e| e.to_string())?;

        // Write each row to the file.
        for row in self.subtree().into_iter().map(Self::row).map(line) {
            file.write_all(row.as_bytes()).map_err(|e| e.to_string())?;
        }

        Ok(())
    }
}

// /// Returns the subtree, with unique integers to reference the parents and children of each `Cluster` in a `HashMap`.
// ///
// /// The Vec contains tuples of:
// ///
// /// - The `Cluster` itself.
// /// - The index of the `Cluster` in the Vec.
// /// - The position of the `Cluster` among its siblings.
// /// - A Vec of tuples of:
// ///  - The index of the parent `Cluster` in the Vec.
// ///  - A tuple of:
// ///     - The index of the parent `Cluster` in the Vec.
// ///    - The extent of the child.
// #[allow(clippy::type_complexity)]
// fn take_subtree(mut self) -> Vec<(Self, usize, usize, Vec<(usize, (usize, U))>)> {
//     let children = self.take_children();
//     let mut clusters = vec![(self, 0, 0, vec![])];

//     for (e, d, children) in children.into_iter().map(|(e, d, c)| (e, d, c.take_subtree())) {
//         let offset = clusters.len();

//         for (ci, (child, parent_index, _, children_indices)) in children.into_iter().enumerate() {
//             let parent_index = parent_index + offset;
//             let children_indices = children_indices.into_iter().map(|(pi, ed)| (pi + offset, ed)).collect();
//             clusters.push((child, parent_index, ci, children_indices));
//         }

//         clusters[0].3.push((offset, (e, d)));
//     }

//     clusters
// }

// /// Returns the subtree as a list of `Cluster`s, with the indices required
// /// to go from a parent to a child and vice versa.
// ///
// /// The Vec contains tuples of:
// ///
// /// - The `Cluster` itself.
// /// - The position of the `Cluster` among its siblings.
// /// - A Vec of tuples of:
// ///  - The index of the parent `Cluster` in the Vec.
// ///  - A tuple of:
// ///     - The index of the parent `Cluster` in the Vec.
// ///    - The extent of the child.
// #[allow(clippy::type_complexity)]
// fn unstack_tree(self) -> Vec<(Self, usize, Vec<(usize, (usize, U))>)> {
//     let mut subtree = self.take_subtree();
//     subtree.sort_by_key(|(_, i, _, _)| *i);
//     subtree
//         .into_iter()
//         .map(|(c, _, ci, children)| (c, ci, children))
//         .collect()
// }

// #[test]
// fn numbered_subtree() {
//     let data = (0..1024).collect::<Vec<_>>();
//     let distance_fn = |x: &u32, y: &u32| x.abs_diff(*y);
//     let metric = Metric::new(distance_fn, false);
//     let data = FlatVec::new(data, metric).unwrap();

//     let seed = Some(42);
//     let criteria = |c: &Ball<_, _, _>| c.cardinality() > 1;
//     let root = Ball::new_tree(&data, &criteria, seed);

//     let mut numbered_subtree = root.clone().take_subtree();
//     let indices = numbered_subtree.iter().map(|(_, i, _, _)| *i).collect::<HashSet<_>>();
//     assert_eq!(indices.len(), numbered_subtree.len());

//     for i in 0..numbered_subtree.len() {
//         assert!(indices.contains(&i));
//     }

//     numbered_subtree.sort_by(|(_, i, _, _), (_, j, _, _)| i.cmp(j));
//     let numbered_subtree = numbered_subtree
//         .into_iter()
//         .map(|(a, _, _, b)| (a, b))
//         .collect::<Vec<_>>();

//     for (ball, child_indices) in &numbered_subtree {
//         for &(i, _) in child_indices {
//             let (child, _) = &numbered_subtree[i];
//             assert!(child.is_descendant_of(ball), "{ball:?} is not parent of {child:?}");
//         }
//     }

//     // let root_list = root.clone().as_indexed_list();
//     // let re_root = Ball::from_indexed_list(root_list);
//     // assert_eq!(root, re_root);

//     // for (l, r) in root.subtree().into_iter().zip(re_root.subtree()) {
//     //     assert_eq!(l, r);
//     // }
// }
