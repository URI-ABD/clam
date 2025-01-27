//! Substitution matrix for the Needleman-Wunsch aligner.

use core::ops::Neg;

use std::collections::HashSet;

use distances::{number::Int, Number};

use super::super::NUM_CHARS;

/// A substitution matrix for the Needleman-Wunsch aligner.
#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "disk-io",
    derive(bitcode::Encode, bitcode::Decode, serde::Serialize, serde::Deserialize)
)]
pub struct CostMatrix<T: Number> {
    /// The cost of substituting one character for another.
    sub_matrix: Vec<Vec<T>>,
    /// The cost to open a gap.
    gap_open: T,
    /// The cost to extend a gap.
    gap_ext: T,
}

impl<T: Number> Default for CostMatrix<T> {
    fn default() -> Self {
        Self::new(T::ONE, T::ONE, T::ONE)
    }
}

impl<T: Number> CostMatrix<T> {
    /// Create a new substitution matrix.
    #[must_use]
    pub fn new(default_sub: T, gap_open: T, gap_ext: T) -> Self {
        // Initialize the substitution matrix.
        let mut sub_matrix = [[default_sub; NUM_CHARS]; NUM_CHARS];

        // Set the diagonal to zero.
        sub_matrix.iter_mut().enumerate().for_each(|(i, row)| row[i] = T::ZERO);

        Self {
            sub_matrix: sub_matrix.iter().map(|row| row.to_vec()).collect(),
            gap_open,
            gap_ext,
        }
    }

    /// Create a new substitution matrix with affine gap penalties.
    ///
    /// All substitution costs are set to 1.
    #[must_use]
    pub fn default_affine(gap_open: Option<usize>) -> Self {
        let gap_open = gap_open.map_or_else(|| T::from(10), T::from);
        Self::new(T::ONE, gap_open, T::ONE)
    }

    /// Add a constant to all substitution costs.
    #[must_use]
    pub fn shift(mut self, shift: T) -> Self {
        for i in 0..NUM_CHARS {
            for j in 0..NUM_CHARS {
                self.sub_matrix[i][j] += shift;
            }
        }
        self
    }

    /// Multiply all substitution costs by a constant.
    #[must_use]
    pub fn scale(mut self, scale: T) -> Self {
        for i in 0..NUM_CHARS {
            for j in 0..NUM_CHARS {
                self.sub_matrix[i][j] *= scale;
            }
        }
        self
    }

    /// Set the cost of substituting one character for another.
    ///
    /// # Arguments
    ///
    /// * `a`: The old character to be substituted.
    /// * `b`: The new character to substitute with.
    /// * `cost`: The cost of the substitution.
    #[must_use]
    pub fn with_sub_cost(mut self, a: u8, b: u8, cost: T) -> Self {
        self.sub_matrix[a as usize][b as usize] = cost;
        self
    }

    /// Set the cost of opening a gap.
    #[must_use]
    pub const fn with_gap_open(mut self, cost: T) -> Self {
        self.gap_open = cost;
        self
    }

    /// Set the cost of extending a gap.
    #[must_use]
    pub const fn with_gap_ext(mut self, cost: T) -> Self {
        self.gap_ext = cost;
        self
    }

    /// Get the cost of substituting one character for another.
    ///
    /// # Arguments
    ///
    /// * `a`: The old character to be substituted.
    /// * `b`: The new character to substitute with.
    pub fn sub_cost(&self, a: u8, b: u8) -> T {
        self.sub_matrix[a as usize][b as usize]
    }

    /// Get the cost of opening a gap.
    pub const fn gap_open_cost(&self) -> T {
        self.gap_open
    }

    /// Get the cost of extending a gap.
    pub const fn gap_ext_cost(&self) -> T {
        self.gap_ext
    }
}

impl<T: Number + Neg<Output = T>> CostMatrix<T> {
    /// Linearly increase all costs in the matrix so that the minimum cost is
    /// zero and all non-zero costs are positive.
    #[must_use]
    pub fn normalize(self) -> Self {
        let shift = self
            .sub_matrix
            .iter()
            .flatten()
            .fold(T::MAX, |a, &b| if a < b { a } else { b });

        self.shift(-shift)
    }
}

impl<T: Number + Neg<Output = T>> Neg for CostMatrix<T> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        for i in 0..NUM_CHARS {
            for j in 0..NUM_CHARS {
                self.sub_matrix[i][j] = -self.sub_matrix[i][j];
            }
        }
        self.gap_open = -self.gap_open;
        self.gap_ext = -self.gap_ext;
        self
    }
}

impl<T: Number + Neg<Output = T>> CostMatrix<T> {
    /// A substitution matrix for the Needleman-Wunsch aligner using the
    /// extendedIUPAC alphabet for nucleotides.
    ///
    /// See [here](https://www.bioinformatics.org/sms/iupac.html) for an
    /// explanation of the IUPAC codes.
    ///
    /// # Arguments
    ///
    /// * `gap_open`: The factor by which it is more expensive to open a gap
    ///   than to extend an existing gap. This defaults to 10.
    pub fn extended_iupac(gap_open: Option<usize>) -> Self {
        let gap_open = gap_open.unwrap_or(10);

        // For each pair of IUPAC characters, the cost is 1 - n / m, where m is
        // the number possible pairs of nucleotides that can be represented by
        // the IUPAC characters, and n is the number of matching pairs.
        #[rustfmt::skip]
        let costs = vec![
            ('A', 'R', 1, 2), ('C', 'Y', 1, 2), ('G', 'R', 1, 2), ('T', 'Y', 1, 2),
            ('A', 'W', 1, 2), ('C', 'S', 1, 2), ('G', 'S', 1, 2), ('T', 'W', 1, 2),
            ('A', 'M', 1, 2), ('C', 'M', 1, 2), ('G', 'K', 1, 2), ('T', 'K', 1, 2),
            ('A', 'D', 1, 3), ('C', 'B', 1, 3), ('G', 'B', 1, 3), ('T', 'B', 1, 3),
            ('A', 'H', 1, 3), ('C', 'H', 1, 3), ('G', 'D', 1, 3), ('T', 'D', 1, 3),
            ('A', 'V', 1, 3), ('C', 'V', 1, 3), ('G', 'V', 1, 3), ('T', 'H', 1, 3),
            ('A', 'N', 1, 4), ('C', 'N', 1, 4), ('G', 'N', 1, 4), ('T', 'N', 1, 4),

            ('R', 'A', 1, 2), ('Y', 'C', 1, 2), ('S', 'G', 1, 2), ('W', 'A', 1, 2), ('K', 'G', 1, 2), ('M', 'A', 1, 2),
            ('R', 'G', 1, 2), ('Y', 'T', 1, 2), ('S', 'C', 1, 2), ('W', 'T', 1, 2), ('K', 'T', 1, 2), ('M', 'C', 1, 2),
            ('R', 'S', 1, 4), ('Y', 'S', 1, 4), ('S', 'R', 1, 4), ('W', 'R', 1, 4), ('K', 'R', 1, 4), ('M', 'R', 1, 4),
            ('R', 'W', 1, 4), ('Y', 'W', 1, 4), ('S', 'Y', 1, 4), ('W', 'Y', 1, 4), ('K', 'Y', 1, 4), ('M', 'Y', 1, 4),
            ('R', 'K', 1, 4), ('Y', 'K', 1, 4), ('S', 'K', 1, 4), ('W', 'K', 1, 4), ('K', 'S', 1, 4), ('M', 'S', 1, 4),
            ('R', 'M', 1, 4), ('Y', 'M', 1, 4), ('S', 'M', 1, 4), ('W', 'M', 1, 4), ('K', 'W', 1, 4), ('M', 'W', 1, 4),
            ('R', 'B', 1, 6), ('Y', 'B', 2, 6), ('S', 'B', 2, 6), ('W', 'B', 1, 6), ('K', 'B', 2, 6), ('M', 'B', 1, 6),
            ('R', 'D', 2, 6), ('Y', 'D', 1, 6), ('S', 'D', 1, 6), ('W', 'D', 2, 6), ('K', 'D', 2, 6), ('M', 'D', 1, 6),
            ('R', 'H', 1, 6), ('Y', 'H', 2, 6), ('S', 'H', 1, 6), ('W', 'H', 2, 6), ('K', 'H', 1, 6), ('M', 'H', 2, 6),
            ('R', 'V', 2, 6), ('Y', 'V', 1, 6), ('S', 'V', 2, 6), ('W', 'V', 1, 6), ('K', 'V', 1, 6), ('M', 'V', 2, 6),
            ('R', 'N', 2, 8), ('Y', 'N', 2, 8), ('S', 'N', 2, 8), ('W', 'N', 2, 8), ('K', 'N', 1, 8), ('M', 'N', 2, 8),
            ('R', 'R', 1, 2), ('Y', 'Y', 1, 2), ('S', 'S', 1, 2), ('W', 'W', 1, 2), ('K', 'K', 1, 2), ('M', 'M', 1, 2),

            ('B', 'C', 1,  3), ('D', 'A', 1,  3), ('H', 'A', 1,  3), ('V', 'A', 1,  3),  ('N', 'A', 1, 4),
            ('B', 'G', 1,  3), ('D', 'G', 1,  3), ('H', 'C', 1,  3), ('V', 'C', 1,  3),  ('N', 'C', 1, 4),
            ('B', 'T', 1,  3), ('D', 'T', 1,  3), ('H', 'T', 1,  3), ('V', 'D', 1,  3),  ('N', 'G', 1, 4),
            ('B', 'R', 1,  6), ('D', 'R', 2,  6), ('H', 'R', 1,  6), ('V', 'R', 2,  6),  ('N', 'T', 1, 4),
            ('B', 'Y', 2,  6), ('D', 'Y', 1,  6), ('H', 'Y', 2,  6), ('V', 'Y', 1,  6),  ('N', 'R', 1, 4),
            ('B', 'S', 2,  6), ('D', 'S', 1,  6), ('H', 'S', 1,  6), ('V', 'S', 2,  6),  ('N', 'Y', 1, 4),
            ('B', 'W', 1,  6), ('D', 'W', 2,  6), ('H', 'W', 2,  6), ('V', 'W', 1,  6),  ('N', 'S', 1, 4),
            ('B', 'K', 2,  6), ('D', 'K', 2,  6), ('H', 'K', 1,  6), ('V', 'K', 1,  6),  ('N', 'W', 1, 4),
            ('B', 'M', 1,  6), ('D', 'M', 1,  6), ('H', 'M', 2,  6), ('V', 'M', 2,  6),  ('N', 'K', 1, 4),
            ('B', 'D', 2,  9), ('D', 'B', 2,  9), ('H', 'B', 2,  9), ('V', 'B', 2,  9),  ('N', 'M', 1, 4),
            ('B', 'H', 2,  9), ('D', 'H', 2,  9), ('H', 'D', 2,  9), ('V', 'D', 2,  9),  ('N', 'B', 1, 4),
            ('B', 'V', 2,  9), ('D', 'V', 2,  9), ('H', 'V', 2,  9), ('V', 'H', 2,  9),  ('N', 'D', 1, 4),
            ('B', 'N', 3, 12), ('D', 'N', 3, 12), ('H', 'N', 3, 12), ('V', 'N', 3, 12),  ('N', 'H', 1, 4),
            ('B', 'B', 1,  3), ('D', 'D', 1,  3), ('H', 'H', 1,  3), ('V', 'V', 1,  3),  ('N', 'V', 1, 4),
                                                                                         ('N', 'N', 1, 4),
        ];

        // Calculate the least common multiple of the denominators so we can
        // scale the costs to integers.
        let lcm = costs
            .iter()
            .map(|&(_, _, _, m)| m)
            .collect::<HashSet<_>>()
            .into_iter()
            .fold(1, |a, b| a.lcm(&b));

        // T and U are interchangeable.
        let t_to_u = costs
            .iter()
            .filter(|&&(a, _, _, _)| a == 'T')
            .map(|&(_, b, n, m)| ('U', b, n, m))
            .chain(
                costs
                    .iter()
                    .filter(|&&(_, b, _, _)| b == 'T')
                    .map(|&(a, _, n, m)| (a, 'U', n, m)),
            )
            .collect::<Vec<_>>();

        // The initial matrix with the default costs, except for gaps which are
        // interchangeable.
        let matrix = Self::default()
            .with_sub_cost(b'-', b'.', T::ZERO)
            .with_sub_cost(b'.', b'-', T::ZERO)
            .scale(T::from(lcm));

        // Add all costs to the matrix.
        costs
            .into_iter()
            .chain(t_to_u)
            // Scale the costs to integers.
            .map(|(a, b, n, m)| (a, b, T::from(n * (lcm / m))))
            .flat_map(|(a, b, cost)| {
                // Add the costs for the upper and lower case versions of the
                // characters.
                [
                    (a, b, cost),
                    (a.to_ascii_lowercase(), b, cost),
                    (a, b.to_ascii_lowercase(), cost),
                    (a.to_ascii_lowercase(), b.to_ascii_lowercase(), cost),
                ]
            })
            // Cast the characters to bytes.
            .map(|(a, b, cost)| (a as u8, b as u8, cost))
            // Add the costs to the substitution matrix.
            .fold(matrix, |matrix, (a, b, cost)| matrix.with_sub_cost(a, b, cost))
            // Add affine gap penalties.
            .with_gap_open(T::from(lcm * gap_open))
            .with_gap_ext(T::from(lcm))
    }

    /// The BLOSUM62 substitution matrix for proteins.
    ///
    /// See [here](https://en.wikipedia.org/wiki/BLOSUM) for more information.
    ///
    /// # Arguments
    ///
    /// * `gap_open`: The factor by which it is more expensive to open a gap
    ///   than to extend an existing gap. This defaults to 10.
    #[must_use]
    pub fn blosum62(gap_open: Option<usize>) -> Self {
        let gap_open = gap_open.unwrap_or(10);

        #[rustfmt::skip]
        let costs = [
            vec![ 9],  // C
            vec![-1,  4],  // S
            vec![-1,  1,  5],  // T
            vec![ 0,  1,  0,  4],  // A
            vec![-3,  0, -2,  0,  6],  // G
            vec![-3, -1, -1, -1, -2,  7],  // P
            vec![-3,  0, -1, -2, -1, -1,  6],  // D
            vec![-4,  0, -1, -1, -2, -1,  2,  5],  // E
            vec![-3,  0, -1, -1, -2, -1,  0,  2,  5],  // Q
            vec![-3,  1,  0, -2,  0, -2,  1,  0,  0,  6],  // N
            vec![-3, -1, -2, -2, -2, -2,  1,  0,  0,  1,  8],  // H
            vec![-3, -1, -1, -1, -2, -2, -2,  0,  1,  0,  0,  5],  // R
            vec![-3,  0, -1, -1, -2, -1, -1,  1,  1,  0, -1,  2,  5],  // K
            vec![-1, -1, -1, -1, -3, -2, -3, -2,  0, -2, -2, -1, -1,  5],  // M
            vec![-1, -2, -1, -1, -4, -3, -3, -3, -3, -3, -3, -3, -3,  1,  4],  // I
            vec![-1, -2, -1, -1, -4, -3, -4, -3, -2, -3, -3, -2, -2,  2,  2,  4],  // L
            vec![-1, -2,  0,  0, -3, -2, -3, -2, -2, -3, -3, -3, -2,  1,  3,  1,  4],  // V
            vec![-2, -3, -2, -3, -2, -4, -4, -3, -2, -4, -2, -3, -3, -1, -3, -2, -3, 11],  // W
            vec![-2, -2, -2, -2, -3, -3, -3, -2, -1, -2,  2, -2, -2, -1, -1, -1, -1,  2,  7],  // Y
            vec![-2, -2, -2, -2, -3, -4, -3, -3, -3, -3, -1, -3, -3,  0,  0,  0, -1,  1,  3,  6],  // F
        ];

        // Calculate the maximum difference between any two substitution costs.
        let max_delta = {
            let (min, max) = costs.iter().flatten().fold((i32::MAX, i32::MIN), |(min, max), &cost| {
                (Ord::min(min, cost), Ord::max(max, cost))
            });
            <usize as Number>::from(max.abs_diff(min))
        };

        // The amino acid codes.
        let codes = "CSTAGPDEQNHRKMILVWYF";

        // The initial matrix with the default costs, except for gaps which are
        // interchangeable.
        let matrix = Self::default()
            .with_sub_cost(b'-', b'.', T::ZERO)
            .with_sub_cost(b'.', b'-', T::ZERO)
            .scale(T::from(max_delta));

        // Flatten the costs into a vector of (a, b, cost) tuples.
        codes
            .chars()
            .zip(costs.iter())
            .flat_map(|(a, costs)| {
                codes
                    .chars()
                    .zip(costs.iter())
                    .map(move |(b, &cost)| (a, b, T::from(cost)))
            })
            .flat_map(|(a, b, cost)| {
                // Add the costs for the upper and lower case versions of the
                // characters.
                [
                    (a, b, cost),
                    (a.to_ascii_lowercase(), b, cost),
                    (a, b.to_ascii_lowercase(), cost),
                    (a.to_ascii_lowercase(), b.to_ascii_lowercase(), cost),
                ]
            })
            // Convert the characters to bytes.
            .map(|(a, b, cost)| (a as u8, b as u8, cost))
            // And combine them into a matrix.
            .fold(matrix, |matrix, (a, b, cost)| {
                matrix.with_sub_cost(a, b, cost).with_sub_cost(b, a, cost)
            })
            // Convert the matrix into a form that can be used to minimize the
            // edit distances.
            .neg()
            .normalize()
            // Add affine gap penalties.
            .with_gap_open(T::from(max_delta * gap_open))
            .with_gap_ext(T::from(max_delta))
    }
}
