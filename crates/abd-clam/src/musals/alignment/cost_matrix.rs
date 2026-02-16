//! Substitution matrix for the Needleman-Wunsch aligner.

use std::collections::{HashMap, HashSet};

use num::Integer;

use crate::DistanceValue;

/// A substitution matrix for the Needleman-Wunsch aligner.
///
/// This matrix defines the costs of substituting one character for another, as well as the costs of opening and extending gaps.
///
/// The default matrix sets all costs to 1 and can be used with genomic or protein sequences. It is possible to fully customize the matrix using the provided
/// methods. We already provide a small number of  specialized matrices. These include:
///   - Extended IUPAC: From the [`CostMatrix::extended_iupac`](CostMatrix::extended_iupac) method, a matrix that uses the extended IUPAC nucleotide code.
///   - BLOSUM62: From the [`CostMatrix::blosum62`](CostMatrix::blosum62) method, a matrix that uses the BLOSUM62 substitution scores for amino-acid sequences.
#[must_use]
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize, databuf::Encode, databuf::Decode))]
#[cfg_attr(feature = "pancakes", derive(deepsize::DeepSizeOf))]
pub struct CostMatrix<T> {
    /// The default cost of substituting one character for another. This is used when a specific cost is not set for a pair of characters.
    default_sub: T,
    /// The cost to open a gap.
    gap_open: T,
    /// The cost to extend a gap.
    gap_ext: T,
    /// The cost of substituting one character for another.
    sub_costs: HashMap<(char, char), T>,
}

impl<T: DistanceValue> Default for CostMatrix<T> {
    fn default() -> Self {
        Self::new(T::one(), T::one(), T::one())
    }
}

impl<T: DistanceValue> CostMatrix<T> {
    /// Create a new substitution matrix.
    pub fn new(default_sub: T, gap_open: T, gap_ext: T) -> Self {
        Self {
            sub_costs: HashMap::new(),
            default_sub,
            gap_open,
            gap_ext,
        }
    }

    /// Cast the cost matrix to another distance value type.
    pub fn cast<U: DistanceValue, F: Fn(T) -> U>(&self, caster: F) -> CostMatrix<U> {
        CostMatrix {
            sub_costs: self.sub_costs.iter().map(|(&(a, b), &cost)| ((a, b), caster(cost))).collect(),
            default_sub: caster(self.default_sub),
            gap_open: caster(self.gap_open),
            gap_ext: caster(self.gap_ext),
        }
    }

    /// Create a new substitution matrix with affine gap penalties.
    ///
    /// All substitution costs are set to 1.
    ///
    /// # Arguments
    ///
    /// * `gap_open`: The factor by which it is more expensive to open a gap than to extend an existing gap. This defaults to 10.
    pub fn default_affine(gap_open: Option<T>) -> Self {
        let gap_open = gap_open.unwrap_or_else(|| T::from_i8(10).unwrap_or_else(|| unreachable!("T::from_i8(10) should be valid for all DistanceValue types")));
        Self::new(T::one(), gap_open, T::one())
    }

    /// Add a constant to all substitution costs.
    pub fn shift_up(mut self, shift: T) -> Self {
        for cost in self.sub_costs.values_mut() {
            *cost += shift;
        }
        self.default_sub += shift;
        self
    }

    /// Subtract a constant from all substitution costs.
    pub fn shift_down(mut self, shift: T) -> Self {
        for cost in self.sub_costs.values_mut() {
            *cost -= shift;
        }
        self.default_sub -= shift;
        self
    }

    /// Multiply all substitution costs by a constant.
    pub fn scale(mut self, scale: T) -> Self {
        for cost in self.sub_costs.values_mut() {
            *cost *= scale;
        }
        self.default_sub *= scale;
        self
    }

    /// Set the cost of substituting one character for another.
    ///
    /// # Arguments
    ///
    /// * `a`: The old character to be substituted.
    /// * `b`: The new character to substitute with.
    /// * `cost`: The cost of the substitution.
    pub fn with_sub_cost(mut self, a: char, b: char, cost: T) -> Self {
        self.sub_costs.insert((a, b), cost);
        self
    }

    /// Set the default cost of substituting one character for another. This is used when a specific cost is not set for a pair of characters.
    pub const fn with_default_sub(mut self, cost: T) -> Self {
        self.default_sub = cost;
        self
    }

    /// Set the cost of opening a gap.
    pub const fn with_gap_open(mut self, cost: T) -> Self {
        self.gap_open = cost;
        self
    }

    /// Set the cost of extending a gap.
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
    pub fn sub_cost(&self, a: char, b: char) -> T {
        if a == b {
            T::zero()
        } else {
            *self.sub_costs.get(&(a, b)).unwrap_or(&self.default_sub)
        }
    }

    /// Get the default cost of substituting one character for another. This is used when a specific cost is not set for a pair of characters.
    pub const fn default_sub(&self) -> T {
        self.default_sub
    }

    /// Get the cost of opening a gap.
    pub const fn gap_open_cost(&self) -> T {
        self.gap_open
    }

    /// Get the cost of extending a gap.
    pub const fn gap_ext_cost(&self) -> T {
        self.gap_ext
    }

    /// Shift all costs up or down so that the minimum cost is zero.
    pub fn normalize(self) -> Self {
        let shift = self.sub_costs.values().fold(T::max_value(), |a, &b| if a < b { a } else { b });
        self.shift_down(shift)
    }

    /// Invert all costs in the matrix, i.e. subtract them from the maximum cost and then normalize.
    pub fn invert(mut self) -> Self {
        let max_cost = self.sub_costs.values().fold(T::min_value(), |a, &b| if a > b { a } else { b });
        for cost in self.sub_costs.values_mut() {
            *cost = max_cost - *cost;
        }
        self.normalize()
    }

    /// Returns the minimum substitution cost in the matrix.
    pub fn min_sub_cost(&self) -> T {
        self.sub_costs.values().fold(T::max_value(), |a, &b| if a < b { a } else { b })
    }

    /// Returns the maximum substitution cost in the matrix.
    pub fn max_sub_cost(&self) -> T {
        self.sub_costs.values().fold(T::min_value(), |a, &b| if a > b { a } else { b })
    }
}

impl<T: DistanceValue> CostMatrix<T> {
    /// A substitution matrix for the Needleman-Wunsch aligner using the extended IUPAC alphabet for nucleotides.
    ///
    /// See [here](https://www.bioinformatics.org/sms/iupac.html) for a non-explanation of the IUPAC codes.
    ///
    /// # Arguments
    ///
    /// * `gap_open`: The factor by which it is more expensive to open a gap than to extend an existing gap. This defaults to 10.
    pub fn extended_iupac(gap_open: Option<T>) -> Self {
        let gap_open = gap_open.unwrap_or_else(|| T::from_u8(10).unwrap_or_else(|| unreachable!("T::from_i8(10) should be valid for all DistanceValue types")));

        // For each pair of IUPAC characters, the cost is 1 - n / m, where m is the number possible pairs of nucleotides that can be represented by the IUPAC
        // characters, and n is the number of matching pairs.
        #[rustfmt::skip]
        let costs = vec![
            ('A', 'R', 1_u8, 2_u8), ('C', 'Y', 1, 2), ('G', 'R', 1, 2), ('T', 'Y', 1, 2),
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

        // Calculate the least common multiple of the denominators so we can scale the costs to integers.
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
            .chain(costs.iter().filter(|&&(_, b, _, _)| b == 'T').map(|&(a, _, n, m)| (a, 'U', n, m)))
            .collect::<Vec<_>>();

        // The initial matrix with the default costs, except for gaps which are interchangeable.
        let matrix = Self::default()
            .with_sub_cost('-', '.', T::zero())
            .with_sub_cost('.', '-', T::zero())
            .scale(T::from_u8(lcm).unwrap_or_else(|| unreachable!("Every distance type should be large enough to hold i8")));

        let lcm_t = T::from_u8(lcm).unwrap_or_else(|| unreachable!("Every distance type should be large enough to hold i8"));

        // Add all costs to the matrix.
        costs
            .into_iter()
            .chain(t_to_u)
            // Scale the costs to integers.
            .map(|(a, b, n, m)| {
                (
                    a,
                    b,
                    T::from_u8(n * (lcm / m)).unwrap_or_else(|| unreachable!("Every distance type should be large enough to hold i8")),
                )
            })
            .flat_map(|(a, b, cost)| {
                // Add the costs for the upper and lower case versions of the characters.
                [
                    (a, b, cost),
                    (a.to_ascii_lowercase(), b, cost),
                    (a, b.to_ascii_lowercase(), cost),
                    (a.to_ascii_lowercase(), b.to_ascii_lowercase(), cost),
                ]
            })
            // Add the costs to the substitution matrix.
            .fold(matrix, |matrix, (a, b, cost)| matrix.with_sub_cost(a, b, cost))
            // Add affine gap penalties.
            .with_gap_open(lcm_t * gap_open)
            .with_gap_ext(lcm_t)
    }

    /// The BLOSUM62 substitution matrix for proteins.
    ///
    /// See [here](https://en.wikipedia.org/wiki/BLOSUM) if you trust Wikipedia.
    ///
    /// # Arguments
    ///
    /// * `gap_open`: The factor by which it is more expensive to open a gap than to extend an existing gap. This defaults to 10.
    pub fn blosum62(gap_open: Option<T>) -> Self {
        let gap_open = gap_open.unwrap_or_else(|| T::from_u8(10).unwrap_or_else(|| unreachable!("Every distance type should be large enough to hold i8")));

        #[rustfmt::skip]
        let costs = [
            vec![ 9_i8],  // C
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
            let (min, max) = costs
                .iter()
                .flatten()
                .fold((i8::MAX, i8::MIN), |(min, max), &cost| (Ord::min(min, cost), Ord::max(max, cost)));
            if max > min { max - min } else { min - max }
        };

        // The amino acid codes.
        let codes = "CSTAGPDEQNHRKMILVWYF";

        // The initial matrix with the default costs, except for gaps which are
        // interchangeable.
        let matrix = Self::default()
            .with_sub_cost('-', '.', T::zero())
            .with_sub_cost('.', '-', T::zero())
            .scale(T::from_i8(max_delta).unwrap_or_else(|| unreachable!("Every distance type should be large enough to hold i8")));

        let max_delta_t = T::from_i8(max_delta).unwrap_or_else(|| unreachable!("Every distance type should be large enough to hold i8: {max_delta}"));

        // Flatten the costs into a vector of (a, b, cost) tuples.
        codes
            .chars()
            .zip(costs.iter())
            .flat_map(|(a, costs)| {
                codes.chars().zip(costs.iter()).map(move |(b, &cost)| {
                    (
                        a,
                        b,
                        T::from_i8(cost).unwrap_or_else(|| unreachable!("Every distance type should be large enough to hold i8: {cost}")),
                    )
                })
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
            // And combine them into a matrix.
            .fold(matrix, |matrix, (a, b, cost)| matrix.with_sub_cost(a, b, cost).with_sub_cost(b, a, cost))
            // Convert the matrix into a form that can be used to minimize the
            // edit distances.
            .invert()
            // Add affine gap penalties.
            .with_gap_open(max_delta_t * gap_open)
            .with_gap_ext(max_delta_t)
    }
}
