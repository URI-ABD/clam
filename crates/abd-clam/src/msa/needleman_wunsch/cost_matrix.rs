//! Substitution matrix for the Needleman-Wunsch aligner.

use core::ops::Neg;

use distances::number::Int;

/// The number of characters.
const NUM_CHARS: usize = 1 + (u8::MAX as usize);

/// A substitution matrix for the Needleman-Wunsch aligner.
#[derive(Clone)]
pub struct CostMatrix<T: Int> {
    /// The cost of substituting one character for another.
    sub_matrix: [[T; NUM_CHARS]; NUM_CHARS],
    /// The cost of inserting a character.
    ins_costs: [T; NUM_CHARS],
    /// The cost of inserting a character immediately after another insertion.
    ins_ext_costs: [T; NUM_CHARS],
    /// The cost of deleting a character.
    del_costs: [T; NUM_CHARS],
    /// The cost of deleting a character immediately after another deletion.
    del_ext_costs: [T; NUM_CHARS],
}

impl<T: Int> Default for CostMatrix<T> {
    fn default() -> Self {
        let mut matrix = Self::new(T::ONE, T::ONE, T::ONE);
        for i in 0..NUM_CHARS {
            matrix.sub_matrix[i][i] = T::ZERO;
        }
        matrix
    }
}

impl<T: Int> CostMatrix<T> {
    /// Create a new substitution matrix.
    #[must_use]
    pub fn new(default_sub_cost: T, default_ins_cost: T, default_del_cost: T) -> Self {
        let mut sub_matrix = [[default_sub_cost; NUM_CHARS]; NUM_CHARS];
        #[allow(clippy::needless_range_loop)]
        for i in 0..NUM_CHARS {
            sub_matrix[i][i] = T::ZERO;
        }
        Self {
            sub_matrix,
            ins_costs: [default_ins_cost; NUM_CHARS],
            ins_ext_costs: [default_ins_cost; NUM_CHARS],
            del_costs: [default_del_cost; NUM_CHARS],
            del_ext_costs: [default_del_cost; NUM_CHARS],
        }
    }

    /// Shift all costs in the matrix by a constant.
    #[must_use]
    pub fn shift(mut self, shift: T) -> Self {
        for i in 0..NUM_CHARS {
            for j in 0..NUM_CHARS {
                self.sub_matrix[i][j] += shift;
            }
            self.ins_costs[i] += shift;
            self.ins_ext_costs[i] += shift;
            self.del_costs[i] += shift;
            self.del_ext_costs[i] += shift;
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
    pub const fn with_sub_cost(mut self, a: u8, b: u8, cost: T) -> Self {
        self.sub_matrix[a as usize][b as usize] = cost;
        self
    }

    /// Set the cost of inserting a character.
    #[must_use]
    pub const fn with_ins_cost(mut self, a: u8, cost: T) -> Self {
        self.ins_costs[a as usize] = cost;
        self
    }

    /// Set the cost of inserting a character immediately after another insertion.
    #[must_use]
    pub const fn with_ins_ext_cost(mut self, a: u8, cost: T) -> Self {
        self.ins_ext_costs[a as usize] = cost;
        self
    }

    /// Set the cost of deleting a character.
    #[must_use]
    pub const fn with_del_cost(mut self, a: u8, cost: T) -> Self {
        self.del_costs[a as usize] = cost;
        self
    }

    /// Set the cost of deleting a character immediately after another deletion.
    #[must_use]
    pub const fn with_del_ext_cost(mut self, a: u8, cost: T) -> Self {
        self.del_ext_costs[a as usize] = cost;
        self
    }

    /// Get the cost of substituting one character for another.
    ///
    /// # Arguments
    ///
    /// * `a`: The old character to be substituted.
    /// * `b`: The new character to substitute with.
    pub const fn sub_cost(&self, a: u8, b: u8) -> T {
        self.sub_matrix[a as usize][b as usize]
    }

    /// Get the cost of inserting a character.
    pub const fn ins_cost(&self, a: u8) -> T {
        self.ins_costs[a as usize]
    }

    /// Get the cost of inserting a character immediately after another insertion.
    pub const fn ins_ext_cost(&self, a: u8) -> T {
        self.ins_ext_costs[a as usize]
    }

    /// Get the cost of deleting a character.
    pub const fn del_cost(&self, a: u8) -> T {
        self.del_costs[a as usize]
    }

    /// Get the cost of deleting a character immediately after another deletion.
    pub const fn del_ext_cost(&self, a: u8) -> T {
        self.del_ext_costs[a as usize]
    }
}

impl<T: Int + Neg<Output = T>> CostMatrix<T> {
    /// Linearly increase all costs in the matrix so that the minimum cost is
    /// zero.
    #[must_use]
    pub fn normalize(self) -> Self {
        let mut shift = T::MAX;
        for i in 0..NUM_CHARS {
            for j in 0..NUM_CHARS {
                shift = shift.min(self.sub_matrix[i][j]);
            }
            shift = shift.min(self.ins_costs[i]);
            shift = shift.min(self.ins_ext_costs[i]);
            shift = shift.min(self.del_costs[i]);
            shift = shift.min(self.del_ext_costs[i]);
        }
        self.shift(-shift)
    }
}

impl<T: Int + Neg<Output = T>> Neg for CostMatrix<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let mut neg_matrix = Self::default();
        for i in 0..NUM_CHARS {
            for j in 0..NUM_CHARS {
                neg_matrix.sub_matrix[i][j] = -self.sub_matrix[i][j];
            }
            neg_matrix.ins_costs[i] = -self.ins_costs[i];
            neg_matrix.ins_ext_costs[i] = -self.ins_ext_costs[i];
            neg_matrix.del_costs[i] = -self.del_costs[i];
            neg_matrix.del_ext_costs[i] = -self.del_ext_costs[i];
        }
        neg_matrix
    }
}
