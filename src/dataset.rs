use super::types::*;
use std::collections::{HashMap};

#[derive(Debug)]
pub struct Dataset<T> {
    pub data: Box<Data<T>>,
    pub metric: &'static str,
    pub history: HashMap<Index, f64>,
}

impl<T> Dataset<T> {
    pub fn new(data: Data<T>, metric: &'static str) -> Dataset<T> {
        let data = Box::new(data);
        let history: HashMap<Index, f64> = [(0, 0.)].iter().cloned().collect();
        Dataset {
            data,
            metric,
            history,
        }
    }

    pub fn len(&self) -> Index {
        self.data.len() as Index
    }
    // pub fn distance(&self, left: Indices, right: Indices) -> Radius {
    //     left.iter().zip(&right).fold(0, |sum, (a, b)| sum + a + b) as f64
    // }

    // fn key(&self, i: Index, j: Index) -> Index {
    //     if i == j { 0 }
    //     else if i < j { (j * (j - 1) / 2 + i + 1) }
    //     else { (i * (i - 1) / 2 + j + 1) }
    // }
    //
    // fn ij(&self, k: Index) -> (Index, Index) {
    //     let i: Index = ((1. + (1. + 8. * k as f64).sqrt()) / 2.).ceil() as Index - 1;
    //     let j: Index = k - 1 - i * (i - 1) / 2;
    //     (i, j)
    // }
    //
    // fn insert(&self, left: Indices, right: Indices) -> () {
    //     let mut keys: HashSet<Index> = HashSet::new();
    //     for &i in left.iter() {
    //         for &j in right.iter() {
    //             keys.insert(self.key(i, j));
    //         }
    //     }
    //
    //     let mut new_keys: Indices = vec![];
    //     for &k in keys.iter() {
    //         if !self.history.contains_key(&k) {
    //             new_keys.push(k);
    //         }
    //     }
    // }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let data = Data::from_elem((2, 2), 1);
        let _d = Dataset::new(data, "euclidean");
    }
}
