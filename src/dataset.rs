use super::types::*;

#[derive(Debug, Eq, PartialEq)]
pub struct Dataset {
    pub data: Box<Data>,
    pub metric: String,
}

impl Dataset {
    pub fn len(&self) -> Index {
        self.data.len()
    }
    pub fn distance(&self, left: Indices, right: Indices) -> Radii {
        let d = &*self.data;
        let l: Vec<&Datum> = d.iter().enumerate().filter(|(i, _)| left.contains(i)).map(|(_, e)| e).collect();
        let r: Vec<&Datum> = d.iter().enumerate().filter(|(i, _)| right.contains(i)).map(|(_, e)| e).collect();
        let s: Radii = l.iter().zip(r.iter()).fold(0.0, |sum, (&a, &b)| sum + a + b);
        vec![0.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        let _d = Dataset { data: Box::new(vec![0, 0]), metric: String::from("euclidean") };
    }
}