use ndarray::Array2;

pub type Data<T> = Array2<T>;
pub type Index = usize;
pub type Indices = Vec<Index>;
pub type Metric = String;
pub type Radius = f64;
pub type Radii = Array2<Radius>;
