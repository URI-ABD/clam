use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

use abd_clam::cakes::{self, SearchAlgorithm, Searchable};
use abd_clam::{Cluster, Metric};
use distances::Number;

#[derive(Debug, Clone, PartialEq)]
pub struct KnnParams {
    pub k: usize,
}

impl fmt::Display for KnnParams {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "k={}", self.k)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct KnnRepeatedRnnParams<T: Number> {
    pub k: usize,
    pub multiplier: T,
}

impl<T: Number> fmt::Display for KnnRepeatedRnnParams<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "k={},multiplier={}", self.k, self.multiplier)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RnnParams<T: Number> {
    pub radius: T,
}

impl<T: Number> fmt::Display for RnnParams<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "radius={}", self.radius)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum QueryAlgorithm<T: Number> {
    KnnLinear(KnnParams),
    KnnRepeatedRnn(KnnRepeatedRnnParams<T>),
    KnnBreadthFirst(KnnParams),
    KnnDepthFirst(KnnParams),
    RnnLinear(RnnParams<T>),
    RnnClustered(RnnParams<T>),
}

impl<T: Number> std::fmt::Display for QueryAlgorithm<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            QueryAlgorithm::KnnLinear(params) => write!(f, "knn-linear({params})"),
            QueryAlgorithm::KnnRepeatedRnn(params) => {
                write!(f, "knn-repeated-rnn(k={},multiplier={})", params.k, params.multiplier)
            }
            QueryAlgorithm::KnnBreadthFirst(params) => write!(f, "knn-breadth-first({params})"),
            QueryAlgorithm::KnnDepthFirst(params) => write!(f, "knn-depth-first({params})"),
            QueryAlgorithm::RnnLinear(params) => write!(f, "rnn-linear({params})"),
            QueryAlgorithm::RnnClustered(params) => write!(f, "rnn-clustered({params})"),
        }
    }
}

impl<T: Number + 'static> QueryAlgorithm<T> {
    pub fn get<I, R, C, M, D>(&self) -> Box<dyn SearchAlgorithm<I, R, C, M, D>>
    where
        R: Number + 'static,
        C: Cluster<R>,
        M: Metric<I, R>,
        D: Searchable<I, R, C, M>,
    {
        match self {
            QueryAlgorithm::KnnLinear(params) => Box::new(cakes::KnnLinear(params.k)),
            QueryAlgorithm::KnnRepeatedRnn(params) => {
                Box::new(cakes::KnnRepeatedRnn(params.k, R::from(params.multiplier)))
            }
            QueryAlgorithm::KnnBreadthFirst(params) => Box::new(cakes::KnnBreadthFirst(params.k)),
            QueryAlgorithm::KnnDepthFirst(params) => Box::new(cakes::KnnDepthFirst(params.k)),
            QueryAlgorithm::RnnLinear(params) => Box::new(cakes::RnnLinear(R::from(params.radius))),
            QueryAlgorithm::RnnClustered(params) => Box::new(cakes::RnnClustered(R::from(params.radius))),
        }
    }
}

/// Parse parameter string into key-value pairs
fn parse_parameters(params_str: &str) -> Result<HashMap<String, String>, String> {
    let mut params = HashMap::new();

    if params_str.is_empty() {
        return Ok(params);
    }

    for pair in params_str.split(',') {
        let pair = pair.trim();
        if let Some((key, value)) = pair.split_once('=') {
            params.insert(key.trim().to_string(), value.trim().to_string());
        } else {
            return Err(format!("Invalid parameter format: '{pair}'. Expected 'key=value'"));
        }
    }
    Ok(params)
}

impl<T: Number> FromStr for QueryAlgorithm<T> {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (algorithm, params_str) = if let Some((alg, params)) = s.split_once(':') {
            (alg, params)
        } else {
            (s, "")
        };
        let params = parse_parameters(params_str)?;

        match algorithm {
            "knn-linear" => Ok(QueryAlgorithm::KnnLinear(KnnParams {
                k: params
                    .get("k")
                    .unwrap_or(&"1".to_string())
                    .parse()
                    .map_err(|_| "Invalid k value")?,
            })),
            "knn-repeated-rnn" => Ok(QueryAlgorithm::KnnRepeatedRnn(KnnRepeatedRnnParams {
                k: params
                    .get("k")
                    .unwrap_or(&"1".to_string())
                    .parse()
                    .map_err(|_| "Invalid k value")?,
                multiplier: params
                    .get("multiplier")
                    .unwrap_or(&"2.0".to_string())
                    .parse()
                    .map_err(|_| "Invalid multiplier value")?,
            })),
            "knn-breadth-first" => Ok(QueryAlgorithm::KnnBreadthFirst(KnnParams {
                k: params
                    .get("k")
                    .unwrap_or(&"1".to_string())
                    .parse()
                    .map_err(|_| "Invalid k value")?,
            })),
            "knn-depth-first" => Ok(QueryAlgorithm::KnnDepthFirst(KnnParams {
                k: params
                    .get("k")
                    .unwrap_or(&"1".to_string())
                    .parse()
                    .map_err(|_| "Invalid k value")?,
            })),
            "rnn-linear" => Ok(QueryAlgorithm::RnnLinear(RnnParams {
                radius: params
                    .get("radius")
                    .unwrap_or(&"1".to_string())
                    .parse()
                    .map_err(|_| "Invalid radius value")?,
            })),
            "rnn-clustered" => Ok(QueryAlgorithm::RnnClustered(RnnParams {
                radius: params
                    .get("radius")
                    .unwrap_or(&"1".to_string())
                    .parse()
                    .map_err(|_| "Invalid radius value")?,
            })),
            _ => Err(format!("Unknown algorithm: '{algorithm}'")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_knn_linear() {
        let query: QueryAlgorithm<f64> = "knn-linear:k=3".parse().unwrap();
        assert_eq!(query, QueryAlgorithm::KnnLinear(KnnParams { k: 3 }));
    }

    #[test]
    fn test_parse_knn_repeated_rnn() {
        let query: QueryAlgorithm<f64> = "knn-repeated-rnn:k=5,multiplier=2.5".parse().unwrap();
        assert_eq!(
            query,
            QueryAlgorithm::KnnRepeatedRnn(KnnRepeatedRnnParams { k: 5, multiplier: 2.5 })
        );
    }

    #[test]
    fn test_parse_rnn() {
        let query: QueryAlgorithm<f64> = "rnn-linear:radius=2.5".parse().unwrap();
        assert_eq!(query, QueryAlgorithm::RnnLinear(RnnParams { radius: 2.5 }));

        let query2: QueryAlgorithm<f64> = "rnn-clustered:radius=1.0".parse().unwrap();
        assert_eq!(query2, QueryAlgorithm::RnnClustered(RnnParams { radius: 1.0 }));
    }

    #[test]
    fn test_display() {
        let query: QueryAlgorithm<f64> = QueryAlgorithm::KnnLinear(KnnParams { k: 3 });
        assert_eq!(query.to_string(), "knn-linear(k=3)");
    }

    #[test]
    fn test_parse_errors() {
        assert!("unknown-algo:k=3".parse::<QueryAlgorithm<f64>>().is_err());
        assert!("knn-linear:k=invalid".parse::<QueryAlgorithm<f64>>().is_err());
        assert!("knn-linear:missing_equals".parse::<QueryAlgorithm<f64>>().is_err());
    }

    #[test]
    fn test_search_algorithm_wrapper_creation() {
        // Test that we can create a SearchAlgorithmWrapper from a QueryAlgorithm
        let query: QueryAlgorithm<f64> = QueryAlgorithm::KnnLinear(KnnParams { k: 5 });
        let alg = query
            .get::<Vec<f64>, f64, abd_clam::Ball<f64>, abd_clam::metric::Euclidean, abd_clam::FlatVec<Vec<f64>, usize>>(
            );

        // Test that the wrapper implements the SearchAlgorithm trait correctly
        assert_eq!(alg.name(), "KnnLinear");
        assert_eq!(alg.k(), Some(5));
        assert_eq!(alg.radius(), None);
    }

    #[test]
    fn test_rnn_wrapper_creation() {
        let query = QueryAlgorithm::RnnLinear(RnnParams { radius: 2.5 });
        let alg = query
            .get::<Vec<f64>, f64, abd_clam::Ball<f64>, abd_clam::metric::Euclidean, abd_clam::FlatVec<Vec<f64>, usize>>(
            );

        assert_eq!(alg.name(), "RnnLinear");
        assert_eq!(alg.k(), None);
        assert_eq!(alg.radius(), Some(2.5));
    }
}
