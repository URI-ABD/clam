use core::str::FromStr;
use std::collections::HashMap;

use abd_clam::{
    DistanceValue,
    cakes::{Cakes, KnnBfs, KnnDfs, KnnLinear, KnnRrnn, RnnChess, RnnLinear},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShellCakes {
    KnnBfs(HashMap<String, String>),
    KnnDfs(HashMap<String, String>),
    KnnLinear(HashMap<String, String>),
    KnnRrnn(HashMap<String, String>),
    RnnChess(HashMap<String, String>),
    RnnLinear(HashMap<String, String>),
    ApproxKnnDfs(HashMap<String, String>),
}

impl core::fmt::Display for ShellCakes {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        match self {
            Self::KnnBfs(params) => write!(f, "KnnBfs({})", display_params(params)),
            Self::KnnDfs(params) => write!(f, "KnnDfs({})", display_params(params)),
            Self::KnnLinear(params) => write!(f, "KnnLinear({})", display_params(params)),
            Self::KnnRrnn(params) => write!(f, "KnnRrnn({})", display_params(params)),
            Self::RnnChess(params) => write!(f, "RnnChess({})", display_params(params)),
            Self::RnnLinear(params) => write!(f, "RnnLinear({})", display_params(params)),
            Self::ApproxKnnDfs(params) => write!(f, "ApproxKnnDfs({})", display_params(params)),
        }
    }
}

fn display_params(params: &HashMap<String, String>) -> String {
    params.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<String>>().join(", ")
}

impl FromStr for ShellCakes {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (alg, params_str) = s.trim().split_once(':').unwrap_or((s, ""));
        let params = parse_parameters(params_str)?;

        match alg.to_lowercase().as_str() {
            "knn-bfs" => Ok(Self::KnnBfs(params)),
            "knn-dfs" => Ok(Self::KnnDfs(params)),
            "knn-linear" => Ok(Self::KnnLinear(params)),
            "knn-rrnn" => Ok(Self::KnnRrnn(params)),
            "rnn-chess" => Ok(Self::RnnChess(params)),
            "rnn-linear" => Ok(Self::RnnLinear(params)),
            "approx-knn-dfs" => Ok(Self::ApproxKnnDfs(params)),
            _ => Err(format!("Unknown search algorithm: {alg}")),
        }
    }
}

impl ShellCakes {
    pub fn get<T>(&self) -> Result<Cakes<T>, String>
    where
        T: DistanceValue + 'static,
        <T as FromStr>::Err: std::fmt::Display,
    {
        match self {
            Self::KnnBfs(params) => {
                let k = parse_param(params, "k", "KnnBfs")?;
                Ok(Cakes::KnnBfs(KnnBfs(k)))
            }
            Self::KnnDfs(params) => {
                let k = parse_param(params, "k", "KnnDfs")?;
                Ok(Cakes::KnnDfs(KnnDfs(k)))
            }
            Self::KnnLinear(params) => {
                let k = parse_param(params, "k", "KnnLinear")?;
                Ok(Cakes::KnnLinear(KnnLinear(k)))
            }
            Self::KnnRrnn(params) => {
                let k = parse_param(params, "k", "KnnRrnn")?;
                Ok(Cakes::KnnRrnn(KnnRrnn(k)))
            }
            Self::RnnChess(params) => {
                let r = parse_param(params, "r", "RnnChess")?;
                Ok(Cakes::RnnChess(RnnChess(r)))
            }
            Self::RnnLinear(params) => {
                let r = parse_param(params, "r", "RnnLinear")?;
                Ok(Cakes::RnnLinear(RnnLinear(r)))
            }
            Self::ApproxKnnDfs(params) => {
                let k = parse_param(params, "k", "ApproxKnnDfs")?;
                let d = parse_param(params, "d", "ApproxKnnDfs")?;
                let l = parse_param(params, "l", "ApproxKnnDfs")?;
                Ok(Cakes::ApproxKnnDfs(abd_clam::cakes::approximate::KnnDfs::new(k, l, d)))
            }
        }
    }
}

/// Parse parameter string into key-value pairs
fn parse_parameters(params_str: &str) -> Result<HashMap<String, String>, String> {
    let mut params = HashMap::new();

    if params_str.is_empty() {
        return Err("No search parameters provided".to_string());
    }

    for pair in params_str.split(',') {
        let pair = pair.trim();
        if let Some((key, value)) = pair.split_once('=') {
            params.insert(key.trim().to_string(), value.trim().to_string());
        } else {
            return Err(format!("Invalid parameter format: '{pair}'. Expected 'key=value'"));
        }
    }

    if params.is_empty() {
        return Err("No valid search parameters found".to_string());
    }

    Ok(params)
}

/// Helper function to parse individual parameters into desired type
fn parse_param<T>(params: &HashMap<String, String>, key: &str, alg_name: &str) -> Result<T, String>
where
    T: FromStr,
    T::Err: std::fmt::Display,
{
    let value_str = params.get(key).ok_or_else(|| format!("Missing parameter '{key}' for {alg_name}"))?;
    value_str.parse::<T>().map_err(|e| format!("Invalid value for '{key}' for {alg_name}: {e}"))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::ShellCakes;

    #[test]
    fn test_parse_knn_linear() {
        let query: ShellCakes = "knn-linear:k=3".parse().unwrap();
        let params = HashMap::from([("k".to_string(), "3".to_string())]);
        assert_eq!(query, ShellCakes::KnnLinear(params));
    }

    #[test]
    fn test_parse_knn_repeated_rnn() {
        let query: ShellCakes = "knn-rrnn:k=5".parse().unwrap();
        let params = HashMap::from([("k".to_string(), "5".to_string())]);
        assert_eq!(query, ShellCakes::KnnRrnn(params));
    }

    #[test]
    fn test_parse_rnn() {
        let query: ShellCakes = "rnn-linear:r=2.5".parse().unwrap();
        let params = HashMap::from([("r".to_string(), "2.5".to_string())]);
        assert_eq!(query, ShellCakes::RnnLinear(params));

        let query2: ShellCakes = "rnn-chess:r=1.0".parse().unwrap();
        let params2 = HashMap::from([("r".to_string(), "1.0".to_string())]);
        assert_eq!(query2, ShellCakes::RnnChess(params2));
    }

    #[test]
    fn test_display() {
        let query = ShellCakes::KnnLinear(HashMap::from([("k".to_string(), "3".to_string())]));
        assert_eq!(query.to_string(), "KnnLinear(k=3)");
    }

    #[test]
    fn test_parse_errors() {
        assert!("unknown-algo:k=3".parse::<ShellCakes>().is_err());
        assert!("knn-linear:missing_equals".parse::<ShellCakes>().is_err());
    }

    #[test]
    fn test_search_algorithm_wrapper_creation() {
        // Test that we can create a SearchAlgorithmWrapper from a ShellSearchAlgorithm
        let query = ShellCakes::KnnLinear(HashMap::from([("k".to_string(), "5".to_string())]));
        let alg = query.get::<f64>();

        // Test that the wrapper implements the SearchAlgorithm trait correctly
        assert!(alg.is_ok());
        let alg = alg.unwrap();
        assert_eq!(alg.name(), "KnnLinear(k=5)");
    }

    #[test]
    fn test_rnn_wrapper_creation() {
        let query = ShellCakes::RnnLinear(HashMap::from([("r".to_string(), "2.5".to_string())]));
        let alg = query.get::<f64>();

        assert!(alg.is_ok());
        let alg = alg.unwrap();
        assert_eq!(alg.name(), "RnnLinear(r=2.5)");
    }
}
