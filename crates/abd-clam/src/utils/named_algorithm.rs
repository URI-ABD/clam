//! Some helpful super-traits for various types in the crate.

/// Algorithms that can be named, and whose configuration can be displayed as, and parsed from, a string.
///
/// This is necessary for using the algorithm in any CLI or web interface.
pub trait NamedAlgorithm: core::fmt::Debug + core::fmt::Display + core::str::FromStr {
    /// Returns the name of the algorithm.
    fn name(&self) -> &'static str;

    /// Returns a regex pattern that can be used to parse the algorithm from a string.
    ///
    /// This should use the [`lazy_regex`] crate to compile the regex pattern at compile time. The pattern should be designed to capture the name and any
    /// parameters of the algorithm, so that the [`FromStr`](core::str::FromStr) implementation can use the captures to construct the algorithm instance. The
    /// [`lazy_regex::regex!`] macro is likely the easiest way to create the regex pattern.
    fn regex_pattern<'a>() -> &'a lazy_regex::Regex;
}

/// A macro for implementing the `NamedAlgorithm` trait for a unit struct.
macro_rules! impl_named_algorithm_for_unit_struct {
    ($name:ident, $str_name:expr, $template:expr) => {
        impl ::core::fmt::Display for $name {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                write!(f, "{}", $str_name)
            }
        }

        impl ::core::str::FromStr for $name {
            type Err = String;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                if s == $str_name {
                    Ok(Self)
                } else {
                    Err(format!("String '{}' does not match expected format '{}'", s, $str_name))
                }
            }
        }

        impl NamedAlgorithm for $name {
            fn name(&self) -> &'static str {
                $str_name
            }

            #[expect(clippy::trivial_regex)]
            fn regex_pattern<'a>() -> &'a lazy_regex::Regex {
                lazy_regex::regex!($template)
            }
        }
    };
}

/// A macro for implementing the `NamedAlgorithm` trait for a Knn-search algorithm tuple struct with a single `k` parameter.
macro_rules! impl_named_algorithm_for_exact_knn {
    ($name:ident, $str_name:expr, $template:expr) => {
        impl ::core::fmt::Display for $name {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                write!(f, "{}::k={}", self.name(), self.k)
            }
        }

        impl ::core::str::FromStr for $name {
            type Err = String;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                let template = lazy_regex::regex!($template);
                if let Some(captures) = template.captures(s) {
                    let k_str = captures.get(1).ok_or_else(|| format!("Missing 'k' value in '{s}'"))?.as_str();
                    let k = k_str.parse::<usize>().map_err(|e| format!("Failed to parse 'k' from '{k_str}': {e}"))?;
                    Ok(Self { k })
                } else {
                    Err(format!("String '{s}' does not match expected format '{}::k=<k>'", $str_name))
                }
            }
        }

        impl NamedAlgorithm for $name {
            fn name(&self) -> &'static str {
                $str_name
            }

            fn regex_pattern<'a>() -> &'a lazy_regex::Regex {
                lazy_regex::regex!($template)
            }
        }
    };
}

/// A macro for implementing the `NamedAlgorithm` trait for a Rnn-search algorithm tuple struct with a single `radius` parameter.
macro_rules! impl_named_algorithm_for_exact_rnn {
    ($name:ident, $str_name:expr, $template:expr) => {
        impl<T: crate::DistanceValue> ::core::fmt::Display for $name<T> {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                write!(f, "{}::radius={}", self.name(), self.radius)
            }
        }

        impl<T: crate::DistanceValue> ::core::str::FromStr for $name<T> {
            type Err = String;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                let template = lazy_regex::regex!($template);
                if let Some(captures) = template.captures(s) {
                    let radius_str = captures.get(1).ok_or_else(|| format!("Missing 'radius' value in '{s}'"))?.as_str();
                    let radius = T::from_str_radix(radius_str, 10).map_err(|_| format!("Failed to parse 'radius' from '{radius_str}'"))?;
                    Ok(Self { radius })
                } else {
                    Err(format!("String '{s}' does not match expected format '{}::radius=<radius>'", $str_name))
                }
            }
        }

        impl<T: crate::DistanceValue> NamedAlgorithm for $name<T> {
            fn name(&self) -> &'static str {
                $str_name
            }

            fn regex_pattern<'a>() -> &'a lazy_regex::Regex {
                lazy_regex::regex!($template)
            }
        }
    };
}

/// A macro for implementing the `NamedAlgorithm` trait for an approximate Knn-search algorithm struct with parameters `k` and `tol`.
macro_rules! impl_named_algorithm_for_approx_knn {
    ($name:ident, $str_name:expr, $template:expr) => {
        impl ::core::fmt::Display for $name {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                write!(f, "{}::k={},tol={}", self.name(), self.k, self.tol)
            }
        }

        impl ::core::str::FromStr for $name {
            type Err = String;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                let template = lazy_regex::regex!($template);
                if let Some(captures) = template.captures(s) {
                    let k_str = captures.get(1).ok_or_else(|| format!("Missing 'k' value in '{s}'"))?.as_str();
                    let tol_str = captures.get(2).ok_or_else(|| format!("Missing 'tol' value in '{s}'"))?.as_str();

                    let k = k_str.parse::<usize>().map_err(|e| format!("Failed to parse 'k' from '{k_str}': {e}"))?;
                    let tol = tol_str
                        .parse::<f64>()
                        .map_err(|e| format!("Failed to parse 'tol' from '{tol_str}': {e}"))?;

                    Ok(Self::new(k, tol))
                } else {
                    Err(format!("String '{s}' does not match expected format '{}::k=<k>,tol=<tol>'", $str_name))
                }
            }
        }

        impl NamedAlgorithm for $name {
            fn name(&self) -> &'static str {
                $str_name
            }

            fn regex_pattern<'a>() -> &'a lazy_regex::Regex {
                lazy_regex::regex!($template)
            }
        }
    };
}
