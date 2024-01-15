//! This file contains the trained meta-ml functions used for the first CHAODA
//! paper. The meta-ml models were first trained in Python and the decision
//! functions were extracted and then translated into Rust.

use crate::core::cluster::Ratios;

/// Returns a list of available `MetaML` scorers as tuples containing their names and boxed functions.
///
/// # Returns
///
/// A `Vec` of tuples, where each tuple consists of a `&str` representing the scorer name and a boxed
/// `MetaMLScorer` function.
///
pub fn get_meta_ml_scorers<'a>() -> Vec<(&'a str, crate::core::graph::MetaMLScorer)> {
    vec![
        ("lr_manhattan_sc", Box::new(lr_manhattan_sc)),
        ("lr_manhattan_cc", Box::new(lr_manhattan_cc)),
        ("lr_manhattan_gn", Box::new(lr_manhattan_gn)),
        ("lr_manhattan_cr", Box::new(lr_manhattan_cr)),
        ("lr_manhattan_sp", Box::new(lr_manhattan_sp)),
        ("lr_manhattan_vd", Box::new(lr_manhattan_vd)),
        ("lr_euclidean_cc", Box::new(lr_euclidean_cc)),
        ("lr_euclidean_sc", Box::new(lr_euclidean_sc)),
        ("lr_euclidean_gn", Box::new(lr_euclidean_gn)),
        ("lr_euclidean_cr", Box::new(lr_euclidean_cr)),
        ("lr_euclidean_sp", Box::new(lr_euclidean_sp)),
        ("lr_euclidean_vd", Box::new(lr_euclidean_vd)),
        ("dt_manhattan_cc", Box::new(dt_manhattan_cc)),
        ("dt_manhattan_sc", Box::new(dt_manhattan_sc)),
        ("dt_manhattan_gn", Box::new(dt_manhattan_gn)),
        ("dt_manhattan_cr", Box::new(dt_manhattan_cr)),
        ("dt_manhattan_sp", Box::new(dt_manhattan_sp)),
        ("dt_manhattan_vd", Box::new(dt_manhattan_vd)),
        ("dt_euclidean_cc", Box::new(dt_euclidean_cc)),
        ("dt_euclidean_sc", Box::new(dt_euclidean_sc)),
        ("dt_euclidean_gn", Box::new(dt_euclidean_gn)),
        ("dt_euclidean_cr", Box::new(dt_euclidean_cr)),
        ("dt_euclidean_sp", Box::new(dt_euclidean_sp)),
        ("dt_euclidean_vd", Box::new(dt_euclidean_vd)),
    ]
}

/// Calculate cluster scores for 'Cluster Cardinality' graph analysis.
///
/// This function uses coefficients obtained from a Linear Regression machine learning model and
/// is designed for clusters partitioned using the Manhattan distance metric.
///
/// # Arguments
///
/// * `ratios`: An array of 6 floats representing child-parent ratios and their exponential moving averages along a branch of the tree.
///
/// # Returns
///
/// A floating-point value representing the cluster score. Higher scores indicate clusters that are better suited for inclusion in the 'Cluster Cardinality' graph.
fn lr_manhattan_cc(ratios: Ratios) -> f64 {
    let a = [
        1.228_800e-01,
        1.227_127e-01,
        7.808_845e-02,
        4.154_137e-02,
        5.657_729e-02,
        3.525_646e-02,
    ];
    a.iter().zip(ratios.iter()).map(|(x, y)| x * y).sum()
}

/// Calculate cluster scores for 'Component Cardinality' graph analysis.
///
/// This function uses coefficients obtained from a Linear Regression machine learning model and
/// is designed for clusters partitioned using the Manhattan distance metric.
///
/// # Arguments
///
/// * `ratios`: An array of 6 floats representing the relationship between a cluster and its parent.
///
/// # Returns
///
/// A floating-point value representing the cluster score. Higher scores indicate clusters that are better suited for inclusion in the 'Component Cardinality' graph.
fn lr_manhattan_sc(ratios: Ratios) -> f64 {
    let a = [
        1.429_870e-02,
        -1.323_484e-02,
        -1.150_261e-02,
        3.896_381e-02,
        4.082_664e-02,
        -1.364_604e-02,
    ];
    a.iter().zip(ratios.iter()).map(|(x, y)| x * y).sum()
}

/// Calculate cluster scores for 'Graph Neighborhood' graph analysis.
///
/// This function uses coefficients obtained from a Linear Regression machine learning model and
/// is designed for clusters partitioned using the Manhattan distance metric.
///
/// # Arguments
///
/// * `ratios`: An array of 6 floats representing the relationship between a cluster and its parent.
///
/// # Returns
///
/// A floating-point value representing the cluster score. Higher scores indicate clusters that are better suited for inclusion in the 'Graph Neighborhood' graph.
fn lr_manhattan_gn(ratios: Ratios) -> f64 {
    let a = [
        8.262_898e-03,
        1.537_685e-02,
        9.422_306e-03,
        3.740_549e-02,
        3.891_843e-02,
        -1.250_707e-02,
    ];
    a.iter().zip(ratios.iter()).map(|(x, y)| x * y).sum()
}

/// Calculate cluster scores for 'Parent Cardinality' graph analysis.
///
/// This function uses coefficients obtained from a Linear Regression machine learning model and
/// is intended for clusters partitioned using the Manhattan distance metric.
///
/// # Arguments
///
/// * `ratios`: An array of 6 floats representing the relationship between a cluster and its parent.
///
/// # Returns
///
/// A floating-point value representing the cluster score. Higher scores indicate clusters that are better suited for inclusion in the 'Parent Cardinality' graph.
fn lr_manhattan_cr(ratios: Ratios) -> f64 {
    let a = [
        8.433_946e-02,
        6.050_625e-02,
        5.882_554e-02,
        3.593_437e-03,
        4.128_473e-02,
        -4.736_846e-02,
    ];
    a.iter().zip(ratios.iter()).map(|(x, y)| x * y).sum()
}

/// Calculate cluster scores for 'Stationary Probabilities' graph analysis.
///
/// This function uses coefficients obtained from a Linear Regression machine learning model and
/// is designed for clusters partitioned using the Manhattan distance metric.
///
/// # Arguments
///
/// * `ratios`: An array of 6 floats representing the relationship between a cluster and its parent.
///
/// # Returns
///
/// A floating-point value representing the cluster score. Higher scores indicate clusters that are better suited for inclusion in the 'Stationary Probabilities' graph.
fn lr_manhattan_sp(ratios: Ratios) -> f64 {
    let a = [
        4.659_433e-02,
        -5.014_006e-02,
        -6.017_402e-02,
        5.812_719e-02,
        1.466_290e-01,
        1.266_893e-03,
    ];
    a.iter().zip(ratios.iter()).map(|(x, y)| x * y).sum()
}

/// Calculate cluster scores for 'Vertex Degree' graph analysis.
///
/// This function uses coefficients obtained from a Linear Regression machine learning model and
/// is designed for clusters partitioned using the Manhattan distance metric.
///
/// # Arguments
///
/// * `ratios`: An array of 6 floats representing the relationship between a cluster and its parent.
///
/// # Returns
///
/// A floating-point value representing the cluster score. Higher scores indicate clusters that are better suited for inclusion in the 'Vertex Degree' graph.
fn lr_manhattan_vd(ratios: Ratios) -> f64 {
    let a = [
        1.032_663e-01,
        1.232_432e-01,
        7.317_461e-02,
        1.084_027e-02,
        9.541_312e-02,
        1.760_110e-02,
    ];
    a.iter().zip(ratios.iter()).map(|(x, y)| x * y).sum()
}

/// Calculate cluster scores for 'Cluster Cardinality' graph analysis.
///
/// This function uses coefficients obtained from a Linear Regression machine learning model and
/// is designed for clusters partitioned using the Euclidean distance metric.
///
/// # Arguments
///
/// * `ratios`: An array of 6 floats representing the relationship between a cluster and its parent.
///
/// # Returns
///
/// A floating-point value representing the cluster score. Higher scores indicate clusters that are better suited for inclusion in the 'Cluster Cardinality' graph.
fn lr_euclidean_cc(ratios: Ratios) -> f64 {
    let a = [
        1.313_924e-01,
        1.326_884e-01,
        9.136_274e-02,
        2.134_787e-02,
        3.100_747e-02,
        3.298_891e-02,
    ];
    a.iter().zip(ratios.iter()).map(|(x, y)| x * y).sum()
}

/// Calculate cluster scores for 'Component Cardinality' graph analysis.
///
/// This function uses coefficients obtained from a Linear Regression machine learning model and
/// is designed for clusters partitioned using the Euclidean distance metric.
///
/// # Arguments
///
/// * `ratios`: An array of 6 floats representing the relationship between a cluster and its parent.
///
/// # Returns
///
/// A floating-point value representing the cluster score. Higher scores indicate clusters that are better suited for inclusion in the 'Component Cardinality' graph.
fn lr_euclidean_sc(ratios: Ratios) -> f64 {
    let a = [
        5.582_536e-02,
        2.442_987e-02,
        8.037_801e-03,
        1.539_072e-02,
        3.654_952e-02,
        1.429_881e-02,
    ];
    a.iter().zip(ratios.iter()).map(|(x, y)| x * y).sum()
}

/// Calculate cluster scores for 'Graph Neighborhood' graph analysis.
///
/// This function uses coefficients obtained from a Linear Regression machine learning model and
/// is designed for clusters partitioned using the Euclidean distance metric.
///
/// # Arguments
///
/// * `ratios`: An array of 6 floats representing the relationship between a cluster and its parent.
///
/// # Returns
///
/// A floating-point value representing the cluster score. Higher scores indicate clusters that are better suited for inclusion in the 'Graph Neighborhood' graph.
fn lr_euclidean_gn(ratios: Ratios) -> f64 {
    let a = [
        9.585_250e-02,
        6.025_230e-02,
        3.753_800e-02,
        -3.118_970e-03,
        7.559_676e-02,
        1.875_789e-02,
    ];
    a.iter().zip(ratios.iter()).map(|(x, y)| x * y).sum()
}

/// Calculate cluster scores for 'Parent Cardinality' graph analysis.
///
/// This function uses coefficients obtained from a Linear Regression machine learning model and
/// is designed for clusters partitioned using the Euclidean distance metric.
///
/// # Arguments
///
/// * `ratios`: An array of 6 floats representing the relationship between a cluster and its parent.
///
/// # Returns
///
/// A floating-point value representing the cluster score. Higher scores indicate clusters that are better suited for inclusion in the 'Parent Cardinality' graph.
fn lr_euclidean_cr(ratios: Ratios) -> f64 {
    let a = [
        9.211_533e-02,
        7.063_269e-02,
        2.331_588e-02,
        7.911_250e-04,
        1.672_235e-02,
        3.891_130e-03,
    ];
    a.iter().zip(ratios.iter()).map(|(x, y)| x * y).sum()
}

/// Calculate cluster scores for 'Stationary Probabilities' graph analysis.
///
/// This function uses coefficients obtained from a Linear Regression machine learning model and
/// is designed for clusters partitioned using the Euclidean distance metric.
///
/// # Arguments
///
/// * `ratios`: An array of 6 floats representing the relationship between a cluster and its parent.
///
/// # Returns
///
/// A floating-point value representing the cluster score. Higher scores indicate clusters that are better suited for inclusion in the 'Stationary Probabilities' graph.
fn lr_euclidean_sp(ratios: Ratios) -> f64 {
    let a = [
        8.131_170e-02,
        1.629_200e-02,
        -4.917_042e-02,
        3.507_954e-02,
        -1.800_446e-03,
        -5.963_697e-03,
    ];
    a.iter().zip(ratios.iter()).map(|(x, y)| x * y).sum()
}

/// Calculate cluster scores for 'Vertex Degree' graph analysis.
///
/// This function uses coefficients obtained from a Linear Regression machine learning model and
/// is designed for clusters partitioned using the Euclidean distance metric.
///
/// # Arguments
///
/// * `ratios`: An array of 6 floats representing the relationship between a cluster and its parent.
///
/// # Returns
///
/// A floating-point value representing the cluster score. Higher scores indicate clusters that are better suited for inclusion in the 'Vertex Degree' graph.
fn lr_euclidean_vd(ratios: Ratios) -> f64 {
    let a = [
        1.096_380e-01,
        1.658_747e-01,
        1.045_492e-01,
        -1.207_694e-03,
        6.186_889e-02,
        2.726_535e-02,
    ];
    a.iter().zip(ratios.iter()).map(|(x, y)| x * y).sum()
}

/// Calculate cluster scores for 'Cluster Cardinality' graph analysis.
///
/// This function uses coefficients obtained from a Decision Tree machine learning model and
/// is designed for clusters partitioned using the Manhattan distance metric.
///
/// # Arguments
///
/// * `ratios`: An array of 6 floats representing the relationship between a cluster and its parent.
///
/// # Returns
///
/// A floating-point value representing the cluster score. Higher scores indicate clusters that are better suited for inclusion in the 'Cluster Cardinality' graph.
fn dt_manhattan_cc(ratios: Ratios) -> f64 {
    let [cardinality, radius, _, cardinality_ema, _, lfd_ema] = ratios;
    if radius <= 1.391_490e-02 {
        if lfd_ema <= 6.940_212e-01 {
            if radius <= 1.341_821e-02 {
                5.454_838e-01
            } else {
                3.113_860e-01
            }
        } else if cardinality <= 2.564_081e-02 {
            3.508_578e-01
        } else {
            7.560_313e-01
        }
    } else if cardinality_ema <= 3.965_033e-01 {
        if cardinality <= 1.411_780e-01 {
            8.795_414e-01
        } else {
            9.287_172e-01
        }
    } else if cardinality_ema <= 7.348_774e-01 {
        9.599_685e-01
    } else {
        9.889_817e-01
    }
}

/// Calculate cluster scores for 'Sub Graph Cardinality' graph analysis.
///
/// This function uses coefficients obtained from a Decision Tree machine learning model and
/// is designed for clusters partitioned using the Manhattan distance metric.
///
/// # Arguments
///
/// * `_ratios`: An array of 6 floats representing the relationship between a cluster and its parent.
///
/// # Returns
///
/// A floating-point value representing the cluster score. Higher scores indicate clusters that are better suited for inclusion in the 'Sub Graph Cardinality' graph.
fn dt_manhattan_sc(ratios: Ratios) -> f64 {
    let [_, radius, _, cardinality_ema, _, lfd_ema] = ratios;
    if radius <= 5.148_810e-03 {
        7.500_000e-01
    } else if cardinality_ema <= 6.438_965e-01 {
        if lfd_ema <= 9.120_842e-01 {
            9.709_770e-01
        } else {
            9.303_896e-01
        }
    } else if lfd_ema <= 8.965_069e-01 {
        9.982_470e-01
    } else {
        9.858_773e-01
    }
}

/// Calculate cluster scores for 'Graph Neighborhood' graph analysis.
///
/// This function uses coefficients obtained from a Decision Tree machine learning model and
/// is designed for clusters partitioned using the Manhattan distance metric.
///
/// # Arguments
///
/// * `_ratios`: An array of 6 floats representing the relationship between a cluster and its parent.
///
/// # Returns
///
/// A floating-point value representing the cluster score. Higher scores indicate clusters that are better suited for inclusion in the 'Graph Neighborhood' graph.
fn dt_manhattan_gn(ratios: Ratios) -> f64 {
    let [cardinality, radius, _, cardinality_ema, _, _] = ratios;
    if radius <= 5.148_810e-03 {
        if cardinality <= 2.540_494e-02 {
            4.437_313e-01
        } else {
            7.500_000e-01
        }
    } else if cardinality_ema <= 6.354_841e-01 {
        if cardinality_ema <= 6.253_555e-01 {
            9.625_763e-01
        } else {
            8.007_593e-01
        }
    } else if cardinality_ema <= 8.838_843e-01 {
        9.887_261e-01
    } else {
        9.979_190e-01
    }
}

/// Calculate cluster scores for 'Parent Cardinality' graph analysis.
///
/// This function uses coefficients obtained from a Decision Tree machine learning model and
/// is designed for clusters partitioned using the Manhattan distance metric.
///
/// # Arguments
///
/// * `_ratios`: An array of 6 floats representing the relationship between a cluster and its parent.
///
/// # Returns
///
/// A floating-point value representing the cluster score. Higher scores indicate clusters that are better suited for inclusion in the 'Parent Cardinality' graph.
fn dt_manhattan_cr(ratios: Ratios) -> f64 {
    let [cardinality, radius, _, cardinality_ema, _, lfd_ema] = ratios;
    if radius <= 1.472_459e-02 {
        if cardinality <= 2.505_734e-02 {
            if radius <= 1.655_350e-04 {
                3.151_504e-01
            } else {
                6.447_252e-01
            }
        } else if cardinality <= 1.076_636e-01 {
            7.414_646e-01
        } else {
            9.229_898e-01
        }
    } else if cardinality <= 5.942_344e-01 {
        if lfd_ema <= 9.527_803e-01 {
            9.468_785e-01
        } else {
            8.911_853e-01
        }
    } else if cardinality_ema <= 9.790_418e-01 {
        9.705_729e-01
    } else {
        9.989_975e-01
    }
}

/// Calculate cluster scores for 'Stationary Probabilities' graph analysis.
///
/// This function uses coefficients obtained from a Decision Tree machine learning model and
/// is designed for clusters partitioned using the Manhattan distance metric.
///
/// # Arguments
///
/// * `_ratios`: An array of 6 floats representing the relationship between a cluster and its parent.
///
/// # Returns
///
/// A floating-point value representing the cluster score. Higher scores indicate clusters that are better suited for inclusion in the 'Stationary Probabilities' graph.
fn dt_manhattan_sp(ratios: Ratios) -> f64 {
    let [_, radius, _, cardinality_ema, radius_ema, _] = ratios;
    if radius <= 2.358_828e-04 {
        if radius_ema <= 1.177_303e-02 {
            if radius_ema <= 8.709_679e-05 {
                1.400_713e-01
            } else {
                1.698_153e-01
            }
        } else {
            4.245_149e-01
        }
    } else if cardinality_ema <= 1.807_206e-02 {
        if radius_ema <= 7.669_807e-01 {
            5.307_418e-01
        } else {
            9.984_011e-01
        }
    } else if cardinality_ema <= 4.135_659e-01 {
        9.372_930e-01
    } else {
        9.907_336e-01
    }
}

/// Calculate cluster scores for 'Vertex Degree' graph analysis.
///
/// This function uses coefficients obtained from a Decision Tree machine learning model and
/// is designed for clusters partitioned using the Manhattan distance metric.
///
/// # Arguments
///
/// * `_ratios`: An array of 6 floats representing the relationship between a cluster and its parent.
///
/// # Returns
///
/// A floating-point value representing the cluster score. Higher scores indicate clusters that are better suited for inclusion in the 'Vertex Degree' graph.
fn dt_manhattan_vd(ratios: Ratios) -> f64 {
    let [cardinality, radius, _, cardinality_ema, radius_ema, lfd_ema] = ratios;
    if radius <= 1.391_490e-02 {
        if lfd_ema <= 6.932_603e-01 {
            if cardinality <= 1.076_636e-01 {
                4.198_158e-01
            } else {
                8.463_385e-01
            }
        } else if cardinality <= 2.564_081e-02 {
            3.269_299e-01
        } else {
            7.921_413e-01
        }
    } else if cardinality_ema <= 7.950_329e-01 {
        if cardinality <= 4.997_393e-01 {
            9.251_187e-01
        } else {
            9.570_666e-01
        }
    } else if radius_ema <= 4.155_880e-01 {
        9.516_691e-01
    } else {
        9.928_017e-01
    }
}

/// Calculate cluster scores for 'Cluster Cardinality' graph analysis.
///
/// This function uses coefficients obtained from a Decision Tree machine learning model and
/// is designed for clusters partitioned using the Euclidean distance metric.
///
/// # Arguments
///
/// * `_ratios`: An array of 6 floats representing the relationship between a cluster and its parent.
///
/// # Returns
///
/// A floating-point value representing the cluster score. Higher scores indicate clusters that are better suited for inclusion in the 'Cluster Cardinality' graph.
fn dt_euclidean_cc(ratios: Ratios) -> f64 {
    let [cardinality, radius, lfd, cardinality_ema, _, lfd_ema] = ratios;
    if radius <= 1.067_198e-02 {
        if cardinality <= 1.079_019e-01 {
            if lfd_ema <= 9.999_740e-01 {
                5.265_128e-01
            } else {
                8.834_613e-01
            }
        } else if cardinality <= 3.330_053e-01 {
            8.695_605e-01
        } else {
            9.694_478e-01
        }
    } else if cardinality <= 4.997_283e-01 {
        if lfd <= 2.929_797e-01 {
            8.696_701e-01
        } else {
            9.258_335e-01
        }
    } else if cardinality_ema <= 7.698_584e-01 {
        9.665_650e-01
    } else {
        9.917_949e-01
    }
}

/// Calculate cluster scores for 'Component (Sub graph) Cardinality' graph analysis.
///
/// This function uses coefficients obtained from a Decision Tree machine learning model and
/// is designed for clusters partitioned using the Euclidean distance metric.
///
/// # Arguments
///
/// * `_ratios`: An array of 6 floats representing the relationship between a cluster and its parent.
///
/// # Returns
///
/// A floating-point value representing the cluster score. Higher scores indicate clusters that are better suited for inclusion in the 'Component (Sub graph) Cardinality' graph.
fn dt_euclidean_sc(ratios: Ratios) -> f64 {
    let [cardinality, radius, _, cardinality_ema, radius_ema, _] = ratios;
    if radius_ema <= 5.111_776e-03 {
        if radius <= 9.307_770e-05 {
            7.500_000e-01
        } else {
            6.997_932e-01
        }
    } else if cardinality <= 1.494_950e-01 {
        if cardinality_ema <= 4.477_562e-02 {
            8.816_036e-01
        } else {
            9.351_654e-01
        }
    } else if cardinality_ema <= 6.924_688e-01 {
        9.702_070e-01
    } else {
        9.932_433e-01
    }
}

/// Calculate cluster scores for 'Graph Neighborhood' graph analysis.
///
/// This function uses coefficients obtained from a Decision Tree machine learning model and
/// is designed for clusters partitioned using the Euclidean distance metric.
///
/// # Arguments
///
/// * `_ratios`: An array of 6 floats representing the relationship between a cluster and its parent.
///
/// # Returns
///
/// A floating-point value representing the cluster score. Higher scores indicate clusters that are better suited for inclusion in the 'Graph Neighborhood' graph.
fn dt_euclidean_gn(ratios: Ratios) -> f64 {
    let [_, _, lfd, cardinality_ema, radius_ema, _] = ratios;
    if radius_ema <= 2.359_367e-05 {
        if lfd <= 3.415_343e-01 {
            1.299_296e-01
        } else {
            4.405_503e-01
        }
    } else if cardinality_ema <= 1.619_149e-02 {
        if cardinality_ema <= 3.227_095e-03 {
            5.649_648e-01
        } else {
            7.366_586e-01
        }
    } else if radius_ema <= 5.773_949e-02 {
        8.322_585e-01
    } else {
        9.700_278e-01
    }
}

/// Calculate cluster scores for 'Parent Cardinality' graph analysis.
///
/// This function uses coefficients obtained from a Decision Tree machine learning model and
/// is designed for clusters partitioned using the Euclidean distance metric.
///
/// # Arguments
///
/// * `_ratios`: An array of 6 floats representing the relationship between a cluster and its parent.
///
/// # Returns
///
/// A floating-point value representing the cluster score. Higher scores indicate clusters that are better suited for inclusion in the 'Parent Cardinality' graph.
fn dt_euclidean_cr(ratios: Ratios) -> f64 {
    let [cardinality, radius, lfd, _, _, _] = ratios;
    if radius <= 1.067_198e-02 {
        if cardinality <= 2.828_062e-02 {
            if radius <= 9.307_770e-05 {
                3.440_542e-01
            } else {
                6.220_238e-01
            }
        } else if lfd <= 2.580_366e-01 {
            5.624_770e-01
        } else {
            8.176_447e-01
        }
    } else if cardinality <= 3.860_220e-02 {
        if cardinality <= 3.826_937e-02 {
            8.636_209e-01
        } else {
            5.281_209e-01
        }
    } else if cardinality <= 4.997_400e-01 {
        9.494_346e-01
    } else {
        9.803_474e-01
    }
}

/// Calculate cluster scores for 'Stationary Probabilities' graph analysis.
///
/// This function uses coefficients obtained from a Decision Tree machine learning model and
/// is designed for clusters partitioned using the Euclidean distance metric.
///
/// # Arguments
///
/// * `_ratios`: An array of 6 floats representing the relationship between a cluster and its parent.
///
/// # Returns
///
/// A floating-point value representing the cluster score. Higher scores indicate clusters that are better suited for inclusion in the 'Stationary Probabilities' graph.
fn dt_euclidean_sp(ratios: Ratios) -> f64 {
    let [cardinality, _, lfd, cardinality_ema, radius_ema, lfd_ema] = ratios;
    if radius_ema <= 3.118_177e-02 {
        if lfd <= 8.090_920e-01 {
            if lfd_ema <= 2.858_897e-01 {
                4.526_594e-01
            } else {
                8.478_856e-01
            }
        } else if lfd_ema <= 6.024_930e-01 {
            5.471_349e-03
        } else {
            2.958_559e-01
        }
    } else if cardinality <= 1.494_950e-01 {
        if radius_ema <= 9.417_503e-01 {
            9.296_579e-01
        } else {
            7.623_177e-01
        }
    } else if cardinality_ema <= 1.749_870e-02 {
        7.646_179e-01
    } else {
        9.863_799e-01
    }
}

/// Calculate cluster scores for 'Vertex Degree' graph analysis.
///
/// This function uses coefficients obtained from a Decision Tree machine learning model and
/// is designed for clusters partitioned using the Euclidean distance metric.
///
/// # Arguments
///
/// * `_ratios`: An array of 6 floats representing the relationship between a cluster and its parent.
///
/// # Returns
///
/// A floating-point value representing the cluster score. Higher scores indicate clusters that are better suited for inclusion in the 'Vertex Degree' graph.
fn dt_euclidean_vd(ratios: Ratios) -> f64 {
    let [cardinality, radius, lfd, _, radius_ema, lfd_ema] = ratios;
    if radius <= 1.067_198e-02 {
        if lfd <= 9.634_169e-01 {
            if lfd_ema <= 9.999_111e-01 {
                4.713_993e-01
            } else {
                9.783_193e-01
            }
        } else if radius <= 1.020_328e-02 {
            9.089_947e-01
        } else {
            4.464_702e-01
        }
    } else if cardinality <= 4.997_769e-01 {
        if lfd <= 9.999_996e-01 {
            9.335_233e-01
        } else {
            7.793_101e-01
        }
    } else if radius_ema <= 7.430_525e-01 {
        9.627_948e-01
    } else {
        9.921_575e-01
    }
}
