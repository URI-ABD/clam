//! This file contains the trained meta-ml functions used for the first CHAODA
//! paper. The meta-ml models were first trained in Python and the decision
//! functions were extracted and then translated into Rust.

use super::Ratios;

/// Returns a list of available `MetaML` scorers as tuples containing their names and boxed functions.
///
/// # Returns
///
/// A `Vec` of tuples, where each tuple consists of a `&str` representing the scorer name and a boxed
/// `MetaMLScorer` function.
///
pub fn get_meta_ml_scorers<'a>() -> Vec<(&'a str, super::graph::MetaMLScorer)> {
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
fn dt_manhattan_sc(_ratios: Ratios) -> f64 {
    todo!()
    // let [_, radius, _, cardinality_ema, _, lfd_ema] = ratios;
    // if radius <= 5.148810e-03 {
    //     7.500000e-01
    // } else if cardinality_ema <= 6.438965e-01 {
    //     if lfd_ema <= 9.120842e-01 {
    //         9.709770e-01
    //     } else {
    //         9.303896e-01
    //     }
    // } else if lfd_ema <= 8.965069e-01 {
    //     9.982470e-01
    // } else {
    //     9.858773e-01
    // }
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
fn dt_manhattan_gn(_ratios: Ratios) -> f64 {
    todo!()

    // let [cardinality, radius, _, cardinality_ema, _, _] = ratios;
    // if radius <= 5.148810e-03 {
    //     if cardinality <= 2.540494e-02 {
    //         4.437313e-01
    //     } else {
    //         7.500000e-01
    //     }
    // } else if cardinality_ema <= 6.354841e-01 {
    //     if cardinality_ema <= 6.253555e-01 {
    //         9.625763e-01
    //     } else {
    //         8.007593e-01
    //     }
    // } else if cardinality_ema <= 8.838843e-01 {
    //     9.887261e-01
    // } else {
    //     9.979190e-01
    // }
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
fn dt_manhattan_cr(_ratios: Ratios) -> f64 {
    todo!()

    // let [cardinality, radius, _, cardinality_ema, _, lfd_ema] = ratios;
    // if radius <= 1.472459e-02 {
    //     if cardinality <= 2.505734e-02 {
    //         if radius <= 1.655350e-04 {
    //             3.151504e-01
    //         } else {
    //             6.447252e-01
    //         }
    //     } else if cardinality <= 1.076636e-01 {
    //         7.414646e-01
    //     } else {
    //         9.229898e-01
    //     }
    // } else if cardinality <= 5.942344e-01 {
    //     if lfd_ema <= 9.527803e-01 {
    //         9.468785e-01
    //     } else {
    //         8.911853e-01
    //     }
    // } else if cardinality_ema <= 9.790418e-01 {
    //     9.705729e-01
    // } else {
    //     9.989975e-01
    // }
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
fn dt_manhattan_sp(_ratios: Ratios) -> f64 {
    todo!()

    // let [_, radius, _, cardinality_ema, radius_ema, _] = ratios;
    // if radius <= 2.358828e-04 {
    //     if radius_ema <= 1.177303e-02 {
    //         if radius_ema <= 8.709679e-05 {
    //             1.400713e-01
    //         } else {
    //             1.698153e-01
    //         }
    //     } else {
    //         4.245149e-01
    //     }
    // } else if cardinality_ema <= 1.807206e-02 {
    //     if radius_ema <= 7.669807e-01 {
    //         5.307418e-01
    //     } else {
    //         9.984011e-01
    //     }
    // } else if cardinality_ema <= 4.135659e-01 {
    //     9.372930e-01
    // } else {
    //     9.907336e-01
    // }
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
fn dt_manhattan_vd(_ratios: Ratios) -> f64 {
    todo!()

    // let [cardinality, radius, _, cardinality_ema, radius_ema, lfd_ema] = ratios;
    // if radius <= 1.391490e-02 {
    //     if lfd_ema <= 6.932603e-01 {
    //         if cardinality <= 1.076636e-01 {
    //             4.198158e-01
    //         } else {
    //             8.463385e-01
    //         }
    //     } else if cardinality <= 2.564081e-02 {
    //         3.269299e-01
    //     } else {
    //         7.921413e-01
    //     }
    // } else if cardinality_ema <= 7.950329e-01 {
    //     if cardinality <= 4.997393e-01 {
    //         9.251187e-01
    //     } else {
    //         9.570666e-01
    //     }
    // } else if radius_ema <= 4.155880e-01 {
    //     9.516691e-01
    // } else {
    //     9.928017e-01
    // }
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
fn dt_euclidean_cc(_ratios: Ratios) -> f64 {
    todo!()

    // let [cardinality, radius, lfd, cardinality_ema, _, lfd_ema] = ratios;
    // if radius <= 1.067198e-02 {
    //     if cardinality <= 1.079019e-01 {
    //         if lfd_ema <= 9.999740e-01 {
    //             5.265128e-01
    //         } else {
    //             8.834613e-01
    //         }
    //     } else if cardinality <= 3.330053e-01 {
    //         8.695605e-01
    //     } else {
    //         9.694478e-01
    //     }
    // } else if cardinality <= 4.997283e-01 {
    //     if lfd <= 2.929797e-01 {
    //         8.696701e-01
    //     } else {
    //         9.258335e-01
    //     }
    // } else if cardinality_ema <= 7.698584e-01 {
    //     9.665650e-01
    // } else {
    //     9.917949e-01
    // }
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
fn dt_euclidean_sc(_ratios: Ratios) -> f64 {
    todo!()

    // let [cardinality, radius, _, cardinality_ema, radius_ema, _] = ratios;
    // if radius_ema <= 5.111776e-03 {
    //     if radius <= 9.307770e-05 {
    //         7.500000e-01
    //     } else {
    //         6.997932e-01
    //     }
    // } else if cardinality <= 1.494950e-01 {
    //     if cardinality_ema <= 4.477562e-02 {
    //         8.816036e-01
    //     } else {
    //         9.351654e-01
    //     }
    // } else if cardinality_ema <= 6.924688e-01 {
    //     9.702070e-01
    // } else {
    //     9.932433e-01
    // }
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
fn dt_euclidean_gn(_ratios: Ratios) -> f64 {
    todo!()

    // let [_, _, lfd, cardinality_ema, radius_ema, _] = ratios;
    // if radius_ema <= 2.359367e-05 {
    //     if lfd <= 3.415343e-01 {
    //         1.299296e-01
    //     } else {
    //         4.405503e-01
    //     }
    // } else if cardinality_ema <= 1.619149e-02 {
    //     if cardinality_ema <= 3.227095e-03 {
    //         5.649648e-01
    //     } else {
    //         7.366586e-01
    //     }
    // } else if radius_ema <= 5.773949e-02 {
    //     8.322585e-01
    // } else {
    //     9.700278e-01
    // }
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
fn dt_euclidean_cr(_ratios: Ratios) -> f64 {
    todo!()

    // let [cardinality, radius, lfd, _, _, _] = ratios;
    // if radius <= 1.067198e-02 {
    //     if cardinality <= 2.828062e-02 {
    //         if radius <= 9.307770e-05 {
    //             3.440542e-01
    //         } else {
    //             6.220238e-01
    //         }
    //     } else if lfd <= 2.580366e-01 {
    //         5.624770e-01
    //     } else {
    //         8.176447e-01
    //     }
    // } else if cardinality <= 3.860220e-02 {
    //     if cardinality <= 3.826937e-02 {
    //         8.636209e-01
    //     } else {
    //         5.281209e-01
    //     }
    // } else if cardinality <= 4.997400e-01 {
    //     9.494346e-01
    // } else {
    //     9.803474e-01
    // }
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
fn dt_euclidean_sp(_ratios: Ratios) -> f64 {
    todo!()

    // let [cardinality, _, lfd, cardinality_ema, radius_ema, lfd_ema] = ratios;
    // if radius_ema <= 3.118177e-02 {
    //     if lfd <= 8.090920e-01 {
    //         if lfd_ema <= 2.858897e-01 {
    //             4.526594e-01
    //         } else {
    //             8.478856e-01
    //         }
    //     } else if lfd_ema <= 6.024930e-01 {
    //         5.471349e-03
    //     } else {
    //         2.958559e-01
    //     }
    // } else if cardinality <= 1.494950e-01 {
    //     if radius_ema <= 9.417503e-01 {
    //         9.296579e-01
    //     } else {
    //         7.623177e-01
    //     }
    // } else if cardinality_ema <= 1.749870e-02 {
    //     7.646179e-01
    // } else {
    //     9.863799e-01
    // }
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
fn dt_euclidean_vd(_ratios: Ratios) -> f64 {
    todo!()

    // let [cardinality, radius, lfd, _, radius_ema, lfd_ema] = ratios;
    // if radius <= 1.067198e-02 {
    //     if lfd <= 9.634169e-01 {
    //         if lfd_ema <= 9.999111e-01 {
    //             4.713993e-01
    //         } else {
    //             9.783193e-01
    //         }
    //     } else if radius <= 1.020328e-02 {
    //         9.089947e-01
    //     } else {
    //         4.464702e-01
    //     }
    // } else if cardinality <= 4.997769e-01 {
    //     if lfd <= 9.999996e-01 {
    //         9.335233e-01
    //     } else {
    //         7.793101e-01
    //     }
    // } else if radius_ema <= 7.430525e-01 {
    //     9.627948e-01
    // } else {
    //     9.921575e-01
    // }
}
