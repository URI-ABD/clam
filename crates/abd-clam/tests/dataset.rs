//! Tests for the `Dataset` trait.

use abd_clam::{Dataset, ParDataset};

use float_cmp::assert_approx_eq;
use test_case::test_case;

mod common;

#[test_case(10, 5 ; "10x5")]
#[test_case(100, 20 ; "100x20")]
#[test_case(1000, 50 ; "1000x50")]
fn one_to_many(car: usize, dim: usize) {
    let (min, max) = (-1.0, 1.0);
    let data = common::data_gen::tabular(car, dim, min, max);
    let range = (0..car).collect::<Vec<_>>();
    let metric = common::metrics::euclidean;

    for &i in &range {
        let expected = data
            .iter()
            .enumerate()
            .map(|(j, y)| (j, metric(&data[i], y)))
            .collect::<Vec<_>>();

        let actual = data.one_to_many(i, &range, &metric);
        for (&(j, d_exp), (k, d_act)) in expected.iter().zip(actual) {
            assert_eq!(j, k, "One-to-many: expected index {j}, got {k}");
            assert_approx_eq!(f32, d_exp, d_act);
        }

        let par_actual = data.par_one_to_many(i, &range, &metric);
        for (&(j, d_exp), (k, d_act)) in expected.iter().zip(par_actual) {
            assert_eq!(j, k, "One-to-many (parallel): expected index {j}, got {k}");
            assert_approx_eq!(f32, d_exp, d_act);
        }
    }
}

#[test_case(10, 5 ; "10x5")]
#[test_case(100, 20 ; "100x20")]
#[test_case(1000, 50 ; "1000x50")]
fn many_to_many(car: usize, dim: usize) {
    let (min, max) = (-1.0, 1.0);
    let data = common::data_gen::tabular(car, dim, min, max);
    let range = (0..car).collect::<Vec<_>>();
    let metric = common::metrics::euclidean;

    let expected = range
        .iter()
        .map(|&i| {
            data.iter()
                .enumerate()
                .map(|(j, y)| (i, j, metric(&data[i], y)))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let actual = data.many_to_many(&range, &range, &metric);
    for (r, (exp_row, act_row)) in expected.iter().zip(actual).enumerate() {
        for (c, (&(i, j, d_exp), (k, l, d_act))) in exp_row.iter().zip(act_row).enumerate() {
            assert_eq!(i, k, "Many-to-many row {r}, col {c}: expected index {i}, got {k}");
            assert_eq!(j, l, "Many-to-many row {r}, col {c}: expected index {j}, got {l}");
            assert_approx_eq!(f32, d_exp, d_act);
        }
    }

    let par_actual = data.par_many_to_many(&range, &range, &metric);
    for (r, (exp_row, act_row)) in expected.iter().zip(par_actual).enumerate() {
        for (c, (&(i, j, d_exp), (k, l, d_act))) in exp_row.iter().zip(act_row).enumerate() {
            assert_eq!(
                i, k,
                "Many-to-many (parallel) row {r}, col {c}: expected index {i}, got {k}"
            );
            assert_eq!(
                j, l,
                "Many-to-many (parallel) row {r}, col {c}: expected index {j}, got {l}"
            );
            assert_approx_eq!(f32, d_exp, d_act);
        }
    }
}
