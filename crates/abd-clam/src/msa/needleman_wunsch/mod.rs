//! Needleman-Wunsch algorithm for global sequence alignment.

mod aligner;
mod cost_matrix;
mod ops;

pub use aligner::Aligner;
pub use cost_matrix::CostMatrix;
pub use ops::{Direction, Edit, Edits};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance() {
        let nw_aligner = Aligner::<i16>::default();

        let x = "NAJIBEATSPEPPERS";
        let y = "NAJIBPEPPERSEATS";
        assert_eq!(nw_aligner.distance(&x, &y), 8);
        assert_eq!(nw_aligner.distance(&y, &x), 8);

        let x = "NOTGUILTY".to_string();
        let y = "NOTGUILTY".to_string();
        assert_eq!(nw_aligner.distance(&x, &y), 0);
        assert_eq!(nw_aligner.distance(&y, &x), 0);
    }

    #[test]
    fn test_compute_table() {
        let x = "NAJIBPEPPERSEATS";
        let y = "NAJIBEATSPEPPERS";
        let nw_aligner = Aligner::default();
        let table = nw_aligner.dp_table(&x, &y);

        #[rustfmt::skip]
        let true_table: [[(i16, Direction); 17]; 17] = [
            [( 0, Direction::Diagonal), ( 1, Direction::Left    ), ( 2, Direction::Left    ), ( 3, Direction::Left    ), ( 4, Direction::Left    ), ( 5, Direction::Left    ), ( 6, Direction::Left    ), (7, Direction::Left    ), (8, Direction::Left    ), (9, Direction::Left    ), (10, Direction::Left    ), (11, Direction::Left    ), (12, Direction::Left    ), (13, Direction::Left    ), (14, Direction::Left    ), (15, Direction::Left    ), (16, Direction::Left    )],
            [( 1, Direction::Up      ), ( 0, Direction::Diagonal), ( 1, Direction::Left    ), ( 2, Direction::Left    ), ( 3, Direction::Left    ), ( 4, Direction::Left    ), ( 5, Direction::Left    ), (6, Direction::Left    ), (7, Direction::Left    ), (8, Direction::Left    ), ( 9, Direction::Left    ), (10, Direction::Left    ), (11, Direction::Left    ), (12, Direction::Left    ), (13, Direction::Left    ), (14, Direction::Left    ), (15, Direction::Left    )],
            [( 2, Direction::Up      ), ( 1, Direction::Up      ), ( 0, Direction::Diagonal), ( 1, Direction::Left    ), ( 2, Direction::Left    ), ( 3, Direction::Left    ), ( 4, Direction::Left    ), (5, Direction::Left    ), (6, Direction::Left    ), (7, Direction::Left    ), ( 8, Direction::Left    ), ( 9, Direction::Left    ), (10, Direction::Left    ), (11, Direction::Left    ), (12, Direction::Diagonal), (13, Direction::Left    ), (14, Direction::Left    )],
            [( 3, Direction::Up      ), ( 2, Direction::Up      ), ( 1, Direction::Up      ), ( 0, Direction::Diagonal), ( 1, Direction::Left    ), ( 2, Direction::Left    ), ( 3, Direction::Left    ), (4, Direction::Left    ), (5, Direction::Left    ), (6, Direction::Left    ), ( 7, Direction::Left    ), ( 8, Direction::Left    ), ( 9, Direction::Left    ), (10, Direction::Left    ), (11, Direction::Left    ), (12, Direction::Left    ), (13, Direction::Left    )],
            [( 4, Direction::Up      ), ( 3, Direction::Up      ), ( 2, Direction::Up      ), ( 1, Direction::Up      ), ( 0, Direction::Diagonal), ( 1, Direction::Left    ), ( 2, Direction::Left    ), (3, Direction::Left    ), (4, Direction::Left    ), (5, Direction::Left    ), ( 6, Direction::Left    ), ( 7, Direction::Left    ), ( 8, Direction::Left    ), ( 9, Direction::Left    ), (10, Direction::Left    ), (11, Direction::Left    ), (12, Direction::Left    )],
            [( 5, Direction::Up      ), ( 4, Direction::Up      ), ( 3, Direction::Up      ), ( 2, Direction::Up      ), ( 1, Direction::Up      ), ( 0, Direction::Diagonal), ( 1, Direction::Left    ), (2, Direction::Left    ), (3, Direction::Left    ), (4, Direction::Left    ), ( 5, Direction::Left    ), ( 6, Direction::Left    ), ( 7, Direction::Left    ), ( 8, Direction::Left    ), ( 9, Direction::Left    ), (10, Direction::Left    ), (11, Direction::Left    )],
            [( 6, Direction::Up      ), ( 5, Direction::Up      ), ( 4, Direction::Up      ), ( 3, Direction::Up      ), ( 2, Direction::Up      ), ( 1, Direction::Up      ), ( 1, Direction::Diagonal), (1, Direction::Diagonal), (2, Direction::Left    ), (3, Direction::Left    ), ( 4, Direction::Diagonal), ( 5, Direction::Left    ), ( 6, Direction::Left    ), ( 7, Direction::Diagonal), ( 8, Direction::Left    ), ( 9, Direction::Left    ), (10, Direction::Left    )],
            [( 7, Direction::Up      ), ( 6, Direction::Up      ), ( 5, Direction::Diagonal), ( 4, Direction::Up      ), ( 3, Direction::Up      ), ( 2, Direction::Up      ), ( 2, Direction::Diagonal), (2, Direction::Diagonal), (2, Direction::Diagonal), (3, Direction::Diagonal), ( 4, Direction::Diagonal), ( 5, Direction::Diagonal), ( 6, Direction::Diagonal), ( 7, Direction::Diagonal), ( 7, Direction::Diagonal), ( 8, Direction::Left    ), ( 9, Direction::Left    )],
            [( 8, Direction::Up      ), ( 7, Direction::Up      ), ( 6, Direction::Up      ), ( 5, Direction::Up      ), ( 4, Direction::Up      ), ( 3, Direction::Up      ), ( 3, Direction::Diagonal), (3, Direction::Diagonal), (3, Direction::Diagonal), (3, Direction::Diagonal), ( 4, Direction::Diagonal), ( 5, Direction::Diagonal), ( 6, Direction::Diagonal), ( 7, Direction::Diagonal), ( 8, Direction::Diagonal), ( 7, Direction::Diagonal), ( 8, Direction::Left    )],
            [( 9, Direction::Up      ), ( 8, Direction::Up      ), ( 7, Direction::Up      ), ( 6, Direction::Up      ), ( 5, Direction::Up      ), ( 4, Direction::Up      ), ( 4, Direction::Diagonal), (4, Direction::Diagonal), (4, Direction::Diagonal), (4, Direction::Diagonal), ( 4, Direction::Diagonal), ( 5, Direction::Diagonal), ( 5, Direction::Diagonal), ( 6, Direction::Left    ), ( 7, Direction::Left    ), ( 8, Direction::Up      ), ( 7, Direction::Diagonal)],
            [(10, Direction::Up      ), ( 9, Direction::Up      ), ( 8, Direction::Up      ), ( 7, Direction::Up      ), ( 6, Direction::Up      ), ( 5, Direction::Up      ), ( 4, Direction::Diagonal), (5, Direction::Diagonal), (4, Direction::Diagonal), (4, Direction::Diagonal), ( 5, Direction::Diagonal), ( 5, Direction::Diagonal), ( 6, Direction::Diagonal), ( 6, Direction::Diagonal), ( 7, Direction::Diagonal), ( 8, Direction::Diagonal), ( 8, Direction::Up      )],
            [(11, Direction::Up      ), (10, Direction::Up      ), ( 9, Direction::Up      ), ( 8, Direction::Up      ), ( 7, Direction::Up      ), ( 6, Direction::Up      ), ( 5, Direction::Up      ), (4, Direction::Diagonal), (5, Direction::Up      ), (5, Direction::Diagonal), ( 4, Direction::Diagonal), ( 5, Direction::Left    ), ( 6, Direction::Diagonal), ( 6, Direction::Diagonal), ( 7, Direction::Diagonal), ( 8, Direction::Diagonal), ( 9, Direction::Diagonal)],
            [(12, Direction::Up      ), (11, Direction::Up      ), (10, Direction::Up      ), ( 9, Direction::Up      ), ( 8, Direction::Up      ), ( 7, Direction::Up      ), ( 6, Direction::Diagonal), (5, Direction::Up      ), (4, Direction::Diagonal), (5, Direction::Diagonal), ( 5, Direction::Up      ), ( 5, Direction::Diagonal), ( 6, Direction::Diagonal), ( 7, Direction::Diagonal), ( 7, Direction::Diagonal), ( 8, Direction::Diagonal), ( 9, Direction::Diagonal)],
            [(13, Direction::Up      ), (12, Direction::Up      ), (11, Direction::Up      ), (10, Direction::Up      ), ( 9, Direction::Up      ), ( 8, Direction::Up      ), ( 7, Direction::Diagonal), (6, Direction::Up      ), (5, Direction::Diagonal), (4, Direction::Diagonal), ( 5, Direction::Left    ), ( 6, Direction::Diagonal), ( 6, Direction::Diagonal), ( 7, Direction::Diagonal), ( 8, Direction::Diagonal), ( 8, Direction::Diagonal), ( 9, Direction::Diagonal)],
            [(14, Direction::Up      ), (13, Direction::Up      ), (12, Direction::Up      ), (11, Direction::Up      ), (10, Direction::Up      ), ( 9, Direction::Up      ), ( 8, Direction::Up      ), (7, Direction::Diagonal), (6, Direction::Up      ), (5, Direction::Up      ), ( 4, Direction::Diagonal), ( 5, Direction::Left    ), ( 6, Direction::Left    ), ( 6, Direction::Diagonal), ( 7, Direction::Left    ), ( 8, Direction::Left    ), ( 9, Direction::Diagonal)],
            [(15, Direction::Up      ), (14, Direction::Up      ), (13, Direction::Up      ), (12, Direction::Up      ), (11, Direction::Up      ), (10, Direction::Up      ), ( 9, Direction::Up      ), (8, Direction::Up      ), (7, Direction::Up      ), (6, Direction::Up      ), ( 5, Direction::Up      ), ( 4, Direction::Diagonal), ( 5, Direction::Left    ), ( 6, Direction::Left    ), ( 7, Direction::Diagonal), ( 8, Direction::Diagonal), ( 9, Direction::Diagonal)],
            [(16, Direction::Up      ), (15, Direction::Up      ), (14, Direction::Up      ), (13, Direction::Up      ), (12, Direction::Up      ), (11, Direction::Up      ), (10, Direction::Up      ), (9, Direction::Up      ), (8, Direction::Up      ), (7, Direction::Up      ), ( 6, Direction::Up      ), ( 5, Direction::Up      ), ( 4, Direction::Diagonal), ( 5, Direction::Left    ), ( 6, Direction::Left    ), ( 7, Direction::Left    ), ( 8, Direction::Diagonal)]
        ];

        assert_eq!(table, true_table);
    }

    #[test]
    fn test_trace_back() {
        let nw_aligner = Aligner::<i16>::default();

        let peppers_x = "NAJIBPEPPERSEATS";
        let peppers_y = "NAJIBEATSPEPPERS";
        let peppers_table = nw_aligner.dp_table(&peppers_x, &peppers_y);

        let (d, [aligned_x, aligned_y]) = nw_aligner.align_str(&peppers_x, &peppers_y, &peppers_table);
        assert_eq!(d, 8);
        assert_eq!(aligned_x, "NAJIB-PEPPERSEATS");
        assert_eq!(aligned_y, "NAJIBEATSPEPPE-RS");

        let guilty_x = "NOTGUILTY";
        let guilty_y = "NOTGUILTY";
        let guilty_table = nw_aligner.dp_table(&guilty_x, &guilty_y);

        let (d, [aligned_x, aligned_y]) = nw_aligner.align_str(&guilty_x, &guilty_y, &guilty_table);
        assert_eq!(d, 0);
        assert_eq!(aligned_x, "NOTGUILTY");
        assert_eq!(aligned_y, "NOTGUILTY");
    }

    #[test]
    fn test_alignment_gaps() {
        let nw_aligner = Aligner::<i16>::default();

        let x = "MDIAIHHPWIRRP---";
        let y = "MDIAIHHPWIRRPF";
        let (d, [x_gaps, y_gaps]) = nw_aligner.alignment_gaps(&x, &y);

        assert_eq!(d, 3);
        assert_eq!(x_gaps, vec![]);
        assert_eq!(y_gaps, vec![13, 13]);

        let (d, [x_gaps, y_gaps]) = nw_aligner.alignment_gaps(&y, &x);
        assert_eq!(d, 3);
        assert_eq!(x_gaps, vec![13, 13]);
        assert_eq!(y_gaps, vec![]);
    }
}
