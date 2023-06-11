use crate::number::Number;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Direction {
    Diagonal,
    Up,
    Left,
    None,
}

// New function to compute the distance and direction table using scoring scheme 0; 1; 1
pub fn compute_nw_table<T: Number>(x: &[T], y: &[T]) -> Vec<Vec<(usize, Direction)>> {
    let len_x = x.len();
    let len_y = y.len();

    // Initializing table; subvecs represent rows
    let mut table = vec![vec![(0, Direction::None); len_x + 1]; len_y + 1];

    let gap_penalty = 1;

    // Initialize top row and left column distance values
    #[allow(clippy::needless_range_loop)]
    for row in 0..(len_y + 1) {
        table[row][0] = (gap_penalty * row, Direction::Up);
    }

    for column in 0..(len_x + 1) {
        table[0][column] = (gap_penalty * column, Direction::Left);
    }

    table[0][0] = (0, Direction::None);

    // Set values for the body of the table
    for row in 1..(len_y + 1) {
        for col in 1..(len_x + 1) {
            // Check if sequences match at position col-1 in x and row-1 in y
            // Reason for subtraction is that NW considers an artificial gap at the start
            // of each sequence, so the dp tables' indices are 1 higher than that of
            // the actual sequences
            let mismatch_penalty = if x[col - 1] == y[row - 1] { 0 } else { 1 };
            let new_cell = [
                table[row - 1][col - 1].0 + mismatch_penalty,
                table[row - 1][col].0 + gap_penalty,
                table[row][col - 1].0 + gap_penalty,
            ]
            .into_iter()
            .zip([Direction::Diagonal, Direction::Up, Direction::Left].into_iter())
            .min_by(|x, y| x.0.cmp(&y.0))
            .unwrap();

            table[row][col] = new_cell;
        }
    }

    table
}

pub enum Edit<T: Number> {
    Deletion(usize),
    Insertion(usize, T),
    Substitution(usize, T),
}

// Given an alignement of two sequences, returns the set of edits needed to turn sequence
// x into sequence y
pub fn alignment_to_edits<T: Number>(aligned_x: &[T], aligned_y: &[T]) -> Vec<Edit<T>> {
    aligned_x
        .iter()
        .zip(aligned_y.iter())
        .filter(|(x, y)| x != y)
        .enumerate()
        .map(|(index, (x, y))| {
            if *x == T::from(b'-').unwrap() {
                Edit::Insertion(index, *y)
            } else if *y == T::from(b'-').unwrap() {
                Edit::Deletion(index)
            } else {
                Edit::Substitution(index, *y)
            }
        })
        .collect()
}

// Iterative version of traceback function so we can benchmark both this and recursive option
pub fn traceback_iterative<T: Number>(
    table: &Vec<Vec<(usize, Direction)>>,
    unaligned_seqs: (&[T], &[T]),
) -> (Vec<T>, Vec<T>) {
    let mut row_index = table.len() - 1;
    let mut column_index = table[0].len() - 1;

    let (mut aligned_x, mut aligned_y) = (Vec::<T>::new(), Vec::<T>::new());
    let (unaligned_x, unaligned_y) = unaligned_seqs;
    let mut direction = table[row_index][column_index].1;

    while direction != Direction::None {
        match direction {
            Direction::Diagonal => {
                aligned_x.push(unaligned_x[column_index - 1]);
                aligned_y.push(unaligned_y[row_index - 1]);
                row_index -= 1;
                column_index -= 1;
            }
            Direction::Left => {
                aligned_x.push(unaligned_x[column_index - 1]);
                aligned_y.push(T::from(b'-').unwrap());
                column_index -= 1;
            }
            Direction::Up => {
                aligned_x.push(T::from(b'-').unwrap());
                aligned_y.push(unaligned_y[row_index - 1]);
                row_index -= 1;
            }
            Direction::None => {}
        }

        direction = table[row_index][column_index].1;
    }

    aligned_x.reverse();
    aligned_y.reverse();

    (aligned_x, aligned_y)
}

// Public function for recurisve traceback to get alignment and edit distance (ignores ties for now)
pub fn traceback_recursive<T: Number>(
    table: &Vec<Vec<(usize, Direction)>>,
    unaligned_seqs: (&[T], &[T]),
) -> (Vec<T>, Vec<T>) {
    let indices = (table.len() - 1, table[0].len() - 1);

    let mut aligned_x: Vec<T> = Vec::new();
    let mut aligned_y: Vec<T> = Vec::new();

    _traceback_recursive(table, indices, unaligned_seqs, (&mut aligned_x, &mut aligned_y));

    (aligned_x, aligned_y)
}

// Private traceback recursive function. Returns a single alignment (disregards ties for best-scoring alignment)
fn _traceback_recursive<T: Number>(
    table: &Vec<Vec<(usize, Direction)>>,
    indices: (usize, usize),
    unaligned_seqs: (&[T], &[T]),
    aligned_seqs: (&mut Vec<T>, &mut Vec<T>),
) {
    /* This function is likely slow because this is a clone happening because an
    immutable thing is now being made mutable. Instead, you should try initializing
    the vectors in the outer function, pass in mutable references to this function,
    and not have this function return anything. */
    let (unaligned_x, unaligned_y) = unaligned_seqs;
    let (aligned_x, aligned_y) = aligned_seqs;

    let (mut row_index, mut column_index) = indices;

    let direction = table[row_index][column_index].1;

    match direction {
        Direction::Diagonal => {
            aligned_x.push(unaligned_x[column_index - 1]);
            aligned_y.push(unaligned_y[row_index - 1]);
            row_index -= 1;
            column_index -= 1;
        }
        Direction::Left => {
            aligned_x.push(unaligned_x[column_index - 1]);
            aligned_y.push(T::from(b'-').unwrap());
            column_index -= 1;
        }
        Direction::Up => {
            aligned_x.push(T::from(b'-').unwrap());
            aligned_y.push(unaligned_y[row_index - 1]);
            row_index -= 1;
        }
        Direction::None => {
            aligned_x.reverse();
            aligned_y.reverse();

            return;
        }
    };

    _traceback_recursive(
        table,
        (row_index, column_index),
        (unaligned_x, unaligned_y),
        (aligned_x, aligned_y),
    )
}

#[cfg(test)]
mod tests {
    use crate::distances::alignment_helpers::compute_nw_table;
    use crate::distances::alignment_helpers::traceback_iterative;
    use crate::distances::alignment_helpers::traceback_recursive;
    use crate::distances::alignment_helpers::Direction;

    #[test]
    fn test_compute_table() {
        let x_u8 = "NAJIBPEPPERSEATS".as_bytes();
        let y_u8 = "NAJIBEATSPEPPERS".as_bytes();
        let table = compute_nw_table(x_u8, y_u8);
        assert_eq!(
            table,
            [
                [
                    (0, Direction::None),
                    (1, Direction::Left),
                    (2, Direction::Left),
                    (3, Direction::Left),
                    (4, Direction::Left),
                    (5, Direction::Left),
                    (6, Direction::Left),
                    (7, Direction::Left),
                    (8, Direction::Left),
                    (9, Direction::Left),
                    (10, Direction::Left),
                    (11, Direction::Left),
                    (12, Direction::Left),
                    (13, Direction::Left),
                    (14, Direction::Left),
                    (15, Direction::Left),
                    (16, Direction::Left)
                ],
                [
                    (1, Direction::Up),
                    (0, Direction::Diagonal),
                    (1, Direction::Left),
                    (2, Direction::Left),
                    (3, Direction::Left),
                    (4, Direction::Left),
                    (5, Direction::Left),
                    (6, Direction::Left),
                    (7, Direction::Left),
                    (8, Direction::Left),
                    (9, Direction::Left),
                    (10, Direction::Left),
                    (11, Direction::Left),
                    (12, Direction::Left),
                    (13, Direction::Left),
                    (14, Direction::Left),
                    (15, Direction::Left)
                ],
                [
                    (2, Direction::Up),
                    (1, Direction::Up),
                    (0, Direction::Diagonal),
                    (1, Direction::Left),
                    (2, Direction::Left),
                    (3, Direction::Left),
                    (4, Direction::Left),
                    (5, Direction::Left),
                    (6, Direction::Left),
                    (7, Direction::Left),
                    (8, Direction::Left),
                    (9, Direction::Left),
                    (10, Direction::Left),
                    (11, Direction::Left),
                    (12, Direction::Diagonal),
                    (13, Direction::Left),
                    (14, Direction::Left)
                ],
                [
                    (3, Direction::Up),
                    (2, Direction::Up),
                    (1, Direction::Up),
                    (0, Direction::Diagonal),
                    (1, Direction::Left),
                    (2, Direction::Left),
                    (3, Direction::Left),
                    (4, Direction::Left),
                    (5, Direction::Left),
                    (6, Direction::Left),
                    (7, Direction::Left),
                    (8, Direction::Left),
                    (9, Direction::Left),
                    (10, Direction::Left),
                    (11, Direction::Left),
                    (12, Direction::Left),
                    (13, Direction::Left)
                ],
                [
                    (4, Direction::Up),
                    (3, Direction::Up),
                    (2, Direction::Up),
                    (1, Direction::Up),
                    (0, Direction::Diagonal),
                    (1, Direction::Left),
                    (2, Direction::Left),
                    (3, Direction::Left),
                    (4, Direction::Left),
                    (5, Direction::Left),
                    (6, Direction::Left),
                    (7, Direction::Left),
                    (8, Direction::Left),
                    (9, Direction::Left),
                    (10, Direction::Left),
                    (11, Direction::Left),
                    (12, Direction::Left)
                ],
                [
                    (5, Direction::Up),
                    (4, Direction::Up),
                    (3, Direction::Up),
                    (2, Direction::Up),
                    (1, Direction::Up),
                    (0, Direction::Diagonal),
                    (1, Direction::Left),
                    (2, Direction::Left),
                    (3, Direction::Left),
                    (4, Direction::Left),
                    (5, Direction::Left),
                    (6, Direction::Left),
                    (7, Direction::Left),
                    (8, Direction::Left),
                    (9, Direction::Left),
                    (10, Direction::Left),
                    (11, Direction::Left)
                ],
                [
                    (6, Direction::Up),
                    (5, Direction::Up),
                    (4, Direction::Up),
                    (3, Direction::Up),
                    (2, Direction::Up),
                    (1, Direction::Up),
                    (1, Direction::Diagonal),
                    (1, Direction::Diagonal),
                    (2, Direction::Left),
                    (3, Direction::Left),
                    (4, Direction::Diagonal),
                    (5, Direction::Left),
                    (6, Direction::Left),
                    (7, Direction::Diagonal),
                    (8, Direction::Left),
                    (9, Direction::Left),
                    (10, Direction::Left)
                ],
                [
                    (7, Direction::Up),
                    (6, Direction::Up),
                    (5, Direction::Diagonal),
                    (4, Direction::Up),
                    (3, Direction::Up),
                    (2, Direction::Up),
                    (2, Direction::Diagonal),
                    (2, Direction::Diagonal),
                    (2, Direction::Diagonal),
                    (3, Direction::Diagonal),
                    (4, Direction::Diagonal),
                    (5, Direction::Diagonal),
                    (6, Direction::Diagonal),
                    (7, Direction::Diagonal),
                    (7, Direction::Diagonal),
                    (8, Direction::Left),
                    (9, Direction::Left)
                ],
                [
                    (8, Direction::Up),
                    (7, Direction::Up),
                    (6, Direction::Up),
                    (5, Direction::Up),
                    (4, Direction::Up),
                    (3, Direction::Up),
                    (3, Direction::Diagonal),
                    (3, Direction::Diagonal),
                    (3, Direction::Diagonal),
                    (3, Direction::Diagonal),
                    (4, Direction::Diagonal),
                    (5, Direction::Diagonal),
                    (6, Direction::Diagonal),
                    (7, Direction::Diagonal),
                    (8, Direction::Diagonal),
                    (7, Direction::Diagonal),
                    (8, Direction::Left)
                ],
                [
                    (9, Direction::Up),
                    (8, Direction::Up),
                    (7, Direction::Up),
                    (6, Direction::Up),
                    (5, Direction::Up),
                    (4, Direction::Up),
                    (4, Direction::Diagonal),
                    (4, Direction::Diagonal),
                    (4, Direction::Diagonal),
                    (4, Direction::Diagonal),
                    (4, Direction::Diagonal),
                    (5, Direction::Diagonal),
                    (5, Direction::Diagonal),
                    (6, Direction::Left),
                    (7, Direction::Left),
                    (8, Direction::Up),
                    (7, Direction::Diagonal)
                ],
                [
                    (10, Direction::Up),
                    (9, Direction::Up),
                    (8, Direction::Up),
                    (7, Direction::Up),
                    (6, Direction::Up),
                    (5, Direction::Up),
                    (4, Direction::Diagonal),
                    (5, Direction::Diagonal),
                    (4, Direction::Diagonal),
                    (4, Direction::Diagonal),
                    (5, Direction::Diagonal),
                    (5, Direction::Diagonal),
                    (6, Direction::Diagonal),
                    (6, Direction::Diagonal),
                    (7, Direction::Diagonal),
                    (8, Direction::Diagonal),
                    (8, Direction::Up)
                ],
                [
                    (11, Direction::Up),
                    (10, Direction::Up),
                    (9, Direction::Up),
                    (8, Direction::Up),
                    (7, Direction::Up),
                    (6, Direction::Up),
                    (5, Direction::Up),
                    (4, Direction::Diagonal),
                    (5, Direction::Up),
                    (5, Direction::Diagonal),
                    (4, Direction::Diagonal),
                    (5, Direction::Left),
                    (6, Direction::Diagonal),
                    (6, Direction::Diagonal),
                    (7, Direction::Diagonal),
                    (8, Direction::Diagonal),
                    (9, Direction::Diagonal)
                ],
                [
                    (12, Direction::Up),
                    (11, Direction::Up),
                    (10, Direction::Up),
                    (9, Direction::Up),
                    (8, Direction::Up),
                    (7, Direction::Up),
                    (6, Direction::Diagonal),
                    (5, Direction::Up),
                    (4, Direction::Diagonal),
                    (5, Direction::Diagonal),
                    (5, Direction::Up),
                    (5, Direction::Diagonal),
                    (6, Direction::Diagonal),
                    (7, Direction::Diagonal),
                    (7, Direction::Diagonal),
                    (8, Direction::Diagonal),
                    (9, Direction::Diagonal)
                ],
                [
                    (13, Direction::Up),
                    (12, Direction::Up),
                    (11, Direction::Up),
                    (10, Direction::Up),
                    (9, Direction::Up),
                    (8, Direction::Up),
                    (7, Direction::Diagonal),
                    (6, Direction::Up),
                    (5, Direction::Diagonal),
                    (4, Direction::Diagonal),
                    (5, Direction::Left),
                    (6, Direction::Diagonal),
                    (6, Direction::Diagonal),
                    (7, Direction::Diagonal),
                    (8, Direction::Diagonal),
                    (8, Direction::Diagonal),
                    (9, Direction::Diagonal)
                ],
                [
                    (14, Direction::Up),
                    (13, Direction::Up),
                    (12, Direction::Up),
                    (11, Direction::Up),
                    (10, Direction::Up),
                    (9, Direction::Up),
                    (8, Direction::Up),
                    (7, Direction::Diagonal),
                    (6, Direction::Up),
                    (5, Direction::Up),
                    (4, Direction::Diagonal),
                    (5, Direction::Left),
                    (6, Direction::Left),
                    (6, Direction::Diagonal),
                    (7, Direction::Left),
                    (8, Direction::Left),
                    (9, Direction::Diagonal)
                ],
                [
                    (15, Direction::Up),
                    (14, Direction::Up),
                    (13, Direction::Up),
                    (12, Direction::Up),
                    (11, Direction::Up),
                    (10, Direction::Up),
                    (9, Direction::Up),
                    (8, Direction::Up),
                    (7, Direction::Up),
                    (6, Direction::Up),
                    (5, Direction::Up),
                    (4, Direction::Diagonal),
                    (5, Direction::Left),
                    (6, Direction::Left),
                    (7, Direction::Diagonal),
                    (8, Direction::Diagonal),
                    (9, Direction::Diagonal)
                ],
                [
                    (16, Direction::Up),
                    (15, Direction::Up),
                    (14, Direction::Up),
                    (13, Direction::Up),
                    (12, Direction::Up),
                    (11, Direction::Up),
                    (10, Direction::Up),
                    (9, Direction::Up),
                    (8, Direction::Up),
                    (7, Direction::Up),
                    (6, Direction::Up),
                    (5, Direction::Up),
                    (4, Direction::Diagonal),
                    (5, Direction::Left),
                    (6, Direction::Left),
                    (7, Direction::Left),
                    (8, Direction::Diagonal)
                ]
            ]
        );
    }

    #[test]
    fn test_traceback_recursive() {
        let peppers_x = "NAJIBPEPPERSEATS".as_bytes();
        let peppers_y = "NAJIBEATSPEPPERS".as_bytes();
        let peppers_table: Vec<Vec<(usize, Direction)>> = compute_nw_table(peppers_x, peppers_y);
        let (aligned_x, aligned_y) = traceback_recursive(&peppers_table, (peppers_x, peppers_y));

        assert_eq!(
            (aligned_x.clone(), aligned_y.clone()),
            (
                [78, 65, 74, 73, 66, 45, 80, 69, 80, 80, 69, 82, 83, 69, 65, 84, 83].to_vec(),
                [78, 65, 74, 73, 66, 69, 65, 84, 83, 80, 69, 80, 80, 69, 45, 82, 83].to_vec()
            )
        );

        assert_eq!(
            (
                std::str::from_utf8(&aligned_x).unwrap(),
                std::str::from_utf8(&aligned_y).unwrap()
            ),
            ("NAJIB-PEPPERSEATS", "NAJIBEATSPEPPE-RS")
        );

        let guilty_x = "NOTGUILTY".as_bytes();
        let guilty_y = "NOTGUILTY".as_bytes();
        let guilty_table: Vec<Vec<(usize, Direction)>> = compute_nw_table(guilty_x, guilty_y);
        let (aligned_x, aligned_y) = traceback_recursive(&guilty_table, (guilty_x, guilty_y));
        assert_eq!(
            (aligned_x.clone(), aligned_y.clone()),
            (
                [78, 79, 84, 71, 85, 73, 76, 84, 89].to_vec(),
                [78, 79, 84, 71, 85, 73, 76, 84, 89].to_vec()
            )
        );
        assert_eq!(
            (
                std::str::from_utf8(&aligned_x).unwrap(),
                std::str::from_utf8(&aligned_y).unwrap()
            ),
            ("NOTGUILTY", "NOTGUILTY")
        );
    }

    #[test]
    fn test_traceback_iterative() {
        let x_u8 = "NAJIBPEPPERSEATS".as_bytes();
        let y_u8 = "NAJIBEATSPEPPERS".as_bytes();
        let table1: Vec<Vec<(usize, Direction)>> = compute_nw_table(x_u8, y_u8);
        let nw_u8 = traceback_iterative(&table1, (x_u8, y_u8));
        assert_eq!(
            nw_u8,
            (
                [78, 65, 74, 73, 66, 45, 80, 69, 80, 80, 69, 82, 83, 69, 65, 84, 83].to_vec(),
                [78, 65, 74, 73, 66, 69, 65, 84, 83, 80, 69, 80, 80, 69, 45, 82, 83].to_vec()
            ),
        );
    }
}
