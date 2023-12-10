#![deny(clippy::correctness)]
#![warn(
    missing_docs,
    clippy::all,
    clippy::suspicious,
    clippy::complexity,
    clippy::perf,
    clippy::style,
    clippy::pedantic,
    clippy::nursery,
    clippy::missing_docs_in_private_items
)]
#![doc = include_str!("../README.md")]

use pyo3::prelude::*;
use rand::Rng;
use std::cmp::Ordering;
use std::io;

pub mod augmentation;
pub mod random_data;

/// The version of the crate.
pub const VERSION: &str = "0.3.0";

/// Guess the number game.
///
/// This is a demo function for the Python module.
///
/// # Panics
///
/// * If the user input is not a number.
/// * If unable to read the user input.
#[pyfunction]
#[allow(clippy::redundant_pub_crate)]
pub fn guess_the_number() {
    println!("Guess the number!");

    let secret_number = rand::thread_rng().gen_range(1..101);

    loop {
        println!("Please input your guess.");

        let mut guess = String::new();

        io::stdin().read_line(&mut guess).expect("Failed to read line");

        let guess: u32 = match guess.trim().parse() {
            Ok(num) => num,
            Err(_) => continue,
        };

        println!("You guessed: {guess}");

        match guess.cmp(&secret_number) {
            Ordering::Less => println!("Too small!"),
            Ordering::Greater => println!("Too big!"),
            Ordering::Equal => {
                println!("You win!");
                break;
            }
        }
    }
}

/// A Python module implemented in Rust.
///
/// The name of this function must match the `lib.name` setting in the `Cargo.toml`,
/// else Python will not be able to import the module.
///
/// # Errors
///
/// * If unable to add the function to the module.
/// * If unable to add the module to the Python interpreter.
#[pymodule]
pub fn symagen(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(guess_the_number, m)?)?;

    Ok(())
}
