use distances::strings::Penalties;




fn are_we_there_yet(seed_string: &str, Penalties: Penalties, target_distance: u16, alphabet: &Vec<char>, radius: u8) -> String {
    let mut edits = vec![];
    let mut total_penalty = 0;
    let mut new_string = seed_string.to_string();
    let mut distance = 0; 
    let lev = levenshtein_custom(Penalties);
    
    while (distance > target_distance + radius/2) || (distance < target_distance - radius/2) {

        while total_penalty < target_distance {

        }

        new_string = compute_edits(new_string, alphabet, &edits);
        distance = lev(new_string, seed_string);

    }
    return new_string;

}

// let penalties = Penalties::new(0, 1, 1);
// let metric = levenshtein_custom(penalties);
// let x = "NAJIBEATSPEPPERS";
// let y = "NAJIBPEPPERSEATS";