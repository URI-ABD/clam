#![allow(missing_docs)]

use rand::prelude::*;
use stringzilla::szs::{DeviceScope, LevenshteinDistances};

#[test]
fn test_levenshtein() -> Result<(), String> {
    for d in 2..=4 {
        let len = 10_usize.pow(d);
        let mut rng = StdRng::seed_from_u64(42);
        let vecs = symagen::random_data::random_string(2, len, len, "ATCGN", &mut rng);
        let (x, y) = (&vecs[0], &vecs[1]);

        let dist = distances::strings::levenshtein::<usize>(x, y);

        let device = DeviceScope::default().map_err(|e| format!("Failed to create DeviceScope: {e}"))?;
        let szla_engine = LevenshteinDistances::new(&device, 0, 1, 1, 1).map_err(|e| format!("Failed to create LevenshteinDistances: {e}"))?;
        let szla_vec = szla_engine
            .compute(&device, &[x], &[y])
            .map_err(|e| format!("Failed to compute distance: {e}"))?;
        let szla = szla_vec[0];

        assert_eq!(dist, szla);
    }

    Ok(())
}
