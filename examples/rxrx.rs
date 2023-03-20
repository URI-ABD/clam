use std::path::PathBuf;

use polars::datatypes::AnyValue;
use polars::prelude::*;

fn main() {
    let data_dir = get_data_root();

    let metadata_df = {
        let mut path = data_dir.clone();
        path.push("metadata_rxrx3.csv");
        assert!(
            path.exists(),
            "Metadata path not found: {path:?}",
        );
        CsvReader::from_path(path).unwrap().finish().unwrap()
    };
    /*
    Shape: (2222103, 10)

    Schema:
        name: well_id, data type: Utf8
        name: experiment_name, data type: Utf8
        name: plate, data type: Int64
        name: address, data type: Utf8
        name: gene, data type: Utf8
        name: treatment, data type: Utf8
        name: SMILES, data type: Utf8
        name: concentration, data type: Float64
        name: perturbation_type, data type: Utf8
        name: cell_type, data type: Utf8
    */

    let row_0 = metadata_df.get_row(0).unwrap().0;
    let well_id = if let AnyValue::Utf8(well_id) = row_0[0] {
        well_id
    } else {
        panic!("Oops");
    };
    // let Row(vec![
    //     well_id,
    //     experiment_name,
    //     plate,
    //     address,
    //     gene,
    //     treatment,
    //     SMILES,
    //     concentration,
    //     perturbation_type,
    //     cell_type,
    // ]) = row_0;

    println!("Data Dir: {data_dir:?}");
    println!("{:?}", row_0);
    println!("well_id: {well_id}");
}

fn get_data_root() -> PathBuf {
    let mut path = std::env::current_dir().unwrap();

    path.pop();
    path.push("data");
    path.push("rxrx3");

    assert!(
        path.exists(),
        "Data path not found: {path:?}",
    );
    assert!(
        path.is_dir(),
        "Data path not a directory: {path:?}",
    );

    path
}
