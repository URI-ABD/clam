// mod data_readers_tmp;
mod chunked_data;
mod chunked_space;
mod readers;

fn main() {
    println!("Hello from search_large!");

    let (in_dir, out_dir) = {
        let mut data_dir = std::env::current_dir().unwrap();
        data_dir.pop();
        data_dir.push("data");

        let in_dir = data_dir.join("search_large");
        assert!(in_dir.exists(), "Data Dir not found: {:?}", &in_dir);

        let out_dir = data_dir.join("working");
        readers::make_dir(&out_dir).unwrap();
        (in_dir, out_dir)
    };

    let data = readers::BigAnnPaths {
        folder: "msft_spacev",
        train: "msft_spacev-1b.i8bin",
        subset_1: None,
        subset_2: None,
        subset_3: None,
        query: "public_query_gt100.bin",
        ground: "msspacev-1B",
    };

    let location = out_dir.join(data.folder).join("train");

    if !location.exists() {
        readers::transform::<i8>(&data, &in_dir.join(data.folder), &out_dir.join(data.folder))
            .map_err(|reason| format!("Failed on {} because {}", data.folder, reason))
            .unwrap();
    }

    let cardinality = 1_000_000_000;
    let dataset = chunked_data::ChunkedTabular::new(&location, cardinality, 100, 1_000_000, data.folder);

    let metric = clam::metric::Euclidean { is_expensive: false };
    let space = chunked_space::ChunkedTabularSpace::<i8>::new(&dataset, &metric);

    let criteria = clam::PartitionCriteria::new(true).with_min_cardinality(10);
    let cakes = clam::CAKES::new(&space).build(&criteria);

    println!(
        "Built Cakes object with radius {:.2e} and depth {}",
        cakes.radius(),
        cakes.depth()
    );
}
