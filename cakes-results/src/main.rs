mod ann_readers;
mod ann_reports;
mod bigann_readers;

fn main() -> Result<(), String> {
    let big_ann = false;

    if big_ann {
        bigann_readers::convert_bigann()
    } else {
        ann_reports::make_reports()
    }?;

    Ok(())
}
