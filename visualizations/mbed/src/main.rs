//! Making some plots for the Mass-Spring System.

use std::path::PathBuf;

use clap::Parser;
use plotters::prelude::*;

mod single_moving_mass;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
struct Args {
    /// Path to the output directory. If not provided, the plots will be written
    /// to the current directory.
    #[arg(short('o'), long)]
    out_dir: Option<PathBuf>,

    // The time-delta for steps in the simulation.
    #[arg(short('d'), long, default_value = "1e-1")]
    dt: f64,

    /// The drag coefficient.
    #[arg(short('b'), long, default_value = "0.02")]
    drag: f64,

    /// The total number of time steps.
    #[arg(short('n'), long, default_value = "2000")]
    n: usize,

    /// The random seed to use.
    #[arg(short('s'), long, default_value = "42")]
    seed: Option<u64>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    println!("Args: {args:?}");

    let out_dir = if let Some(out_dir) = args.out_dir {
        if out_dir.exists() {
            if !out_dir.is_dir() {
                return Err(format!("{out_dir:?} is not a directory").into());
            }
        } else {
            std::fs::create_dir_all(&out_dir)?;
        }
        out_dir
    } else {
        PathBuf::from(".")
    };
    let out_file = out_dir.join("mass-spring.png");

    let [ts, x1s, x2s, kes, pes] = single_moving_mass::one_moving_mass(args.drag, args.dt, args.n);

    let multiplier = 100;

    let root_area = BitMapBackend::new(&out_file, (16 * multiplier, 10 * multiplier)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let root_area = root_area.titled("Moving Masses", ("sans-serif", 60))?;
    let drawing_areas = root_area.split_evenly((2, 1));

    if drawing_areas.len() != 2 {
        return Err("Expected two drawing areas".into());
    }
    let (upper, lower) = (&drawing_areas[0], &drawing_areas[1]);

    let (x_min, x_max) = ts.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &x| {
        (min.min(x), max.max(x))
    });
    let (y_min, y_max) = x1s
        .iter()
        .chain(x2s.iter())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &y| {
            (min.min(y), max.max(y))
        });

    let mut cc = ChartBuilder::on(upper)
        .margin(5)
        .set_all_label_area_size(50)
        .caption("Positions", ("sans-serif", 40))
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    cc.configure_mesh()
        .x_labels(20)
        .y_labels(10)
        .disable_mesh()
        .x_label_formatter(&|v| format!("{:.1}", v))
        .y_label_formatter(&|v| format!("{:.1}", v))
        .draw()?;

    cc.draw_series(LineSeries::new(ts.iter().zip(x1s.iter()).map(|(&x, &y)| (x, y)), &RED))?
        .label("M1 Position")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    cc.draw_series(LineSeries::new(ts.iter().zip(x2s.iter()).map(|(&x, &y)| (x, y)), &BLUE))?
        .label("M2 Position")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    cc.configure_series_labels().border_style(BLACK).draw()?;

    let (e_max, e_min) = kes
        .iter()
        .chain(pes.iter())
        .fold((f64::NEG_INFINITY, f64::INFINITY), |(max, min), &e| {
            (max.max(e), min.min(e))
        });

    let mut cc = ChartBuilder::on(lower)
        .margin(5)
        .set_all_label_area_size(50)
        .caption("Energies", ("sans-serif", 40))
        .build_cartesian_2d(x_min..x_max, e_min..e_max)?;

    cc.configure_mesh()
        .x_labels(20)
        .y_labels(10)
        .disable_mesh()
        .x_label_formatter(&|v| format!("{:.1}", v))
        .y_label_formatter(&|v| format!("{:.1}", v))
        .draw()?;

    cc.draw_series(LineSeries::new(ts.iter().zip(kes.iter()).map(|(&x, &y)| (x, y)), &RED))?
        .label("Kinetic Energy")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    cc.draw_series(LineSeries::new(ts.iter().zip(pes.iter()).map(|(&x, &y)| (x, y)), &BLUE))?
        .label("Potential Energy")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    cc.configure_series_labels().border_style(BLACK).draw()?;

    root_area.present()?;

    println!("Saved plot to {out_file:?}");
    Ok(())
}

#[test]
fn entry_point() {
    main().unwrap()
}
