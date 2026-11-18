use clap::Parser;
use doppler_locate::archive;
use doppler_locate::tracks::Frame;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(long, num_args = 1..)]
    archive: Vec<PathBuf>,
    #[arg(long)]
    sites: PathBuf,
    #[arg(long)]
    start: f64,
    #[arg(long)]
    end: f64,
    #[arg(long, default_value_t = 25000.0)]
    radius: f64,
    #[arg(long)]
    out: PathBuf,
}

#[derive(Deserialize)]
struct Site {
    lat: f64,
    lon: f64,
}

#[derive(Serialize, Default)]
struct Trace {
    hex: String,
    r#type: String,
    t: Vec<f64>,
    lat: Vec<f64>,
    lon: Vec<f64>,
    baro: Vec<Option<f64>>,
    geom: Vec<Option<f64>>,
}

fn main() {
    let args = Args::parse();
    let sites: HashMap<String, Site> = serde_json::from_slice(&std::fs::read(&args.sites).expect("sites file")).expect("sites json");
    let frames: Vec<(&String, Frame)> = sites.iter().map(|(name, s)| (name, Frame::new(s.lat, s.lon))).collect();
    let aircraft = archive::read(&args.archive, (args.start, args.end), &|_| true);
    std::fs::create_dir_all(&args.out).expect("output directory");
    for (name, frame) in &frames {
        let traces: Vec<Trace> = aircraft
            .iter()
            .filter_map(|a| {
                let near: Vec<_> = a
                    .fixes
                    .iter()
                    .filter(|f| {
                        let (x, y) = frame.to_xy(f.lat, f.lon);
                        x.hypot(y) < args.radius
                    })
                    .collect();
                (near.len() >= 3).then(|| Trace {
                    hex: a.hex.clone(),
                    r#type: a.type_code.clone(),
                    t: near.iter().map(|f| f.t).collect(),
                    lat: near.iter().map(|f| f.lat).collect(),
                    lon: near.iter().map(|f| f.lon).collect(),
                    baro: near.iter().map(|f| f.baro).collect(),
                    geom: near.iter().map(|f| f.geom).collect(),
                })
            })
            .collect();
        eprintln!("{name}: {} aircraft", traces.len());
        std::fs::write(args.out.join(format!("{name}.json")), serde_json::to_vec(&traces).expect("serializable traces")).expect("writable traces");
    }
}
