use clap::Parser;
use doppler_locate::{archive, dem, search, spectrogram, tracks, world};
use rayon::prelude::*;
use search::OFFSETS_S;
use serde::Serialize;
use std::path::PathBuf;
use std::time::Instant;
use tracks::{Frame, Types};

const SCAN_ZOOM: i32 = 6;
const VERIFY_ZOOM: i32 = 11;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    audio: PathBuf,
    #[arg(long, num_args = 1..)]
    archive: Vec<PathBuf>,
    #[arg(long)]
    adsb: Vec<PathBuf>,
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    archive_shift_days: f64,
    #[arg(long, default_value = "../ext/types.js")]
    types: PathBuf,
    #[arg(long, default_value = "../dem")]
    dem: PathBuf,
    #[arg(long, default_value_t = 2.048)]
    segment_s: f64,
    #[arg(long, default_value_t = 7000.0)]
    radius: f64,
    #[arg(long, value_enum, default_value_t = search::Statistic::Peak)]
    statistic: search::Statistic,
    #[arg(long, default_value_t = 3000)]
    screen: usize,
    #[arg(long, default_value_t = 5000)]
    null_cells: usize,
    #[arg(long, default_value_t = 20)]
    candidates: usize,
    #[arg(long, default_value_t = 3000.0)]
    separation: f64,
    #[arg(long, default_value_t = 600.0)]
    half: f64,
    #[arg(long, default_value_t = 100.0)]
    step: f64,
    #[arg(long, default_value_t = 5)]
    finalists: usize,
    #[arg(long, default_value_t = 25.0)]
    final_step: f64,
    #[arg(long)]
    out: PathBuf,
    #[arg(long)]
    spectrogram_out: Option<PathBuf>,
}

#[derive(Serialize)]
struct Located {
    lat: f64,
    lon: f64,
    significance: f64,
    offset_s: f64,
    passes: usize,
    scan_rank: usize,
    scan_significance: f64,
    screen_significance: f64,
}

const SCREEN_OFFSET_REACH_S: f64 = 4.0;
const REFINE_OFFSET_REACH_S: f64 = 2.0;

struct Best {
    point: (f64, f64),
    offset: f64,
    significance: f64,
    passes: usize,
}

fn grids(centers: impl Iterator<Item = (f64, f64)>, half: f64, step: f64) -> Vec<world::Grid> {
    let side = (2.0 * half / step).round() as usize + 1;
    centers.map(|(lat, lon)| world::Grid { frame: Frame::new(lat, lon), cell: step, side }).collect()
}

fn main() {
    let args = Args::parse();
    let clock = Instant::now();
    let log = |stage: &str| eprintln!("{stage} at {:.1}s", clock.elapsed().as_secs_f64());
    let spec = spectrogram::compute(&args.audio, args.segment_s);
    if let Some(path) = &args.spectrogram_out {
        spec.save(path);
    }
    let types = Types::load(&args.types);
    let shift = args.archive_shift_days * 86400.0;
    let window = (spec.first_frame_time - 120.0 - shift, spec.first_frame_time + spec.frames as f64 * spectrogram::HOP_S + 120.0 - shift);
    let mut tracks: Vec<tracks::GeoTrack> = archive::read(&args.archive, window, &|t| types.is_propeller(t)).iter().flat_map(|a| tracks::segments(&a.fixes)).collect();
    tracks.iter_mut().for_each(|t| t.tau.iter_mut().for_each(|tau| *tau += shift));
    tracks.extend(args.adsb.iter().flat_map(|path| tracks::load(path, &types)));
    log(&format!("{} track segments", tracks.len()));
    let runs = world::runs(&tracks, args.radius);
    let terrain = dem::Dem::fetch(&args.dem, SCAN_ZOOM, world::tile_corners(&runs).into_iter());
    let bank = search::TemplateBank::build(&spec, args.radius, args.statistic);
    log("templates");
    let cells = world::scan(&runs, &tracks, &bank, &terrain, args.radius);
    let mut ranked: Vec<(f64, f64, &world::Cell)> = cells
        .par_iter()
        .map(|c| {
            let (offset_index, significance) = c.significance();
            (significance, OFFSETS_S[offset_index], c)
        })
        .collect();
    ranked.par_sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    ranked.truncate(args.screen);
    log(&format!("{} cells scanned", cells.len()));
    let tile_runs = world::tile_runs(&runs);
    let exact = |probes: &[world::Probe], dem: &dem::Dem| world::exact(probes, &tile_runs, &tracks, &spec, dem, args.radius, args.statistic);
    let null_probes: Vec<world::Probe> = cells.iter().step_by(cells.len().div_ceil(args.null_cells)).map(|c| world::Probe::around((c.lat, c.lon), 0.0, 0.0)).collect();
    let null = world::Null::fit(&exact(&null_probes, &terrain));
    let screen_probes: Vec<world::Probe> = ranked.iter().map(|&(_, offset, c)| world::Probe::around((c.lat, c.lon), offset, SCREEN_OFFSET_REACH_S)).collect();
    let screen_points: Vec<(f64, f64)> = screen_probes.iter().map(|p| p.point).collect();
    let mut screened: Vec<(usize, f64, f64)> = exact(&screen_probes, &terrain)
        .iter()
        .zip(&screen_probes)
        .map(|(p, probe)| null.significance(p, &probe.offsets))
        .enumerate()
        .map(|(i, (offset, significance))| (i, offset, significance))
        .collect();
    screened.sort_by(|a, b| b.2.total_cmp(&a.2));
    log("screened");
    let mut picked: Vec<(usize, f64, f64)> = Vec::new();
    for &(rank, offset, significance) in &screened {
        if picked.len() < args.candidates && picked.iter().all(|&(p, _, _)| world::distance_m(screen_points[p], screen_points[rank]) > args.separation) {
            picked.push((rank, offset, significance));
        }
    }
    let coarse_grids = grids(picked.iter().map(|&(i, _, _)| screen_points[i]), args.half, args.step);
    let fine_terrain = dem::Dem::fetch(&args.dem, VERIFY_ZOOM, coarse_grids.iter().flat_map(|g| g.corners()));
    let best_in = |grid_list: &[world::Grid], offsets: &[f64]| -> Vec<Best> {
        let probes: Vec<world::Probe> = grid_list
            .iter()
            .zip(offsets)
            .flat_map(|(g, &offset)| (0..g.side * g.side).map(move |c| world::Probe::around(g.latlon(c), offset, REFINE_OFFSET_REACH_S)))
            .collect();
        let passes = exact(&probes, &fine_terrain);
        let per_grid = probes.len() / grid_list.len();
        probes
            .chunks(per_grid)
            .zip(passes.chunks(per_grid))
            .map(|(grid_probes, grid_passes)| {
                grid_probes
                    .iter()
                    .zip(grid_passes)
                    .map(|(probe, p)| {
                        let (offset, significance) = null.significance(p, &probe.offsets);
                        Best { point: probe.point, offset, significance, passes: p.len() }
                    })
                    .max_by(|a, b| a.significance.total_cmp(&b.significance))
                    .expect("nonempty grid")
            })
            .collect()
    };
    let coarse = best_in(&coarse_grids, &picked.iter().map(|&(_, offset, _)| offset).collect::<Vec<_>>());
    let mut order: Vec<usize> = (0..coarse.len()).collect();
    order.sort_by(|&a, &b| coarse[b].significance.total_cmp(&coarse[a].significance));
    order.truncate(args.finalists);
    let fine = best_in(&grids(order.iter().map(|&i| coarse[i].point), args.step, args.final_step), &order.iter().map(|&i| coarse[i].offset).collect::<Vec<_>>());
    log("refined");
    let mut located: Vec<Located> = order
        .iter()
        .zip(&fine)
        .map(|(&i, best)| {
            let (scan_rank, _, screen_significance) = picked[i];
            Located {
                lat: best.point.0,
                lon: best.point.1,
                significance: best.significance,
                offset_s: best.offset,
                passes: best.passes,
                scan_rank,
                scan_significance: ranked[scan_rank].0,
                screen_significance,
            }
        })
        .collect();
    located.sort_by(|a, b| b.significance.total_cmp(&a.significance));
    std::fs::write(&args.out, serde_json::to_vec(&located).expect("serializable result")).expect("writable output");
    for l in &located {
        println!("{:.5} {:.5} significance {:.2} offset {} passes {} scan rank {}", l.lat, l.lon, l.significance, l.offset_s, l.passes, l.scan_rank);
    }
}
