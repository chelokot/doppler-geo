use crate::dem::Dem;
use crate::search::{Geometry, OFFSETS_S, SPEED_OF_SOUND, Statistic, TemplateBank, Workspace, evaluate};
use crate::spectrogram::Spectrogram;
use crate::stats::fisher_significance;
use crate::tracks::{Frame, GeoTrack, STEP_S, Track};
use rayon::prelude::*;
use std::collections::HashMap;

const SCAN_CELL_M: f64 = 500.0;
const TILE_CELLS: usize = 64;
const TILE_M: f64 = SCAN_CELL_M * TILE_CELLS as f64;
const METERS_PER_DEGREE_LAT: f64 = 110540.0;
const METERS_PER_DEGREE_LON: f64 = 111320.0;
const MARK_STRIDE: usize = 8;
const TAIL_FRACTION: f64 = 0.05;
const ASSIGN_MARGIN_M: f64 = 600.0;

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileKey {
    band: i32,
    column: i32,
}

impl TileKey {
    fn lon_span(band: i32) -> f64 {
        let center = (band as f64 + 0.5) * TILE_M / METERS_PER_DEGREE_LAT;
        TILE_M / (METERS_PER_DEGREE_LON * center.to_radians().cos())
    }

    pub fn of(lat: f64, lon: f64) -> TileKey {
        let band = (lat * METERS_PER_DEGREE_LAT / TILE_M).floor() as i32;
        TileKey { band, column: (lon / Self::lon_span(band)).floor() as i32 }
    }

    pub fn touched(lat: f64, lon: f64, reach: f64) -> Vec<TileKey> {
        let (dlat, dlon) = (reach / METERS_PER_DEGREE_LAT, reach / (METERS_PER_DEGREE_LON * lat.to_radians().cos()));
        let mut keys: Vec<TileKey> = [-1.0, 1.0].into_iter().flat_map(|sy| [-1.0, 1.0].map(|sx| TileKey::of(lat + sy * dlat, lon + sx * dlon))).collect();
        keys.sort();
        keys.dedup();
        keys
    }

    fn grid(&self) -> Grid {
        let frame = Frame::new((self.band as f64 + 0.5) * TILE_M / METERS_PER_DEGREE_LAT, (self.column as f64 + 0.5) * Self::lon_span(self.band));
        Grid { frame, cell: SCAN_CELL_M, side: TILE_CELLS }
    }
}

pub struct Grid {
    pub frame: Frame,
    pub cell: f64,
    pub side: usize,
}

impl Grid {
    fn half(&self) -> f64 {
        self.cell * self.side as f64 / 2.0
    }

    fn center(&self, c: usize) -> [f64; 2] {
        [((c % self.side) as f64 + 0.5) * self.cell - self.half(), ((c / self.side) as f64 + 0.5) * self.cell - self.half()]
    }

    pub fn corners(&self) -> [(f64, f64); 4] {
        [0, self.side - 1, self.side * (self.side - 1), self.side * self.side - 1].map(|c| self.latlon(c))
    }

    pub fn latlon(&self, c: usize) -> (f64, f64) {
        let [x, y] = self.center(c);
        self.frame.to_latlon(x, y)
    }
}

pub struct Span {
    pub track: usize,
    pub start: usize,
    pub end: usize,
}

pub struct Run {
    pub tile: TileKey,
    pub span: Span,
}

pub struct Cell {
    pub lat: f64,
    pub lon: f64,
    pub total: [f32; OFFSETS_S.len()],
    pub passes: u32,
}

impl Cell {
    pub fn significance(&self) -> (usize, f64) {
        self.total.iter().enumerate().fold((0, f64::NEG_INFINITY), |best, (k, &t)| {
            let significance = fisher_significance(t as f64, self.passes as usize);
            if significance > best.1 { (k, significance) } else { best }
        })
    }
}

pub fn runs(tracks: &[GeoTrack], radius: f64) -> Vec<Run> {
    let mut runs: Vec<Run> = tracks
        .par_iter()
        .enumerate()
        .flat_map_iter(|(track, geo)| {
            let n = geo.tau.len();
            let mut open: HashMap<TileKey, Vec<(usize, usize)>> = HashMap::new();
            for i in (0..n).step_by(MARK_STRIDE) {
                let [lat, lon, _] = geo.pos[i];
                let (start, end) = (i.saturating_sub(MARK_STRIDE), (i + MARK_STRIDE + 1).min(n));
                for tile in TileKey::touched(lat, lon, radius + ASSIGN_MARGIN_M) {
                    let spans = open.entry(tile).or_default();
                    match spans.last_mut() {
                        Some(last) if last.1 >= start => last.1 = end,
                        _ => spans.push((start, end)),
                    }
                }
            }
            open.into_iter().flat_map(move |(tile, spans)| spans.into_iter().map(move |(start, end)| Run { tile, span: Span { track, start, end } }))
        })
        .collect();
    runs.par_sort_unstable_by_key(|r| (r.tile, r.span.track, r.span.start));
    runs
}

pub fn score_grid<'a>(grid: &Grid, spans: impl Iterator<Item = &'a Span>, tracks: &[GeoTrack], bank: &TemplateBank, dem: &Dem, radius: f64) -> Vec<Cell> {
    let cells = grid.side * grid.side;
    let ground: Vec<f64> = (0..cells)
        .map(|c| {
            let (lat, lon) = grid.latlon(c);
            dem.ground(lat, lon)
        })
        .collect();
    let mut total = vec![[0f32; OFFSETS_S.len()]; cells];
    let mut passes = vec![0u32; cells];
    let mut nearest = vec![(f64::INFINITY, 0usize); cells];
    let mut touched: Vec<usize> = Vec::new();
    let lags = OFFSETS_S.map(|o| (o / bank.hop).round() as i64);
    let reach_cells = (radius / grid.cell).ceil() as i64;
    let bound = grid.half() + radius;
    for span in spans {
        let geo = &tracks[span.track];
        let local: Vec<[f64; 3]> = geo.pos[span.start..span.end]
            .iter()
            .map(|&[lat, lon, alt]| {
                let (x, y) = grid.frame.to_xy(lat, lon);
                [x, y, alt]
            })
            .collect();
        for (i, p) in local.iter().enumerate().step_by(MARK_STRIDE) {
            if p[0].abs() > bound || p[1].abs() > bound {
                continue;
            }
            let (cx, cy) = (((p[0] + grid.half()) / grid.cell).floor() as i64, ((p[1] + grid.half()) / grid.cell).floor() as i64);
            for iy in (cy - reach_cells).max(0)..(cy + reach_cells + 1).min(grid.side as i64) {
                for ix in (cx - reach_cells).max(0)..(cx + reach_cells + 1).min(grid.side as i64) {
                    let c = iy as usize * grid.side + ix as usize;
                    let [x, y] = grid.center(c);
                    let d2 = (x - p[0]).powi(2) + (y - p[1]).powi(2);
                    if d2 < radius * radius && d2 < nearest[c].0 {
                        if nearest[c].0.is_infinite() {
                            touched.push(c);
                        }
                        nearest[c] = (d2, i);
                    }
                }
            }
        }
        for &c in &touched {
            let [x, y] = grid.center(c);
            let slant2 = |p: &[f64; 3]| (x - p[0]).powi(2) + (y - p[1]).powi(2) + (p[2] - ground[c]).powi(2);
            let around = nearest[c].1.saturating_sub(MARK_STRIDE)..(nearest[c].1 + MARK_STRIDE + 1).min(local.len());
            let i = around.min_by(|&a, &b| slant2(&local[a]).total_cmp(&slant2(&local[b]))).expect("nonempty neighborhood");
            nearest[c] = (f64::INFINITY, 0);
            if i == 0 || i + 1 == local.len() {
                continue;
            }
            let distance = slant2(&local[i]).sqrt();
            let speed = (0..3).map(|k| (local[i + 1][k] - local[i - 1][k]).powi(2)).sum::<f64>().sqrt() / (2.0 * STEP_S);
            let anchor = ((geo.tau[span.start + i] + distance / SPEED_OF_SOUND - bank.first_frame_time) / bank.hop).round() as i64;
            let scores: Option<Vec<f32>> = lags.iter().map(|&lag| bank.score(speed, distance, anchor + lag)).collect();
            if let Some(scores) = scores {
                total[c].iter_mut().zip(scores).for_each(|(t, s)| *t += s);
                passes[c] += 1;
            }
        }
        touched.clear();
    }
    (0..cells)
        .filter(|&c| passes[c] > 0)
        .map(|c| {
            let (lat, lon) = grid.latlon(c);
            Cell { lat, lon, total: total[c], passes: passes[c] }
        })
        .collect()
}

pub fn scan(runs: &[Run], tracks: &[GeoTrack], bank: &TemplateBank, dem: &Dem, radius: f64) -> Vec<Cell> {
    let groups: Vec<&[Run]> = runs.chunk_by(|a, b| a.tile == b.tile).collect();
    groups.par_iter().flat_map_iter(|group| score_grid(&group[0].tile.grid(), group.iter().map(|r| &r.span), tracks, bank, dem, radius)).collect()
}

pub fn tile_corners(runs: &[Run]) -> Vec<(f64, f64)> {
    let mut tiles: Vec<TileKey> = runs.iter().map(|r| r.tile).collect();
    tiles.dedup();
    tiles.iter().flat_map(|tile| tile.grid().corners()).collect()
}

pub fn distance_m(a: (f64, f64), b: (f64, f64)) -> f64 {
    let (x, y) = Frame::new(a.0, a.1).to_xy(b.0, b.1);
    x.hypot(y)
}

pub struct Null {
    sorted: Vec<f64>,
    threshold: f64,
    scale: f64,
}

impl Null {
    pub fn fit(points: &[Vec<Vec<f64>>]) -> Null {
        let mut sorted: Vec<f64> = points.iter().flatten().flatten().copied().collect();
        sorted.sort_by(f64::total_cmp);
        let start = ((1.0 - TAIL_FRACTION) * sorted.len() as f64) as usize;
        let threshold = sorted[start];
        let scale = sorted[start..].iter().map(|v| v - threshold).sum::<f64>() / (sorted.len() - start) as f64;
        Null { sorted, threshold, scale }
    }

    fn ln_tail(&self, score: f64) -> f64 {
        if score > self.threshold {
            return TAIL_FRACTION.ln() - (score - self.threshold) / self.scale;
        }
        let above = self.sorted.len() - self.sorted.partition_point(|&v| v < score);
        ((above + 1) as f64 / (self.sorted.len() + 1) as f64).ln()
    }

    pub fn significance(&self, passes: &[Vec<f64>], offsets: &[f64]) -> (f64, f64) {
        offsets.iter().enumerate().fold((offsets[0], f64::NEG_INFINITY), |best, (k, &offset)| {
            let significance = fisher_significance(-passes.iter().map(|p| self.ln_tail(p[k])).sum::<f64>(), passes.len());
            if significance > best.1 { (offset, significance) } else { best }
        })
    }
}

pub fn tile_runs(runs: &[Run]) -> HashMap<TileKey, &[Run]> {
    runs.chunk_by(|a, b| a.tile == b.tile).map(|group| (group[0].tile, group)).collect()
}

pub struct Probe {
    pub point: (f64, f64),
    pub offsets: Vec<f64>,
}

impl Probe {
    pub fn around(point: (f64, f64), offset: f64, reach: f64) -> Probe {
        let offsets = OFFSETS_S.iter().copied().filter(|o| (o - offset).abs() <= reach).collect();
        Probe { point, offsets }
    }
}

pub fn exact(probes: &[Probe], tile_runs: &HashMap<TileKey, &[Run]>, tracks: &[GeoTrack], spec: &Spectrogram, dem: &Dem, radius: f64, statistic: Statistic) -> Vec<Vec<Vec<f64>>> {
    probes
        .par_iter()
        .map_init(
            || Workspace::new(statistic),
            |ws, probe| {
                let (lat, lon) = probe.point;
                let frame = Frame::new(lat, lon);
                let geometry = Geometry { ground: dem.ground(lat, lon), radius };
                let runs = tile_runs.get(&TileKey::of(lat, lon)).copied().unwrap_or_default();
                runs.iter()
                    .filter_map(|run| {
                        let geo = &tracks[run.span.track];
                        let span = run.span.start..run.span.end;
                        let pos = geo.pos[span.clone()]
                            .iter()
                            .map(|&[plat, plon, alt]| {
                                let (x, y) = frame.to_xy(plat, plon);
                                [x, y, alt]
                            })
                            .collect();
                        evaluate([0.0, 0.0], &Track { tau: geo.tau[span].to_vec(), pos }, spec, &geometry, &probe.offsets, ws)
                    })
                    .collect()
            },
        )
        .collect()
}
