use crate::spectrogram::{HOP_S, LOG_STEP, Spectrogram};
use crate::tracks::STEP_S;
use crate::stats::{robust_peak_z, round_half_even};
use crate::tracks::Track;
use rayon::prelude::*;

pub const SPEED_OF_SOUND: f64 = 343.0;
const OFFSET_COUNT: usize = 31;
pub const OFFSETS_S: [f64; OFFSET_COUNT] = {
    let mut offsets = [0.0; OFFSET_COUNT];
    let mut i = 0;
    while i < OFFSET_COUNT {
        offsets[i] = -30.0 + 2.0 * i as f64;
        i += 1;
    }
    offsets
};
const MIN_NEAR_SAMPLES: usize = 20;
const MIN_WEIGHT: f64 = 0.1;
const MIN_PROFILE_BINS: i64 = 50;

const TEMPLATE_TIME_POOL: usize = 2;
const TEMPLATE_FREQ_POOL: usize = 4;
const TEMPLATE_SPEED_MIN: f64 = 20.0;
const TEMPLATE_SPEED_STEP: f64 = 4.0;
const TEMPLATE_SPEEDS: usize = 23;
const TEMPLATE_DISTANCE_MIN: f64 = 80.0;
const TEMPLATE_DISTANCE_RATIO: f64 = 1.1;

pub struct Geometry {
    pub ground: f64,
    pub radius: f64,
}



#[derive(Clone, Copy, clap::ValueEnum)]
pub enum Statistic {
    Peak,
    Glr,
}

pub struct Workspace {
    sum: Vec<f32>,
    sum_sq: Vec<f32>,
    profile: Vec<f64>,
    scratch: Vec<f64>,
    statistic: Statistic,
}

impl Workspace {
    pub fn new(statistic: Statistic) -> Self {
        Workspace { sum: Vec::new(), sum_sq: Vec::new(), profile: Vec::new(), scratch: Vec::new(), statistic }
    }
}

fn gradient(values: &[f64], coords: &[f64]) -> Vec<f64> {
    let n = values.len();
    (0..n)
        .map(|i| match i {
            0 => (values[1] - values[0]) / (coords[1] - coords[0]),
            _ if i == n - 1 => (values[n - 1] - values[n - 2]) / (coords[n - 1] - coords[n - 2]),
            _ => {
                let hs = coords[i] - coords[i - 1];
                let hd = coords[i + 1] - coords[i];
                (hs * hs * values[i + 1] + (hd * hd - hs * hs) * values[i] - hd * hd * values[i - 1]) / (hs * hd * (hd + hs))
            }
        })
        .collect()
}

struct Weights {
    values: Vec<f32>,
    sum: f64,
    effective: f64,
}

impl Weights {
    fn new(values: Vec<f64>) -> Weights {
        let sum: f64 = values.iter().sum();
        let effective = sum * sum / values.iter().map(|w| w * w).sum::<f64>();
        Weights { values: values.into_iter().map(|w| w as f32).collect(), sum, effective }
    }
}

fn pass_score(spec: &Spectrogram, frames: &[usize], shifts: &[i64], weights: &Weights, sign: i64, ws: &mut Workspace) -> f64 {
    let (lo, hi) = if sign > 0 {
        (0.max(-shifts.iter().min().unwrap()), spec.bins as i64 - shifts.iter().max().unwrap())
    } else {
        (0.max(*shifts.iter().max().unwrap()), spec.bins as i64 + shifts.iter().min().unwrap())
    };
    let len = (hi - lo) as usize;
    ws.sum.clear();
    ws.sum.resize(len, 0.0);
    ws.sum_sq.clear();
    ws.sum_sq.resize(len, 0.0);
    for ((&frame, &shift), &w) in frames.iter().zip(shifts).zip(&weights.values) {
        let start = (lo + sign * shift) as usize;
        let row = &spec.row(frame)[start..start + len];
        for ((acc, acc_sq), &s) in ws.sum.iter_mut().zip(ws.sum_sq.iter_mut()).zip(row) {
            *acc += w * s;
            *acc_sq += w * s * s;
        }
    }
    match ws.statistic {
        Statistic::Peak => {
            ws.profile.clear();
            ws.profile.extend(ws.sum.iter().map(|&v| v as f64 / weights.sum));
            robust_peak_z(&ws.profile, &mut ws.scratch)
        }
        Statistic::Glr => {
            let explained: f64 = ws
                .sum
                .iter()
                .zip(&ws.sum_sq)
                .map(|(&sum, &sum_sq)| {
                    let mean = sum as f64 / weights.sum;
                    let variance = (sum_sq as f64 / weights.sum - mean * mean).max(0.0);
                    let t = mean / (variance / weights.effective + 1e-9).sqrt();
                    t.max(0.0).powi(2)
                })
                .sum();
            explained / len as f64
        }
    }
}

pub fn evaluate(point: [f64; 2], track: &Track, spec: &Spectrogram, geometry: &Geometry, offsets: &[f64], ws: &mut Workspace) -> Option<Vec<f64>> {
    let r: Vec<f64> = track
        .pos
        .iter()
        .map(|p| ((p[0] - point[0]).powi(2) + (p[1] - point[1]).powi(2) + (p[2] - geometry.ground).powi(2)).sqrt())
        .collect();
    let r_min = r.iter().copied().fold(f64::INFINITY, f64::min);
    let cutoff = geometry.radius.min(r_min / MIN_WEIGHT.sqrt());
    let near: Vec<usize> = (0..r.len()).filter(|&i| r[i] < cutoff).collect();
    if near.len() < MIN_NEAR_SAMPLES {
        return None;
    }
    let radial_speed = gradient(&r, &track.tau);
    let shifts: Vec<i64> = near
        .iter()
        .map(|&i| round_half_even((SPEED_OF_SOUND / (SPEED_OF_SOUND + radial_speed[i])).ln() / LOG_STEP))
        .collect();
    let base_frames: Vec<i64> = near
        .iter()
        .map(|&i| round_half_even((track.tau[i] + r[i] / SPEED_OF_SOUND - spec.first_frame_time) / HOP_S))
        .collect();
    let lag = (OFFSETS_S[OFFSETS_S.len() - 1] / HOP_S) as i64;
    if base_frames.iter().min().unwrap() - lag < 0 || base_frames.iter().max().unwrap() + lag >= spec.frames as i64 {
        return None;
    }
    let (min_shift, max_shift) = (*shifts.iter().min().unwrap(), *shifts.iter().max().unwrap());
    let bins = spec.bins as i64;
    if (bins - max_shift) - 0.max(-min_shift) < MIN_PROFILE_BINS || (bins + min_shift) - 0.max(max_shift) < MIN_PROFILE_BINS {
        return None;
    }
    let weights = Weights::new(near.iter().map(|&i| (r_min / r[i]).powi(2)).collect());
    let mut frames = vec![0usize; near.len()];
    Some(
        offsets
            .iter()
            .map(|offset| {
                let delta = (offset / HOP_S) as i64;
                for (f, &b) in frames.iter_mut().zip(&base_frames) {
                    *f = (b + delta) as usize;
                }
                pass_score(spec, &frames, &shifts, &weights, 1, ws) - pass_score(spec, &frames, &shifts, &weights, -1, ws)
            })
            .collect(),
    )
}

struct Shape {
    lags: Vec<i64>,
    shifts: Vec<i64>,
    weights: Weights,
}

impl Shape {
    fn of(speed: f64, distance: f64, radius: f64, hop: f64, step: f64, min_bins: i64, bins: usize) -> Option<Shape> {
        let cutoff = radius.min(distance / MIN_WEIGHT.sqrt());
        let reach = (cutoff * cutoff - distance * distance).sqrt() / speed;
        let half = (reach / hop).ceil() as i64;
        let (mut lags, mut shifts, mut weights) = (Vec::new(), Vec::new(), Vec::new());
        for j in -half..=half {
            let tau = j as f64 * hop;
            let r = distance.hypot(speed * tau);
            if r >= cutoff {
                continue;
            }
            lags.push(round_half_even((tau + (r - distance) / SPEED_OF_SOUND) / hop));
            shifts.push(round_half_even((SPEED_OF_SOUND / (SPEED_OF_SOUND + speed * speed * tau / r)).ln() / step));
            weights.push((distance / r).powi(2));
        }
        let (min_shift, max_shift) = (*shifts.iter().min()?, *shifts.iter().max()?);
        let bins = bins as i64;
        let narrow = (bins - max_shift) - 0.max(-min_shift) < min_bins || (bins + min_shift) - 0.max(max_shift) < min_bins;
        if (lags.len() as f64) < MIN_NEAR_SAMPLES as f64 * STEP_S / hop || narrow {
            return None;
        }
        Some(Shape { lags, shifts, weights: Weights::new(weights) })
    }
}

pub struct TemplateBank {
    pub first_frame_time: f64,
    pub hop: f64,
    frames: usize,
    distances: usize,
    scores: Vec<f32>,
}

impl TemplateBank {
    pub fn build(spec: &Spectrogram, radius: f64, statistic: Statistic) -> TemplateBank {
        let pooled = spec.pooled(TEMPLATE_TIME_POOL, TEMPLATE_FREQ_POOL);
        let hop = HOP_S * TEMPLATE_TIME_POOL as f64;
        let step = LOG_STEP * TEMPLATE_FREQ_POOL as f64;
        let min_bins = MIN_PROFILE_BINS / TEMPLATE_FREQ_POOL as i64;
        let distances = (radius / TEMPLATE_DISTANCE_MIN).ln().div_euclid(TEMPLATE_DISTANCE_RATIO.ln()) as usize + 1;
        let shapes: Vec<Option<Shape>> = (0..TEMPLATE_SPEEDS * distances)
            .map(|k| {
                let speed = TEMPLATE_SPEED_MIN + (k / distances) as f64 * TEMPLATE_SPEED_STEP;
                let distance = TEMPLATE_DISTANCE_MIN * TEMPLATE_DISTANCE_RATIO.powi((k % distances) as i32);
                Shape::of(speed, distance, radius, hop, step, min_bins, pooled.bins)
            })
            .collect();
        let frames = pooled.frames;
        let scores = shapes
            .par_iter()
            .flat_map_iter(|shape| {
                let mut ws = Workspace::new(statistic);
                let mut rows = Vec::new();
                let raw: Vec<f32> = (0..frames as i64)
                    .map(|anchor| {
                        let Some(shape) = shape else {
                            return f32::NAN;
                        };
                        let (first, last) = (anchor + shape.lags[0], anchor + shape.lags[shape.lags.len() - 1]);
                        if first < 0 || last >= frames as i64 {
                            return f32::NAN;
                        }
                        rows.clear();
                        rows.extend(shape.lags.iter().map(|&lag| (anchor + lag) as usize));
                        let matched = pass_score(&pooled, &rows, &shape.shifts, &shape.weights, 1, &mut ws);
                        (matched - pass_score(&pooled, &rows, &shape.shifts, &shape.weights, -1, &mut ws)) as f32
                    })
                    .collect();
                let mut valid: Vec<f32> = raw.iter().copied().filter(|v| !v.is_nan()).collect();
                valid.sort_by(f32::total_cmp);
                let count = valid.len() as f64 + 1.0;
                raw.into_iter().map(move |v| {
                    let above = valid.len() - valid.partition_point(|&x| x < v);
                    if v.is_nan() { f32::NAN } else { -((above + 1) as f64 / count).ln() as f32 }
                })
            })
            .collect();
        TemplateBank { first_frame_time: pooled.first_frame_time, hop, frames, distances, scores }
    }

    pub fn score(&self, speed: f64, distance: f64, frame: i64) -> Option<f32> {
        let speed_index = (((speed - TEMPLATE_SPEED_MIN) / TEMPLATE_SPEED_STEP).round().max(0.0) as usize).min(TEMPLATE_SPEEDS - 1);
        let distance_index = ((distance / TEMPLATE_DISTANCE_MIN).ln() / TEMPLATE_DISTANCE_RATIO.ln()).round().max(0.0) as usize;
        if distance_index >= self.distances || frame < 0 || frame >= self.frames as i64 {
            return None;
        }
        let value = self.scores[(speed_index * self.distances + distance_index) * self.frames + frame as usize];
        (!value.is_nan()).then_some(value)
    }
}
