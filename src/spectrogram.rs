use crate::stats::median;
use rayon::prelude::*;
use realfft::RealFftPlanner;
use std::path::Path;

pub const LOG_STEP: f64 = 0.005;
pub const HOP_S: f64 = 0.5;
const LOG_FREQ_MIN: f64 = 40.0;
const LOG_FREQ_MAX: f64 = 4000.0;
const RUNNING_MEDIAN_HALF: usize = 60;
const LINE_MEDIAN_HALF: usize = 20;
const POWER_RATIO_CLIP: f64 = 12.0;

pub struct Spectrogram {
    pub first_frame_time: f64,
    pub frames: usize,
    pub bins: usize,
    pub data: Vec<f32>,
}

impl Spectrogram {
    pub fn row(&self, frame: usize) -> &[f32] {
        &self.data[frame * self.bins..(frame + 1) * self.bins]
    }

    pub fn pooled(&self, time: usize, freq: usize) -> Spectrogram {
        let (frames, bins) = (self.frames / time, self.bins / freq);
        let scale = 1.0 / (time * freq) as f32;
        let data = (0..frames)
            .flat_map(|f| {
                (0..bins).map(move |b| {
                    let rows = (f * time..(f + 1) * time).map(|r| self.row(r)[b * freq..(b + 1) * freq].iter().sum::<f32>());
                    rows.sum::<f32>() * scale
                })
            })
            .collect();
        let first_frame_time = self.first_frame_time + (time - 1) as f64 * HOP_S / 2.0;
        Spectrogram { first_frame_time, frames, bins, data }
    }

    pub fn save(&self, path: &Path) {
        let header = format!("{} {} {}\n", self.first_frame_time, self.frames, self.bins);
        let body = self.data.iter().flat_map(|v| v.to_le_bytes());
        std::fs::write(path, header.bytes().chain(body).collect::<Vec<u8>>()).expect("writable spectrogram");
    }
}

fn arange_len(start: f64, stop: f64, step: f64) -> usize {
    ((stop - start) / step).ceil() as usize
}

pub fn log_freqs() -> Vec<f64> {
    let start = LOG_FREQ_MIN.ln();
    (0..arange_len(start, LOG_FREQ_MAX.ln(), LOG_STEP)).map(|i| start + i as f64 * LOG_STEP).collect()
}

fn read_audio(path: &Path) -> (u32, Vec<f32>) {
    let mut reader = hound::WavReader::open(path).expect("readable wav");
    let rate = reader.spec().sample_rate;
    let samples = reader.samples::<i16>().map(|s| s.expect("pcm16 sample") as f32).collect();
    (rate, samples)
}

struct LogBins {
    interp: Vec<(usize, f64)>,
    edges: Vec<usize>,
    first_pooled: usize,
}

impl LogBins {
    fn new(rate: u32, segment: usize) -> LogBins {
        let frequency = |k: usize| k as f64 * rate as f64 / segment as f64;
        let log_bin: Vec<f64> = (1..=segment / 2).map(|k| frequency(k).ln()).collect();
        let centers = log_freqs();
        let interp = centers
            .iter()
            .map(|&u| {
                let j = log_bin.partition_point(|&v| v <= u) - 1;
                (j, (u - log_bin[j]) / (log_bin[j + 1] - log_bin[j]))
            })
            .collect();
        let edges: Vec<usize> = (0..=centers.len()).map(|b| (centers[0] + (b as f64 - 0.5) * LOG_STEP).exp()).map(|edge| ((edge / frequency(1)).ceil() as usize - 1).min(segment / 2)).collect();
        let first_pooled = edges.windows(2).position(|e| e[1] - e[0] >= 2).expect("fft finer than the log grid at high frequencies");
        LogBins { interp, edges, first_pooled }
    }

    fn row(&self, power: &[f32]) -> Vec<f64> {
        let level = |p: f32| (p as f64 + 1e-3).ln();
        (0..self.interp.len())
            .map(|b| {
                if b < self.first_pooled {
                    let (j, w) = self.interp[b];
                    level(power[j]) * (1.0 - w) + level(power[j + 1]) * w
                } else {
                    level(power[self.edges[b]..self.edges[b + 1]].iter().copied().fold(0.0, f32::max))
                }
            })
            .collect()
    }
}

fn log_power_frames(samples: &[f32], hop: usize, segment: usize, bins: &LogBins) -> Vec<Vec<f64>> {
    let window: Vec<f32> = (0..segment).map(|n| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / segment as f64).cos()) as f32).collect();
    let window_sum: f32 = window.iter().sum();
    let fft = RealFftPlanner::<f32>::new().plan_fft_forward(segment);
    let frames = (samples.len() - segment) / hop + 1;
    (0..frames)
        .into_par_iter()
        .map_init(
            || (fft.make_input_vec(), fft.make_output_vec(), fft.make_scratch_vec(), Vec::new()),
            |(input, output, scratch, power), i| {
                for (dst, (x, w)) in input.iter_mut().zip(samples[i * hop..i * hop + segment].iter().zip(&window)) {
                    *dst = x * w;
                }
                fft.process_with_scratch(input, output, scratch).expect("fft sizes match");
                power.clear();
                power.extend(output[1..].iter().map(|z| (z / window_sum).norm_sqr()));
                let mut row = bins.row(power);
                let med = median(&mut row.clone());
                row.iter_mut().for_each(|v| *v -= med);
                row
            },
        )
        .collect()
}

fn running_median(row: &[f64], half: usize) -> Vec<f64> {
    let n = row.len();
    let padded = |i: usize| row[i.saturating_sub(half).min(n - 1)];
    let mut window: Vec<f64> = (0..2 * half + 1).map(padded).collect();
    window.sort_by(f64::total_cmp);
    let mut out = Vec::with_capacity(n);
    for t in 0..n {
        out.push(window[half]);
        if t + 1 == n {
            break;
        }
        let outgoing = padded(t);
        let incoming = padded(t + 2 * half + 1);
        let at = window.partition_point(|v| v.total_cmp(&outgoing).is_lt());
        window.remove(at);
        let at = window.partition_point(|v| v.total_cmp(&incoming).is_lt());
        window.insert(at, incoming);
    }
    out
}

pub fn compute(audio: &Path, segment_s: f64) -> Spectrogram {
    let start: f64 = std::fs::read_to_string(audio.with_extension("wav.start")).expect("start file").trim().parse().expect("start time");
    let (rate, samples) = read_audio(audio);
    let hop = (HOP_S * rate as f64) as usize;
    let segment = (segment_s * rate as f64) as usize;
    let mut by_frame = log_power_frames(&samples, hop, segment, &LogBins::new(rate, segment));
    let frames = by_frame.len();
    let bins = by_frame[0].len();
    let by_bin: Vec<Vec<f64>> = (0..bins)
        .into_par_iter()
        .map(|b| {
            let series: Vec<f64> = by_frame.iter().map(|r| r[b]).collect();
            let baseline = running_median(&series, RUNNING_MEDIAN_HALF);
            series.iter().zip(baseline).map(|(v, m)| v - m).collect()
        })
        .collect();
    by_frame.par_iter_mut().enumerate().for_each(|(t, row)| {
        for (b, v) in row.iter_mut().enumerate() {
            *v = by_bin[b][t];
        }
        let lines = running_median(row, LINE_MEDIAN_HALF);
        row.iter_mut().zip(lines).for_each(|(v, m)| *v = (*v - m).exp().min(POWER_RATIO_CLIP) - 1.0);
    });
    let scales: Vec<f64> =
        (0..bins).into_par_iter().map(|b| 1.4826 * median(&mut by_frame.iter().map(|row| row[b].abs()).collect::<Vec<_>>())).map(|scale: f64| scale.max(f64::MIN_POSITIVE)).collect();
    let data = by_frame.iter().flat_map(|row| row.iter().zip(&scales).map(|(v, scale)| (v / scale).clamp(-3.0, 6.0) as f32)).collect();
    Spectrogram { first_frame_time: start + (segment / 2) as f64 / rate as f64, frames, bins, data }
}
