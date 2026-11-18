use doppler_locate::dem::mercator;
use doppler_locate::tracks::{FEET, Frame};
use flate2::Compression;
use flate2::write::GzEncoder;
use serde::Deserialize;
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::process::Command;

const SITE: (f64, f64) = (46.2, 8.9);
const RECORDING_START: f64 = 1_800_000_000.0;
const RECORDING_S: f64 = 1800.0;
const FLIGHT_S: (f64, f64) = (-150.0, 1950.0);
const RATE: u32 = 16000;
const SPEED_OF_SOUND: f64 = 343.0;
const TONE_AT_1KM: f64 = 0.006;
const HARMONICS: [f64; 4] = [1.0, 0.7, 0.5, 0.35];
const NOISE: f64 = 0.05;
const PCM_SCALE: f64 = 8000.0;
const EMISSION_STEP_S: f64 = 0.05;
const TERRAIN_MARGIN_DEG: f64 = 0.6;
const SEA_LEVEL_TERRARIUM: [u8; 3] = [128, 0, 0];
const SILENT_AIRCRAFT: usize = 200;
const SILENT_DISTANCE_M: (f64, f64) = (10_000.0, 150_000.0);

struct Pass {
    hex: String,
    closest_s: f64,
    closest_xy: [f64; 2],
    heading_deg: f64,
    altitude: f64,
    speed: f64,
}

fn abeam(heading_deg: f64, miss_m: f64) -> [f64; 2] {
    let heading = heading_deg.to_radians();
    [heading.cos() * miss_m, -heading.sin() * miss_m]
}

fn audible() -> [(Pass, f64); 4] {
    [
        (Pass { hex: "b2c3d4".into(), closest_s: 300.0, closest_xy: abeam(200.0, 1200.0), heading_deg: 200.0, altitude: 600.0, speed: 60.0 }, 105.0),
        (Pass { hex: "a1b2c3".into(), closest_s: 600.0, closest_xy: abeam(90.0, 600.0), heading_deg: 90.0, altitude: 300.0, speed: 45.0 }, 92.0),
        (Pass { hex: "c3d4e5".into(), closest_s: 1200.0, closest_xy: abeam(340.0, -900.0), heading_deg: 340.0, altitude: 450.0, speed: 55.0 }, 118.0),
        (Pass { hex: "e5f6a7".into(), closest_s: 1500.0, closest_xy: abeam(45.0, -400.0), heading_deg: 45.0, altitude: 250.0, speed: 40.0 }, 80.0),
    ]
}

fn silent() -> Vec<Pass> {
    let mut rng = Noise(0x2545_F491_4F6C_DD1D);
    (0..SILENT_AIRCRAFT)
        .map(|i| {
            let bearing = 2.0 * PI * rng.uniform();
            let distance = SILENT_DISTANCE_M.0 + (SILENT_DISTANCE_M.1 - SILENT_DISTANCE_M.0) * rng.uniform();
            Pass {
                hex: format!("d{i:05x}"),
                closest_s: RECORDING_S * rng.uniform(),
                closest_xy: [distance * bearing.sin(), distance * bearing.cos()],
                heading_deg: bearing.to_degrees() + 90.0,
                altitude: 300.0 + 1200.0 * rng.uniform(),
                speed: 40.0 + 40.0 * rng.uniform(),
            }
        })
        .collect()
}

impl Pass {
    fn position(&self, t: f64) -> [f64; 3] {
        let heading = self.heading_deg.to_radians();
        let flown = (t - RECORDING_START - self.closest_s) * self.speed;
        [self.closest_xy[0] + heading.sin() * flown, self.closest_xy[1] + heading.cos() * flown, self.altitude]
    }

    fn range(&self, t: f64) -> f64 {
        self.position(t).iter().map(|v| v * v).sum::<f64>().sqrt()
    }
}

#[derive(Clone, Copy)]
struct Bounds {
    south: f64,
    north: f64,
    west: f64,
    east: f64,
}

#[derive(Deserialize)]
struct Located {
    lat: f64,
    lon: f64,
    significance: f64,
}

struct Noise(u64);

impl Noise {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn gaussian(&mut self) -> f64 {
        (-2.0 * (1.0 - self.uniform()).ln()).sqrt() * (2.0 * PI * self.uniform()).cos()
    }
}

fn received((pass, blade_pass_hz): &(Pass, f64)) -> Vec<f64> {
    let emissions = ((FLIGHT_S.1 - FLIGHT_S.0) / EMISSION_STEP_S) as usize;
    let emitted: Vec<f64> = (0..emissions).map(|k| RECORDING_START + FLIGHT_S.0 + k as f64 * EMISSION_STEP_S).collect();
    let ranges: Vec<f64> = emitted.iter().map(|&t| pass.range(t)).collect();
    let arrivals: Vec<f64> = emitted.iter().zip(&ranges).map(|(t, r)| t + r / SPEED_OF_SOUND).collect();
    (0..(RECORDING_S * RATE as f64) as usize)
        .map(|n| {
            let t = RECORDING_START + n as f64 / RATE as f64;
            let k = arrivals.partition_point(|&a| a <= t);
            if k == 0 || k == arrivals.len() {
                return 0.0;
            }
            let w = (t - arrivals[k - 1]) / (arrivals[k] - arrivals[k - 1]);
            let phase = 2.0 * PI * blade_pass_hz * (emitted[k - 1] + w * EMISSION_STEP_S - RECORDING_START);
            let r = ranges[k - 1] + w * (ranges[k] - ranges[k - 1]);
            TONE_AT_1KM * 1000.0 / r * HARMONICS.iter().enumerate().map(|(h, a)| a * ((h + 1) as f64 * phase).sin()).sum::<f64>()
        })
        .collect()
}

fn write_audio(path: &Path) {
    let tones: Vec<Vec<f64>> = audible().iter().map(received).collect();
    let spec = hound::WavSpec { channels: 1, sample_rate: RATE, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    let mut noise = Noise(0x9E37_79B9_7F4A_7C15);
    for n in 0..tones[0].len() {
        let sample = tones.iter().map(|tone| tone[n]).sum::<f64>() + NOISE * noise.gaussian();
        writer.write_sample((sample * PCM_SCALE).round() as i16).unwrap();
    }
    writer.finalize().unwrap();
    std::fs::write(path.with_extension("wav.start"), RECORDING_START.to_string()).unwrap();
}

fn write_adsb(path: &Path) -> Bounds {
    let frame = Frame::new(SITE.0, SITE.1);
    let mut bounds = Bounds { south: SITE.0, north: SITE.0, west: SITE.1, east: SITE.1 };
    let traffic: Vec<Pass> = audible().into_iter().map(|(pass, _)| pass).chain(silent()).collect();
    let mut out = BufWriter::new(File::create(path).unwrap());
    for second in FLIGHT_S.0 as i64..=FLIGHT_S.1 as i64 {
        let now = RECORDING_START + second as f64;
        let aircraft: Vec<serde_json::Value> = traffic
            .iter()
            .map(|pass| {
                let [x, y, altitude] = pass.position(now);
                let (lat, lon) = frame.to_latlon(x, y);
                bounds = Bounds { south: bounds.south.min(lat), north: bounds.north.max(lat), west: bounds.west.min(lon), east: bounds.east.max(lon) };
                serde_json::json!({"hex": pass.hex, "t": "C172", "lat": lat, "lon": lon, "seen_pos": 0.0, "alt_baro": altitude / FEET, "alt_geom": altitude / FEET})
            })
            .collect();
        writeln!(out, "{}", serde_json::json!({"now": now, "ac": aircraft})).unwrap();
    }
    bounds
}

fn write_types(path: &Path) {
    let mut gz = GzEncoder::new(File::create(path).unwrap(), Compression::default());
    gz.write_all(b"{}").unwrap();
    gz.finish().unwrap();
}

fn write_flat_terrain(cache: &Path, traffic: Bounds) {
    let mut png_bytes = Vec::new();
    let mut encoder = png::Encoder::new(&mut png_bytes, 256, 256);
    encoder.set_color(png::ColorType::Rgb);
    encoder.write_header().unwrap().write_image_data(&SEA_LEVEL_TERRARIUM.repeat(256 * 256)).unwrap();
    for zoom in [6, 11] {
        let (x0, y0) = mercator(zoom, traffic.north + TERRAIN_MARGIN_DEG, traffic.west - TERRAIN_MARGIN_DEG);
        let (x1, y1) = mercator(zoom, traffic.south - TERRAIN_MARGIN_DEG, traffic.east + TERRAIN_MARGIN_DEG);
        for x in x0 as u32..=x1 as u32 {
            for y in y0 as u32..=y1 as u32 {
                let dir = cache.join(format!("{zoom}/{x}"));
                std::fs::create_dir_all(&dir).unwrap();
                std::fs::write(dir.join(format!("{y}.png")), &png_bytes).unwrap();
            }
        }
    }
}

#[test]
fn locates_a_recording_from_four_propeller_passes_within_100_m() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("synthetic_flight");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    write_audio(&dir.join("recording.wav"));
    let traffic = write_adsb(&dir.join("adsb.jsonl"));
    write_types(&dir.join("types.js"));
    write_flat_terrain(&dir.join("dem"), traffic);
    let status = Command::new(env!("CARGO_BIN_EXE_doppler-locate"))
        .current_dir(&dir)
        .args(["--audio", "recording.wav", "--adsb", "adsb.jsonl", "--types", "types.js", "--dem", "dem", "--out", "located.json"])
        .status()
        .unwrap();
    assert!(status.success());
    let located: Vec<Located> = serde_json::from_slice(&std::fs::read(dir.join("located.json")).unwrap()).unwrap();
    let (x, y) = Frame::new(SITE.0, SITE.1).to_xy(located[0].lat, located[0].lon);
    let error = x.hypot(y);
    let margin = located[0].significance - located[1].significance;
    eprintln!("error {error:.0} m, significance {:.1}, runner-up {:.1}", located[0].significance, located[1].significance);
    assert!(error < 100.0, "located {error:.0} m from the recorder");
    assert!(margin > 5.0, "best candidate leads the runner-up by only {margin:.1}");
}
