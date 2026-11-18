use flate2::read::GzDecoder;
use serde::Deserialize;
use serde::de::IgnoredAny;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub const FEET: f64 = 0.3048;
const MAX_ALTITUDE_M: f64 = 5000.0;
const MAX_GAP_S: f64 = 30.0;
const MIN_POINT_SPACING_S: f64 = 1.0;
const MAX_SPEED: f64 = 200.0;
const MAX_CLIMB: f64 = 30.0;
pub const STEP_S: f64 = 0.5;

#[derive(Clone, Copy)]
pub struct Frame {
    lat0: f64,
    lon0: f64,
    kx: f64,
    ky: f64,
}

impl Frame {
    pub fn new(lat0: f64, lon0: f64) -> Self {
        Frame { lat0, lon0, kx: lat0.to_radians().cos() * 111320.0, ky: 110540.0 }
    }

    pub fn to_xy(&self, lat: f64, lon: f64) -> (f64, f64) {
        ((lon - self.lon0) * self.kx, (lat - self.lat0) * self.ky)
    }

    pub fn to_latlon(&self, x: f64, y: f64) -> (f64, f64) {
        (self.lat0 + y / self.ky, self.lon0 + x / self.kx)
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum Altitude {
    Feet(f64),
    Label(IgnoredAny),
}

#[derive(Deserialize)]
struct Aircraft {
    hex: String,
    t: Option<String>,
    lat: Option<f64>,
    lon: Option<f64>,
    seen_pos: Option<f64>,
    alt_baro: Option<Altitude>,
    alt_geom: Option<f64>,
}

#[derive(Deserialize)]
struct Snapshot {
    now: f64,
    ac: Vec<Aircraft>,
}

#[derive(Deserialize)]
struct TypeInfo {
    desc: String,
}

pub struct Types(HashMap<String, TypeInfo>);

impl Types {
    pub fn load(path: &Path) -> Self {
        Types(serde_json::from_reader(GzDecoder::new(File::open(path).expect("types db"))).expect("types json"))
    }

    pub fn is_propeller(&self, type_code: &str) -> bool {
        self.0.get(type_code).is_none_or(|info| {
            let desc = info.desc.as_bytes();
            b"HG".contains(&desc[0]) || b"PT".contains(&desc[2])
        })
    }
}

#[derive(Clone, Copy)]
pub struct Fix {
    pub t: f64,
    pub lat: f64,
    pub lon: f64,
    pub baro: Option<f64>,
    pub geom: Option<f64>,
}

pub struct Track {
    pub tau: Vec<f64>,
    pub pos: Vec<[f64; 3]>,
}

pub struct GeoTrack {
    pub tau: Vec<f64>,
    pub pos: Vec<[f64; 3]>,
}

fn tenths(seconds: f64) -> i64 {
    format!("{seconds:.1}").replace('.', "").parse().expect("formatted decimal")
}

fn interp(x: f64, xs: &[f64], ys: &[f64]) -> f64 {
    let j = xs.partition_point(|&v| v <= x).clamp(1, xs.len() - 1);
    let w = (x - xs[j - 1]) / (xs[j] - xs[j - 1]);
    ys[j - 1] + w * (ys[j] - ys[j - 1])
}

pub fn segments(fixes: &[Fix]) -> Vec<GeoTrack> {
    let Some(first) = fixes.first() else {
        return Vec::new();
    };
    let frame = Frame::new(first.lat, first.lon);
    let geometric = fixes.iter().all(|f| f.geom.is_some());
    let mut kept: Vec<[f64; 4]> = Vec::new();
    for fix in fixes {
        let Some(alt) = (if geometric { fix.geom } else { fix.baro }) else {
            continue;
        };
        let (x, y) = frame.to_xy(fix.lat, fix.lon);
        if let Some(&[pt, plat, plon, pz]) = kept.last() {
            let dt = fix.t - pt;
            let (px, py) = frame.to_xy(plat, plon);
            if dt < MIN_POINT_SPACING_S || (x - px).hypot(y - py) > MAX_SPEED * dt || (alt - pz).abs() > MAX_CLIMB * dt {
                continue;
            }
        }
        kept.push([fix.t, fix.lat, fix.lon, alt]);
    }
    kept.chunk_by(|a, b| b[0] - a[0] <= MAX_GAP_S)
        .filter(|seg| seg.len() >= 3 && seg.iter().map(|p| p[3]).fold(f64::INFINITY, f64::min) <= MAX_ALTITUDE_M)
        .map(|seg| {
            let columns: [Vec<f64>; 4] = std::array::from_fn(|c| seg.iter().map(|p| p[c]).collect());
            let (t0, t1) = (columns[0][0], columns[0][seg.len() - 1]);
            let tau: Vec<f64> = (0..((t1 - t0) / STEP_S).ceil() as usize).map(|k| t0 + k as f64 * STEP_S).collect();
            let pos = tau.iter().map(|&t| std::array::from_fn(|c| interp(t, &columns[0], &columns[c + 1]))).collect();
            GeoTrack { tau, pos }
        })
        .collect()
}

pub fn load(adsb: &Path, types: &Types) -> Vec<GeoTrack> {
    let mut observed: HashMap<String, (String, BTreeMap<i64, Fix>)> = HashMap::new();
    let mut order: Vec<String> = Vec::new();
    for line in BufReader::new(File::open(adsb).expect("adsb log")).lines() {
        let snapshot: Snapshot = serde_json::from_str(&line.expect("adsb line")).expect("adsb snapshot");
        for a in snapshot.ac {
            let (Some(lat), Some(lon), Some(seen_pos), Some(Altitude::Feet(alt_ft))) = (a.lat, a.lon, a.seen_pos, a.alt_baro) else {
                continue;
            };
            let entry = observed.entry(a.hex.clone()).or_insert_with(|| {
                order.push(a.hex.clone());
                (a.t.unwrap_or_default(), BTreeMap::new())
            });
            let key = tenths(snapshot.now - seen_pos);
            let fix = Fix { t: key as f64 / 10.0, lat, lon, baro: Some(alt_ft * FEET), geom: a.alt_geom.map(|ft| ft * FEET) };
            entry.1.insert(key, fix);
        }
    }
    order.iter().map(|hex| &observed[hex]).filter(|(type_code, _)| types.is_propeller(type_code)).flat_map(|(_, fixes)| segments(&fixes.values().copied().collect::<Vec<_>>())).collect()
}
