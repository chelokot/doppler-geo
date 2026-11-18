use crate::tracks::{Altitude, FEET, Fix};
use flate2::read::GzDecoder;
use rayon::prelude::*;
use serde::Deserialize;
use serde::de::{Deserializer, Error, IgnoredAny, SeqAccess, Visitor};
use serde_json::value::RawValue;
use std::fmt;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::PathBuf;
use std::sync::mpsc;

const GEOMETRIC_ALTITUDE_FLAG: u32 = 8;
const TRACE_PREFIX: &str = "trace_full_";

pub struct Aircraft {
    pub hex: String,
    pub type_code: String,
    pub fixes: Vec<Fix>,
}

#[derive(Deserialize)]
struct TraceFile<'a> {
    icao: String,
    t: Option<String>,
    timestamp: f64,
    #[serde(borrow)]
    trace: &'a RawValue,
}

struct TracePoint {
    dt: f64,
    lat: f64,
    lon: f64,
    altitude: Altitude,
    flags: u32,
    geometric: Option<f64>,
}

fn required<'de, T: Deserialize<'de>, A: SeqAccess<'de>>(seq: &mut A) -> Result<T, A::Error> {
    seq.next_element()?.ok_or_else(|| A::Error::custom("short trace point"))
}

impl<'de> Deserialize<'de> for TracePoint {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct PointVisitor;
        impl<'de> Visitor<'de> for PointVisitor {
            type Value = TracePoint;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("readsb trace point")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<TracePoint, A::Error> {
                let dt = required(&mut seq)?;
                let lat = required(&mut seq)?;
                let lon = required(&mut seq)?;
                let altitude = required(&mut seq)?;
                let _: [IgnoredAny; 2] = [required(&mut seq)?, required(&mut seq)?];
                let flags = required(&mut seq)?;
                let _: [IgnoredAny; 3] = [required(&mut seq)?, required(&mut seq)?, required(&mut seq)?];
                let geometric = seq.next_element::<Option<f64>>()?.flatten();
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(TracePoint { dt, lat, lon, altitude, flags, geometric })
            }
        }
        deserializer.deserialize_seq(PointVisitor)
    }
}

fn parse(compressed: &[u8], (start, end): (f64, f64), accept: &(dyn Fn(&str) -> bool + Sync)) -> Option<Aircraft> {
    let mut json = Vec::new();
    GzDecoder::new(compressed).read_to_end(&mut json).expect("trace gzip");
    let file: TraceFile = serde_json::from_slice(&json).expect("trace json");
    let type_code = file.t.unwrap_or_default();
    if !accept(&type_code) {
        return None;
    }
    let points: Vec<TracePoint> = serde_json::from_str(file.trace.get()).expect("trace points");
    let fixes: Vec<Fix> = points
        .iter()
        .filter_map(|p| {
            let t = file.timestamp + p.dt;
            let Altitude::Feet(feet) = p.altitude else {
                return None;
            };
            let baro = (p.flags & GEOMETRIC_ALTITUDE_FLAG == 0).then_some(feet * FEET);
            (t > start && t < end).then_some(Fix { t, lat: p.lat, lon: p.lon, baro, geom: p.geometric.map(|ft| ft * FEET) })
        })
        .collect();
    (!fixes.is_empty()).then_some(Aircraft { hex: file.icao, type_code, fixes })
}

pub fn read(parts: &[PathBuf], window: (f64, f64), accept: &(dyn Fn(&str) -> bool + Sync)) -> Vec<Aircraft> {
    let (sender, receiver) = mpsc::sync_channel::<Vec<u8>>(4096);
    std::thread::scope(|scope| {
        scope.spawn(move || {
            let stream = parts.iter().fold(Box::new(std::io::empty()) as Box<dyn Read>, |acc, part| {
                Box::new(acc.chain(BufReader::with_capacity(1 << 22, File::open(part).expect("archive part"))))
            });
            let mut archive = tar::Archive::new(stream);
            for entry in archive.entries().expect("tar entries") {
                let mut entry = entry.expect("tar entry");
                let is_trace = entry.path().expect("tar path").file_name().is_some_and(|name| name.to_string_lossy().starts_with(TRACE_PREFIX));
                if is_trace {
                    let mut compressed = Vec::with_capacity(entry.size() as usize);
                    entry.read_to_end(&mut compressed).expect("tar read");
                    sender.send(compressed).expect("trace consumer");
                }
            }
        });
        receiver.into_iter().par_bridge().filter_map(|compressed| parse(&compressed, window, accept)).collect()
    })
}
