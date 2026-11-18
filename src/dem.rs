use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;
use std::io::Cursor;
use std::path::Path;

const TILE_PX: usize = 256;
const TERRARIUM: &str = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium";

pub struct Dem {
    zoom: i32,
    tiles: HashMap<(u32, u32), Vec<f32>>,
}

pub fn mercator(zoom: i32, lat: f64, lon: f64) -> (f64, f64) {
    let n = 2f64.powi(zoom);
    ((lon + 180.0) / 360.0 * n, (1.0 - lat.to_radians().tan().asinh() / PI) / 2.0 * n)
}

fn download(cache: &Path, zoom: i32, (x, y): (u32, u32)) -> Vec<u8> {
    let path = cache.join(format!("{zoom}/{x}/{y}.png"));
    if let Ok(bytes) = std::fs::read(&path) {
        return bytes;
    }
    let bytes = ureq::get(format!("{TERRARIUM}/{zoom}/{x}/{y}.png")).call().expect("terrain tile").body_mut().read_to_vec().expect("terrain body");
    std::fs::create_dir_all(path.parent().expect("tile directory")).expect("terrain cache");
    std::fs::write(&path, &bytes).expect("terrain cache write");
    bytes
}

fn decode(png_bytes: &[u8]) -> Vec<f32> {
    let mut decoder = png::Decoder::new(Cursor::new(png_bytes));
    decoder.set_transformations(png::Transformations::EXPAND);
    let mut reader = decoder.read_info().expect("terrain png");
    let mut pixels = vec![0; reader.output_buffer_size().expect("terrain size")];
    let info = reader.next_frame(&mut pixels).expect("terrain frame");
    let channels = info.color_type.samples();
    pixels[..info.buffer_size()].chunks_exact(channels).map(|p| p[0] as f32 * 256.0 + p[1] as f32 + p[2] as f32 / 256.0 - 32768.0).collect()
}

impl Dem {
    pub fn fetch(cache: &Path, zoom: i32, points: impl Iterator<Item = (f64, f64)>) -> Dem {
        let keys: HashSet<(u32, u32)> = points
            .map(|(lat, lon)| {
                let (x, y) = mercator(zoom, lat, lon);
                (x as u32, y as u32)
            })
            .collect();
        let tiles = keys.into_par_iter().map(|key| (key, decode(&download(cache, zoom, key)))).collect();
        Dem { zoom, tiles }
    }

    pub fn ground(&self, lat: f64, lon: f64) -> f64 {
        let (x, y) = mercator(self.zoom, lat, lon);
        let tile = &self.tiles[&(x as u32, y as u32)];
        let (px, py) = ((x.fract() * TILE_PX as f64) as usize, (y.fract() * TILE_PX as f64) as usize);
        tile[py * TILE_PX + px] as f64
    }
}
