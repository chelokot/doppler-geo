import os, json, gzip, time, datetime, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from data import SITES, Frame, FEET, tracks_path

DISCOVERY_RADIUS = 12000.0
DISCOVERY_CEILING = 25000 // 25
TRACK_RADIUS = 25000.0
SLICE_MARK = 0xE7F7C9D
INFO_FLAG = 1 << 30
GEOMETRIC_ALTITUDE_FLAG = 8
RECORD = np.dtype([("hex", "<i4"), ("lat", "<i4"), ("lon", "<i4"), ("alt", "<i2"), ("gs", "<i2")])


def get(url):
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://globe.adsb.lol/"})
    for attempt in range(8):
        try:
            body = urllib.request.urlopen(request, timeout=60).read()
            return gzip.decompress(body) if body[:2] == b"\x1f\x8b" else body
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            time.sleep(2 * (attempt + 1))
        except OSError:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(url)


def positions(half_hour):
    day = datetime.datetime.fromtimestamp(half_hour * 1800, datetime.UTC)
    raw = get(f"https://adsb.lol/globe_history/{day:%Y/%m/%d}/heatmap/{day.hour * 2 + day.minute // 30:02d}.bin.ttf")
    records = np.frombuffer(raw[: len(raw) // RECORD.itemsize * RECORD.itemsize], RECORD)
    return records[(records["hex"] != SLICE_MARK) & (records["lat"] & INFO_FLAG == 0) & (records["alt"] < DISCOVERY_CEILING)]


def trace(hex_id):
    body = get(f"https://adsb.lol/data/traces/{hex_id[-2:]}/trace_full_{hex_id}.json")
    return body and json.loads(body)


frames = {name: Frame(s["lat"], s["lon"]) for name, s in SITES.items()}
seen = {name: set() for name in SITES}
first = int(min(s["start"] for s in SITES.values()) // 1800)
for half_hour in range(first, int(time.time() // 1800)):
    p = positions(half_hour)
    for name, frame in frames.items():
        x, y = frame.to_xy(p["lat"] / 1e6, p["lon"] / 1e6)
        seen[name] |= {f"{h & 0xFFFFFF:06x}" for h in p["hex"][np.hypot(x, y) < DISCOVERY_RADIUS]}
    print(half_hour, sum(map(len, seen.values())), flush=True)
wanted = sorted(set().union(*seen.values()))
print(len(wanted), "aircraft", flush=True)
with ThreadPoolExecutor(3) as pool:
    traces = dict(zip(wanted, pool.map(trace, wanted)))
os.makedirs("tracks_live", exist_ok=True)
for name, frame in frames.items():
    out = []
    for hex_id in sorted(seen[name]):
        t = traces[hex_id]
        if t is None:
            continue
        points = [p for p in t["trace"] if isinstance(p[3], (int, float))]
        x, y = frame.to_xy(np.array([p[1] for p in points]), np.array([p[2] for p in points]))
        points = [p for p, near in zip(points, np.hypot(x, y) < TRACK_RADIUS) if near]
        out.append(dict(hex=hex_id, type=t.get("t", ""), t=[t["timestamp"] + p[0] for p in points], lat=[p[1] for p in points], lon=[p[2] for p in points],
                        baro=[None if p[6] & GEOMETRIC_ALTITUDE_FLAG else p[3] * FEET for p in points], geom=[p[10] * FEET if p[10] is not None else None for p in points]))
    json.dump(out, open(tracks_path(name), "w"))
    print(name, len(out), flush=True)
