import json, gzip, subprocess
import numpy as np
import scipy.signal as signal

SPEED_OF_SOUND = 343.0
FEET = 0.3048
TYPES = json.load(gzip.open("../ext/types.js"))
SITES = json.load(open("../sites.json"))
RATE = 16000


class Frame:
    def __init__(self, lat0, lon0):
        self.lat0, self.lon0 = lat0, lon0
        self.kx = np.cos(np.radians(lat0)) * 111320.0
        self.ky = 110540.0

    def to_xy(self, lat, lon):
        return (np.asarray(lon) - self.lon0) * self.kx, (np.asarray(lat) - self.lat0) * self.ky


def kind(type_code):
    desc = TYPES.get(type_code or "", {}).get("desc")
    if desc is None:
        return "unknown"
    if desc[0] in "HG":
        return "heli"
    return {"J": "jet", "P": "piston", "T": "turboprop"}.get(desc[2], "other")


def segment(fixes, step=0.5, max_gap=90.0, max_speed=350.0, max_climb=60.0):
    fixes = sorted(fixes)
    kept = []
    for f in fixes:
        if kept:
            dt = f[0] - kept[-1][0]
            if dt < 1.0 or np.hypot(f[1] - kept[-1][1], f[2] - kept[-1][2]) > max_speed * dt or abs(f[3] - kept[-1][3]) > max_climb * dt:
                continue
        kept.append(f)
    if len(kept) < 3:
        return []
    k = np.array(kept)
    out = []
    for seg in np.split(k, np.where(np.diff(k[:, 0]) > max_gap)[0] + 1):
        if len(seg) < 3:
            continue
        tau = np.arange(seg[0, 0], seg[-1, 0], step)
        out.append((tau, np.stack([np.interp(tau, seg[:, 0], seg[:, c]) for c in (1, 2, 3)], axis=1)))
    return out


def load_tracks_json(path, frame):
    tracks = []
    for t in json.load(open(path)):
        geometric = all(g is not None for g in t["geom"])
        alt = t["geom"] if geometric else t["baro"]
        x, y = frame.to_xy(np.array(t["lat"]), np.array(t["lon"]))
        fixes = [(tt, xx, yy, a) for tt, xx, yy, a in zip(t["t"], x, y, alt) if a is not None]
        for tau, pos in segment(fixes):
            tracks.append(dict(hex=t["hex"], type=t["type"], kind=kind(t["type"]), tau=tau, pos=pos))
    return tracks


def audio_path(name):
    return f"../dvr/{name}.opus"


def tracks_path(name):
    return f"tracks_live/{name}.json"


def site_tracks(name):
    s = SITES[name]
    return load_tracks_json(tracks_path(name), Frame(s["lat"], s["lon"]))


def read_audio(name):
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", audio_path(name), "-ac", "1", "-ar", str(RATE), "-f", "s16le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(pcm, np.int16), SITES[name]["start"]


def geometry(track, ground, point=(0.0, 0.0)):
    p = track["pos"]
    r = np.sqrt((p[:, 0] - point[0]) ** 2 + (p[:, 1] - point[1]) ** 2 + (p[:, 2] - ground) ** 2)
    recv = track["tau"] + r / SPEED_OF_SOUND
    radial = np.gradient(r, track["tau"])
    return r, recv, SPEED_OF_SOUND / (SPEED_OF_SOUND + radial)


def passes(tracks, ground, max_range=15000.0, point=(0.0, 0.0)):
    out = []
    for i, tr in enumerate(tracks):
        r, recv, k = geometry(tr, ground, point)
        mins = signal.argrelmin(r, order=40)[0]
        cands = list(mins) + [j for j in (0, len(r) - 1) if r[j] == r.min()]
        for j in cands:
            if r[j] < max_range and 0 < j < len(r) - 1:
                v = np.linalg.norm(tr["pos"][min(j + 1, len(r) - 1)] - tr["pos"][j - 1]) / (tr["tau"][min(j + 1, len(r) - 1)] - tr["tau"][j - 1])
                out.append(dict(track=i, hex=tr["hex"], type=tr["type"], kind=tr["kind"], recv=recv[j], r=r[j], alt=tr["pos"][j, 2] - ground, speed=v))
    return sorted(out, key=lambda p: p["recv"])
