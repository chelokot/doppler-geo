import sys, json
import numpy as np
import bench, scorers
from data import SITES, site_tracks

STRONG = 8.0
SPEED_TOLERANCE = 0.1
IMPOSTORS_PER_PASS = 40
NEGATIVES = 1500


def velocity(track, index):
    before, after = max(index - 4, 0), min(index + 4, len(track["tau"]) - 1)
    return (track["pos"][after] - track["pos"][before]) / (track["tau"][after] - track["tau"][before])


def straight(track, index, span):
    tau = track["tau"][span]
    return dict(tau=tau, pos=track["pos"][index] + velocity(track, index) * (tau - track["tau"][index])[:, None])


def transplant(foreign, at, anchor, heading, span):
    u = velocity(foreign, at)
    angle = heading - np.arctan2(u[1], u[0])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    moved = foreign["pos"][span] - foreign["pos"][at]
    return dict(tau=foreign["tau"][span], pos=np.column_stack([moved[:, :2] @ rotation.T, moved[:, 2]]) + anchor)


nper, line_width, names = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
pool = [t for n in SITES for t in site_tracks(n)]
for name in names:
    rng = np.random.default_rng(0)
    d = np.load(f"survey_32768_log/{name}.npz")
    positive = d["label"] == 1
    reference = bench.Tail(d["line"][~positive, 3][np.isfinite(d["line"][~positive, 3])])
    site = bench.Site(name, bench.SURVEY_RADIUS, nper, "log", line_width, bench.ALL_KINDS, bench.clock(name))

    def score(track, point=(0.0, 0.0)):
        view = site.pass_view(track, point)
        return scorers.line_and_comb(site, view)[0] if view else np.nan

    null = np.array([score(site.tracks[i], p) for i, p in bench.pairs(site, NEGATIVES, rng)[1]])
    tail = bench.Tail(null[np.isfinite(null)])
    evidence = lambda value: -np.log10(tail.p(value))
    spread = np.nanstd(null)
    for index, reference_score, distance in zip(d["track"][positive], d["line"][positive, 3], d["range"][positive]):
        track = site.tracks[index]
        print("pass", name, index, round(float(distance)), score(track) / spread, evidence(score(track)), flush=True)
        if -np.log10(reference.p(reference_score)) < STRONG:
            continue
        view = site.pass_view(track, (0.0, 0.0))
        closest = np.searchsorted(track["tau"], view["tau"][np.argmin(view["r"])])
        lo, hi = view["tau"][0], view["tau"][-1]
        v = velocity(track, closest)
        heading, speed = np.arctan2(v[1], v[0]), np.hypot(v[0], v[1])
        true = score(track)
        own = slice(np.searchsorted(track["tau"], lo), np.searchsorted(track["tau"], hi) + 1)
        flat = score(straight(track, closest, own))
        print("straight", name, index, round(float(distance)), true / spread, flat / spread, 0.0, evidence(true), evidence(flat), flush=True)
        candidates = []
        for foreign in pool:
            if foreign["hex"] == track["hex"] or foreign["tau"][0] > lo or foreign["tau"][-1] < hi:
                continue
            at = np.searchsorted(foreign["tau"], track["tau"][closest])
            u = velocity(foreign, at)
            ratio = np.hypot(u[0], u[1]) / speed
            if abs(ratio - 1) < SPEED_TOLERANCE:
                candidates.append((foreign, at, ratio))
        for pick in rng.permutation(len(candidates))[:IMPOSTORS_PER_PASS]:
            foreign, at, ratio = candidates[pick]
            span = slice(np.searchsorted(foreign["tau"], lo), np.searchsorted(foreign["tau"], hi) + 1)
            fake = score(transplant(foreign, at, track["pos"][closest], heading, span))
            print("impostor", name, index, 0, true / spread, fake / spread, ratio - 1, evidence(true), evidence(fake), flush=True)
