import sys, os
import numpy as np
import bench, scorers

RADIUS = 25000.0
MIN_RANGE = 4000.0
MIN_ELEVATION = np.radians(15.0)
NEGATIVES = 4000

name = sys.argv[1]
site = bench.Site(name, RADIUS, 32768, "log", 41, bench.SOUNDING, bench.clock(name))
positives, negatives = bench.pairs(site, NEGATIVES, np.random.default_rng(0))
rows = []
for label, pairs in ((1, positives), (0, negatives)):
    for index, point in pairs:
        track = site.tracks[index]
        view = site.pass_view(track, point)
        if not view:
            continue
        closest = np.argmin(view["r"])
        distance = view["r"][closest]
        height = track["pos"][np.searchsorted(track["tau"], view["tau"][closest]), 2] - site.ground
        if distance < MIN_RANGE or height < distance * np.sin(MIN_ELEVATION):
            continue
        rows.append((label, track["kind"], track["type"], distance, height, scorers.line_and_comb(site, view)[0]))
os.makedirs("survey_far", exist_ok=True)
np.savez(f"survey_far/{name}.npz", **{c: np.array([r[i] for r in rows]) for i, c in enumerate(["label", "kind", "type", "range", "height", "line"])})
print(name, "positives", sum(r[0] for r in rows), "negatives", sum(1 - r[0] for r in rows), flush=True)
