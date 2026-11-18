import sys, glob, os, json
import numpy as np
import bench

RADIUS = 25000.0
STRONG = 8.0
CORE = 1.5
OUTSIDE = 2.0
CONTROL_BINS = (-25, -15, 15, 25)
DRIFT_BINS = 3

rows = []
for path in sorted(glob.glob("survey_32768_log/*.npz")):
    name = os.path.basename(path)[:-4]
    d = np.load(path)
    positive = d["label"] == 1
    null = d["line"][~positive, 3]
    if json.load(open(f"clock/{name}.json"))["strength"] < 5 or len(null) < 200:
        continue
    tail = bench.Tail(null[np.isfinite(null)])
    strong = [i for i, s in zip(d["track"][positive], d["line"][positive, 3]) if -np.log10(tail.p(s)) >= STRONG]
    if not strong:
        continue
    site = bench.Site(name, RADIUS, 32768, "log", 41, bench.ALL_KINDS, bench.clock(name))
    for index in strong:
        track = site.tracks[index]
        view = site.pass_view(track, (0.0, 0.0))
        if not view:
            continue
        D = site.dewarp(view)
        r = view["r"]
        height = track["pos"][np.searchsorted(track["tau"], view["tau"]), 2] - site.ground
        core = r < CORE * r.min()
        line = int(np.nanargmax(np.nanmean(D[core], 0)[40:-40])) + 40
        control = np.nanmean(D[:, [line + c for c in CONTROL_BINS]], 1)
        window = lambda center: np.nanmax(D[:, center - DRIFT_BINS : center + DRIFT_BINS + 1], 1)
        drifting = window(line) - np.mean([window(line + c) for c in CONTROL_BINS], 0)
        for k in np.flatnonzero(r > OUTSIDE * r.min()):
            rows.append((r[k], np.degrees(np.arcsin(height[k] / r[k])), D[k, line] - control[k], r.min(), track["kind"] == "jet", drifting[k]))
        rows.append((0.0, 90.0, np.nanmean(D[core, line] - control[core]), r.min(), track["kind"] == "jet", np.nanmean(drifting[core])))
np.save(sys.argv[1], np.array(rows, float))
