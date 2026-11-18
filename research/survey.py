import sys, os
import numpy as np
import bench, scorers

LOCAL_OFFSETS = np.arange(-3, 4)
NEGATIVES_PER_PASS = 3

name, nper, domain = sys.argv[1], int(sys.argv[2]), sys.argv[3]
site = bench.Site(name, bench.SURVEY_RADIUS, nper, domain, 41, bench.ALL_KINDS, bench.clock(name))
rng = np.random.default_rng(0)
positives, negatives = bench.pairs(site, NEGATIVES_PER_PASS * len(site.tracks), rng)
rows = []
for label, pairs in ((1, positives), (0, negatives)):
    for index, point in pairs:
        track = site.tracks[index]
        view = site.pass_view(track, point)
        views = view and site.shifted(view, LOCAL_OFFSETS)
        if not views:
            continue
        closest = np.argmin(view["r"])
        position = np.searchsorted(track["tau"], view["tau"][closest])
        before, after = max(position - 2, 0), min(position + 2, len(track["tau"]) - 1)
        speed = np.linalg.norm(track["pos"][after] - track["pos"][before]) / (track["tau"][after] - track["tau"][before])
        rows.append((label, index, track["kind"], track["type"], track["hex"], view["r"][closest], track["pos"][position, 2] - site.ground, speed, view["recv"][closest], len(view["r"]), *zip(*[scorers.line_and_comb(site, v) for v in views]), [scorers.envelope(site, v) for v in views]))
os.makedirs(f"survey_{nper}_{domain}", exist_ok=True)
columns = ["label", "track", "kind", "type", "hex", "range", "height", "speed", "recv", "frames", "line", "comb", "envelope"]
np.savez(f"survey_{nper}_{domain}/{name}.npz", **{c: np.array([r[i] for r in rows]) for i, c in enumerate(columns)})
print(name, "positives", sum(r[0] for r in rows), "negatives", sum(1 - r[0] for r in rows), flush=True)
