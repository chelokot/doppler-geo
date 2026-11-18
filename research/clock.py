import sys, os, json
import numpy as np
import bench, scorers
from survey_report import evidence

OFFSETS = np.arange(-60, 31)
RADIUS = 7000.0
CLOSE = 4000.0
NEGATIVES = 250

name = sys.argv[1]
site = bench.Site(name, RADIUS, 32768, "log", 41, bench.ALL_KINDS, 0.0)
rng = np.random.default_rng(0)
positives, negatives = bench.pairs(site, NEGATIVES, rng)
scores, labels = [], []
for label, pairs in ((1, positives), (0, negatives)):
    for index, point in pairs:
        view = site.pass_view(site.tracks[index], point)
        views = view and view["r"].min() < CLOSE and site.shifted(view, OFFSETS)
        if views:
            scores.append([scorers.base(site, v) for v in views])
            labels.append(label)
scores, positive = np.array(scores), np.array(labels) == 1
strength = np.clip(evidence(scores, positive) - 2, 0, None).sum(0) if positive.any() else np.zeros(len(OFFSETS))
os.makedirs("clock", exist_ok=True)
json.dump(dict(offset=int(OFFSETS[np.argmax(strength)]), strength=float(strength.max()), passes=int(positive.sum()), profile=[round(float(x), 1) for x in strength]), open(f"clock/{name}.json", "w"))
print(name, "offset", int(OFFSETS[np.argmax(strength)]), "strength", round(float(strength.max()), 1), "close passes", int(positive.sum()), flush=True)
