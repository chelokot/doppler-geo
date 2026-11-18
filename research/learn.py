import sys, glob, json, os
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from survey_report import evidence, RANGES, DETECTED, MIN_CLOCK_STRENGTH, MIN_NEGATIVES

KINDS = ["piston", "turboprop", "heli", "jet", "unknown", "other"]


def features(d):
    center = d["line"].shape[1] // 2
    envelope = d["envelope"]
    return np.column_stack([
        d["line"][:, center], d["line"].max(1), d["comb"][:, center], d["comb"].max(1), envelope[:, center, :], envelope.max(1),
        np.log(d["range"].astype(float)), d["height"].astype(float), d["speed"].astype(float), np.log(d["frames"].astype(float)),
        [KINDS.index(k) for k in d["kind"]],
    ])


data = {}
for path in sorted(glob.glob(f"{sys.argv[1]}/*.npz")):
    name = os.path.basename(path)[:-4]
    d = np.load(path)
    if json.load(open(f"clock/{name}.json"))["strength"] >= MIN_CLOCK_STRENGTH and (d["label"] == 0).sum() >= MIN_NEGATIVES:
        data[name] = (features(d), d["label"], d["kind"], d["range"].astype(float))
rows = []
for held, (X, y, kind, distance) in data.items():
    Xt = np.concatenate([v[0] for k, v in data.items() if k != held])
    yt = np.concatenate([v[1] for k, v in data.items() if k != held])
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=40, l2_regularization=1.0, categorical_features=[X.shape[1] - 1], random_state=0).fit(Xt, yt)
    positive = y == 1
    e = np.column_stack([evidence(s[:, None], positive)[:, 0] for s in (X[:, 0], X[:, 1], model.decision_function(X))])
    rows += list(zip(kind[positive], distance[positive], e))
print(len(data), "sites; share of passes detected at p<0.001:     line  line±3s  learned   | mean evidence")
for kind in KINDS:
    for lo, hi in zip(RANGES, RANGES[1:]):
        v = np.array([v for k, r, v in rows if k == kind and lo <= r < hi])
        if len(v) >= 5:
            print(f"{kind:9s} {lo/1000:2.0f}-{hi/1000:2.0f} km n {len(v):4d}                    ", " ".join(f"{x:8.1%}" for x in (v > DETECTED).mean(0)), "  |", " ".join(f"{x:5.2f}" for x in v.mean(0)))
v = np.array([v for _, _, v in rows])
print("all", len(v), "detected", (v > DETECTED).sum(0), "total evidence", np.round(v.sum(0)))
