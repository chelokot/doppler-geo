import sys, glob, json, os
import numpy as np
from bench import Tail
from data import SITES

RANGES = [0, 1000, 2000, 4000, 7000, 12000]
DETECTED = 3.0
MIN_CLOCK_STRENGTH = 5.0
MIN_NEGATIVES = 200


def evidence(scores, positive):
    null = scores[~positive]
    tail = Tail(null[np.isfinite(null)])
    return np.array([[-np.log10(tail.p(s)) for s in row] for row in scores[positive]])


def variants(d):
    center = d["line"].shape[1] // 2
    return {"line": d["line"][:, [center]], "line±3s": d["line"].max(1, keepdims=True), "comb": d["comb"][:, [center]], "comb±3s": d["comb"].max(1, keepdims=True)}


if __name__ == "__main__":
    rows = []
    for path in sorted(glob.glob(f"{sys.argv[1]}/*.npz")):
        name = os.path.basename(path)[:-4]
        d, clock = np.load(path), json.load(open(f"clock/{name}.json"))
        positive = d["label"] == 1
        if clock["strength"] < MIN_CLOCK_STRENGTH or (~positive).sum() < MIN_NEGATIVES:
            print(f"{SITES[name]['title'][:26]:26s} skipped: clock strength {clock['strength']:.1f}, negatives {(~positive).sum()}")
            continue
        e = {k: evidence(v, positive)[:, 0] for k, v in variants(d).items()}
        print(f"{SITES[name]['title'][:26]:26s} passes {positive.sum():4d} offset {clock['offset']:+3d}s", " ".join(f"{k} {v.sum():6.1f}" for k, v in e.items()), f"null {0.434 * positive.sum():5.1f}")
        rows += list(zip(d["kind"][positive], d["range"][positive].astype(float), np.column_stack(list(e.values()))))
    print(f"\nshare of passes detected at p<0.001:", " ".join(f"{k:>8s}" for k in e))
    for kind in sorted({k for k, _, _ in rows}):
        for lo, hi in zip(RANGES, RANGES[1:]):
            v = np.array([v for k, r, v in rows if k == kind and lo <= r < hi])
            if len(v) >= 5:
                print(f"{kind:9s} {lo/1000:2.0f}-{hi/1000:2.0f} km n {len(v):4d}            ", " ".join(f"{x:8.1%}" for x in (v > DETECTED).mean(0)))
