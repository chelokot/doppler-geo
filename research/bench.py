import json
import numpy as np
from scipy import stats
from data import SITES, site_tracks, geometry, SPEED_OF_SOUND
from spec import compute, levels, HOP_S, LOG_STEP

PAD = 200
SOUNDING = {"piston", "turboprop", "heli", "unknown"}
ALL_KINDS = SOUNDING | {"jet", "other"}
SURVEY_RADIUS = 12000.0


def clock(name):
    return json.load(open(f"clock/{name}.json"))["offset"]


class Site:
    def __init__(self, name, radius, nper, domain, line_width, kinds, offset):
        self.name, self.radius = name, radius
        S, self.t0 = compute(name, nper, line_width, domain)
        self.S = np.pad(S, ((0, 0), (PAD, PAD)), constant_values=np.nan)
        self.bins = S.shape[1]
        self.L = levels(name)
        self.tracks = [t for t in site_tracks(name) if t["kind"] in kinds]
        end = self.t0 + len(S) * HOP_S
        self.tracks = [t for t in self.tracks if t["tau"][-1] > self.t0 and t["tau"][0] < end]
        self.ground, self.offset = SITES[name]["ground"], offset

    def pass_view(self, track, point, offset=None):
        r, recv, k = geometry(track, self.ground, point)
        near = r < self.radius
        if near.sum() < 20:
            return None
        frames = np.round((recv[near] + (self.offset if offset is None else offset) - self.t0) / HOP_S).astype(int)
        if frames.min() < 0 or frames.max() >= len(self.S):
            return None
        shifts = np.round(np.log(k[near]) / LOG_STEP).astype(int)
        if np.abs(shifts).max() >= PAD:
            return None
        tau = track["tau"][near]
        return dict(frames=frames, shifts=shifts, r=r[near], tau=tau, recv=recv[near], k=k[near])

    def shifted(self, view, offsets):
        views = [view | dict(frames=view["frames"] + int(round(o / HOP_S))) for o in offsets]
        return views if views[0]["frames"].min() >= 0 and views[-1]["frames"].max() < len(self.S) else None

    def dewarp(self, view, sign=1):
        cols = PAD + np.arange(self.bins)[None, :] + sign * view["shifts"][:, None]
        return self.S[view["frames"][:, None], cols]


def pairs(site, n_neg, rng):
    pos = [(i, (0.0, 0.0)) for i in range(len(site.tracks))]
    neg = []
    while len(neg) < n_neg:
        p = rng.uniform(-15000, 15000, 2)
        if np.hypot(*p) < 1500:
            continue
        for i, t in enumerate(site.tracks):
            d = np.hypot(t["pos"][:, 0] - p[0], t["pos"][:, 1] - p[1]).min()
            if d < site.radius:
                neg.append((i, tuple(p)))
    return pos, neg[:n_neg]


def evaluate(site, scorer, pos, neg):
    def run(lst):
        out = []
        for i, p in lst:
            v = site.pass_view(site.tracks[i], p)
            if v is not None:
                out.append((i, scorer(site, v)))
        return out
    ps, ns = run(pos), np.sort([s for _, s in run(neg)])
    ns = ns[np.isfinite(ns)]
    tail = Tail(ns)
    pv = [tail.p(s) for _, s in ps]
    evidence = -np.log10(pv)
    auc = np.mean([(ns < s).mean() + 0.5 * (ns == s).mean() for _, s in ps])
    return dict(n_pos=len(ps), n_neg=len(ns), auc=auc, evidence=evidence.sum(), per=sorted(np.round(evidence, 1))[::-1])


class Tail:
    def __init__(self, null, fraction=0.05):
        self.sorted = np.sort(null)
        self.threshold = np.quantile(null, 1 - fraction)
        self.fraction = fraction
        self.shape, _, self.scale = stats.genpareto.fit(self.sorted[self.sorted > self.threshold] - self.threshold, floc=0)

    def p(self, score):
        if not np.isfinite(score):
            return 1.0
        if score <= self.threshold:
            return (len(self.sorted) - np.searchsorted(self.sorted, score) + 1) / (len(self.sorted) + 1)
        return max(self.fraction * stats.genpareto.sf(score - self.threshold, self.shape, 0, self.scale), 1e-9)
