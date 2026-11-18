import glob, os, json
import numpy as np
from data import SITES

MIN_SNR_DB = 12.0
MIN_VALID = 0.3
LAGS = np.arange(-60, 61)
MIN_OVERLAP_S = 1800
SAME_GRID = 0.8

series = {}
for path in sorted(glob.glob("enf/*.npz")):
    name = os.path.basename(path)[:-4]
    d = np.load(path)
    snr = 10 * np.log10(d["snr"])
    valid = (snr > MIN_SNR_DB).mean(2)
    m, h = np.unravel_index(valid.argmax(), valid.shape)
    ok = snr[m, h] > MIN_SNR_DB
    print(f"{SITES[name]['title'][:26]:26s} lon {SITES[name]['lon']:7.1f} mains {(50, 60)[m]} harmonic {h + 1} valid {valid[m, h]:5.1%} median snr {np.median(snr[m, h]):5.1f} dB std {d['frequency'][m, h][ok].std() * 1000 if ok.any() else 0:5.1f} mHz")
    if valid[m, h] > MIN_VALID:
        series[name] = (d["t"][ok], d["frequency"][m, h][ok], d["t"])
print(len(series), "of", len(glob.glob("enf/*.npz")), "recordings carry mains hum")
grid = np.arange(min(s[2][0] for s in series.values()), max(s[2][-1] for s in series.values()))
X = {}
for name, (t, f, all_t) in series.items():
    x = np.interp(grid, t, f)
    gap = np.abs(grid - t[np.clip(np.searchsorted(t, grid), 0, len(t) - 1)]) > 8
    x[gap | (grid < all_t[0]) | (grid > all_t[-1])] = np.nan
    X[name] = x


def match(a, b):
    best = (0.0, 0)
    for lag in LAGS:
        u, v = (a[lag:], b[: len(b) - lag]) if lag >= 0 else (a[:lag], b[-lag:])
        both = np.isfinite(u) & np.isfinite(v)
        if both.sum() > MIN_OVERLAP_S:
            best = max(best, (float(np.corrcoef(u[both], v[both])[0, 1]), int(lag)))
    return best


names = list(X)
clock = {n: json.load(open(f"clock/{n}.json")) for n in names}
pairs = {(a, b): match(X[a], X[b]) for i, a in enumerate(names) for b in names[i + 1 :]}
group = {n: n for n in names}
for (a, b), (c, _) in pairs.items():
    if c > SAME_GRID:
        old, new = group[b], group[a]
        group = {n: new if g == old else g for n, g in group.items()}
for g in sorted(set(group.values()), key=lambda g: -sum(v == g for v in group.values())):
    members = [n for n in names if group[n] == g]
    print(f"\ngrid of {len(members)}:", "; ".join(f"{SITES[n]['title'][:18]} ({SITES[n]['lon']:.0f})" for n in members))
    for a in members[1:]:
        c, lag = pairs.get((members[0], a)) or pairs[(a, members[0])]
        print(f"  {SITES[a]['title'][:22]:22s} corr {c:5.2f} hum lag {lag:+3d}s aircraft clocks {clock[members[0]]['offset']:+3d} {clock[a]['offset']:+3d} strength {clock[members[0]]['strength']:.0f} {clock[a]['strength']:.0f}")
cross = [c for (a, b), (c, _) in pairs.items() if group[a] != group[b]]
print("\ncorrelation across grids: median", np.round(np.median(cross), 2), "max", np.round(np.max(cross), 2))
