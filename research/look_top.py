import sys, glob, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import bench, scorers
from spec import LOGF
from survey_report import evidence, MIN_CLOCK_STRENGTH, MIN_NEGATIVES

survey, out, kinds, lo, hi, count, order = sys.argv[1], sys.argv[2], sys.argv[3].split(","), float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
picks = []
for path in sorted(glob.glob(f"{survey}/*.npz")):
    name = os.path.basename(path)[:-4]
    d = np.load(path)
    positive = d["label"] == 1
    if json.load(open(f"clock/{name}.json"))["strength"] < MIN_CLOCK_STRENGTH or (~positive).sum() < MIN_NEGATIVES:
        continue
    e = evidence(d["line"][:, [d["line"].shape[1] // 2]], positive)[:, 0]
    distance = d["range"][positive].astype(float)
    for i in np.where(np.isin(d["kind"][positive], kinds) & (distance >= lo) & (distance < hi))[0]:
        picks.append((e[i], name, int(d["track"][positive][i]), d["type"][positive][i], distance[i]))
picks = sorted(picks, reverse=order == "top")[:count]
ticks = [int(np.argmin(abs(np.exp(LOGF) - f))) for f in (50, 100, 200, 400, 800, 1600, 3200)]
labels = ["50", "100", "200", "400", "800", "1.6k", "3.2k"]
fig, axes = plt.subplots(len(picks), 3, figsize=(20, 3.2 * len(picks)), squeeze=False, gridspec_kw=dict(width_ratios=[3, 3, 2]))
sites = {}
for row, (e, name, index, type_code, r) in zip(axes, picks):
    site = sites.get(name) or sites.setdefault(name, bench.Site(name, bench.SURVEY_RADIUS, 32768, "log", 41, bench.ALL_KINDS, bench.clock(name)))
    view = site.pass_view(site.tracks[index], (0.0, 0.0))
    near = view["r"] < max(3 * view["r"].min(), 2500.0)
    view = {k: v[near] for k, v in view.items()}
    row[0].imshow(site.S[view["frames"].min():view["frames"].max() + 1, bench.PAD:-bench.PAD].T, aspect="auto", origin="lower", cmap="magma", vmin=-1, vmax=4)
    row[1].imshow(site.dewarp(view).T, aspect="auto", origin="lower", cmap="magma", vmin=-1, vmax=4)
    for ax in row:
        (ax.set_xticks if ax is row[2] else ax.set_yticks)(ticks, labels)
    row[0].set_title(f"{type_code} ({site.tracks[index]['kind']}), closest {r:.0f} m: spectrogram, Hz over 0.5 s frames")
    row[1].set_title(f"de-warped along the Doppler curve predicted from ADS-B, evidence {e:.1f}")
    row[2].set_title("frequency profile: matched (blue) vs mirrored (orange)")
    w = scorers.weights(view)
    row[2].plot(scorers.profile(site, view, 1, w), lw=0.8)
    row[2].plot(scorers.profile(site, view, -1, w), lw=0.5, alpha=0.6)
fig.tight_layout()
fig.savefig(out, dpi=70)
