import sys
import numpy as np
import scipy.signal as signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import site_tracks, read_audio, passes, geometry, RATE

site, ground, selector, out = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
x, start = read_audio(site)
rate = RATE
tracks = site_tracks(site)
ps = [p for p in passes(tracks, ground) if start + 120 < p["recv"] < start + len(x) / rate - 120]
ps = sorted([p for p in ps if eval(selector, {}, p)], key=lambda p: p["r"])[:12]
hop, nper = int(0.25 * rate), 4096
cols = 3
fig, axes = plt.subplots((len(ps) + cols - 1) // cols, cols, figsize=(8 * cols, 4.2 * ((len(ps) + cols - 1) // cols)), squeeze=False)
for ax, p in zip(axes.flat, ps):
    a = int((p["recv"] - 120 - start) * rate)
    seg = x[a:a + 240 * rate]
    f, t, Z = signal.stft(seg, fs=rate, nperseg=nper, noverlap=nper - hop, boundary=None, padded=False)
    P = 10 * np.log10(np.abs(Z) ** 2 + 1e-3)
    P -= np.median(P, axis=1, keepdims=True)
    keep = f >= 30
    ax.pcolormesh(t - 120 + (nper / 2) / rate, f[keep], P[keep], shading="auto", vmin=-3, vmax=15, cmap="magma")
    ax.set_yscale("log")
    ax.set_ylim(30, rate / 2)
    tr = tracks[p["track"]]
    r, recv, k = geometry(tr, ground)
    m = (recv > p["recv"] - 120) & (recv < p["recv"] + 120)
    for f0 in (100, 300, 1000, 3000):
        ax.plot(recv[m] - p["recv"], f0 * k[m], color="cyan", lw=0.6, alpha=0.6)
    ax.axvline(0, color="white", lw=0.5, ls="--")
    ax.set_title(f"{p['kind']} {p['type']} r={p['r']:.0f} alt={p['alt']:.0f} v={p['speed']:.0f}", fontsize=10)
for ax in axes.flat[len(ps):]:
    ax.axis("off")
fig.tight_layout()
fig.savefig(out, dpi=60)
print(len(ps), "passes")
