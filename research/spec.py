import os
import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndimage
from data import read_audio, RATE

HOP_S = 0.5
LOG_STEP = 0.005
FMIN, FMAX = 40.0, 4000.0
LOGF = np.arange(np.log(FMIN), np.log(FMAX), LOG_STEP)
CLIP = (-3.0, 6.0)
POWER_RATIO_CLIP = 12.0
BLOCK = 512


BANDS = [(40, 80), (80, 160), (160, 315), (315, 630), (630, 1250), (1250, 2500), (2500, 5000), (5000, 7900)]
DRIFT_FRAMES = 1201


def frame_power(x, nper):
    hop = int(HOP_S * RATE)
    window = signal.get_window("hann", nper).astype(np.float32)
    frames = (len(x) - nper) // hop + 1
    for first in range(0, frames, BLOCK):
        rows = np.arange(first, min(first + BLOCK, frames))[:, None] * hop + np.arange(nper)[None, :]
        yield np.abs(np.fft.rfft(x[rows] * window, axis=1)[:, 1:] / window.sum()) ** 2


def log_spectrogram(x, nper):
    f = np.fft.rfftfreq(nper, 1 / RATE)[1:]
    logf = np.log(f)
    upper = np.searchsorted(logf, LOGF)
    weight = ((LOGF - logf[upper - 1]) / (logf[upper] - logf[upper - 1])).astype(np.float32)
    edges = np.searchsorted(f, np.exp(np.append(LOGF, LOGF[-1] + LOG_STEP) - LOG_STEP / 2))
    wide = np.diff(edges) >= 2
    first_wide = int(np.argmax(wide))
    blocks = []
    for power in frame_power(x, nper):
        interpolated = np.log(power[:, upper - 1] + 1e-3) * (1 - weight) + np.log(power[:, upper] + 1e-3) * weight
        interpolated[:, first_wide:] = np.log(np.maximum.reduceat(power, edges[first_wide:], axis=1)[:, :-1] + 1e-3)
        blocks.append(interpolated)
    return np.concatenate(blocks)


def levels(site, nper=8192):
    path = f"cache/{site}_levels.npy"
    if not os.path.exists(path):
        f = np.fft.rfftfreq(nper, 1 / RATE)[1:]
        x, _ = read_audio(site)
        L = np.concatenate([np.stack([10 * np.log10(power[:, (f >= a) & (f < b)].mean(1) + 1e-9) for a, b in BANDS], 1) for power in frame_power(x, nper)]).astype(np.float32)
        np.save(path, L - ndimage.median_filter(L, size=(DRIFT_FRAMES, 1), mode="nearest"))
    return np.load(path)


def compute(site, nper, line_width, domain):
    path = f"cache/{site}_{nper}_{line_width}_{domain}.npz"
    if not os.path.exists(path):
        x, start = read_audio(site)
        S = log_spectrogram(x, nper)
        S -= np.median(S, axis=1, keepdims=True)
        S -= ndimage.median_filter(S, size=(121, 1), mode="nearest")
        S -= ndimage.median_filter(S, size=(1, line_width), mode="nearest")
        if domain == "power":
            S = np.minimum(np.exp(S), POWER_RATIO_CLIP) - 1
        S /= np.maximum(1.4826 * np.median(np.abs(S), axis=0, keepdims=True), np.finfo(np.float32).tiny)
        quantized = np.round((np.clip(S, *CLIP) - CLIP[0]) / (CLIP[1] - CLIP[0]) * 255).astype(np.uint8)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, S=quantized, t0=start + nper / (2 * RATE))
    d = np.load(path)
    return d["S"].astype(np.float32) / 255 * (CLIP[1] - CLIP[0]) + CLIP[0], float(d["t0"])
