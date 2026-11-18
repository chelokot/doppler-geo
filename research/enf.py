import sys, os, subprocess
import numpy as np
from scipy import signal
from data import SITES, audio_path

RATE = 400
BASEBAND = 2
MAINS = (50.0, 60.0)
HARMONICS = (1, 2, 3)
WINDOW, HOP, PAD = 32, 8, 1024
DEVIATION = 0.1


def line(x, frequency, harmonic):
    mixed = x * np.exp(-2j * np.pi * frequency * harmonic * np.arange(len(x)) / RATE)
    z = signal.resample_poly(mixed, 1, RATE // BASEBAND)
    f, t, Z = signal.stft(z, fs=BASEBAND, nperseg=WINDOW, noverlap=WINDOW - HOP, nfft=PAD, return_onesided=False, boundary=None)
    P = np.abs(Z) ** 2
    near = np.flatnonzero(np.abs(f) <= DEVIATION * harmonic)
    k = near[P[near].argmax(0)]
    cols = np.arange(P.shape[1])
    lo, mid, hi = (np.log(P[(k + d) % PAD, cols]) for d in (-1, 0, 1))
    peak = f[k] + np.clip(0.5 * (lo - hi) / (lo - 2 * mid + hi), -0.5, 0.5) * BASEBAND / PAD
    return frequency + peak / harmonic, P[k, cols] / np.median(P, axis=0), t


name = sys.argv[1]
raw = subprocess.run(["ffmpeg", "-v", "error", "-i", audio_path(name), "-ac", "1", "-ar", str(RATE), "-f", "s16le", "-"], capture_output=True, check=True).stdout
x = np.frombuffer(raw, np.int16).astype(np.float64)
results = [[line(x, m, h) for h in HARMONICS] for m in MAINS]
os.makedirs("enf", exist_ok=True)
np.savez(f"enf/{name}.npz", t=SITES[name]["start"] + results[0][0][2], frequency=np.array([[r[0] for r in row] for row in results], np.float32), snr=np.array([[r[1] for r in row] for row in results], np.float32))
