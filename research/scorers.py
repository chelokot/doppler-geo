import numpy as np
import scipy.ndimage as ndimage
from spec import LOG_STEP, LOGF


def robust_z(profile):
    p = profile[np.isfinite(profile)]
    if len(p) < 50:
        return np.nan
    med = np.median(p)
    mad = np.median(np.abs(p - med))
    return (p.max() - med) / (1.4826 * mad + 1e-9)


def weights(view, power=2):
    return (view["r"].min() / view["r"]) ** power


def profile(site, view, sign, w):
    D = site.dewarp(view, sign)
    m = np.isfinite(D)
    return np.where(m.sum(0) > 0.8 * len(D), np.nansum(D * w[:, None], 0) / np.maximum((m * w[:, None]).sum(0), 1e-9), np.nan)


def base(site, view):
    w = weights(view)
    return robust_z(profile(site, view, 1, w)) - robust_z(profile(site, view, -1, w))


LEVEL_CLIP_DB = 6.0


def envelope(site, view):
    w = weights(view)
    w = w - w.mean()
    return np.clip(site.L[view["frames"]], -LEVEL_CLIP_DB, LEVEL_CLIP_DB).T @ w / (np.linalg.norm(w) + 1e-9)


COMB_HARMONICS = np.arange(1, 13)
COMB_FUNDAMENTALS = np.arange(np.log(20.0), np.log(320.0), LOG_STEP)
COMB_BINS = np.round((COMB_FUNDAMENTALS[:, None] + np.log(COMB_HARMONICS)[None, :] - LOGF[0]) / LOG_STEP).astype(int)
COMB_VALID = (COMB_BINS >= 0) & (COMB_BINS < len(LOGF))
COMB_MIN_HARMONICS = 3
COMB_TOLERANCE_BINS = 2


def comb_peak(P):
    ok = np.isfinite(P)
    if ok.sum() < 50:
        return np.nan
    med = np.median(P[ok])
    z = np.where(ok, (P - med) / (1.4826 * np.median(np.abs(P[ok] - med)) + 1e-9), 0.0)
    pooled = ndimage.maximum_filter1d(z, 2 * COMB_TOLERANCE_BINS + 1)
    count = COMB_VALID.sum(1)
    ladder = (pooled[np.clip(COMB_BINS, 0, len(LOGF) - 1)] * COMB_VALID).sum(1) / np.sqrt(np.maximum(count, 1))
    return ladder[count >= COMB_MIN_HARMONICS].max()


def line_and_comb(site, view):
    w = weights(view)
    match, mirror = profile(site, view, 1, w), profile(site, view, -1, w)
    return robust_z(match) - robust_z(mirror), comb_peak(match) - comb_peak(mirror)
