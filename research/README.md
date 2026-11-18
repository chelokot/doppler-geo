# Research

These are the Python experiments behind the locator's design. They work on the benchmark cameras with their true locations known. That separates two questions: whether an aircraft can be heard at all, and whether the search can find the point.

Run everything from this directory with the packages from `../requirements.txt`. The scripts read the following inputs:

- `../sites.json` (site list)
- `../ext/types.js` (aircraft types)
- `../dvr/<site>.opus` (recordings)
- `tracks_live/<site>.json` (aircraft tracks)

They cache spectrograms in `cache/`.

| Step | Script | Output |
|---|---|---|
| tracks of every aircraft seen within 12 km of each site, from the adsb.lol heatmaps and trace API | `live_tracks.py` | `tracks_live/` |
| recording clock offset against ADS-B, from close passes | `clock.py <site>` | `clock/<site>.json` |
| score of every real pass within 12 km, and of random wrong points as the null | `survey.py <site> 32768 log` | `survey_32768_log/<site>.npz` |
| detection rates by aircraft kind and range | `survey_report.py survey_32768_log` | stdout |
| gradient-boosted detector over survey features, leave one site out | `learn.py survey_32768_log` | stdout |
| look-alike test for single passes | `impostor.py 32768 41 <sites…>` | stdout |
| oracle tone test | `audibility.py <out.npy>` | array |
| passes up to 25 km at elevation ≥ 15° | `survey_far.py <site>` | `survey_far/<site>.npz` |
| mains hum frequency, and its grids and clocks | `enf.py <site>`, then `enf_report.py` | `enf/<site>.npz`, stdout |
| plots | `look.py`, `look_top.py` | PNG |

`survey_report.py`, `learn.py`, `audibility.py` and `look_top.py` only use sites whose clock offset is well determined.

## Findings

### Elevation angle, not distance, decides audibility

Detection here means p < 0.001 against the null. Across 3,344 propeller passes within 12 km of 35 sites, the share detected rises with elevation angle:

| Elevation | 0–5° | 5–10° | 10–15° | 15–25° | 25–45° | 45–90° |
|---|---|---|---|---|---|---|
| detected | 0.6% | 4% | 9% | 20% | 37% | 44% |

The 19% of passes above 15° give 80% of the detections. Turboprops at about 6 km altitude are still detected 21% of the time at 7–12 km slant range. Jets are never detected beyond 4 km, at any elevation.

### Beyond 7 km the tone is gone, not just hard to extract

The oracle tone test takes 105 strong propeller passes and fixes each tone's frequency from frames near the closest approach. It then measures that same frequency once the same aircraft is far away. Nothing is searched for, so this is a ceiling for any detector. The table gives the mean excess per 0.5 s frame, in units of noise σ, for one harmonic:

| Range | 0–5° | 5–15° | above 15° |
|---|---|---|---|
| near closest approach | | | 2.35 |
| 2–4 km | 0.20 | 0.26 | 0.61 |
| 4–7 km | 0.07 | 0.16 | 0.34 |
| 7–12 km | 0.01 | 0.00 | 0.04 |
| 12–18 km | 0.00 | 0.01 | 0.04 |

Allowing the tone to drift by ±1.5% changes nothing.

**2–7 km at low elevation.** The tone there is weak but real. A pass of a few hundred frames would reach z ≈ 3–4 per harmonic with the frequency known, while the blind detector catches only 1–5% of such passes. A prior on the frequency, from the aircraft type or from the same aircraft's earlier loud pass, is the open lever there.

### One pass is not unique

A straight pass at constant speed comes down to three numbers: time, speed and miss distance. About 3,000 aircraft are below 10,000 ft at any moment, and 40–80 of them fly within 1% of a given pass's speed.

`impostor.py` transplants other real tracks onto the geometry of 114 strong passes and compares scores:

| Track | Share of the true score kept |
|---|---|
| straightened copy of the true track | median 0.87 |
| other aircraft within 1% of the speed | median 0.47; 15% keep at least 0.9 |

So the curvature of real tracks rejects most look-alikes, but a single pass still leaves roughly 20–50 candidate points worldwide. Combining passes is what makes a fix unique.

### A learned detector helps a little

Over 6,979 surveyed passes:

| Detector | Passes detected |
|---|---|
| matched line (what the locator uses) | 323 |
| same, with ±3 s of clock slack | 315 |
| gradient-boosted classifier over line, harmonic comb, level envelope, range, height, speed and type | 361 |

The learned detector is not yet in the locator.

### Mains hum gives the grid and the clock, not the point

28 of 58 recordings carry a usable 50/60 Hz hum. Its frequency wander correlates at 0.8–0.99 between cameras on the same grid and at 0.03 across grids. That cleanly separates the US Eastern (18 cameras) and Western (6) interconnections. Lags in the hum match the aircraft-derived clock offsets within 1–2 s.

### Time to a fix

From the per-pass evidence at each site, these are the estimated shares of cameras that reach a confident fix:

| Audio | 10 min | 30 min | 1 h | 2 h | 4 h | 8 h |
|---|---|---|---|---|---|---|
| cameras | <1% | 4% | 9% | 20% | 40% | 50% |

20 of the 58 cameras hear too few propeller aircraft ever to get there. At sites where aircraft are heard at all, the median wait for one detected pass is 49 minutes.
