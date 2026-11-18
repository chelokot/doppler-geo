import sys, time, json
import requests

name, lat, lon, radius_nm, duration_s = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
sources = [
    ("lol", f"https://api.adsb.lol/v2/point/{lat}/{lon}/{radius_nm}", lambda j: (j["now"] / 1000, j["ac"])),
    ("fi", f"https://opendata.adsb.fi/api/v2/lat/{lat}/lon/{lon}/dist/{radius_nm}", lambda j: (j["now"], j["aircraft"])),
]
deadline = time.time() + duration_s
with open(f"adsb_{name}.jsonl", "a") as out:
    i = 0
    while time.time() < deadline:
        src, url, unpack = sources[i % 2]
        i += 1
        started = time.time()
        try:
            now, aircraft = unpack(requests.get(url, timeout=8, headers={"User-Agent": "doppler-geo"}).json())
            out.write(json.dumps({"src": src, "fetched": started, "now": now, "ac": aircraft}) + "\n")
            out.flush()
        except (requests.RequestException, ValueError, KeyError) as error:
            print(src, "error", repr(error)[:80], flush=True)
        time.sleep(max(0.0, 1.5 - (time.time() - started)))
