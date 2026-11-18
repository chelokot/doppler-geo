import os, io, re, json, glob, math
import numpy as np
import requests
from PIL import Image

cams = json.load(open("dvr/cams.json"))


def ground(lat, lon, zoom=11):
    n = 2 ** zoom
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    png = requests.get(f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{zoom}/{int(x)}/{int(y)}.png", timeout=30).content
    r, g, b = np.array(Image.open(io.BytesIO(png)).convert("RGB"), float)[int(y % 1 * 256), int(x % 1 * 256)]
    return r * 256 + g + b / 256 - 32768


sites = {}
for path in sorted(glob.glob("dvr/*.opus")):
    name = os.path.basename(path)[:-5]
    video_id, start = name.rsplit("_", 1)
    cam = cams[video_id]
    if not cam["coords"]:
        continue
    pin = re.search(r"!3d(-?\d+\.\d+)!4d(-?\d+\.\d+)", cam["url"])
    lat, lon = (float(pin.group(1)), float(pin.group(2))) if pin else cam["coords"]
    sites[name] = dict(title=cam["title"].split("|")[0].strip(), start=float(start), lat=lat, lon=lon, ground=round(ground(lat, lon), 1))
json.dump(sites, open("sites.json", "w"), indent=1)
print(len(sites), "sites")
