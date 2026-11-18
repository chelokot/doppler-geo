import os, re, sys, json, subprocess
from concurrent.futures import ThreadPoolExecutor
import requests

STREAMS = "dvr/streams.txt"
CAMS = "dvr/cams.json"
COOKIES = ["--cookies-from-browser", sys.argv[1]] if len(sys.argv) > 1 else []
COORDINATES = [r"@(-?\d+\.\d+),(-?\d+\.\d+)", r"[?&](?:q|query|ll)=(-?\d+\.\d+)(?:,|%2C)\+?(-?\d+\.\d+)", r"!3d(-?\d+\.\d+)!4d(-?\d+\.\d+)", r"/search/(-?\d+\.\d+),\+?(-?\d+\.\d+)"]

chosen = {l.split("|")[0]: l.split("|", 2)[2].strip() for l in open(STREAMS) if "|is_live|" in l}
known = json.load(open(CAMS)) if os.path.exists(CAMS) else {}
todo = [v for v in chosen if not (known.get(v) or {}).get("coords")]


def describe(video_id):
    for _ in range(3):
        out = subprocess.run(["yt-dlp", *COOKIES, "-j", "--skip-download", f"https://www.youtube.com/watch?v={video_id}"], capture_output=True, text=True, timeout=120).stdout
        if out:
            return json.loads(out)["description"]
    return ""


def resolve(video_id):
    link = re.search(r"Camera Location:\s*(https?://\S+)", describe(video_id))
    if not link:
        return video_id, None, None
    url = link.group(1)
    for _ in range(6):
        r = requests.get(url, allow_redirects=False, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        if "location" not in r.headers:
            break
        url = r.headers["location"]
    found = next(filter(None, (re.search(pattern, url) for pattern in COORDINATES)), None)
    return video_id, url, (float(found.group(1)), float(found.group(2))) if found else None


with ThreadPoolExecutor(2) as pool:
    for video_id, url, coords in pool.map(resolve, todo):
        known[video_id] = dict(title=chosen[video_id], url=url, coords=coords)
        print(video_id, coords, (url or "")[:120], flush=True)
json.dump(known, open(CAMS, "w"), indent=1)
