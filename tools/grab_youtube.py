import sys, re, json, subprocess, datetime, os, time
from concurrent.futures import ThreadPoolExecutor
import requests

DEPTH = 6500
SEGMENT_S = 5.0
MIN_RUN = 360
OUT = "dvr"
COOKIES = ["--cookies-from-browser", sys.argv[4]] if len(sys.argv) > 4 else []


class MembersOnly(Exception):
    pass


def stream_url(video_id):
    while True:
        result = subprocess.run(["yt-dlp", *COOKIES, "-g", "-f", "91", f"https://www.youtube.com/watch?v={video_id}"], capture_output=True, text=True, timeout=120)
        if result.stdout.strip():
            time.sleep(15)
            return result.stdout.split()[0]
        if "members" in result.stderr:
            raise MembersOnly(video_id)
        print("blocked, waiting", video_id, flush=True)
        time.sleep(180)


def playlist(video_id):
    url = stream_url(video_id)
    text = requests.get(url, timeout=30).text
    sequence = int(re.search(r"#EXT-X-MEDIA-SEQUENCE:(\d+)", text).group(1))
    pdt = datetime.datetime.fromisoformat(re.search(r"#EXT-X-PROGRAM-DATE-TIME:(\S+)", text).group(1)).timestamp()
    template = next(line for line in text.splitlines() if line and not line.startswith("#"))
    last = max(int(s) for s in re.findall(r"/sq/(\d+)/", text))
    return sequence, pdt, template, last


def grab(video_id):
    sequence, pdt, template, last = playlist(video_id)
    start_of = lambda sq: pdt + (sq - sequence) * SEGMENT_S
    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter(max_retries=requests.adapters.Retry(total=5, backoff_factor=1.0)))
    def fetch(sq):
        r = session.get(re.sub(r"/sq/\d+/", f"/sq/{sq}/", template), timeout=30)
        return sq, r.content if r.status_code == 200 else None
    with ThreadPoolExecutor(12) as pool:
        segments = [s for s in pool.map(fetch, range(last - DEPTH, last + 1)) if s[1]]
    runs = []
    for sq, data in segments:
        if runs and sq == runs[-1][-1][0] + 1:
            runs[-1].append((sq, data))
        else:
            runs.append([(sq, data)])
    kept = [r for r in runs if len(r) >= MIN_RUN]
    for run in kept:
        begin = start_of(run[0][0])
        path = f"{OUT}/{video_id}_{begin:.3f}.opus"
        proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-i", "pipe:0", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "libopus", "-b:a", "24k", path], stdin=subprocess.PIPE)
        for _, data in run:
            proc.stdin.write(data)
        proc.stdin.close()
        proc.wait()
    return video_id, [(round(len(r) * SEGMENT_S / 3600, 2), datetime.datetime.fromtimestamp(start_of(r[0][0]), datetime.UTC).strftime("%H:%M")) for r in kept]


def place(title):
    return re.sub(r"\s+", " ", title.split("|")[0].split("(")[0].split(" - ")[0]).strip().lower()


def first_public(candidates):
    for video_id, title in candidates:
        try:
            return grab(video_id)
        except MembersOnly:
            continue
        except Exception as error:
            return video_id, f"failed: {error!r}"[:200]
    return candidates[0][0], "members only"


os.makedirs(OUT, exist_ok=True)
done = {f.rsplit("_", 1)[0] for f in os.listdir(OUT) if f.endswith(".opus")}
rows = [l.rstrip("\n").split("|", 2) for l in open(sys.argv[1]) if "|is_live|" in l]
places = {}
for video_id, _, title in rows:
    places.setdefault(place(title), []).append((video_id, title))
cams = json.load(open(f"{OUT}/cams.json"))
located = lambda video_id: bool(cams.get(video_id, {}).get("coords"))
pending = [sorted(c, key=lambda c: (not located(c[0]), "fixed" in c[1].lower())) for c in places.values() if any(located(v) for v, _ in c) and not any(v in done for v, _ in c)]
print(len(pending), "places pending", flush=True)
shard, shards = int(sys.argv[2]), int(sys.argv[3])
for candidates in pending[shard::shards]:
    print(*first_public(candidates), flush=True)
