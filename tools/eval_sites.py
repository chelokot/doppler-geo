import sys, os, json, subprocess, math

archive, out_dir, binary, extra = sys.argv[1].split(","), sys.argv[2], sys.argv[3], sys.argv[4:]
sites = json.load(open("sites.json"))
os.makedirs(out_dir, exist_ok=True)


def distance(a, b):
    kx = math.cos(math.radians(a[0])) * 111320.0
    return math.hypot((a[1] - b[1]) * kx, (a[0] - b[0]) * 110540.0)


for name, site in sites.items():
    result = f"{out_dir}/{name}.json"
    if os.path.exists(result):
        continue
    wav = f"{out_dir}/{name}.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"dvr/{name}.opus", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav], check=True)
    open(wav + ".start", "w").write(str(site["start"]))
    log = subprocess.run([binary, "--audio", wav, "--archive", *archive, "--types", "ext/types.js", "--dem", "dem", "--out", result, *extra], capture_output=True, text=True)
    os.remove(wav)
    os.remove(wav + ".start")
    found = json.load(open(result))
    print(name, site["title"][:28], "error_m", round(distance((site["lat"], site["lon"]), (found[0]["lat"], found[0]["lon"]))), "significance", round(found[0]["significance"], 1), "runner_up", round(found[1]["significance"], 1), "passes", found[0]["passes"], log.stderr.strip().splitlines()[-1], flush=True)
