"""Triage diagnostics of one served package: /v1/status and /v1/models before and after a chat and an image request (full error bodies),
device memory counters, host MemAvailable. usage: triage_diag.py URL MODEL OUT.json"""
import base64, glob, json, struct, sys, urllib.request, urllib.error, zlib

url, model, out = sys.argv[1:4]


def http(method, path, body=None, timeout=600):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode() if body is not None else None, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        return {"code": r.status, "body": r.read().decode(errors="replace")[:6000]}
    except urllib.error.HTTPError as e:
        return {"code": e.code, "body": e.read().decode(errors="replace")[:6000]}
    except Exception as e:  # noqa: BLE001
        return {"code": None, "body": repr(e)}


def mem():
    d = {}
    for l in open("/proc/meminfo"):
        k, v = l.split(":")
        if k in ("MemTotal", "MemAvailable", "Shmem"):
            d[k] = v.strip()
    for c in sorted(glob.glob("/sys/class/drm/card*/device")):
        for n in ("mem_info_vram_total", "mem_info_vram_used", "mem_info_gtt_total", "mem_info_gtt_used"):
            try:
                d[f"{c.split('/')[4]}.{n}"] = int(open(f"{c}/{n}").read())
            except OSError:
                pass
    return d


def png(w, h, rgb):
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    def ch(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + ch(b"IDAT", zlib.compress(raw)) + ch(b"IEND", b"")


R = {"mem_start": mem(), "status0": http("GET", "/v1/status"), "models": http("GET", "/v1/models")}
R["chat"] = http("POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "Capital of France? One word."}], "max_tokens": 16, "temperature": 0})
R["mem_after_chat"] = mem()
R["status1"] = http("GET", "/v1/status")
img = "data:image/png;base64," + base64.b64encode(png(224, 224, (220, 20, 20))).decode()
R["image"] = http("POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": img}},
                  {"type": "text", "text": "What color is this image? Answer in one word."}]}], "max_tokens": 16, "temperature": 0})
R["mem_after_image"] = mem()
R["status2"] = http("GET", "/v1/status")
json.dump(R, open(out, "w"), indent=1)
print("diag:", R["chat"]["code"], R["image"]["code"], (R["image"]["body"] or "")[:200])
