"""Image-input gate: Loom vs llama.cpp (llama-server --mmproj) on the same images and questions.

usage:
  record:  python vision_golden.py record OUT.json --llama http://127.0.0.1:8130 [--images a.png,b.jpg] [--tokens 96]
  compare: python vision_golden.py compare PACKAGE OUT.json --url http://127.0.0.1:8160 [--alt OUT2.json]

Both engines get the same OpenAI chat request (an image_url part + a text part, greedy, thinking off).
Per case:
  - prompt tokens must match exactly. A mismatch means the image token count (resize rule, merge,
    min/max token limits), the image begin/end markers or the chat rendering of the image part differ:
    that is a port bug, not a near-tie, and the gate FAILs.
  - the replies are compared token by token (the package tokenizer). At the first divergence, llama.cpp's
    top-5 log-probs at that position decide: a near-tie (bench_common.near_tie's rule on log-probs) is
    PASS, anything else FAIL. llama.cpp is not run-to-run deterministic at near-ties: record twice
    (record OUT2.json) and pass --alt, so a case where llama.cpp disagrees with itself is not a FAIL.
Images: a built-in deterministic set (shapes + text, a chart, a dense text card, a photo-like gradient,
a tall and a wide aspect ratio: they exercise resize, both grid directions and the token limits) plus
--images. Prints per-case lines and VISION_RESULT {"verdict": "PASS"|"FAIL", ...}.
Run llama.cpp and Loom one at a time (record, stop llama-server, then serve Loom and compare).
"""
import argparse, base64, io, json, math, os, sys

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def builtin_images():
    from PIL import Image, ImageDraw, ImageFont
    def font(size):
        for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "C:/Windows/Fonts/arialbd.ttf",
                  "/System/Library/Fonts/Supplemental/Arial Bold.ttf"):
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
        return ImageFont.load_default()
    out = {}
    im = Image.new("RGB", (640, 480), "white"); d = ImageDraw.Draw(im)
    d.ellipse((40, 60, 240, 260), fill="red"); d.rectangle((380, 60, 600, 260), fill="blue")
    d.polygon([(320, 300), (240, 440), (400, 440)], fill="green"); d.text((430, 380), "LOOM 42", fill="black", font=font(48), anchor="mm")
    out["shapes"] = (im, "Describe every shape in this image with its color and position, and transcribe any text exactly.")
    im = Image.new("RGB", (800, 500), (250, 250, 245)); d = ImageDraw.Draw(im)
    vals = [3, 7, 4, 9, 6]; names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    for i, (v, n) in enumerate(zip(vals, names)):
        x = 100 + i * 130; d.rectangle((x, 420 - v * 40, x + 80, 420), fill=(40, 90, 200)); d.text((x + 40, 445), n, fill="black", font=font(24), anchor="mm")
    d.text((400, 30), "Tickets closed per day", fill="black", font=font(30), anchor="mm")
    out["chart"] = (im, "What is the title of this chart, and which day has the tallest bar?")
    im = Image.new("RGB", (900, 600), "white"); d = ImageDraw.Draw(im)
    lines = ["INVOICE #20931", "Date: 2026-11-03", "2 x Widget A      $14.00", "1 x Gadget B      $27.50", "Shipping           $5.25", "TOTAL             $46.75"]
    for i, t in enumerate(lines):
        d.text((60, 50 + i * 80), t, fill="black", font=font(40))
    out["text"] = (im, "Transcribe this document exactly, line by line.")
    w, h = 1024, 768; im = Image.new("RGB", (w, h)); px = im.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = (int(128 + 100 * math.sin(x / 50.0)), int(128 + 100 * math.cos(y / 37.0)), int((x * y) % 255))
    d = ImageDraw.Draw(im); d.ellipse((400, 250, 640, 520), fill=(250, 200, 30)); d.rectangle((100, 600, 300, 700), fill=(20, 20, 20))
    out["photo"] = (im, "Describe this image in two sentences.")
    im = Image.new("RGB", (300, 1200), (230, 240, 255)); d = ImageDraw.Draw(im)
    for i, c in enumerate(["red", "orange", "yellow", "green", "blue", "purple"]):
        d.ellipse((90, 40 + i * 190, 210, 160 + i * 190), fill=c)
    out["tall"] = (im, "List the colors of the circles from top to bottom.")
    im = Image.new("RGB", (1600, 260), "white"); d = ImageDraw.Draw(im)
    d.text((800, 130), "THE QUICK BROWN FOX 1234", fill=(120, 0, 0), font=font(90), anchor="mm")
    out["wide"] = (im, "What text is written in this image?")
    return out


def data_url(im):
    buf = io.BytesIO(); im.save(buf, format="PNG"); return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def request(url, model, durl, question, tokens, llama):
    body = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": durl}}, {"type": "text", "text": question}]}],
            "temperature": 0, "max_tokens": tokens, "enable_thinking": False, "chat_template_kwargs": {"enable_thinking": False}}
    if model:
        body["model"] = model
    if llama:
        body.update({"logprobs": True, "top_logprobs": 5, "cache_prompt": False})
    r = requests.post(url + "/v1/chat/completions", json=body, headers={"Content-Type": "application/json"}, timeout=1800)
    if not r.ok:
        return {"error": f"HTTP {r.status_code}: {r.text[:300]}"}
    o = r.json(); ch = o["choices"][0]
    res = {"content": ch["message"].get("content") or "", "prompt_tokens": o.get("usage", {}).get("prompt_tokens"),
           "finish": ch.get("finish_reason")}
    if llama:
        res["tokens"] = [{"token": t["token"], "logprob": t["logprob"], "top": {x["token"]: x["logprob"] for x in t.get("top_logprobs", [])}}
                         for t in (ch.get("logprobs") or {}).get("content", [])]
    return res


def cases(images_arg):
    items = {k: (data_url(im), q, im.size) for k, (im, q) in builtin_images().items()}
    for p in [x for x in (images_arg or "").split(",") if x]:
        from PIL import Image
        im = Image.open(p).convert("RGB"); items[os.path.basename(p)] = (data_url(im), "Describe this image in detail.", im.size)
    return items


def near_tie(top, gold, loom, max_gap=1.0):
    """bench_common.near_tie on token strings (llama.cpp's chat logprobs carry strings, not ids)."""
    if gold not in top or loom not in top:
        return False
    if top[gold] - top[loom] <= max_gap:
        return True
    ranked = sorted(top, key=top.get, reverse=True)
    return ranked[:2] == [gold, loom] and math.exp(top[gold]) < 0.5


def judge(pkg, ref, got):
    """-> (status, detail). Token-aligned via llama.cpp's own token strings."""
    if got.get("error"):
        return "FAIL", got["error"]
    if ref["prompt_tokens"] != got["prompt_tokens"]:
        return "FAIL", f"prompt tokens differ: llama.cpp {ref['prompt_tokens']} vs Loom {got['prompt_tokens']} (image token count, markers or rendering)"
    if got["content"] == ref["content"]:
        return "PASS", f"identical ({len(ref['tokens'])} tokens)"
    text = ""; toks = ref["tokens"]
    for k, t in enumerate(toks):
        if got["content"].startswith(text + t["token"]):
            text += t["token"]; continue
        rest = got["content"][len(text):]
        loom_tok = pkg.tok.decode(pkg.ids(rest)[:1]) if rest else ""
        if not rest and ref["finish"] != "length":
            return "FAIL", f"Loom stopped at token {k}, llama.cpp continued with {t['token']!r}"
        nt = near_tie(t["top"], t["token"], loom_tok)
        gap = t["top"].get(t["token"], 0) - t["top"].get(loom_tok, float("-inf"))
        return ("PASS" if nt else "FAIL"), f"diverged at token {k}: llama.cpp {t['token']!r} vs Loom {loom_tok!r} (gap {gap:.2f}{', near-tie' if nt else ''})"
    return "PASS", "Loom continued past llama.cpp's end (stop/length difference)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["record", "compare"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--llama", default="http://127.0.0.1:8130")
    ap.add_argument("--url", default="http://127.0.0.1:8160")
    ap.add_argument("--alt", default="")
    ap.add_argument("--images", default="")
    ap.add_argument("--tokens", type=int, default=96)
    a = ap.parse_args()
    if a.mode == "record":
        out = {}
        for name, (durl, q, size) in cases(a.images).items():
            out[name] = dict(question=q, size=size, **request(a.llama, None, durl, q, a.tokens, True))
            print(name, size, out[name].get("prompt_tokens"), repr(out[name].get("content", out[name].get("error", "")))[:100], flush=True)
        json.dump({"tokens": a.tokens, "images": a.images, "cases": out}, open(a.args[0], "w"), indent=1)
        return
    from bench_common import Package
    pkg = Package(a.args[0]); gold = json.load(open(a.args[1])); alt = json.load(open(a.alt))["cases"] if a.alt else {}
    model = requests.get(a.url + "/v1/models", timeout=30).json()["data"][0]["id"]
    items = cases(gold.get("images", "")); bad = []; rows = []
    for name, ref in gold["cases"].items():
        durl, q, size = items[name]
        got = request(a.url, model, durl, q, gold["tokens"], False)
        st, detail = judge(pkg, ref, got)
        if st == "FAIL" and name in alt:
            st2, d2 = judge(pkg, alt[name], got)
            if st2 == "PASS":
                st, detail = "PASS", d2 + " (matches the alternate llama.cpp recording)"
        print(f"{st} {name} {size}: {detail}", flush=True)
        rows.append({"case": name, "status": st, "detail": detail, "loom": got.get("content", ""), "llama": ref.get("content", "")})
        if st == "FAIL":
            bad.append(name)
    print("VISION_RESULT " + json.dumps({"verdict": "FAIL" if bad else "PASS", "cases": len(rows), "failed": bad}))
    json.dump(rows, open(os.path.splitext(a.args[1])[0] + "_compare.json", "w"), indent=1)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
