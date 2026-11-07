"""Image-input gate: Loom vs llama.cpp (llama-server --mmproj) on the same images and questions.

usage: python vision_golden.py PACKAGE --url http://127.0.0.1:8160 --llama http://127.0.0.1:8130 [--images a.png,b.jpg] [--tokens 96]
                                [--cases shapes,wide]   (a subset of the built-in images: gates.sh LEVEL=dev)

Like golden.py, llama.cpp gets the package's own rendering of the conversation (render_messages, with
llama.cpp's media marker where the image goes) plus the image through /completion: both engines see the
same prompt, and a template's moving parts (a current-date line) cannot make them differ. Loom gets the
OpenAI chat request (an image_url part + a text part, greedy, thinking off). Per case:
  - prompt tokens must match exactly. A mismatch means the image token count (resize rule, merge,
    min/max token limits), the image begin/end markers or the image request's chat rendering differ
    from the text path: a port bug, not a near-tie.
  - the reply (raw_from_chat) is compared token by token with llama.cpp's greedy ids; the first
    divergence must be a near-tie (bench_common.near_tie) by llama.cpp's top-5 log-probs, else it is
    re-recorded once (llama.cpp flips near-ties run to run) and judged against both.
References go through bench_common.llama_completion: cached in $LLAMA_CACHE, deferred with
$LLAMA_DEFER (gates.sh fills them with only llama.cpp loaded, started with MMPROJ=).
Images: a built-in deterministic set (shapes + text, a chart, a dense text card, a photo-like gradient,
a tall and a wide aspect ratio: they exercise resize, both grid directions and the token limits) plus
--images. Prints per-case lines and VISION_RESULT {"verdict": "PASS"|"FAIL", ...}.
"""
import argparse, base64, io, json, math, os, re, sys

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


SHAPES_QUESTION = "What shapes and colors are in this image?"


def about_shapes(reply):
    """Whether a reply to SHAPES_QUESTION on the "shapes" image (a red circle, a blue rectangle, a green triangle on
    white) is about that image: an English answer naming, as whole words, two of its three colors and one of its
    shapes. A non-empty reply is not enough: a broken port answers in fluent, often multilingual, garbage, which
    can still hold "red" inside other words. port.sh's image smoke check."""
    text = (reply or "").lower()
    letters = [c for c in text if c.isalpha()]
    if not letters or sum(c.isascii() for c in letters) < 0.9 * len(letters):
        return False
    words = {w.rstrip("s") for w in re.findall(r"[a-z]+", text)}
    return len(words & {"red", "blue", "green"}) >= 2 and bool(words & {"circle", "rectangle", "square", "triangle", "oval", "ellipse"})


def data_url(im):
    buf = io.BytesIO(); im.save(buf, format="PNG"); return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


MARKER = "<__media__>"   # llamacpp_server.sh pins llama-server's media marker (LLAMA_MEDIA_MARKER) to this


def request(url, model, durl, question, tokens, llama=False):
    """Loom: the OpenAI chat request."""
    body = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": durl}}, {"type": "text", "text": question}]}],
            "temperature": 0, "max_tokens": tokens, "enable_thinking": False, "chat_template_kwargs": {"enable_thinking": False}}
    if model:
        body["model"] = model
    r = requests.post(url + "/v1/chat/completions", json=body, headers={"Content-Type": "application/json"}, timeout=1800)
    if not r.ok:
        return {"error": f"HTTP {r.status_code}: {r.text[:300]}"}
    o = r.json(); ch = o["choices"][0]
    return {"content": ch["message"].get("content") or "", "reasoning": ch["message"].get("reasoning_content") or "",
            "message": ch["message"], "prompt_tokens": o.get("usage", {}).get("prompt_tokens"), "finish": ch.get("finish_reason")}


_BOS = {}
def adds_bos(gguf_path):
    if gguf_path not in _BOS:
        import gguf
        f = gguf.GGUFReader(gguf_path).fields.get("tokenizer.ggml.add_bos_token")
        _BOS[gguf_path] = bool(f.parts[f.data[0]][0]) if f is not None else True
    return _BOS[gguf_path]


def reference(pkg, llama, durl, question, tokens, salt=""):
    """llama.cpp: the package's rendering + the image, greedy, top-5 log-probs by id."""
    from bench_common import GREEDY, llama_completion
    prompt = pkg.render_messages([{"role": "user", "content": MARKER + question}], enable_thinking=False)
    bos = (pkg.chat or {}).get("bos", "")
    if bos and prompt.startswith(bos) and adds_bos(pkg.cfg["gguf"]):   # llama.cpp tokenizes prompt_string with BOS
        prompt = prompt[len(bos):]
    body = dict(GREEDY, prompt={"prompt_string": prompt, "multimodal_data": [durl.split(",", 1)[1]]},
                n_predict=tokens, n_probs=5, return_tokens=True)
    return llama_completion(llama, body, pkg, salt)


def cases(images_arg):
    items = {k: (data_url(im), q, im.size) for k, (im, q) in builtin_images().items()}
    for p in [x for x in (images_arg or "").split(",") if x]:
        from PIL import Image
        im = Image.open(p).convert("RGB"); items[os.path.basename(p)] = (data_url(im), "Describe this image in detail.", im.size)
    return items


def judge(pkg, ref, got):
    """-> (status, detail)."""
    from bench_common import near_tie
    if got.get("error"):
        return "FAIL", got["error"]
    n_ref = ref.get("tokens_evaluated") or (ref.get("timings") or {}).get("prompt_n")
    if n_ref != got["prompt_tokens"]:
        return "FAIL", f"prompt tokens differ: llama.cpp {n_ref} vs Loom {got['prompt_tokens']} (image token count, markers or rendering)"
    cp = ref.get("completion_probabilities", []); gold = ref.get("tokens") or [x["id"] for x in cp]
    finished = got["finish"] == "stop"
    raws = [pkg.raw_from_chat(got["message"], finished, False)]
    import chat_probe as _cp
    if hasattr(_cp, "raw_alternatives"):   # a channel format ambiguous in the parsed message (Glimmer's optional <|eom|>)
        raws += _cp.raw_alternatives(pkg.chat, got["message"], finished, False)[1:]
    best = None
    for raw in raws:   # the raw text consistent with the parsed reply that agrees longest with llama.cpp
        ids = pkg.ids(raw)
        kk = next((i for i, (a, b) in enumerate(zip(gold, ids)) if a != b), min(len(gold), len(ids)))
        if best is None or kk > best[0]:
            best = (kk, ids)
    k, mine = best
    # Loom's ids are its text re-tokenized, which can merge what it generated as two tokens ("GB" + "P" -> "GBP"): the
    # divergence is the first llama.cpp token whose text Loom's text does not continue with; Loom's token there is its
    # remaining text's first token.
    text = pkg.tok.decode(mine)
    if k < len(gold) and k < len(mine) and text.startswith(pkg.tok.decode(gold[:k + 1])):
        j = k
        while j < len(gold) and text.startswith(pkg.tok.decode(gold[:j + 1])):
            j += 1
        rest = pkg.ids(text[len(pkg.tok.decode(gold[:j])):])
        k, mine = j, list(gold[:j]) + rest
    if k >= len(gold) or k >= len(mine):
        return "PASS", f"identical ({k} tokens)" if len(gold) == len(mine) else f"identical for {k} tokens (one stopped)"
    top = {x["id"]: x["logprob"] for x in cp[k]["top_logprobs"]} if k < len(cp) else {}
    nt = near_tie(top, gold[k], mine[k])
    gap = top.get(gold[k], 0) - top.get(mine[k], float("-inf"))
    return ("PASS" if nt else "FAIL"), (f"diverged at token {k}: llama.cpp {pkg.tok.decode([gold[k]])!r} vs Loom "
                                        f"{pkg.tok.decode([mine[k]])!r} (gap {gap:.2f}{', near-tie' if nt else ''})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("package")
    ap.add_argument("--url", default="http://127.0.0.1:8160")
    ap.add_argument("--llama", default="http://127.0.0.1:8130")
    ap.add_argument("--images", default="")
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument("--cases", default="", help="comma-separated built-in image names to run (default: all)")
    a = ap.parse_args()
    from bench_common import Package
    pkg = Package(a.package)
    items = cases(a.images); refs = {}
    if a.cases:
        want = a.cases.split(",")
        unknown = [c for c in want if c not in items]
        if unknown:
            raise SystemExit(f"unknown --cases {unknown}; have {sorted(items)}")
        items = {k: items[k] for k in want}
    for name, (durl, q, size) in items.items():   # all references first: with LLAMA_DEFER they are queued
        refs[name] = reference(pkg, a.llama, durl, q, a.tokens)
    if any(r.get("_pending") for r in refs.values()):
        print("VISION_RESULT " + json.dumps({"verdict": "PENDING", "pending": sum(bool(r.get("_pending")) for r in refs.values())}))
        return 0
    model = requests.get(a.url + "/v1/models", timeout=30).json()["data"][0]["id"]
    bad = []; rows = []
    for name, (durl, q, size) in items.items():
        got = request(a.url, model, durl, q, a.tokens)
        st, detail = judge(pkg, refs[name], got)
        if st == "FAIL" and not got.get("error"):
            alt = reference(pkg, a.llama, durl, q, a.tokens, salt="2")
            if not alt.get("_pending"):
                st2, d2 = judge(pkg, alt, got)
                if st2 == "PASS":
                    st, detail = "PASS", d2 + " (matches llama.cpp's re-recording)"
        print(f"{st} {name} {size}: {detail}", flush=True)
        rows.append({"case": name, "status": st, "detail": detail, "loom": got.get("reasoning", "") + got.get("content", ""),
                     "llama": refs[name].get("content", "")})
        if st == "FAIL":
            bad.append(name)
    print("VISION_RESULT " + json.dumps({"verdict": "FAIL" if bad else "PASS", "cases": len(rows), "failed": bad}))
    json.dump(rows, open("vision_golden.json", "w"), indent=1)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
