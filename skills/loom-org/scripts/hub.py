"""Minimal Hugging Face Hub reads for the loom-org scripts: the user's token when present, retry on rate limits."""
import json
import os
import sys
import time
import urllib.error
import urllib.request

HF = "https://huggingface.co"


def _token():
    t = os.environ.get("HF_TOKEN")
    if t:
        return t.strip()
    p = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "token")
    return open(p).read().strip() if os.path.exists(p) else None


TOKEN = _token()


def fetch(url, tries=5):
    """Bytes at url, or None for a 404/401 (missing file or repo). Retries 429 and transient errors."""
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"} if TOKEN else {})
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                return None
            if e.code == 429 or e.code >= 500:
                wait = int(e.headers.get("Retry-After") or 0) or 2 ** (i + 1)
                print(f"hub: HTTP {e.code} for {url}, retrying in {wait}s", file=sys.stderr)
                time.sleep(min(wait, 120))
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"hub: {e} for {url}, retrying", file=sys.stderr)
            time.sleep(2 ** (i + 1))
    raise SystemExit(f"hub: giving up on {url}" + ("" if TOKEN else " (no HF token: set HF_TOKEN or `hf auth login`)"))


def get_json(url):
    b = fetch(url)
    return json.loads(b) if b is not None else None
