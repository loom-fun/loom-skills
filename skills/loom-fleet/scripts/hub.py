"""Minimal Hugging Face Hub reads for the loom-fleet scripts: the user's token when present, retry on rate limits."""
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HF = "https://huggingface.co"


def _token():
    t = os.environ.get("HF_TOKEN")
    if t:
        return t.strip()
    p = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "token")
    return open(p).read().strip() if os.path.exists(p) else None


TOKEN = _token()


def is_hub_url(url):
    parts = urllib.parse.urlsplit(url)
    return parts.scheme == "https" and parts.netloc == "huggingface.co"


class HubRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        authorization = req.get_header("Authorization")
        if redirected is not None and authorization and is_hub_url(redirected.full_url):
            redirected.add_unredirected_header("Authorization", authorization)
        return redirected


OPENER = urllib.request.build_opener(HubRedirects())


def fetch(url, tries=5):
    """Bytes at url, or None for a 404/401 (missing file or repo). Retries 429 and transient errors."""
    if not is_hub_url(url):
        raise ValueError("only HTTPS Hugging Face Hub URLs are supported")
    req = urllib.request.Request(url)
    if TOKEN:
        req.add_unredirected_header("Authorization", f"Bearer {TOKEN}")
    for i in range(tries):
        try:
            with OPENER.open(req, timeout=60) as r:
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
