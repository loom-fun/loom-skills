#!/usr/bin/env python3
"""GGUF header dumps and header-only stub files (stdlib only).

The programs a build generates depend on a GGUF's metadata, tensor infos and file size, and on the data of a few tiny tensors
(rope_freqs.weight), never on the weights. A dump keeps exactly that:

    {"version", "alignment", "size", "header_sha256", "kv": [{"key", "type", "raw" | "blob"}],
     "tensors": [[name, dims, type, offset]], "data": {name: {"offset", "b64"}}}

`raw` is the KV value's bytes (base64); values over BLOB_MIN bytes (vocab arrays, merges, chat templates) go to the shared
content-addressed store fleet/blobs/<sha>.xz so the models of one vocabulary share one copy.  write_stub() rebuilds a sparse
file whose first data_start bytes are the original header byte for byte (checked against header_sha256), zeros elsewhere,
the original size, and the saved tensor data.

usage: gguf_stub.py dump SOURCE OUT.json.gz [--blobs DIR]     SOURCE: file path, or URL (HTTP range reads)
       gguf_stub.py stub DUMP.json.gz OUT.gguf [--blobs DIR]
"""
import argparse
import base64
import gzip
import hashlib
import json
import lzma
import os
import struct
import threading
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BLOBS = os.path.join(HERE, "blobs")
BLOB_MIN = 4096
SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
STRING, ARRAY = 8, 9
KEEP_DATA = ("rope_freqs.weight",)   # tensors whose data the generators read (inspect_gguf.py, draft/gen_assist.py)
F32 = 0   # the only type a KEEP_DATA tensor has today


class Reader:
    """Sequential reader over a growing prefix: `fetch(n)` returns the first n bytes (or fewer at EOF)."""

    def __init__(self, fetch, first=1 << 23):
        self.fetch, self.n, self.pos = fetch, first, 0
        self.buf = fetch(first)

    def take(self, k):
        while self.pos + k > len(self.buf):
            if len(self.buf) < self.n:
                raise EOFError("truncated GGUF")
            self.n *= 2
            self.buf = self.fetch(self.n)
        b = self.buf[self.pos:self.pos + k]
        self.pos += k
        return b

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def string(self):
        return self.take(self.u64())

    def value_span(self, t):
        """Skips one value of type t; the bytes consumed are returned."""
        start = self.pos
        if t == STRING:
            self.string()
        elif t == ARRAY:
            et, n = self.u32(), self.u64()
            if et in SCALAR:
                self.take(n * struct.calcsize(SCALAR[et]))
            elif et == STRING:
                for _ in range(n):
                    self.string()
            else:
                for _ in range(n):
                    self.value_span(et)
        else:
            self.take(struct.calcsize(SCALAR[t]))
        return bytes(self.buf[start:self.pos])


def fetcher(source):
    if os.path.exists(source):
        def f(n):
            with open(source, "rb") as fh:
                return fh.read(n)
        return f, os.path.getsize(source)

    def f(n):
        req = urllib.request.Request(source, headers={"Range": f"bytes=0-{n - 1}"})
        return urllib.request.urlopen(req, timeout=600).read()
    r = urllib.request.urlopen(urllib.request.Request(source, headers={"Range": "bytes=0-0"}), timeout=120)
    return f, int(r.headers["Content-Range"].split("/")[1])


def blob_put(data, blobs):
    sha = hashlib.sha256(data).hexdigest()[:24]
    p = os.path.join(blobs, sha + ".xz")
    if not os.path.exists(p):
        os.makedirs(blobs, exist_ok=True)
        tmp = f"{p}.{os.getpid()}.{threading.get_ident()}.tmp"
        with open(tmp, "wb") as fo:
            fo.write(lzma.compress(data, preset=9 | lzma.PRESET_EXTREME))
        os.replace(tmp, p)
    return sha


def blob_get(sha, blobs):
    with open(os.path.join(blobs, sha + ".xz"), "rb") as fh:
        return lzma.decompress(fh.read())


def dump(source, blobs=BLOBS):
    fetch, size = fetcher(source)
    r = Reader(fetch)
    if r.take(4) != b"GGUF":
        raise ValueError("not a GGUF")
    version, ntensors, nkv = r.u32(), r.u64(), r.u64()
    kvs, alignment = [], 32
    for _ in range(nkv):
        key = r.string()
        t = r.u32()
        raw = r.value_span(t)
        if key == b"general.alignment":
            alignment = struct.unpack("<I", raw)[0]
        e = {"key": key.decode(), "type": t}
        if len(raw) > BLOB_MIN:
            e["blob"] = blob_put(raw, blobs)
        else:
            e["raw"] = base64.b64encode(raw).decode()
        kvs.append(e)
    tensors = []
    for _ in range(ntensors):
        name = r.string().decode()
        nd = r.u32()
        dims = [r.u64() for _ in range(nd)]
        tt, off = r.u32(), r.u64()
        tensors.append([name, dims, tt, off])
    end = r.pos
    data_start = -(-end // alignment) * alignment
    r.take(data_start - end)
    header_sha = hashlib.sha256(r.buf[:end]).hexdigest()
    data = {}
    for name, dims, tt, off in tensors:
        if name in KEEP_DATA:
            assert tt == F32, f"{name}: type {tt}"
            n = 4
            for d in dims:
                n *= d
            data[name] = {"offset": off, "b64": base64.b64encode(fetcher_range(source, data_start + off, n)).decode()}
    return {"version": version, "alignment": alignment, "size": size, "data_start": data_start, "header_sha256": header_sha,
            "kv": kvs, "tensors": tensors, "data": data}


def fetcher_range(source, lo, n):
    if os.path.exists(source):
        with open(source, "rb") as fh:
            fh.seek(lo)
            return fh.read(n)
    req = urllib.request.Request(source, headers={"Range": f"bytes={lo}-{lo + n - 1}"})
    return urllib.request.urlopen(req, timeout=600).read()


def header_bytes(d, blobs=BLOBS):
    out = [b"GGUF", struct.pack("<IQQ", d["version"], len(d["tensors"]), len(d["kv"]))]
    for e in d["kv"]:
        k = e["key"].encode()
        raw = blob_get(e["blob"], blobs) if "blob" in e else base64.b64decode(e["raw"])
        out += [struct.pack("<Q", len(k)), k, struct.pack("<I", e["type"]), raw]
    for name, dims, tt, off in d["tensors"]:
        nb = name.encode()
        out += [struct.pack("<Q", len(nb)), nb, struct.pack("<I", len(dims))] + [struct.pack("<Q", x) for x in dims]
        out += [struct.pack("<IQ", tt, off)]
    h = b"".join(out)
    if hashlib.sha256(h).hexdigest() != d["header_sha256"]:
        raise ValueError("header does not match header_sha256")
    return h


def write_stub(d, path, blobs=BLOBS):
    h = header_bytes(d, blobs)
    with open(path, "wb") as fo:
        fo.write(h)
        fo.write(b"\0" * (d["data_start"] - len(h)))
        for name, v in d["data"].items():
            fo.seek(d["data_start"] + v["offset"])
            fo.write(base64.b64decode(v["b64"]))
        fo.truncate(d["size"])
    return path


def save(d, path):
    with gzip.GzipFile(path, "wb", mtime=0) as fo:
        fo.write(json.dumps(d, separators=(",", ":")).encode())


def load(path):
    with gzip.open(path, "rb") as fh:
        return json.loads(fh.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dump", "stub"])
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--blobs", default=BLOBS)
    a = ap.parse_args()
    if a.cmd == "dump":
        save(dump(a.src, a.blobs), a.out)
    else:
        write_stub(load(a.src), a.out, a.blobs)


if __name__ == "__main__":
    main()
