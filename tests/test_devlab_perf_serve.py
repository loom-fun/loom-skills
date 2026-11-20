"""devlab_perf.sh serve_alone: every GPU is served as its card's Quick start serves it (device.json card.serve_args).

Strix Halo and the R9700 take --pool_backing=fixed with no --memory_bytes; a variant with its own serve_args --pool_backing, a memory_bytes
or "elastic": true keeps --memory_bytes (the elastic backing). serve_alone is run with the real function text and a stub serve.sh that
records its arguments (no GPU)."""
import json, os, subprocess, tempfile, unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SCRIPT = os.path.join(ROOT, ".github", "scripts", "devlab_perf.sh")


def _fn(name):
    lines, on = [], False
    for ln in open(SCRIPT):
        if ln.startswith(name + "()"):
            on = True
        if on:
            lines.append(ln)
            if ln.startswith("}"):
                break
    return "".join(lines)


def _line(prefix):
    return next(ln for ln in open(SCRIPT) if ln.startswith(prefix)).rstrip("\n")


def serve_line(variant, rdev="", rsuf="-", mem="17000000000", weights=None):
    """The serve.sh arguments serve_alone passes for a variant of a one-preset package on the runner of `rdev` ("" = Strix Halo)."""
    with tempfile.TemporaryDirectory() as w:
        os.makedirs(f"{w}/D/v/pkg"); os.makedirs(f"{w}/state")
        model = {"presets": {"chat": {}}}
        if weights is not None:   # one parameter file of this many bytes, sized by the package's lock (what the card's Quick start decides on)
            model["variants"] = {"v": {"services": {"s": {"parameters": {"model": {"files": [{"origin": "@m/x.gguf"}]}}}}}}
            json.dump({"files": [{"origin": "@m/x.gguf", "size": weights}]}, open(f"{w}/D/v/pkg/model.json.lock", "w"))
        json.dump(model, open(f"{w}/D/v/pkg/model.json", "w"))
        open(f"{w}/sv_serve.sh", "w").write('#!/bin/bash\necho "serve.sh $*" > "$REC"\n')
        os.chmod(f"{w}/sv_serve.sh", 0o755)
        json.dump({"jobs": [{"name": "j", "variants": [variant]}]}, open(f"{w}/exp.json", "w"))
        script = f'''
T=python3; here={ROOT}; D={w}/D; out={w}; REC={w}/rec; LOOM_SERVE_STATE={w}/state; B=/nonexistent; LLAMA_PORT=1; PORT=1; url=http://127.0.0.1:1
RDEV={rdev!r}; RSUF={rsuf}; RUNNER=x; mem={mem}
export REC
mkdir -p {w}/SV; cp {w}/sv_serve.sh {w}/SV/serve.sh; printf '#!/bin/bash\\n' > {w}/SV/stop.sh; chmod +x {w}/SV/stop.sh; SV={w}/SV
exp={w}/exp.json; job=j
{_line("qv()")}
sdk_root() {{ SDKR=/sdk; }}
step() {{ :; }}
gufo_down() {{ :; }}; mem_guard() {{ :; }}
echo '{{}}' > {w}/serve_v.out
{_fn("serve_alone")}
serve_alone v >/dev/null 2>&1; true
cat {w}/rec
'''
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True).stdout.strip()


@unittest.skipIf(os.name == "nt", "devlab_perf.sh runs on the Linux DevLab runners only")
class ServeAlone(unittest.TestCase):
    def test_strix_serves_fixed(self):
        got = serve_line({"name": "v"})
        self.assertIn(" none ", got)                          # MEMORY_BYTES=none: serve.sh passes no --memory_bytes
        self.assertTrue(got.endswith("-- --pool_backing=fixed"), got)   # weights unknown: no preload

    def test_r9700_serves_fixed_and_preloaded(self):
        got = serve_line({"name": "v"}, rdev="radeon-r9700", rsuf="-r9700", mem="34000000000")
        self.assertIn(" none ", got)
        self.assertTrue(got.endswith("-- --pool_backing=fixed --parameter_mode=preload"), got)   # no weight limit on the R9700

    def test_strix_preloads_up_to_40_gb_of_weights(self):
        self.assertTrue(serve_line({"name": "v"}, weights=40_000_000_000).endswith("-- --pool_backing=fixed --parameter_mode=preload"))
        self.assertTrue(serve_line({"name": "v"}, weights=40_000_000_001).endswith("-- --pool_backing=fixed"))

    def test_elastic_variant_is_not_preloaded(self):
        self.assertNotIn("preload", serve_line({"name": "v", "elastic": True}, rdev="radeon-r9700", rsuf="-r9700"))

    def test_elastic_variant_keeps_memory_bytes(self):
        got = serve_line({"name": "v", "elastic": True})
        self.assertIn(" 17000000000 ", got)
        self.assertNotIn("pool_backing", got)

    def test_variant_memory_bytes_keeps_it(self):
        got = serve_line({"name": "v", "memory_bytes": 6000000000})
        self.assertIn(" 6000000000 ", got)
        self.assertNotIn("pool_backing", got)

    def test_variant_serve_args_pool_backing_wins(self):
        got = serve_line({"name": "v", "serve_args": ["--pool_backing=elastic"]}, mem="9000000000")
        self.assertIn(" 9000000000 ", got)
        self.assertTrue(got.endswith("-- --pool_backing=elastic"), got)


if __name__ == "__main__":
    unittest.main()
