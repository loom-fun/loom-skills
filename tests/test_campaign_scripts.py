"""The campaign scaffolding of skills/loom-fleet: new_campaign.sh, status.py, waitmail.sh, templates, doc links."""
import json
import os
import re
import runpy
import subprocess
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMPROVE = ROOT / "skills" / "loom-fleet"
CAMP = IMPROVE / "scripts" / "campaign"


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


@unittest.skipIf(sys.platform == "win32", "shell tests require a Bash environment with python3")
class Scaffold(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / "camp"

    def tearDown(self):
        self.tmp.cleanup()

    def new(self):
        r = run(["bash", str(CAMP / "new_campaign.sh"), str(self.dir), "demo", "demo", "main"])
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_new_campaign_skeleton(self):
        self.new()
        for f in ("BRIEF.md", "DECISIONS.md", "TECHLEAD.md", "status.json", "log.md", "waitmail.sh", "status.py",
                  "notes/ENG_CONTRACT.md", "notes/task_TEMPLATE.md", "inbox/techlead.md", "inbox/pm.md", "inbox/dashboard.md"):
            self.assertTrue((self.dir / f).exists(), f)
        self.assertIn("demo-<name>", (self.dir / "notes/ENG_CONTRACT.md").read_text())
        self.assertNotIn("@CAMPAIGN@", (self.dir / "BRIEF.md").read_text())
        self.assertNotIn("@DIR@", (self.dir / "notes/ENG_CONTRACT.md").read_text())
        self.assertIn("campaign demo created", (self.dir / "log.md").read_text())
        self.assertEqual(run(["bash", str(CAMP / "new_campaign.sh"), str(self.dir), "demo"]).returncode, 1)

    def test_waitmail_returns_new_lines(self):
        self.new()
        env = dict(os.environ, WAITMAIL_POLL_S="0.1")
        p = subprocess.Popen(["bash", str(CAMP / "waitmail.sh"), "techlead", str(self.dir)], stdout=subprocess.PIPE, text=True, env=env)
        time.sleep(1.0)  # the waiter records the current length first
        with open(self.dir / "inbox/techlead.md", "a") as fh:
            fh.write("2026-01-01 00:00 eng-x: PR ready\n")
        out, _ = p.communicate(timeout=20)
        self.assertEqual(out.strip(), "2026-01-01 00:00 eng-x: PR ready")


class Status(unittest.TestCase):
    def setUp(self):
        self.dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.status_path = self.dir / "status.json"
        self.log_path = self.dir / "log.md"
        self.status_path.write_text('{"workstreams": {}}', encoding="utf-8")
        # Campaigns created with the original helper can have empty lock files.
        for path in (self.status_path, self.log_path):
            path.with_suffix(path.suffix + ".lock").touch()

    def status(self, *args):
        result = run([sys.executable, str(CAMP / "status.py"), "--dir", str(self.dir), *args], timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_status_set_get_log(self):
        self.status("set", "workstreams.a.stage", "review", "workstreams.a.pr", "7")
        doc = json.loads(self.status_path.read_text(encoding="utf-8"))
        self.assertEqual(doc["workstreams"]["a"], {"stage": "review", "pr": 7})
        self.assertEqual(json.loads(self.status("get", "workstreams.a.pr")), 7)
        self.status("log", "eng-a", "did", "it")
        self.assertRegex(self.log_path.read_text(encoding="utf-8").strip(), r"^\d{4}-\d\d-\d\d \d\d:\d\d eng-a: did it$")
        st = [sys.executable, str(CAMP / "status.py")]
        result = run(st + ["get"], env=dict(os.environ, CAMPAIGN_DIR=str(self.dir)), timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), doc)
        env = dict(os.environ)
        env.pop("CAMPAIGN_DIR", None)
        self.assertEqual(run(st + ["set", "only_a_path"], cwd=self.dir, env=env, timeout=30).returncode, 2)

    def test_status_and_log_concurrent_writers(self):
        with ThreadPoolExecutor(max_workers=16) as pool:
            writes = []
            for i in range(16):
                writes.append(pool.submit(self.status, "set", f"workstreams.w{i}.n", str(i)))
                writes.append(pool.submit(self.status, "log", f"eng-{i}", "wrote", str(i)))
            for write in writes:
                write.result()
        self.assertEqual(json.loads(self.status("get", "workstreams")), {f"w{i}": {"n": i} for i in range(16)})
        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        self.assertCountEqual([line[17:] for line in lines], [f"eng-{i}: wrote {i}" for i in range(16)])

    def test_contended_lock_releases_after_exception(self):
        locked = runpy.run_path(str(CAMP / "status.py"))["locked"]
        with ThreadPoolExecutor(max_workers=1) as pool:
            with self.assertRaisesRegex(RuntimeError, "forced exception"):
                with locked(str(self.status_path)):
                    update = pool.submit(self.status, "set", "workstreams.a.pr", "7")
                    with self.assertRaises(TimeoutError):
                        update.result(timeout=0.5)
                    raise RuntimeError("forced exception")
            update.result()
        self.assertEqual(json.loads(self.status("get", "workstreams.a.pr")), 7)


class Docs(unittest.TestCase):
    def test_relative_links_resolve(self):
        files = list((IMPROVE / "reference").glob("*.md")) + [IMPROVE / "SKILL.md"]
        for f in files:
            for target in re.findall(r"\]\(([^)#\s]+)\)", f.read_text()):
                if "://" in target:
                    continue
                self.assertTrue((f.parent / target).exists(), f"{f.name}: {target}")

    def test_referenced_campaign_files_exist(self):
        text = "".join(p.read_text() for p in (IMPROVE / "reference").glob("*.md"))
        for m in set(re.findall(r"(?:scripts/campaign|templates)/[\w.]+\.\w+", text)):
            self.assertTrue((IMPROVE / m).exists(), m)



class CommitTrailerPolicy(unittest.TestCase):
    """One commit-trailer rule everywhere: exactly the plain Claude line, no model names (two contradicting rules once got every commit denied)."""
    TRAILER = "Co-Authored-By: Claude <noreply@anthropic.com>"

    def test_policy_files_carry_the_exact_trailer(self):
        for rel in ("skills/loom-fleet/templates/ENG_CONTRACT.md", "skills/loom-fleet/reference/campaign.md"):
            text = (ROOT / rel).read_text()
            self.assertIn(self.TRAILER, text, rel)
            self.assertNotRegex(text, r"own `?Co-Authored-By|own credit line", rel)

    def test_tools_write_no_model_name_in_commit_trailers(self):
        files = [p for p in list((ROOT / "fleet").rglob("*.py")) + list((ROOT / ".github").rglob("*")) + list((ROOT / "skills").rglob("*"))
                 if p.is_file() and p.suffix in (".py", ".sh", ".md", ".yml", ".yaml", ".json", "")]
        for p in files:
            if p.is_file():
                for m in re.finditer(r"Co-Authored-By: [^\n\"'`]*", p.read_text(errors="ignore"), re.I):
                    self.assertEqual(m.group(0).strip(), self.TRAILER, f"{p.relative_to(ROOT)}: {m.group(0)}")

if __name__ == "__main__":
    unittest.main()
