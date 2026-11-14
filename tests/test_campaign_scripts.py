"""The campaign scaffolding of skills/loom-improve: new_campaign.sh, status.py, waitmail.sh, templates, doc links."""
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMPROVE = ROOT / "skills" / "loom-improve"
CAMP = IMPROVE / "scripts" / "campaign"


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


@unittest.skipIf(sys.platform == "win32", "campaign scripts are POSIX (bash, fcntl): campaigns run on the Linux dev host")
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

    def test_status_set_get_log(self):
        self.new()
        st = [sys.executable, str(CAMP / "status.py"), "--dir", str(self.dir)]
        self.assertEqual(run(st + ["set", "workstreams.a.stage", "review", "workstreams.a.pr", "7"]).returncode, 0)
        doc = json.loads((self.dir / "status.json").read_text())
        self.assertEqual(doc["workstreams"]["a"], {"stage": "review", "pr": 7})
        self.assertEqual(json.loads(run(st + ["get", "workstreams.a.pr"]).stdout), 7)
        self.assertEqual(run(st + ["log", "eng-a", "did", "it"]).returncode, 0)
        self.assertRegex((self.dir / "log.md").read_text().splitlines()[-1], r"^\d{4}-\d\d-\d\d \d\d:\d\d eng-a: did it$")
        self.assertEqual(run(st + ["set", "only_a_path"]).returncode, 2)

    def test_status_concurrent_writers(self):
        self.new()
        st = [sys.executable, str(CAMP / "status.py"), "--dir", str(self.dir)]
        ts = [threading.Thread(target=run, args=(st + ["set", f"workstreams.w{i}.n", str(i)],)) for i in range(8)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        doc = json.loads((self.dir / "status.json").read_text())
        self.assertEqual(sorted(doc["workstreams"]), sorted(f"w{i}" for i in range(8)))

    def test_waitmail_returns_new_lines(self):
        self.new()
        env = dict(os.environ, WAITMAIL_POLL_S="0.1")
        p = subprocess.Popen(["bash", str(CAMP / "waitmail.sh"), "techlead", str(self.dir)], stdout=subprocess.PIPE, text=True, env=env)
        time.sleep(1.0)  # the waiter records the current length first
        with open(self.dir / "inbox/techlead.md", "a") as fh:
            fh.write("2026-01-01 00:00 eng-x: PR ready\n")
        out, _ = p.communicate(timeout=20)
        self.assertEqual(out.strip(), "2026-01-01 00:00 eng-x: PR ready")


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


if __name__ == "__main__":
    unittest.main()
