"""lint_card.py runs check (DECISIONS D6 a): a card cites the job; a successful cited job stands even when its batch run holds other
models' failed jobs, a failed cited job or a run-only citation of a failed run is a problem."""
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "skills", "loom-benchmark", "scripts", "card"))
import lint_card  # noqa: E402


class RunsCheck(unittest.TestCase):
    def test_lint_accepts_a_successful_job_in_a_failed_batch_run(self):
        runs = lint_card.Runs()
        runs.run = lambda repo, rid: "failure"   # another model's job failed in the same batch run
        runs.job = lambda repo, jid: "success" if jid == "2" else "failure"
        cite = lambda jid: [("info.json source s1", "loom-fun/loom-skills", "1", jid, None)]
        with mock.patch.object(lint_card, "cited_runs", lambda card, info: cite("2")):
            self.assertEqual(lint_card.run_problems("c", {}, runs), [])
        with mock.patch.object(lint_card, "cited_runs", lambda card, info: cite("3")):
            self.assertEqual(len(lint_card.run_problems("c", {}, runs)), 1)
        with mock.patch.object(lint_card, "cited_runs", lambda card, info: cite(None)):   # a run-only citation needs a successful run
            self.assertEqual(len(lint_card.run_problems("c", {}, runs)), 1)


if __name__ == "__main__":
    unittest.main()
