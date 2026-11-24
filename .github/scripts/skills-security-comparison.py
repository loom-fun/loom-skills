"""Verify the unchanged public UI and exact installed files for both revisions."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys


directory = Path(sys.argv[1])
candidate = sys.argv[2]
baseline = (directory / "current/repository-before.txt").read_text().split()[0]
subprocess.run(["git", "fetch", "--quiet", "--no-tags", "origin", baseline, candidate], check=True)


def verify(name, commit):
    case = directory / name
    report = json.loads((case / "installed-files.json").read_text())
    transcript = (case / "public-install.txt").read_text()
    for item in report["files"]:
        relative = Path(item["path"]).relative_to(".agents")
        data = subprocess.check_output(["git", "show", f"{commit}:{relative.as_posix()}"])
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise RuntimeError(f"installed file differs from {commit}: {relative}")
    return report, tuple(prompt for prompt in (
        "Which agents do you want to install to?",
        "Installation scope",
        "Installation method",
        "Security Risk Assessments",
        "Proceed with installation?",
    ) if prompt in transcript)


before, original_prompts = verify("current", baseline)
if len(sys.argv) > 3 and sys.argv[3] == "postmerge":
    assert "Security Risk Assessments" in original_prompts
    subprocess.run(["git", "merge-base", "--is-ancestor", candidate, baseline], check=True)
    resolved = {"source": baseline, "includes_merge": candidate, "skills": before["skills"],
                "installed_files": len(before["files"]), "prompts": original_prompts}
    (directory / "postmerge-verification.json").write_text(json.dumps(resolved, indent=2) + "\n")
    sys.exit(0)
after, updated_prompts = verify("candidate", candidate)
assert before["skills"] == after["skills"], "the available skill choices differ"
assert original_prompts == updated_prompts and "Security Risk Assessments" in updated_prompts
summary = {
    "before": baseline,
    "after": candidate,
    "skills": after["skills"],
    "matching_prompts": updated_prompts,
    "before_files": len(before["files"]),
    "after_files": len(after["files"]),
}
(directory / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
