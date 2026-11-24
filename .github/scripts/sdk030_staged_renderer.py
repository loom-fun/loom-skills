"""Render the seven pinned packages and preserve the qualification evidence."""
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, "skills/loom-benchmark/scripts/card")
import render_card

work = Path(sys.argv[1])
output = work / "evidence"
output.mkdir()
targets = json.loads(Path(".github/devlab/sdk030-renderer-heads.json").read_text())
assert len(targets) == 7
for target in targets:
    repo, revision = target["staging_repo"], target["commit"]
    slug = repo.split("/")[-1]
    directory = work / slug
    snapshot_download(repo, revision=revision, local_dir=directory, token=False,
                      allow_patterns=["*.json", "*.lock", "README.md", "REPORT.md", "program/*.loom"])
    readme = render_card.render(str(directory))
    report = render_card.render_report(str(directory))
    evidence = output / slug
    evidence.mkdir()
    (evidence / "README.md").write_text(readme)
    (evidence / "REPORT.md").write_text(report)
    info = json.loads((directory / "info.json").read_text())
    qualification = json.loads((directory / "sdk-qualification.json").read_text())
    digest = render_card.programs_sha256(directory)
    assert qualification["programs_sha256"] == digest, slug
    comparison = info.get("promotion_hold") or qualification["performance_disposition"]["previous_hold"]
    assert "## SDK evidence" not in readme and "## Production promotion hold" not in readme
    assert "[SDK qualification](sdk-qualification.json)" in readme
    assert comparison["comparison_url"] in readme and comparison["comparison_url"] in report
    for metric in comparison["rows"].values():
        delta = f"{metric['percent']:+.2f}%"
        assert delta in readme and delta in report, (slug, delta)
    for device in info["devices"].values():
        for session in (device.get("performance") or {}).get("sessions", []):
            pin = info["sources"][session["source"]].get("hrx_loom")
            if isinstance(pin, dict) and pin.get("hrx_loom"):
                assert pin["hrx_loom"] in report, (slug, pin)
    if "performance_disposition" in qualification:
        assert not render_card.sdk_performance_comparison(info, str(directory), "stale", summary=True)
    print(repo, revision, digest, "PASS", flush=True)
