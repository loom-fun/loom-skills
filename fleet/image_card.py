#!/usr/bin/env python3
"""The SDK migration card of an image package (campaign decision D6 c: image cards have no info.json generator): its production card at the
inventory revision with only the migration's mechanical changes and the SDK-only delta line generated from the regen image job (image_sdk_compare.py
image_result.json: identical pixels and seconds per image, the same programs on both SDKs in one job).

Changes, and nothing else: the pin of the `pipx install` line; the `loom-fetch` line fetches from ORG without `--materialize=copy` (gone in 0.3.0,
skills/loom-migrate/versions/v0.2.0-to-v0.3.0.md 2.5) and the sentence that explained it; "supported by hrx-loom OLD" -> NEW; the Loom row of every timing
table marked as measured on OLD in the cross-engine session; one "Change from the previous card" block before "## Files"; the Versions
line names the SDK, the SDK the values were measured on and this script's commit.

usage: image_card.py SLUG JOB_DIR JOB_URL OUT_DIR [--org loom-fun-staging] [--publish]
  JOB_DIR: the downloaded image job (image_result.json); OUT_DIR: written fresh (the production card's small files + the migrated README.md).
  --publish: one commit to ORG/SLUG on top of its head: these files, and the files there the card format does not define deleted
  (card_files.card_file, e.g. a qualification record)."""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "skills", "loom-benchmark", "scripts", "card"))
import card_files  # noqa: E402

STAGING_ORG = "loom-fun-staging"


def not_card_format(paths):
    """The repository paths the card format does not define (card_files.card_file): the only files --publish deletes."""
    return sorted(p for p in paths if not card_files.card_file(p))


def skills_commit():
    r = subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else "unknown"


LOOM_ROW = "| Loom (this package) |"


def mark_loom_rows(text, old, new):
    """The Loom row of every timing table (a table whose header has s/image): marked as measured on the old SDK in the cross-engine session, so no
    old-SDK value renders unmarked under the new pin (the new SDK's re-measurement is the delta block's)."""
    out, header = [], None
    for line in text.split("\n"):
        if line.startswith("|"):
            header = header or line
            if "s/image" in header and line.startswith(LOOM_ROW):
                line = (f"| Loom (this package; hrx-loom {old}, Windows cross-engine session; SDK {new} re-measured on Linux below) |"
                        + line[len(LOOM_ROW):])
        else:
            header = None
        out.append(line)
    return "\n".join(out)


def migrate(text, slug, result, url, org, commit):
    """The migrated README text of one image card (refuses a job that did not show identical pixels on the same programs)."""
    old, new = result["sdk"]["old"], result["sdk"]["new"]
    if not (result.get("pixels_match") and result.get("programs_match")):
        raise ValueError(f"{slug}: the image job did not show identical pixels on the same programs")
    rev = result["revision"]
    t = text.replace(f'pipx install "hrx-loom[hf]=={old}"', f'pipx install "hrx-loom[hf]=={new}"')
    t = re.sub(r"loom-fetch huggingface:loom-fun/(\S+) --materialize=copy", rf"loom-fetch huggingface:{org}/\1", t)
    t = t.replace("- `--materialize=copy` puts copies of the fetched package files in the output directory. ", "- ")
    t = t.replace(f"supported by hrx-loom {old} can fetch", f"supported by hrx-loom {new} can fetch")
    t = mark_loom_rows(t, old, new)
    o, n = result["runs"]["old"], result["runs"]["new"]
    images = sum(len(s["seconds"]) for s in n["sessions"])
    block = (f"**Change from the previous card** ([revision `{rev[:7]}`](https://huggingface.co/loom-fun/{slug}/tree/{rev}), hrx-loom {old}; "
             f"this card: hrx-loom {new}):\n\n"
             f"- Strix Halo: identical pixels (RGB sha256 `{n['sessions'][0]['rgb_sha256'][0][:7]}` on both, {images} images each), "
             f"s/image {o['median_s']:g} → {n['median_s']:g} ({(n['median_s'] / o['median_s'] - 1) * 100:+.1f} %; lower is faster): "
             f"the same programs on both SDKs in one [job]({url}).\n\n")
    if t.count("\n## Files\n") != 1:
        raise ValueError(f"{slug}: README has no single '## Files' section")
    t = t.replace("\n## Files\n", "\n" + block + "## Files\n")
    t, k = re.subn(r"(\nVersions: card schema `model-card\.v1`; )hrx-loom `" + re.escape(old) + "`;",
                   rf"\1hrx-loom `{new}` (values above measured on `{old}` unless marked; card migrated by loom-skills `fleet/image_card.py` `{commit[:7]}`);", t)
    if k != 1 or "--materialize=copy" in t:
        raise ValueError(f"{slug}: README is not in the expected image-card form (Versions line / fetch line)")
    return t


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slug")
    ap.add_argument("job_dir")
    ap.add_argument("job_url")
    ap.add_argument("out")
    ap.add_argument("--org", default=STAGING_ORG, choices=(STAGING_ORG,), help="the staging org (the only one this script writes)")
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args(argv)
    from huggingface_hub import HfApi, snapshot_download, CommitOperationAdd, CommitOperationDelete
    result = json.load(open(os.path.join(a.job_dir, "image_result.json")))
    shutil.rmtree(a.out, ignore_errors=True)
    snapshot_download(f"loom-fun/{a.slug}", revision=result["revision"], local_dir=a.out, allow_patterns=["*.md", "*.json", "*.lock"])
    shutil.rmtree(os.path.join(a.out, ".cache"), ignore_errors=True)
    p = os.path.join(a.out, "README.md")
    text = migrate(open(p, encoding="utf-8").read(), a.slug, result, a.job_url, a.org, skills_commit())
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"{a.slug}: README migrated in {a.out}")
    if a.publish:
        api, repo = HfApi(), f"{a.org}/{a.slug}"
        info = api.model_info(repo, files_metadata=True)
        local = sorted(f for f in os.listdir(a.out) if os.path.isfile(os.path.join(a.out, f)) and not f.startswith("."))
        gone = not_card_format(s.rfilename for s in info.siblings if s.rfilename not in local)
        ops = [CommitOperationAdd(path_in_repo=f, path_or_fileobj=os.path.join(a.out, f)) for f in local] + [CommitOperationDelete(path_in_repo=f) for f in gone]
        c = api.create_commit(repo, ops, parent_commit=info.sha,
                              commit_message=f"Card on hrx-loom {result['sdk']['new']}: production card + migration, SDK-only delta from the image job")
        print(f"{repo}: published {c.oid} (deleted: {', '.join(gone) or 'nothing'})")


if __name__ == "__main__":
    main()
