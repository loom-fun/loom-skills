"""fleet/image_card.py: the SDK migration of an image card (README text only, generated from the image job)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fleet"))
import image_card  # noqa: E402

README = """# x

```bash
pipx install "hrx-loom[hf]==0.2.0"
loom-fetch huggingface:loom-fun/acme-Img-loom:image --materialize=copy --output=img
```

- `--materialize=copy` puts copies of the fetched package files in the output directory. The weights come from acme at the pinned revision.

Other AMD GPUs supported by hrx-loom 0.2.0 can fetch and serve the same package.

| backend | s/image ↓ | Loom speedup |
|---|---|---|
| Loom (this package) | **7.1** | — |
| other | 9.0 | 1.3× |

| backend | PSNR ↑ |
|---|---|
| Loom (this package) | 25.69 |

## Files

Versions: card schema `model-card.v1`; hrx-loom `0.2.0`; programs from somewhere.
"""
RESULT = {"revision": "a" * 40, "sdk": {"old": "0.2.0", "new": "0.3.0"}, "pixels_match": True, "programs_match": True,
          "runs": {"old": {"median_s": 10.0, "sessions": [{"seconds": [10, 10], "rgb_sha256": ["b" * 64] * 2}]},
                   "new": {"median_s": 9.9, "sessions": [{"seconds": [9.9, 9.9], "rgb_sha256": ["b" * 64] * 2}] * 2}}}


class Migrate(unittest.TestCase):
    def test_mechanical_changes_and_the_delta_block(self):
        t = image_card.migrate(README, "acme-Img-loom", RESULT, "https://x/job/1", "loom-fun-staging", "c" * 40)
        self.assertIn('pipx install "hrx-loom[hf]==0.3.0"', t)
        self.assertIn("loom-fetch huggingface:loom-fun-staging/acme-Img-loom:image --output=img", t)
        self.assertNotIn("materialize", t)
        self.assertIn("- The weights come from acme at the pinned revision.", t)
        self.assertIn("supported by hrx-loom 0.3.0 can fetch", t)
        self.assertIn("| Loom (this package; hrx-loom 0.2.0, Windows cross-engine session; SDK 0.3.0 re-measured on Linux below) | **7.1** | — |", t)
        self.assertIn("| Loom (this package) | 25.69 |", t)   # not a timing table: unchanged
        self.assertIn("- Strix Halo: identical pixels (RGB sha256 `bbbbbbb` on both, 4 images each), s/image 10 → 9.9 (-1.0 %; lower is faster): "
                      "the same programs on both SDKs in one [job](https://x/job/1).\n\n## Files", t)
        self.assertIn("hrx-loom `0.3.0` (values above measured on `0.2.0` unless marked; card migrated by loom-skills `fleet/image_card.py` `ccccccc`);", t)

    def test_refuses_other_pixels_or_form(self):
        with self.assertRaises(ValueError):
            image_card.migrate(README, "s", dict(RESULT, pixels_match=False), "u", "loom-fun-staging", "c")
        with self.assertRaises(ValueError):
            image_card.migrate(README.replace("## Files", "## Other"), "s", RESULT, "u", "loom-fun-staging", "c")


class Publish(unittest.TestCase):
    def test_only_files_outside_the_card_format_are_deleted(self):
        self.assertEqual(image_card.not_card_format(["README.md", "model.json", "tokenizer.json", "program/a.loom", "library/motif/x.loom",
                                                     ".gitattributes", "sdk-qualification.json", "dev/x.json"]),
                         ["dev/x.json", "sdk-qualification.json"])

    def test_staging_org_only(self):
        for org in ("loom-fun", "someone-else"):
            with self.assertRaises(SystemExit):
                image_card.main(["s", "d", "u", "o", "--org", org])


if __name__ == "__main__":
    unittest.main()
