# Public release entry points

Both Hugging Face organization profiles still install SDK 0.2.0 and print the old fetch and serve flags. The reviewed READMEs use the verified Qwen3 0.6B Quick start for SDK 0.3.0. The production profile links to the canonical catalog and documents the outgoing `v0.2.0` tag. The staging profile links to the staging catalog view and directs readers to production for historical releases. [Production profile](https://huggingface.co/loom-fun) · [Staging profile](https://huggingface.co/loom-fun-staging) · [Staged reference card](https://huggingface.co/loom-fun-staging/unsloth-Qwen3-0.6B-loom/blob/483f4e580c6e3a3ddd35942e3c0f54faf5825de0/README.md)

The catalog homepage has one hard-coded SDK 0.2.0 install line. The reviewed `docs/index.html` changes that pin to 0.3.0. The website already defaults to production, offers a staging opt-in, and derives model-specific Quick starts from the source cards. The existing summary workflow refreshes both catalogs hourly and accepts on-demand dispatch. [Live catalog](https://loom-fun.dev/) · [Website source](https://github.com/loom-fun/loom-dev/blob/78559c0320df1d7be8365a71a03347ad22f58bda/docs/index.html) · [Quick start extraction](https://github.com/loom-fun/loom-dev/blob/78559c0320df1d7be8365a71a03347ad22f58bda/docs/quickstart-source.md) · [Summary refresh](https://github.com/loom-fun/loom-dev/actions/workflows/summaries.yml)

Include `publish-entry-points.py`, `manifest.json` and the `reviewed/` directory in the release bundle. With the bundle's Python environment and the release operator's Hugging Face and GitHub authentication, run the following from this directory. Apply after the outgoing tags and production model receipts are complete.

```sh
python publish-entry-points.py
python publish-entry-points.py --apply
gh workflow run summaries.yml --repo loom-fun/loom-dev --ref main
```

The first command checks current content and prints the three expected parents. The apply command uses those parents as compare-and-swap guards and verifies the returned commit bytes and live file digests. Preserve the printed commit receipts. Rerun the helper after any interruption. It skips already verified content and stops on an unreviewed file change. Allow the summary workflow and GitHub Pages deployment to complete, then verify the homepage install line and follow the production and staging catalog links.

Read-only preflight matched all three reviewed source preimages. Publishing remains an owner release action. [Release issue](https://github.com/loom-fun/loom-skills/issues/1391)
