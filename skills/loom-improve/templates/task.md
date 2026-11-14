# Task: <one line> (engineer "<name>")
Read first: BRIEF.md, DECISIONS.md (sections ...), notes/ENG_CONTRACT.md, <the files and PRs this task builds on>.
Branch @INTEGRATION@-<name>, PR -> @INTEGRATION@. Dependencies in flight: <engineer: files, so you code against their schema or an adapter>.

## Scope (loom-scope block to put in the PR)
kind: <targeted | horizontal | architecture | tooling | docs | card>
scopes: <selector nodes>   kernels: <families>   (files you own; files you must not touch)

## Build
1. <numbered, concrete deliverables; file names; commands; the invariant to prove (for example an empty fleet program diff)>
2. Tests: <which test files>.

## Prove
<the evidence the review needs: command + expected result; DevLab A/B representatives and rows if behaviour changes>

## Done when
<PR open + CI green + specific checks>. Hand back: PR link, files, evidence, open issues.
