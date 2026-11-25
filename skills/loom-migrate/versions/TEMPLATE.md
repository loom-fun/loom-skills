# vX -> vY: SDK migration entry (copy to `vX-to-vY.md`, keep these section titles and order)

Pair, release page, release notes, the tags and commits compared, library tag, wheel sha256, date written, who verified what.

## 1. Summary
One table WITH a severity column (HARD parse / HARD load / HARD deployment / INSTALL / SILENT / N/A), the change, the fleet impact (how many
packages, which). The measured numbers (sweep: parse failures, link failures) from the control run, with the run's link; a run that failed or was
superseded is never cited as proof.

## 2. Breaking changes
One subsection per change, same fields every time:
- **Class** (source IR, program ABI, card schema, install and pin, device runtime, request semantics) and severity
- **Before / after**: the exact text or signature, upstream PR
- **Touches**: files in loom-skills (path:line), count in the fleet
- **Step**: the mechanical step and the script that does it (or "hand edit" with the fixture to copy)
- **Verify**: the check that proves it (sweep class gone, link, GPU smoke, test)

## 3. Mechanical steps
The ordered list to apply, each a command. A user migrating their own package follows only this and section 5.

## 4. Scripts
Which scripts apply to this pair, what they do not do (the manual cases).

## 5. Verification
The fleet proof (sweep, explained diff), the GPU smoke set, the A/B metrics, the expected program diff classes and counts.

### 5b. Per-model per-GPU SDK delta report
Procedure step 7. Generated from the sweep's evidence, never typed: the generation command (`fleet/regen.py evidence ...` then
`fleet/sdk_delta_report.py evidence.json --strict`), the pass rule (equal or better on a supermajority of rows, every regression over 3 %
explained), and the generated table between `SDK-DELTA-REPORT` markers. Leave the markers and a placeholder row until the final sweep.

## 6. Behaviour changes to measure
Silent changes (semantics, defaults, scheduling, completion policy): one row each with the number it is expected to move and what was
measured (job or run link). A change without an expected number is not ready.

## 7. Simplifications available
The simplification ledger, one row per item: landed / listed (with the reason it did not land) / measuring, plus the evidence. What the new SDK
lets us delete or shorten (a flag, a computation, a workaround); an item that depends on behaviour is measured before it is claimed. Items kept
on purpose (an explicit flag the A/B says is still needed) are rows too, with their numbers.

## 8. Opportunities
Optimizations the release opens, each with its evidence and the filed issue link (an item with no measured or source-proven gain is listed
as "not filed" with the reason, never as a vague "can be evaluated"). Optimization lanes that run in the campaign link their own issues.

## 9. Gotchas
`symptom -> cause -> fix`, one per line, in the order found.

## 10. Not applicable
Changes in the release notes that do not touch this repo, with the reason (so the next reader need not re-check).

## 11. Promotion
Point to the reviewed bundle procedure (`reference/prepared-promotion.md`: freeze `eligible.json` and `held.json`, `replay_promotion.py prepare`, the owner approves one manifest
SHA256) and restate the order of `reference/release-checklist.md` section 5, which is the only order: 1. outgoing tags `scripts/tag_org.py --bundle DIR --manifest-sha256 SHA`
(every repo at its reviewed parent), 2. merge the integration branch and tag the merge commit `sdk<to>-u0` (`loom-fleet` publish.md "Release tags"), 3. `replay_promotion.py replay
--execute --receipt FILE`, 4. production org page PR, 5. website PR, 6. extension / weaver / Lemonade / Gufo, 7. cleanup. Nothing tags or merges before step 1. The card-tag rule:
`loom-fleet` reference/publish.md "Card tags". Release-specific facts (held repos, exceptions, stock-host findings) go here.

## 12. Campaign evidence
Links: the campaign PRs, DevLab run ids (job links for a batch run: cite the job whose conclusion is success), the published staging cards, the tracking issue, the final report.
Fill what is final; mark the rest with an explicit placeholder owner (for example TECHLEAD) so nothing reads as evidence before it exists.
