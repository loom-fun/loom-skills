# vX -> vY: SDK migration entry (copy to `vX-to-vY.md`, keep these section titles and order)

Pair, release page, release notes, the tags and commits compared, library tag, wheel sha256, date written, who verified what.

## 1. Summary
One table: severity (HARD parse / HARD load / HARD deployment / INSTALL / SILENT / N/A), the change, the fleet impact (how many
packages, which). The measured numbers (sweep: parse failures, link failures) from the control run.

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

## 6. Behaviour changes to measure
Silent changes (semantics, defaults, scheduling, completion policy) and the number each is expected to move.

## 7. Simplifications available
What the new SDK lets us delete or shorten; landed / listed / measuring.

## 8. Opportunities
Optimizations the release opens, with the evidence and the issue link.

## 9. Gotchas
`symptom -> cause -> fix`, one per line, in the order found.

## 10. Not applicable
Changes in the release notes that do not touch this repo, with the reason (so the next reader need not re-check).

## 11. Promotion
Before the staging cards go to production: `scripts/tag_org.py --org loom-fun --tag v<from>` (the outgoing cards stay addressable); after it, tag the new state `v<to>`.

## 12. Campaign evidence
Links: the campaign PRs, DevLab run ids, the published staging cards, the final report.
