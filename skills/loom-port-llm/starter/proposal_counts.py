"""Choice telemetry.proposal_counts: publish_epoch reports the candidates each span presented to verification (SDK 0.3.0).

SDK 0.3.0 (hrx-system #3360) reads speculative proposals from the controller instead of deriving them from the configured
draft depth. A version-1 publish_epoch may end with `%progress: buffer` (13 arguments) or add `%proposal_counts: buffer`
(14 arguments, loom/src/loom/serve/service/text/control.c accepts both for ABI 1). With 13 arguments every verified span
reports an unknown count, and /metrics then omits draft_proposed_tokens for the life of the server: the acceptance rate
of a drafted package is unmeasurable (loom-skills #1401). With the 14th buffer, entry i (little-endian u32) is the number
of candidates span i presented: zero for ordinary work, the speculative span's planned length minus its anchor for a
verified drafted span (the same count SDK 0.2.0 derived as verifications x draft depth). Tokens, the device programs and
the version-1 service ABI are unchanged; the package needs hrx-loom >= 0.3.0 (0.2.0 binds only the 13-argument form).
"""
import re

OLD_ARGS = "%next_results: buffer, %progress: buffer)"
NEW_ARGS = "%next_results: buffer, %progress: buffer, %proposal_counts: buffer)"
OLD_CALL_ARGS = "%next_results, %progress) : (buffer, buffer, buffer, i32, i32, i32, i32, i32, buffer, buffer, buffer, buffer, buffer)"
NEW_CALL_ARGS = ("%next_results, %progress, %proposal_counts) : "
                 "(buffer, buffer, buffer, i32, i32, i32, i32, i32, buffer, buffer, buffer, buffer, buffer, buffer)")

# control.loom's ordinary publish_epoch: every span presented no candidates
PLAIN_ZERO_OLD = """  %published = buffer.view %progress[%origin] : buffer -> view<[%length]x21xi32>
  scf.for %i = [%c0 to %length step %c1] {
    %request = index.assume %i [lt(%i, %plan_extent)] : index"""
PLAIN_ZERO_NEW = """  %published = buffer.view %progress[%origin] : buffer -> view<[%length]x21xi32>
  %proposals = buffer.view %proposal_counts[%origin] : buffer -> view<[%length]xi32>
  scf.for %i = [%c0 to %length step %c1] {
    view.store %c0_i32, %proposals[%i] : i32, view<[%length]xi32>
    %request = index.assume %i [lt(%i, %plan_extent)] : index"""

# draft/build_draft.py's verified publish: a speculative span presented its planned length minus the anchor
DRAFT_OLD = """      %verification = scalar.extui %speculative : i1 to i32
"""
DRAFT_NEW = """      %verification = scalar.extui %speculative : i1 to i32
      %planned = view.load %requests[%ir, 0] : view<[%plan_extent]x9xi32> -> i32
      %candidates = scalar.subi %planned, %c1_i32 : i32
      %presented = scf.select %speculative, %candidates, %c0_i32 : i32
      view.store %presented, %proposals[%i] : i32, view<[%length]xi32>
"""
DRAFT_VIEW_OLD = """    %published = buffer.view %progress[%origin] : buffer -> view<[%length]x21xi32>
    scf.for %i = [%c0 to %length step %c1] {
      %ir = index.assume %i [lt(%i, %plan_extent)] : index"""
DRAFT_VIEW_NEW = """    %published = buffer.view %progress[%origin] : buffer -> view<[%length]x21xi32>
    %proposals = buffer.view %proposal_counts[%origin] : buffer -> view<[%length]xi32>
    scf.for %i = [%c0 to %length step %c1] {
      %ir = index.assume %i [lt(%i, %plan_extent)] : index"""


def _sub(text, old, new, what, count=1):
    n = text.count(old)
    assert n == count if count else n >= 1, f"proposal_counts: {what}: {n} occurrences"
    return text.replace(old, new)


def _publish_defs(text):
    """The publish_epoch definitions and declarations (header lines) of TEXT."""
    return [m.group(0) for m in re.finditer(r"^func\.(?:def|decl)[^\n]*@\w*publish_epoch\w*\([^\n]*$", text, re.M)]


def patch_service(text):
    """service.loom: every publish_epoch wrapper and its call take the 14th buffer."""
    heads = _publish_defs(text)
    assert heads, "proposal_counts: service.loom has no publish_epoch"
    for h in heads:
        text = _sub(text, h, h.replace(OLD_ARGS, NEW_ARGS), f"service {h[:60]}")
    return _sub(text, OLD_CALL_ARGS, NEW_CALL_ARGS, "service publish calls", count=0)


def patch_control(text):
    """control.loom (after the draft patch, when the package drafts): publish_epoch writes one count per span."""
    for h in _publish_defs(text):
        text = _sub(text, h, h.replace(OLD_ARGS, NEW_ARGS), f"control {h[:60]}")
    text = _sub(text, PLAIN_ZERO_OLD, PLAIN_ZERO_NEW, "plain publish loop")
    if "func.def @plain_publish_epoch(" in text:   # the drafted package's verified publish (draft/build_draft.py)
        text = _sub(text, OLD_CALL_ARGS, NEW_CALL_ARGS, "draft call of plain_publish_epoch")
        text = _sub(text, DRAFT_VIEW_OLD, DRAFT_VIEW_NEW, "draft publish view")
        text = _sub(text, DRAFT_OLD, DRAFT_NEW, "draft publish count")
    assert OLD_ARGS not in text, "proposal_counts: a publish_epoch kept 13 arguments"
    return text
