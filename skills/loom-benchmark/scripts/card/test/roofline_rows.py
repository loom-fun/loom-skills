"""check.sh helper: which acceptance each drafted roofline row counts (make_info.py drafted_acceptance +
roofline_block) and the card's note about it (render_card.drafted_note), without the GGUF or the gguf package.

usage: roofline_rows.py CARD_DIR (make_info.py and render_card.py's directory) PKG INFO.json -> JSON on stdout

A stand-in roofline module makes every single-pass ceiling 100 tok/s and a drafted ceiling 100 x (1 + acceptance x
depth), so the output shows only the acceptance chosen per row: the session's own where it recorded one (perf.py
decode rows), the spec gate's elsewhere."""
import os
import sys

sys.path.insert(0, sys.argv[1])
from card_lib import ROOFLINE_METRIC, dumps, load_json, load_package, package_facts  # noqa: E402
from make_info import drafted_acceptance, roofline_block  # noqa: E402
from render_card import drafted_note  # noqa: E402


class StubRoofline:
    DEVICE, BW, FLOPS = "stub device", 1.0, 1.0

    @staticmethod
    def ceilings(m, bw, flops, docs):
        return {rk: 100.0 for rk in ROOFLINE_METRIC.values()}

    @staticmethod
    def drafted_ceiling(m, depth, acceptance, bw, ctx=0, streams=1):
        return 100.0 * (1 + acceptance * depth)

    @staticmethod
    def drafted_row_ceiling(m, metric, depth, acceptance, bw):
        return StubRoofline.drafted_ceiling(m, depth, acceptance, bw)

    @staticmethod
    def efficiency(m, metric, ceiling, tokens_per_pass=None):
        return 0.5 if tokens_per_pass else 0.9


facts = package_facts(load_package(sys.argv[2]))
info = load_json(sys.argv[3])
gate, rows = drafted_acceptance(facts, info["performance"]["sessions"])
block = roofline_block(facts, None, gate, StubRoofline, 1.0, 1.0, {}, os.path.basename(facts["checkpoint"]["file"]), rows)
sys.stdout.write(dumps({"metrics": {k: r for k, r in block["metrics"].items() if "single_pass" in r},
                        "drafting": block["drafting"], "note": drafted_note(block["drafting"], block["metrics"])}))
