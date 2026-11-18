"""Timing ablations of the gfx12 flash2s twin (FLASH2_ABL=comma list; results are garbage: for profiling only).

nostv   no V^T staging stores        nokld  no global K/V loads (zeros are staged)   nobar  no workgroup barriers
noqk    no S = K Q^T MMAs            nopv   no O += V^T P MMAs (hot path)           nosm   no exp
"""
import re


def ablate(text, flags):
    f = set(x for x in flags.split(",") if x)
    if "nostv" in f:
        text = re.sub(r"      scf\.for %j = \[%c0 to %cseg_w step %c1\] unroll \{\n(?:        .*\n)*?      \}\n", "", text)
    if "nokld" in f:
        text = re.sub(r"(%key_valid\w* = index\.cmp) ule, (%key_pos\w*), %pos_\w+ : index", r"\1 ult, \2, %c0 : index", text)
    if "nobar" in f:
        text = re.sub(r"^ *kernel\.barrier<workgroup>[^\n]*\n", "", text, flags=re.M)
    if "noqk" in f:
        text = re.sub(r"(%s\d+_\d+) = vector\.mma %kl\d+_\d+, %qf\d+, (%\w+) : [^\n]*", r"\1 = vector.addf \2, %zero8 : vector<8xf32>", text)
    if "nopv" in f:
        text = re.sub(r"(%pl\d+_\d+) = vector\.mma %vl\d+_\d+, %pt\d+, (%\w+) : [^\n]*", r"\1 = vector.addf \2, %zero8 : vector<8xf32>", text)
    if "nosm" in f:
        text = re.sub(r"(%pe\d+) = vector\.expf<afn> (%sh\d+) : vector<8xf32>", r"\1 = vector.addf \2, %zero8 : vector<8xf32>", text)
    return text
