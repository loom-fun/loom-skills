"""GDN_AB_WMMA (choice prefill.gdn.ab_wmma): the gated delta-net alpha / beta projections of GEMM-shape epochs as one WMMA GEMM.

Qwen3.5/3.8 GGUFs keep ssm_alpha / ssm_beta (HV x H each, 48 x 5120 on the 27B) in F32. The default path runs each as the 4-token
f32 matvec g_mv_f32_k<H>_t4 over the f32 norm row %h (weight-restream bound: 1 MB of weights per 4 tokens; ~0.021 ms/token of
the Strix Halo 8k prefill). Here both run as one GEMM on the f16 operand the qkv/z GEMMs already read (%h16 + row scales %s_h):
one workgroup per 64 rows, 8 waves (4 row groups x 2: alpha | beta, 3 16-column fragments each), K step 64, the F32 weights
rounded to F16 while staging (no prepared copy), next step's loads issued before the MMAs. Results: %galpha / %gbeta
([rows] x HV f32, the matvec's layout). Numerics: f16 activations (as every trunk GEMM) and f16-rounded weights instead of f32.

kernel(H, HV) -> gfx11 kernel text + its gfx12 twin (gen_gemm_rdna4.with_twins).
fuse(text) -> (text, layers): @llm_advance's GEMM-epoch alpha matvec becomes the GEMM launch, the beta matvec's GEMM branch is dropped.
"""
import re

NAME = "g_gemm_ab_f32w"


def kernel(H, HV):
    assert HV % 16 == 0 and H % 64 == 0, (H, HV)
    NC = 2 * HV            # stacked output columns (alpha | beta)
    FR = HV // 16          # 16-column fragments per wave (one weight per wave column group)
    WR = NC                # staged weight rows
    assert WR <= 128 and (WR * 4) <= 512, WR
    s1_rows = WR - 64      # rows staged by the second slot (threads t < s1_rows*4); threads above repeat a valid slot
    assert 0 < s1_rows <= 64
    L = [f'''kernel.def export("{NAME}") @{NAME}(%rows: index) {{
  %c1 = index.constant 1 : index
  %c64 = index.constant 64 : index
  %c256 = index.constant 256 : index
  %row_tiles = index.div %rows, %c64 : index
  kernel.launch.config workgroups(%row_tiles, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%rows_arg: index, %x: buffer, %scales: buffer, %wa: buffer, %wb: buffer, %ya: buffer, %yb: buffer) {{
  %rows = index.assume %rows_arg [range(%rows_arg, 64, 8192), multiple_of(%rows_arg, 64)] : index
  %origin = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %c16 = index.constant 16 : index
  %c64 = index.constant 64 : index
  %c128 = index.constant 128 : index
  %chv = index.constant {HV} : index
  %k_steps = index.constant {H // 64} : index
  %k_last = index.constant {H - 64} : index
  %row_tile_count = index.div %rows, %c64 : index
  %row_group_id = kernel.workgroup.id<x> : index
  %row_group = index.assume %row_group_id [lt(%row_group_id, %row_tile_count)] : index
  %workitem_id = kernel.workitem.id<x> : index
  %workitem = index.assume %workitem_id [range(%workitem_id, 0, 255)] : index
  %subgroup = kernel.subgroup.id : index
  %bounded_subgroup = index.assume %subgroup [range(%subgroup, 0, 7)] : index
  %lane = kernel.subgroup.lane.id : index
  %lane_row = index.rem %lane, %c16 : index
  %wave_row_group = index.div %bounded_subgroup, %c2 : index
  %wave_col_group = index.rem %bounded_subgroup, %c2 : index
  %wave_row_v = index.mul %wave_row_group, %c16 : index
  %wave_row = index.assume %wave_row_v [range(%wave_row_v, 0, 48)] : index
  %wave_col = index.mul %wave_col_group, %chv : index
  %row_origin = index.mul %row_group, %c64 : index
  %x_view = buffer.view %x[%origin] : buffer -> view<[%rows]x{H}xf16>
  %s_view = buffer.view %scales[%origin] : buffer -> view<[%rows]x2xf32>
  %x_bytes = index.constant 9216 : offset
  %x_stage = buffer.alloca<workgroup> align(16) %x_bytes : buffer
  %x_stage_view = buffer.view %x_stage[%origin] : buffer -> view<64x72xf16>
  %w_bytes = index.constant {WR * 144} : offset
  %w_stage = buffer.alloca<workgroup> align(16) %w_bytes : buffer
  %w_stage_view = buffer.view %w_stage[%origin] : buffer -> view<{WR}x72xf16>
  // thread t stages x row t/4 (16 values at (t%4)*16); weight slot 0 = stacked row t/4, slot 1 = stacked row 64 + (t%{s1_rows * 4})/4
  %stage_row_v = index.div %workitem, %c4 : index
  %stage_row = index.assume %stage_row_v [range(%stage_row_v, 0, 63)] : index
  %stage_q = index.rem %workitem, %c4 : index
  %stage_col = index.mul %stage_q, %c16 : index
  %x_row_value = index.add %row_origin, %stage_row : index
  %x_row = index.assume %x_row_value [lt(%x_row_value, %rows)] : index
  %w0_is_b = index.cmp uge, %stage_row, %chv : index
  %w0_rb = index.sub %stage_row, %chv : index
  %w0_r = scf.select %w0_is_b, %w0_rb, %stage_row : index
  %w0_row = index.assume %w0_r [range(%w0_r, 0, {HV - 1})] : index
  %w0_buf = scf.select %w0_is_b, %wb, %wa : buffer
  %w0_view = buffer.view %w0_buf[%origin] : buffer -> view<{HV}x{H}xf32>
  %s1_span = index.constant {s1_rows * 4} : index
  %s1_t = index.rem %workitem, %s1_span : index
  %s1_r4 = index.div %s1_t, %c4 : index
  %s1_rv = index.add %s1_r4, %c64 : index
  %s1_row = index.assume %s1_rv [range(%s1_rv, 64, {WR - 1})] : index
  %w1_r = index.sub %s1_row, %chv : index
  %w1_row = index.assume %w1_r [range(%w1_r, 0, {HV - 1})] : index
  %w1_view = buffer.view %wb[%origin] : buffer -> view<{HV}x{H}xf32>
  %zero8 = vector.splatf 0.0 : vector<8xf32>
  %init = vector.fragment<init> %zero8 shape [%c16, %c16] : vector<8xf32>
  %x_first = vector.load %x_view[%x_row, %stage_col] : view<[%rows]x{H}xf16> -> vector<16xf16>
  %w0_first = vector.load %w0_view[%w0_row, %stage_col] : view<{HV}x{H}xf32> -> vector<16xf32>
  %w1_first = vector.load %w1_view[%w1_row, %stage_col] : view<{HV}x{H}xf32> -> vector<16xf32>''']
    accs = [f"a{c}" for c in range(FR)]
    L.append("  " + ", ".join(f"%{a}_raw" for a in accs) + ", %x_end, %w0_end, %w1_end = scf.for %step = [%c0 to %k_steps step %c1]("
             + ", ".join(f"%{a}_acc = %init : vector<8xf32>" for a in accs)
             + ", %x_cur = %x_first : vector<16xf16>, %w0_cur = %w0_first : vector<16xf32>, %w1_cur = %w1_first : vector<16xf32>) -> ("
             + ", ".join("vector<8xf32>" for _ in accs) + ", vector<16xf16>, vector<16xf32>, vector<16xf32>) {")
    L.append(f'''    vector.store %x_cur, %x_stage_view[%stage_row, %stage_col] : vector<16xf16>, view<64x72xf16>
    %w0_h = vector.fptrunc %w0_cur : vector<16xf32> to vector<16xf16>
    %w1_h = vector.fptrunc %w1_cur : vector<16xf32> to vector<16xf16>
    vector.store %w0_h, %w_stage_view[%stage_row, %stage_col] : vector<16xf16>, view<{WR}x72xf16>
    vector.store %w1_h, %w_stage_view[%s1_row, %stage_col] : vector<16xf16>, view<{WR}x72xf16>
    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
    %k_origin = index.mul %step, %c64 : index
    %k_next_raw = index.add %k_origin, %c64 : index
    %k_next = index.min %k_next_raw, %k_last : index
    %n_col = index.add %k_next, %stage_col : index
    %x_nx = vector.load %x_view[%x_row, %n_col] : view<[%rows]x{H}xf16> -> vector<16xf16>
    %w0_nx = vector.load %w0_view[%w0_row, %n_col] : view<{HV}x{H}xf32> -> vector<16xf32>
    %w1_nx = vector.load %w1_view[%w1_row, %n_col] : view<{HV}x{H}xf32> -> vector<16xf32>''')
    cur = {a: f"%{a}_acc" for a in accs}
    for c in range(FR):
        L.append(f"    %wr{c}_c = index.constant {c * 16} : index\n    %wr{c}_b = index.add %wave_col, %wr{c}_c : index\n    %wr{c} = index.add %wr{c}_b, %lane_row : index")
    for kk in range(4):
        L.append(f"    %fk{kk} = index.constant {kk * 16} : index\n"
                 f"    %l{kk} = vector.fragment.load<lhs> %x_stage_view[%wave_row, %fk{kk}] shape [%c16, %c16] : view<64x72xf16> -> vector<16xf16>")
        for c in range(FR):
            L.append(f'''    %rv{c}_{kk} = vector.load %w_stage_view[%wr{c}, %fk{kk}] : view<{WR}x72xf16> -> vector<16xf16>
    %rf{c}_{kk} = vector.fragment<rhs> %rv{c}_{kk} shape [%c16, %c16] : vector<16xf16>
    %m{c}_{kk} = vector.mma %l{kk}, %rf{c}_{kk}, {cur[f"a{c}"]} : vector<16xf16>, vector<16xf16>, vector<8xf32>''')
            cur[f"a{c}"] = f"%m{c}_{kk}"
    L.append("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    L.append("    scf.yield " + ", ".join(cur[a] for a in accs) + ", %x_nx, %w0_nx, %w1_nx : " + ", ".join("vector<8xf32>" for _ in accs)
             + ", vector<16xf16>, vector<16xf32>, vector<16xf32>\n  }")
    L.append(f'''  %inv_view = buffer.view %x_stage[%origin] : buffer -> view<64x16xf32>
  %writer = index.cmp ult, %workitem, %c64 : index
  scf.if %writer {{
    %bw = index.assume %workitem [range(%workitem, 0, 63)] : index
    %inv_row_value = index.add %row_origin, %bw : index
    %inv_row = index.assume %inv_row_value [lt(%inv_row_value, %rows)] : index
    %inverse = view.load %s_view[%inv_row, %c1] : view<[%rows]x2xf32> -> f32
    %inverse16 = vector.splat %inverse : vector<16xf32>
    vector.store %inverse16, %inv_view[%bw, %c0] : vector<16xf32>, view<64x16xf32>
  }}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %inv = vector.fragment.load<result> %inv_view[%wave_row, %c0] shape [%c16, %c16] : view<64x16xf32> -> vector<8xf32>
  %is_b = index.cmp eq, %wave_col_group, %c1 : index
  %y_layout = encoding.layout.strided [%chv, %c1] : encoding<layout>
  %y_first = index.mul %row_origin, %chv : index
  %four = index.constant 4 : offset
  %y_offset = index.scale %y_first, %four : index, offset -> offset''')
    for c in range(FR):
        L.append(f"  %v{c} = vector.mulf %a{c}_raw, %inv : vector<8xf32>\n  %oc{c} = index.constant {c * 16} : index")
    for side, buf in (("then", "%yb"), ("else", "%ya")):
        L.append("  scf.if %is_b {" if side == "then" else "  } else {")
        L.append(f"    %y_{side} = buffer.view {buf}[%y_offset] : buffer -> view<64x{HV}xf32, %y_layout>")
        for c in range(FR):
            L.append(f"    vector.fragment.store<result> %v{c}, %y_{side}[%wave_row, %oc{c}] shape [%c16, %c16] : vector<8xf32>, view<64x{HV}xf32, %y_layout>")
    L.append("  }")
    L.append("  kernel.return\n}\n")
    gfx11 = "\n".join(L)
    import gen_gemm_rdna4 as _r4
    return _r4.with_twins(gfx11)


_ALPHA = re.compile(r"(?m)^(?P<ind> *)kernel\.launch @g_mv_f32_k(?P<k>\d+)_t4\[%nHV, %t\]\(%nHV, %t, %zero_i32, %h, (?P<wa>%walpha_(?P<l>\d+)), %galpha\) : \[index, index\]\(index, index, i32, buffer, (?P<va>view<\d+xi8>), buffer\)\n")


def fuse(text):
    """@llm_advance: per GDN layer, `scf.if %gemm { g_mv_f32_k<H>_t4(alpha) } else {...}` launches the GEMM instead, and the beta block's
    `scf.if %gemm { t4(beta) } else { E }` becomes `scf.if %ab_ngemm { E }`. Returns (text, layers)."""
    a = text.find("command.def public @llm_advance(")
    if a < 0:
        return text, 0
    b = text.find("\ncommand.def ", a + 10)
    b = len(text) if b < 0 else b + 1
    body = text[a:b]
    n, out, pos = 0, [], 0
    for m in _ALPHA.finditer(body):
        l = m.group("l")
        assert body[:m.start()].rstrip(" ").endswith("scf.if %gemm {\n"), f"ab: layer {l}: alpha launch not under scf.if %gemm"
        beta = re.compile(r"(?m)^(?P<ind> *)scf\.if %gemm \{\n *kernel\.launch @g_mv_f32_k\d+_t4\[%nHV, %t\]\(%nHV, %t, %zero_i32, %h, %wbeta_" + l
                          + r", %gbeta\) : [^\n]*\n *\} else \{\n")
        bm = beta.search(body, m.end())
        assert bm, f"ab: layer {l}: beta block"
        launch = (f"{m.group('ind')}kernel.launch @{NAME}[%tp](%tp, %h16, %s_h, {m.group('wa')}, %wbeta_{l}, %galpha, %gbeta) : "
                  f"[index](index, buffer, buffer, {m.group('va')}, {m.group('va')}, buffer, buffer)\n")
        out.append(body[pos:m.start()] + launch + body[m.end():bm.start()] + f"{bm.group('ind')}scf.if %ab_ngemm {{\n")
        # the beta block's else branch stays; its closing `}` now closes scf.if %ab_ngemm
        pos = bm.end()
        n += 1
    if not n:
        return text, 0
    body = "".join(out) + body[pos:]
    anchor = re.search(r"(?m)^  %gemm = index\.cmp uge, %t, %c64 : index\n", body)
    assert anchor, "ab: %gemm"
    body = body[:anchor.end()] + "  %ab_ngemm = index.cmp ult, %t, %c64 : index\n" + body[anchor.end():]
    return text[:a] + body + text[b:], n
