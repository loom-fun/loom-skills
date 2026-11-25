"""Ternary rotated-basis checkpoints (PrismML Ternary Bonsai; llama_config "hadamard" + PTQ1_0 weights).

Two pieces, both emitted only when the model has them (no other model's programs change):

1. PTQ1_0 weights (GGML type 143, gguf_ext.py): 128 trits {-1, 0, 1} and one F16 scale per 28-byte block. The value helper
   (@g_ptq1_0_values, GEMM dequant and embedding) and the int8 code helper (@g_ptq1_0_icodes, the decode matvecs and the head:
   signed trit codes x one scale, as q8_0) decode one 32-value group of a block.

2. prism.hadamard: a folded weight W D H expects the input H (D x) (D: +/-1 per input column, H: normalized 1024-point Sylvester
   Walsh-Hadamard transform per block). The rotation is applied where the activations are quantized for the matmuls (the int8
   copy of the decode matvecs, the F16 row-scaled copy of the GEMMs): every trunk quantizer and the head's become rotating ones
   (wiring()), so the f32 activation stays in the model basis for the unrotated F32 projections that read it (gated delta-net
   alpha / beta) and for the MTP block (Q4_0, not folded). ssm_out takes its input permuted from the layer's tiled head order to
   the grouped order of the fold (gdn_v_grouped). The token embedding stores rotated rows: the lookup applies D (H z).
"""
import re

BLK = 1024   # prism.hadamard.block_size this module implements (inspect_gguf.py records it; asserted in setup())

PTQ1_0_HELPERS = r"""
// PTQ1_0 trits (0, 1, 2) of 16 bytes at weight 3^n (%p, per 16-bit lane: the even and odd bytes of a lane pair take the same power):
// ((q * 3^n mod 256) * 3) >> 8, on 16-bit lanes (even bytes, then odd bytes), so no byte is widened alone.
func.def inline @g_ptq1_0_tcw(%b: vector<16xi8>, %p: vector<8xi16>) -> (vector<16xi8>) {
  %w = vector.bitcast %b : vector<16xi8> to vector<8xi16>
  %ff = vector.splati 255 : vector<8xi16>
  %c8 = vector.splati 8 : vector<8xi16>
  %c3 = vector.splati 3 : vector<8xi16>
  %e = vector.andi %w, %ff : vector<8xi16>
  %o = vector.shrui %w, %c8 : vector<8xi16>
  %me = vector.muli %e, %p : vector<8xi16>
  %mo = vector.muli %o, %p : vector<8xi16>
  %me8 = vector.andi %me, %ff : vector<8xi16>
  %mo8 = vector.andi %mo, %ff : vector<8xi16>
  %te3 = vector.muli %me8, %c3 : vector<8xi16>
  %to3 = vector.muli %mo8, %c3 : vector<8xi16>
  %te = vector.shrui %te3, %c8 : vector<8xi16>
  %to = vector.shrui %to3, %c8 : vector<8xi16>
  %tos = vector.shli %to, %c8 : vector<8xi16>
  %t = vector.ori %te, %tos : vector<8xi16>
  %c = vector.bitcast %t : vector<8xi16> to vector<16xi8>
  func.return %c : vector<16xi8>
}

// PTQ1_0 block (28 bytes): qs[24] then qh[2] then F16 d. Value v of the block: v < 80: byte v % 16, trit v / 16; v < 120: byte
// 16 + (v - 80) % 8, trit (v - 80) / 8; else byte 24 + (v - 120) % 2, trit (v - 120) / 2. Group %group (0..3) = values 32g..32g+31.
// Branch-free (lanes of one wave hold different groups of a block): each half of the group is 16 bytes x 16 powers of 3, picked by
// selects: g0 = A*1 | A*3, g1 = A*9 | A*27, g2 = A*81 | B*(1,3), g3 = B*(9,27) | B*81, Q*(1,1,3,3,9,9,27,27)  (A = bytes 0..15, B =
// bytes 16..23 twice, Q = bytes 24, 25 four times).
func.def inline @g_ptq1_0_icodes(%w: buffer, %origin: offset, %group: index) -> (vector<32xi8>, f32, f32) {
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c3 = index.constant 3 : index
  %o16 = index.constant 16 : offset
  %o24 = index.constant 24 : offset
  %o26 = index.constant 26 : offset
  %a_view = buffer.view %w[%origin] : buffer -> view<16xi8>
  %b_origin = index.add %origin, %o16 : offset
  %b_view = buffer.view %w[%b_origin] : buffer -> view<8xi8>
  %h_origin = index.add %origin, %o24 : offset
  %h_view = buffer.view %w[%h_origin] : buffer -> view<2xi8>
  %d_origin = index.add %origin, %o26 : offset
  %d_view = buffer.view %w[%d_origin] : buffer -> view<1xf16>
  %d_half = view.load %d_view[0] : view<1xf16> -> f16
  %d = scalar.extf %d_half : f16 to f32
  %a = vector.load %a_view[0] : view<16xi8> -> vector<16xi8>
  %b8 = vector.load %b_view[0] : view<8xi8> -> vector<8xi8>
  %h2 = vector.load %h_view[0] : view<2xi8> -> vector<2xi8>
  %bb = vector.concat<0> %b8, %b8 : vector<8xi8>, vector<8xi8> -> vector<16xi8>
  %h4 = vector.concat<0> %h2, %h2 : vector<2xi8>, vector<2xi8> -> vector<4xi8>
  %h8 = vector.concat<0> %h4, %h4 : vector<4xi8>, vector<4xi8> -> vector<8xi8>
  %bq = vector.concat<0> %b8, %h8 : vector<8xi8>, vector<8xi8> -> vector<16xi8>
  %is1 = index.cmp eq, %group, %c1 : index
  %is2 = index.cmp eq, %group, %c2 : index
  %is3 = index.cmp eq, %group, %c3 : index
  %le1 = index.cmp ult, %group, %c2 : index
  %m1 = vector.splat %is1 : vector<16xi1>
  %m2 = vector.splat %is2 : vector<16xi1>
  %m3 = vector.splat %is3 : vector<16xi1>
  %ml1 = vector.splat %le1 : vector<16xi1>
  %lo = vector.select %m3, %bb, %a : vector<16xi8>
  %hi_23 = vector.select %m2, %bb, %bq : vector<16xi8>
  %hi = vector.select %ml1, %a, %hi_23 : vector<16xi8>
  %i1 = scalar.constant 1 : i16
  %i3 = scalar.constant 3 : i16
  %i9 = scalar.constant 9 : i16
  %i27 = scalar.constant 27 : i16
  %i81 = scalar.constant 81 : i16
  %v1 = vector.splat %i1 : vector<8xi16>
  %v3 = vector.splat %i3 : vector<8xi16>
  %v9 = vector.splat %i9 : vector<8xi16>
  %v27 = vector.splat %i27 : vector<8xi16>
  %v81 = vector.splat %i81 : vector<8xi16>
  %p3lo = vector.from_elements %i9, %i9, %i9, %i9, %i27, %i27, %i27, %i27 : vector<8xi16>
  %p2hi = vector.from_elements %i1, %i1, %i1, %i1, %i3, %i3, %i3, %i3 : vector<8xi16>
  %p3hi = vector.from_elements %i81, %i81, %i81, %i81, %i1, %i3, %i9, %i27 : vector<8xi16>
  %n1 = vector.splat %is1 : vector<8xi1>
  %n2 = vector.splat %is2 : vector<8xi1>
  %n3 = vector.splat %is3 : vector<8xi1>
  %plo_01 = vector.select %n1, %v9, %v1 : vector<8xi16>
  %plo_012 = vector.select %n2, %v81, %plo_01 : vector<8xi16>
  %plo = vector.select %n3, %p3lo, %plo_012 : vector<8xi16>
  %phi_01 = vector.select %n1, %v27, %v3 : vector<8xi16>
  %phi_012 = vector.select %n2, %p2hi, %phi_01 : vector<8xi16>
  %phi = vector.select %n3, %p3hi, %phi_012 : vector<8xi16>
  %clo = func.call @g_ptq1_0_tcw(%lo, %plo) : (vector<16xi8>, vector<8xi16>) -> (vector<16xi8>)
  %chi = func.call @g_ptq1_0_tcw(%hi, %phi) : (vector<16xi8>, vector<8xi16>) -> (vector<16xi8>)
  %codes = vector.concat<0> %clo, %chi : vector<16xi8>, vector<16xi8> -> vector<32xi8>
  func.return %codes, %d, %d : vector<32xi8>, f32, f32
}

// Values 16h..16h+15 of a block (h = 0..7): h < 5: bytes 0..15 at 3^h; h = 5: bytes 16..23 twice at (1 x8, 3 x8); h = 6: the same at
// (9 x8, 27 x8); h = 7: bytes 16..23 then 24, 25 four times at (81 x8, 1, 1, 3, 3, 9, 9, 27, 27). The GEMMs' 16-value staging.
func.def inline @g_ptq1_0_half16(%w: buffer, %origin: offset, %h: index) -> (vector<16xf32>) {
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c3 = index.constant 3 : index
  %c4 = index.constant 4 : index
  %c5 = index.constant 5 : index
  %c6 = index.constant 6 : index
  %c7 = index.constant 7 : index
  %o16 = index.constant 16 : offset
  %o24 = index.constant 24 : offset
  %o26 = index.constant 26 : offset
  %a_view = buffer.view %w[%origin] : buffer -> view<16xi8>
  %b_origin = index.add %origin, %o16 : offset
  %b_view = buffer.view %w[%b_origin] : buffer -> view<8xi8>
  %h_origin = index.add %origin, %o24 : offset
  %h_view = buffer.view %w[%h_origin] : buffer -> view<2xi8>
  %d_origin = index.add %origin, %o26 : offset
  %d_view = buffer.view %w[%d_origin] : buffer -> view<1xf16>
  %d_half = view.load %d_view[0] : view<1xf16> -> f16
  %d = scalar.extf %d_half : f16 to f32
  %a = vector.load %a_view[0] : view<16xi8> -> vector<16xi8>
  %b8 = vector.load %b_view[0] : view<8xi8> -> vector<8xi8>
  %h2 = vector.load %h_view[0] : view<2xi8> -> vector<2xi8>
  %bb = vector.concat<0> %b8, %b8 : vector<8xi8>, vector<8xi8> -> vector<16xi8>
  %h4 = vector.concat<0> %h2, %h2 : vector<2xi8>, vector<2xi8> -> vector<4xi8>
  %h8 = vector.concat<0> %h4, %h4 : vector<4xi8>, vector<4xi8> -> vector<8xi8>
  %bq = vector.concat<0> %b8, %h8 : vector<8xi8>, vector<8xi8> -> vector<16xi8>
  %ge5 = index.cmp uge, %h, %c5 : index
  %is7 = index.cmp eq, %h, %c7 : index
  %m5 = vector.splat %ge5 : vector<16xi1>
  %m7 = vector.splat %is7 : vector<16xi1>
  %bsel = vector.select %m7, %bq, %bb : vector<16xi8>
  %bytes = vector.select %m5, %bsel, %a : vector<16xi8>
  %i1 = scalar.constant 1 : i16
  %i3 = scalar.constant 3 : i16
  %i9 = scalar.constant 9 : i16
  %i27 = scalar.constant 27 : i16
  %i81 = scalar.constant 81 : i16
  %is1 = index.cmp eq, %h, %c1 : index
  %is2 = index.cmp eq, %h, %c2 : index
  %is3 = index.cmp eq, %h, %c3 : index
  %is4 = index.cmp eq, %h, %c4 : index
  %s1 = scf.select %is1, %i3, %i1 : i16
  %s2 = scf.select %is2, %i9, %s1 : i16
  %s3 = scf.select %is3, %i27, %s2 : i16
  %s4 = scf.select %is4, %i81, %s3 : i16
  %ps = vector.splat %s4 : vector<8xi16>
  %p5 = vector.from_elements %i1, %i1, %i1, %i1, %i3, %i3, %i3, %i3 : vector<8xi16>
  %p6 = vector.from_elements %i9, %i9, %i9, %i9, %i27, %i27, %i27, %i27 : vector<8xi16>
  %p7 = vector.from_elements %i81, %i81, %i81, %i81, %i1, %i3, %i9, %i27 : vector<8xi16>
  %n7 = vector.splat %is7 : vector<8xi1>
  %p67 = vector.select %n7, %p7, %p6 : vector<8xi16>
  %is5 = index.cmp eq, %h, %c5 : index
  %n5e = vector.splat %is5 : vector<8xi1>
  %p567 = vector.select %n5e, %p5, %p67 : vector<8xi16>
  %n5 = vector.splat %ge5 : vector<8xi1>
  %pw = vector.select %n5, %p567, %ps : vector<8xi16>
  %codes = func.call @g_ptq1_0_tcw(%bytes, %pw) : (vector<16xi8>, vector<8xi16>) -> (vector<16xi8>)
  %f = vector.sitofp %codes : vector<16xi8> to vector<16xf32>
  %one = vector.splatf 1.0 : vector<16xf32>
  %fc = vector.subf %f, %one : vector<16xf32>
  %ds = vector.splat %d : vector<16xf32>
  %values = vector.mulf %fc, %ds : vector<16xf32>
  func.return %values : vector<16xf32>
}

func.def inline @g_ptq1_0_values(%w: buffer, %origin: offset, %group: index) -> (vector<32xf32>) {
  %codes, %d, %dm = func.call @g_ptq1_0_icodes(%w, %origin, %group) : (buffer, offset, index) -> (vector<32xi8>, f32, f32)
  %f = vector.sitofp %codes : vector<32xi8> to vector<32xf32>
  %one = vector.splatf 1.0 : vector<32xf32>
  %c = vector.subf %f, %one : vector<32xf32>
  %ds = vector.splat %d : vector<32xf32>
  %values = vector.mulf %c, %ds : vector<32xf32>
  func.return %values : vector<32xf32>
}
"""


Q2_0_HELPERS = r"""
// Q2_0 (GGML type 42, upstream llama.cpp; Ternary Bonsai's stock-runnable packing): 64 values per 18-byte block, F16 d then 16 bytes of
// 2-bit codes, value j = byte j / 4 bits 2 (j % 4): code q, value (q - 1) d. A 32-value group is 8 bytes: (w >> 2k) & 0x03030303 holds
// values 4i + k in byte i; a shuffle puts them in value order.
func.def inline @g_q2_0_codes8(%w8: vector<2xi32>) -> (vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>) {
  %m = vector.splati 50529027 : vector<2xi32>
  %s2 = vector.splati 2 : vector<2xi32>
  %s4 = vector.splati 4 : vector<2xi32>
  %s6 = vector.splati 6 : vector<2xi32>
  %r0 = vector.andi %w8, %m : vector<2xi32>
  %t2 = vector.shrui %w8, %s2 : vector<2xi32>
  %r1 = vector.andi %t2, %m : vector<2xi32>
  %t4 = vector.shrui %w8, %s4 : vector<2xi32>
  %r2 = vector.andi %t4, %m : vector<2xi32>
  %t6 = vector.shrui %w8, %s6 : vector<2xi32>
  %r3 = vector.andi %t6, %m : vector<2xi32>
  %b0 = vector.bitcast %r0 : vector<2xi32> to vector<8xi8>
  %b1 = vector.bitcast %r1 : vector<2xi32> to vector<8xi8>
  %b2 = vector.bitcast %r2 : vector<2xi32> to vector<8xi8>
  %b3 = vector.bitcast %r3 : vector<2xi32> to vector<8xi8>
  func.return %b0, %b1, %b2, %b3 : vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>
}

func.def inline @g_q2_0_icodes(%w: buffer, %origin: offset, %group: index) -> (vector<32xi8>, f32, f32) {
  %two = index.constant 2 : offset
  %eight = index.constant 8 : offset
  %d_view = buffer.view %w[%origin] : buffer -> view<1xf16>
  %d_half = view.load %d_view[0] : view<1xf16> -> f16
  %d = scalar.extf %d_half : f16 to f32
  %qo = index.add %origin, %two : offset
  %go = index.scale %group, %eight : index, offset -> offset
  %co = index.add %qo, %go : offset
  %c_view = buffer.view %w[%co] : buffer -> view<8xi8>
  %c8 = vector.load %c_view[0] : view<8xi8> -> vector<8xi8>
  %cw = vector.bitcast %c8 : vector<8xi8> to vector<2xi32>
  %b0, %b1, %b2, %b3 = func.call @g_q2_0_codes8(%cw) : (vector<2xi32>) -> (vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>)
  %b01 = vector.concat<0> %b0, %b1 : vector<8xi8>, vector<8xi8> -> vector<16xi8>
  %b23 = vector.concat<0> %b2, %b3 : vector<8xi8>, vector<8xi8> -> vector<16xi8>
  %all = vector.concat<0> %b01, %b23 : vector<16xi8>, vector<16xi8> -> vector<32xi8>
  // all[8k + i] = value 4i + k: value j = 4i + k comes from lane 8 (j % 4) + j / 4
  %codes = vector.shuffle<[0, 8, 16, 24, 1, 9, 17, 25, 2, 10, 18, 26, 3, 11, 19, 27, 4, 12, 20, 28, 5, 13, 21, 29, 6, 14, 22, 30, 7, 15, 23, 31]> %all : vector<32xi8>
  func.return %codes, %d, %d : vector<32xi8>, f32, f32
}

func.def inline @g_q2_0_values(%w: buffer, %origin: offset, %group: index) -> (vector<32xf32>) {
  %codes, %d, %dm = func.call @g_q2_0_icodes(%w, %origin, %group) : (buffer, offset, index) -> (vector<32xi8>, f32, f32)
  %f = vector.sitofp %codes : vector<32xi8> to vector<32xf32>
  %one = vector.splatf 1.0 : vector<32xf32>
  %c = vector.subf %f, %one : vector<32xf32>
  %ds = vector.splat %d : vector<32xf32>
  %values = vector.mulf %c, %ds : vector<32xf32>
  func.return %values : vector<32xf32>
}
"""

def signs_rodata(cfg):
    """One read-only table per sign width: the +/-1 signs as bits (set = -1), little-endian 32-bit words."""
    had = cfg["hadamard"]
    return "".join(f'global.rodata.def @g_had_signs_{w} = align(64) bytes("{bits}")\n' for w, bits in sorted(had["signs"].items(), key=lambda kv: int(kv[0])))


def fwht(lds, nblk, lanes, pfx):
    """IR: in-place normalized Walsh-Hadamard transform of `nblk` consecutive 1024-value blocks held in workgroup buffer %{lds}
    (f32), all `lanes` lanes of the workgroup taking part (a barrier before and after every stage), then x 1/32."""
    n_bf = nblk * BLK // 2   # butterflies per stage
    per = (n_bf + lanes - 1) // lanes
    return f"""  %{pfx}_n = index.constant {nblk * BLK} : index
  %{pfx}_v = buffer.view %{lds}[%zero] : buffer -> view<{nblk * BLK}xf32>
  %{pfx}_nbf = index.constant {n_bf} : index
  %{pfx}_per = index.constant {per} : index
  %{pfx}_lanes = index.constant {lanes} : index
  %{pfx}_c1 = index.constant 1 : index
  %{pfx}_c0 = index.constant 0 : index
  %{pfx}_c10 = index.constant 10 : index
  %{pfx}_c2 = index.constant 2 : index
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %{pfx}_hend = scf.for %{pfx}_s = [%{pfx}_c0 to %{pfx}_c10 step %{pfx}_c1](%{pfx}_hc = %{pfx}_c1 : index) -> (index) {{
    %{pfx}_h = index.assume %{pfx}_hc [range(%{pfx}_hc, 1, 512)] : index
    scf.for %{pfx}_k = [%{pfx}_c0 to %{pfx}_per step %{pfx}_c1] unroll {{
      %{pfx}_p0 = index.mul %{pfx}_k, %{pfx}_lanes : index
      %{pfx}_p = index.add %{pfx}_p0, %lane : index
      %{pfx}_ok = index.cmp ult, %{pfx}_p, %{pfx}_nbf : index
      scf.if %{pfx}_ok {{
        %{pfx}_hi = index.div %{pfx}_p, %{pfx}_h : index
        %{pfx}_lo = index.rem %{pfx}_p, %{pfx}_h : index
        %{pfx}_hi2 = index.mul %{pfx}_hi, %{pfx}_h : index
        %{pfx}_hi3 = index.add %{pfx}_hi2, %{pfx}_hi2 : index
        %{pfx}_i0 = index.add %{pfx}_hi3, %{pfx}_lo : index
        %{pfx}_i = index.assume %{pfx}_i0 [range(%{pfx}_i0, 0, {nblk * BLK - 1})] : index
        %{pfx}_j0 = index.add %{pfx}_i, %{pfx}_h : index
        %{pfx}_j = index.assume %{pfx}_j0 [range(%{pfx}_j0, 0, {nblk * BLK - 1})] : index
        %{pfx}_a = view.load %{pfx}_v[%{pfx}_i] : view<{nblk * BLK}xf32> -> f32
        %{pfx}_b = view.load %{pfx}_v[%{pfx}_j] : view<{nblk * BLK}xf32> -> f32
        %{pfx}_sum = scalar.addf %{pfx}_a, %{pfx}_b : f32
        %{pfx}_dif = scalar.subf %{pfx}_a, %{pfx}_b : f32
        view.store %{pfx}_sum, %{pfx}_v[%{pfx}_i] : f32, view<{nblk * BLK}xf32>
        view.store %{pfx}_dif, %{pfx}_v[%{pfx}_j] : f32, view<{nblk * BLK}xf32>
      }}
    }}
    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
    %{pfx}_h2 = index.mul %{pfx}_h, %{pfx}_c2 : index
    scf.yield %{pfx}_h2 : index
  }}
"""


def _uniq(text, p):
    """Prefix every value `text` defines (and its uses) with p: the same snippet emitted twice in one kernel."""
    names = set(re.findall(r"%(\w+) = ", text)) | set(re.findall(r"scf\.for %(\w+) = ", text)) | set(re.findall(r"\(%(\w+) = ", text))
    for n in sorted(names, key=len, reverse=True):
        text = re.sub(r"%" + n + r"\b", "%" + p + n, text)
    return text


def _sign_of(width, idx, pfx):
    """IR: f32 %{pfx}_sg = -1.0 or 1.0, the sign of input column %{idx} (an index < width) of a width-`width` input."""
    return f"""  %{pfx}_tab = global.load @g_had_signs_{width} : buffer
  %{pfx}_tv = buffer.view %{pfx}_tab[%zero] : buffer -> view<{width // 32}xi32>
  %{pfx}_c32 = index.constant 32 : index
  %{pfx}_wi0 = index.div %{idx}, %{pfx}_c32 : index
  %{pfx}_wi = index.assume %{pfx}_wi0 [range(%{pfx}_wi0, 0, {width // 32 - 1})] : index
  %{pfx}_bi = index.rem %{idx}, %{pfx}_c32 : index
  %{pfx}_word = view.load %{pfx}_tv[%{pfx}_wi] : view<{width // 32}xi32> -> i32
  %{pfx}_sh = index.cast %{pfx}_bi : index to i32
  %{pfx}_bit0 = scalar.shrui %{pfx}_word, %{pfx}_sh : i32
  %{pfx}_one = scalar.constant 1 : i32
  %{pfx}_bit = scalar.andi %{pfx}_bit0, %{pfx}_one : i32
  %{pfx}_neg = scalar.cmpi ne, %{pfx}_bit, %zero_i : i32
  %{pfx}_pf = scalar.constant 1.0 : f32
  %{pfx}_nf = scalar.constant -1.0 : f32
  %{pfx}_sg = scf.select %{pfx}_neg, %{pfx}_nf, %{pfx}_pf : f32
"""


def _gather(width, perm, gdn):
    """IR (lanes of one workgroup, block %blk of row %row): load this lane's 4 inputs of the block into LDS %buf, in the fold's
    column order (perm: tiled -> grouped heads) and multiplied by the column's sign."""
    head = gdn["head_v"] if gdn else 128
    nk = gdn["k_heads"] if gdn else 1
    rep = (gdn["v_heads"] // gdn["k_heads"]) if gdn else 1
    src = "%col"
    pm = ""
    if perm:   # grouped column g = d + hd (r + rep k) reads tiled column d + hd (k + nk r)
        src = "%src"
        pm = f"""      %hd = index.constant {head} : index
      %rep = index.constant {rep} : index
      %nk = index.constant {nk} : index
      %hidx = index.div %col, %hd : index
      %dd = index.rem %col, %hd : index
      %rr = index.rem %hidx, %rep : index
      %kk = index.div %hidx, %rep : index
      %t0 = index.mul %nk, %rr : index
      %t1 = index.add %kk, %t0 : index
      %t2 = index.mul %t1, %hd : index
      %src0 = index.add %t2, %dd : index
      %src = index.assume %src0 [range(%src0, 0, {width - 1})] : index
"""
    return f"""  %buf_v = buffer.view %buf[%zero] : buffer -> view<{BLK}xf32>
  %xrow = buffer.view %x[%zero] : buffer -> view<[%rows]x{width}xf32>
  %blk_base = index.mul %blk, %c1024 : index
  scf.for %e = [%c0 to %c4 step %c1] unroll {{
      %e4 = index.mul %lane, %c4 : index
      %ei0 = index.add %e4, %e : index
      %ei = index.assume %ei0 [range(%ei0, 0, 1023)] : index
      %col0 = index.add %blk_base, %ei : index
      %col = index.assume %col0 [range(%col0, 0, {width - 1})] : index
{pm}      %xv = view.load %xrow[%row, {src}] : view<[%rows]x{width}xf32> -> f32
{_sign_of(width, "col", "sg").replace(chr(10) + "  ", chr(10) + "      ")}      %xs = scalar.mulf %xv, %sg_sg : f32
      view.store %xs, %buf_v[%ei] : f32, view<{BLK}xf32>
  }}
"""


def q8_kernel(width, perm, gdn):
    """g_quant_q8 with the rotation: grid (width / 1024 blocks, rows), 256 lanes per block; the same arguments and the same
    output layout as g_quant_q8 (int8 codes per 32-value group, then (d as F16-rounded f32, code sum) per group)."""
    name = f"g_had_q8_w{width}{'p' if perm else ''}"
    return f"""
kernel.def export("{name}") @{name}(%rows: index, %width: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %nb = index.constant {width // BLK} : index
  kernel.launch.config workgroups(%nb, %rows, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%rows_arg: index, %width_arg: index, %x: buffer, %xq: buffer) {{
  %rows = index.assume %rows_arg [range(%rows_arg, 1, 8192)] : index
  %width = index.constant {width} : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c4 = index.constant 4 : index
  %c32 = index.constant 32 : index
  %c1024 = index.constant 1024 : index
  %sg_c32w = index.constant {width // 32} : index
  %zero_f = scalar.constant 0.0 : f32
  %zero_i = scalar.constant 0 : i32
  %c127 = scalar.constant 127.0 : f32
  %zero8i = vector.splati 0 : vector<8xi32>
  %ones8 = vector.splati 1 : vector<32xi8>
  %zero32 = vector.splatf 0.0 : vector<32xf32>
  %blk_id = kernel.workgroup.id<x> : index
  %nb = index.constant {width // BLK} : index
  %blk = index.assume %blk_id [lt(%blk_id, %nb)] : index
  %row_id = kernel.workgroup.id<y> : index
  %row = index.assume %row_id [lt(%row_id, %rows)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %buf_bytes = index.constant {BLK * 4} : offset
  %buf = buffer.alloca<workgroup> align(16) %buf_bytes : buffer
{_gather(width, perm, gdn)}{fwht("buf", 1, 256, "fw")}  %is_q = index.cmp ult, %lane, %c32 : index
  scf.if %is_q {{
    %gl = index.assume %lane [range(%lane, 0, 31)] : index
    %groups = index.constant {width // 32} : index
    %bv32 = buffer.view %buf[%zero] : buffer -> view<32x32xf32>
    %raw = vector.load %bv32[%gl, %c0] : view<32x32xf32> -> vector<32xf32>
    %inv = vector.splatf 0.03125 : vector<32xf32>
    %v = vector.mulf %raw, %inv : vector<32xf32>
    %g0 = index.mul %blk, %c32 : index
    %g1 = index.add %g0, %gl : index
    %g = index.assume %g1 [range(%g1, 0, {width // 32 - 1})] : index
    %q_view = buffer.view %xq[%zero] : buffer -> view<[%rows]x{width // 32}x8xi32>
    %m_bytes = index.constant {width} : offset
    %m_off = index.scale %rows, %m_bytes : index, offset -> offset
    %m_view = buffer.view %xq[%m_off] : buffer -> view<[%rows]x{width // 32}x2xf32>
    %a = vector.absf %v : vector<32xf32>
    %amax = vector.reduce<maxnumf> %a, %zero_f : vector<32xf32>, f32
    %d = scalar.divf %amax, %c127 : f32
    %nz = scalar.cmpf one, %amax, %zero_f : f32
    %dv = vector.splat %d : vector<32xf32>
    %sc = vector.divf %v, %dv : vector<32xf32>
    %rd = vector.roundf %sc : vector<32xf32>
    %nzv = vector.splat %nz : vector<32xi1>
    %rs = vector.select %nzv, %rd, %zero32 : vector<32xf32>
    %q = vector.fptosi %rs : vector<32xf32> to vector<32xi8>
    %qw = vector.bitcast %q : vector<32xi8> to vector<8xi32>
    vector.store %qw, %q_view[%row, %g, %c0] : vector<8xi32>, view<[%rows]x{width // 32}x8xi32>
    %qs = vector.dot4i<s8s8> %q, %ones8, %zero8i : vector<32xi8>, vector<32xi8>, vector<8xi32>
    %qsi = vector.reduce<addi> %qs, %zero_i : vector<8xi32>, i32
    %qsf = scalar.sitofp %qsi : i32 to f32
    %dh = scalar.fptrunc %d : f32 to f16
    %da = scalar.extf %dh : f16 to f32
    %meta = vector.from_elements %da, %qsf : vector<2xf32>
    vector.store %meta, %m_view[%row, %g, %c0] : vector<2xf32>, view<[%rows]x{width // 32}x2xf32>
  }}
  kernel.return
}}
"""


def rs16_kernel(width, perm, gdn):
    """g_row_scale_f16 with the rotation: one workgroup per row (256 lanes); pass 1 rotates every block and takes the row's
    max |value|, pass 2 rotates again and writes value * scale as F16 (the same power-of-two scale rule and outputs as
    g_row_scale_f16)."""
    name = f"g_had_rs16_w{width}{'p' if perm else ''}"
    nb = width // BLK
    body = _gather(width, perm, gdn) + fwht("buf", 1, 256, "fw")
    body2 = _uniq(body, "p2")
    return f"""
kernel.def export("{name}") @{name}(%rows: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%rows, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%rows_arg: index, %k_arg: index, %x: buffer, %scales: buffer, %x16: buffer) {{
  %rows = index.assume %rows_arg [range(%rows_arg, 1, 2097152)] : index
  %width = index.constant {width} : index
  %row_id = kernel.workgroup.id<x> : index
  %row = index.assume %row_id [lt(%row_id, %rows)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c4 = index.constant 4 : index
  %c1024 = index.constant 1024 : index
  %sg_c32w = index.constant {width // 32} : index
  %zero_f = scalar.constant 0.0 : f32
  %zero_i = scalar.constant 0 : i32
  %inv32 = scalar.constant 0.03125 : f32
  %nb = index.constant {nb} : index
  %buf_bytes = index.constant {BLK * 4} : offset
  %buf = buffer.alloca<workgroup> align(16) %buf_bytes : buffer
  %local = scf.for %blk = [%c0 to %nb step %c1](%m = %zero_f : f32) -> (f32) {{
{body}    %bv = buffer.view %buf[%zero] : buffer -> view<256x4xf32>
    %v = vector.load %bv[%lane, %c0] : view<256x4xf32> -> vector<4xf32>
    %a = vector.absf %v : vector<4xf32>
    %n = vector.reduce<maxnumf> %a, %m : vector<4xf32>, f32
    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
    scf.yield %n : f32
  }}
  %maximum0 = kernel.workgroup.reduce<maxnumf> %local : f32
  %maximum = scalar.mulf %maximum0, %inv32 : f32
  %bits = scalar.bitcast %maximum : f32 to i32
  %c23 = scalar.constant 23 : i32
  %c255 = scalar.constant 255 : i32
  %shifted = scalar.shrui %bits, %c23 : i32
  %biased = scalar.andi %shifted, %c255 : i32
  %c127 = scalar.constant 127 : i32
  %c13 = scalar.constant 13 : i32
  %c0_i32 = scalar.constant 0 : i32
  %c100 = scalar.constant 100 : i32
  %unbiased = scalar.subi %biased, %c127 : i32
  %raw_e = scalar.subi %unbiased, %c13 : i32
  %positive = scalar.maxsi %raw_e, %c0_i32 : i32
  %e = scalar.minsi %positive, %c100 : i32
  %scale_exp = scalar.subi %c127, %e : i32
  %inverse_exp = scalar.addi %c127, %e : i32
  %scale_bits = scalar.shli %scale_exp, %c23 : i32
  %inverse_bits = scalar.shli %inverse_exp, %c23 : i32
  %scale = scalar.bitcast %scale_bits : i32 to f32
  %inverse = scalar.bitcast %inverse_bits : i32 to f32
  %scale32 = scalar.mulf %scale, %inv32 : f32
  %scale4 = vector.splat %scale32 : vector<4xf32>
  %k4 = index.constant {width // 4} : index
  %h4_view = buffer.view %x16[%zero] : buffer -> view<[%rows]x{width // 4}x4xf16>
  scf.for %blk = [%c0 to %nb step %c1] {{
{body2}    %bv = buffer.view %buf[%zero] : buffer -> view<256x4xf32>
    %v = vector.load %bv[%lane, %c0] : view<256x4xf32> -> vector<4xf32>
    %sv = vector.mulf %v, %scale4 : vector<4xf32>
    %hv = vector.fptrunc %sv : vector<4xf32> to vector<4xf16>
    %c256 = index.constant 256 : index
    %ob0 = index.mul %blk, %c256 : index
    %ob1 = index.add %ob0, %lane : index
    %ob = index.assume %ob1 [range(%ob1, 0, {width // 4 - 1})] : index
    vector.store %hv, %h4_view[%row, %ob, %c0] : vector<4xf16>, view<[%rows]x{width // 4}x4xf16>
    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  }}
  %first = index.cmp eq, %lane, %c0 : index
  scf.if %first {{
    %s_view = buffer.view %scales[%zero] : buffer -> view<[%rows]x2xf32>
    view.store %scale, %s_view[%row, %c0] : f32, view<[%rows]x2xf32>
    view.store %inverse, %s_view[%row, %c1] : f32, view<[%rows]x2xf32>
  }}
  kernel.return
}}
"""


def sign_vec32(word, pfx):
    """IR: vector<32xf32> %{pfx}_s of +/-1 from the 32 sign bits of i32 %{word} (bit k = column k of the group)."""
    return f"""  %{pfx}_ws = vector.splat {word} : vector<32xi32>
  %{pfx}_z0 = scalar.constant 0 : i32
  %{pfx}_s1 = scalar.constant 1 : i32
  %{pfx}_k = vector.iota %{pfx}_z0 step %{pfx}_s1 : vector<32xi32>
  %{pfx}_sh = vector.shrui %{pfx}_ws, %{pfx}_k : vector<32xi32>
  %{pfx}_one = vector.splati 1 : vector<32xi32>
  %{pfx}_bit = vector.andi %{pfx}_sh, %{pfx}_one : vector<32xi32>
  %{pfx}_zi = vector.splati 0 : vector<32xi32>
  %{pfx}_neg = vector.cmpi ne, %{pfx}_bit, %{pfx}_zi : vector<32xi32> -> vector<32xi1>
  %{pfx}_pf = vector.splatf 1.0 : vector<32xf32>
  %{pfx}_nf = vector.splatf -1.0 : vector<32xf32>
  %{pfx}_s = vector.select %{pfx}_neg, %{pfx}_nf, %{pfx}_pf : vector<32xf32>
"""


def embed_patch(text, H):
    """The lookup of a Hadamard-latent token embedding (g_embed: one workgroup of 256 lanes per token, lane l < H/32 holding
    values 32l..32l+31 in %values): h = D (H z) for table rows; image feature rows (%is_feat) are in the model basis already."""
    anchor = "  %normalized = vector.addf %values, %zero32 : vector<32xf32>\n"
    m = re.search(r'kernel\.def export\("g_embed"\).*?\nkernel\.return\n?\}', text, re.S) or \
        re.search(r'kernel\.def export\("g_embed"\).*?\n  kernel\.return\n\}', text, re.S)
    assert m and anchor in m.group(0), "ternary.embed_patch: g_embed layout changed"
    nblk, groups = H // BLK, H // 32
    rot = f"""  // prism.hadamard: the table stores rotated rows z; the model row is D (H z) (llama.cpp build_tok)
  %he_bytes = index.constant {H * 4} : offset
  %he_buf = buffer.alloca<workgroup> align(16) %he_bytes : buffer
  %he_v32 = buffer.view %he_buf[%zero] : buffer -> view<{groups}x32xf32>
  scf.if %active {{
    %he_l = index.assume %lane [range(%lane, 0, {groups - 1})] : index
    vector.store %values, %he_v32[%he_l, %c0] : vector<32xf32>, view<{groups}x32xf32>
  }}
{fwht("he_buf", nblk, 256, "hf")}  %rotated = scf.if %active -> (vector<32xf32>) {{
    %hr_l = index.assume %lane [range(%lane, 0, {groups - 1})] : index
    %hr_raw = vector.load %he_v32[%hr_l, %c0] : view<{groups}x32xf32> -> vector<32xf32>
    %hr_inv = vector.splatf 0.03125 : vector<32xf32>
    %hr_n = vector.mulf %hr_raw, %hr_inv : vector<32xf32>
    %hr_tab = global.load @g_had_signs_{H} : buffer
    %hr_tv = buffer.view %hr_tab[%zero] : buffer -> view<{groups}xi32>
    %hr_word = view.load %hr_tv[%hr_l] : view<{groups}xi32> -> i32
{sign_vec32("%hr_word", "hs")}    %hr_v = vector.mulf %hr_n, %hs_s : vector<32xf32>
    scf.yield %hr_v : vector<32xf32>
  }} else {{
    scf.yield %zero32 : vector<32xf32>
  }}
FEATSEL  %normalized = vector.addf %he_values, %zero32 : vector<32xf32>
"""
    rot = rot.replace("FEATSEL", """  %he_featv = vector.splat %is_feat : vector<32xi1>
  %he_values = vector.select %he_featv, %values, %rotated : vector<32xf32>
""" if "%is_feat" in m.group(0) else "  %he_values = vector.addf %rotated, %zero32 : vector<32xf32>\n")
    k = m.group(0).replace(anchor, rot)
    return text[:m.start()] + k + text[m.end():]


_QLAUNCH = re.compile(r"kernel\.launch @(g_row_scale_f16|g_quant_q8)\[[^\]]*\]\((%\w+), (%\w+), (%\w+), (%\w+)(?:, (%\w+))?\)")


def wiring(program, cfg):
    """Rename every activation quantizer whose copy feeds only folded weights to its rotating variant. Per command program: the
    copy a quantizer writes (X16 of g_row_scale_f16, Xq of g_quant_q8) is followed to the launches that read it before the next
    quantizer rewrites it; their weight operands (command.parameter bindings) decide: all folded -> rotate (width = the weights'
    input width; ssm_out -> permuted variant), none folded -> keep, mixed -> error. Returns (program, {(kind, width, perm)})."""
    had = cfg["hadamard"]
    folded = set(had["weights"])
    shapes = {n: t["shape"] for n, t in cfg["tensors"].items()}
    used = set()
    out, pos = [], 0
    for pm in re.finditer(r"command\.def public[^\n]*\{\n.*?\n\}\n", program, re.S):
        out.append(program[pos:pm.start()])
        body = pm.group(0)
        params = dict(re.findall(r"(%\w+) = command\.parameter %\w+, \"([^\"]+)\"", body))
        launches = list(_QLAUNCH.finditer(body))
        edits = []
        for i, q in enumerate(launches):
            kind = q.group(1)
            dst = q.group(6) if kind == "g_row_scale_f16" else q.group(5)
            end = len(body)
            for q2 in launches[i + 1:]:
                d2 = q2.group(6) if q2.group(1) == "g_row_scale_f16" else q2.group(5)
                if d2 == dst:
                    end = q2.start()
                    break
            seg = body[q.end():end]
            names = set()
            for ln in re.findall(r"kernel\.launch @\w+\[[^\]]*\]\(([^)]*)\)", seg):
                args = [a.strip() for a in ln.split(",")]
                if dst in args:
                    names |= {params[a] for a in args if a in params}
            mats = {n for n in names if len(shapes.get(n, [])) == 2}
            if not mats or not (mats & folded):
                continue
            assert mats <= folded, f"ternary.wiring: {dst} feeds folded and unfolded weights: {sorted(mats)[:6]}"
            widths = {shapes[n][0] for n in mats}
            assert len(widths) == 1, f"ternary.wiring: {dst} feeds weights of input widths {widths}"
            w = widths.pop()
            perm = had.get("gdn_v_grouped", False) and any(n.endswith(".ssm_out.weight") for n in mats)
            assert not perm or all(n.endswith(".ssm_out.weight") for n in mats)
            assert str(w) in had["signs"] and w % BLK == 0, f"ternary.wiring: no sign vector / block split for width {w}"
            k = "rs16" if kind == "g_row_scale_f16" else "q8"
            used.add((k, w, perm))
            edits.append((q.start(), q.start() + len(f"kernel.launch @{kind}"), f"kernel.launch @g_had_{k}_w{w}{'p' if perm else ''}"))
        for a, b, new in reversed(edits):
            body = body[:a] + new + body[b:]
        out.append(body)
        pos = pm.end()
    out.append(program[pos:])
    return "".join(out), used


def kernels(cfg, used):
    """Sign tables and the rotating quantizers wiring() launched."""
    gdn = cfg.get("gdn")
    text = signs_rodata(cfg)
    for k, w, perm in sorted(used):
        text += (q8_kernel if k == "q8" else rs16_kernel)(w, perm, gdn)
    return text


def setup(cfg):
    """True when the model is a prism.hadamard checkpoint this module handles (asserts the parts it relies on)."""
    had = cfg.get("hadamard")
    if not had:
        return False
    assert had["block"] == BLK, f"prism.hadamard block {had['block']}: only {BLK} is implemented"
    assert set(had["inverse"]) <= {"token_embd.weight"}
    assert "token_embd.weight" in had["inverse"], "a Hadamard model whose embedding is not latent: not seen yet"
    return True


# ---- numpy references (make_kernel_tests.py) ----
def np_signs(cfg, width):
    import numpy as np
    bits = np.unpackbits(np.frombuffer(bytes.fromhex(cfg["hadamard"]["signs"][str(width)]), np.uint8), bitorder="little")[:width]
    return np.where(bits == 1, -1.0, 1.0)


def np_hadamard(n=BLK):
    """The normalized Sylvester Walsh-Hadamard matrix (llama.cpp load_tensors_hadamard: entry (r, c) = (-1)^popcount(r & c) / sqrt(n))."""
    import numpy as np
    r = np.arange(n)
    par = np.vectorize(lambda v: bin(v).count("1") & 1)(r[:, None] & r[None, :])
    return np.where(par == 1, -1.0, 1.0) / np.sqrt(n)


def np_perm(cfg):
    """Grouped column g -> the tiled source column it reads (ssm_out input with gdn_v_grouped)."""
    import numpy as np
    gd = cfg["gdn"]; hd, nk = gd["head_v"], gd["k_heads"]; rep = gd["v_heads"] // nk
    g = np.arange(hd * nk * rep); d, h = g % hd, g // hd; r, k = h % rep, h // rep
    return d + hd * (k + nk * r)


def np_rotate(x, cfg, width, perm=False):
    """H (D x) per 1024-block of the last axis (x: rows x width), after the grouped permutation when perm."""
    import numpy as np
    x = np.asarray(x, np.float64)
    if perm:
        x = x[:, np_perm(cfg)]
    x = x * np_signs(cfg, width)
    Hm = np_hadamard()
    return (x.reshape(x.shape[0], -1, BLK) @ Hm).reshape(x.shape)


def np_unrotate_embed(z, cfg):
    """D (H z) of embedding rows (the lookup's inverse transform)."""
    import numpy as np
    z = np.asarray(z, np.float64)
    w = z.shape[-1]
    return (z.reshape(z.shape[0], -1, BLK) @ np_hadamard()).reshape(z.shape) * np_signs(cfg, w)


# ---- whole-block decode matvec (choice decode.ptq1_0.mv = "block") ----
def _block_codes(pfx):
    """IR: the 4 groups' int8 codes %{pfx}c0..3 of the block in vector<7xi32> %{pfx}blk (as _icodes, no selects)."""
    return f"""      %{pfx}by = vector.bitcast %{pfx}blk : vector<7xi32> to vector<28xi8>
      %{pfx}a = vector.slice %{pfx}by[0] : vector<28xi8> -> vector<16xi8>
      %{pfx}b8 = vector.slice %{pfx}by[16] : vector<28xi8> -> vector<8xi8>
      %{pfx}h2 = vector.slice %{pfx}by[24] : vector<28xi8> -> vector<2xi8>
      %{pfx}dw = vector.extract %{pfx}blk[6] : vector<7xi32> -> i32
      %{pfx}c16 = scalar.constant 16 : i32
      %{pfx}dsh = scalar.shrui %{pfx}dw, %{pfx}c16 : i32
      %{pfx}d16 = scalar.trunci %{pfx}dsh : i32 to i16
      %{pfx}dh = scalar.bitcast %{pfx}d16 : i16 to f16
      %{pfx}d = scalar.extf %{pfx}dh : f16 to f32
      %{pfx}bb = vector.concat<0> %{pfx}b8, %{pfx}b8 : vector<8xi8>, vector<8xi8> -> vector<16xi8>
      %{pfx}h4 = vector.concat<0> %{pfx}h2, %{pfx}h2 : vector<2xi8>, vector<2xi8> -> vector<4xi8>
      %{pfx}h8 = vector.concat<0> %{pfx}h4, %{pfx}h4 : vector<4xi8>, vector<4xi8> -> vector<8xi8>
      %{pfx}bq = vector.concat<0> %{pfx}b8, %{pfx}h8 : vector<8xi8>, vector<8xi8> -> vector<16xi8>
      %{pfx}x0 = func.call @g_ptq1_0_tcw(%{pfx}a, %tp1) : (vector<16xi8>, vector<8xi16>) -> (vector<16xi8>)
      %{pfx}x1 = func.call @g_ptq1_0_tcw(%{pfx}a, %tp3) : (vector<16xi8>, vector<8xi16>) -> (vector<16xi8>)
      %{pfx}x2 = func.call @g_ptq1_0_tcw(%{pfx}a, %tp9) : (vector<16xi8>, vector<8xi16>) -> (vector<16xi8>)
      %{pfx}x3 = func.call @g_ptq1_0_tcw(%{pfx}a, %tp27) : (vector<16xi8>, vector<8xi16>) -> (vector<16xi8>)
      %{pfx}x4 = func.call @g_ptq1_0_tcw(%{pfx}a, %tp81) : (vector<16xi8>, vector<8xi16>) -> (vector<16xi8>)
      %{pfx}x5 = func.call @g_ptq1_0_tcw(%{pfx}bb, %tp2hi) : (vector<16xi8>, vector<8xi16>) -> (vector<16xi8>)
      %{pfx}x6 = func.call @g_ptq1_0_tcw(%{pfx}bb, %tp3lo) : (vector<16xi8>, vector<8xi16>) -> (vector<16xi8>)
      %{pfx}x7 = func.call @g_ptq1_0_tcw(%{pfx}bq, %tp3hi) : (vector<16xi8>, vector<8xi16>) -> (vector<16xi8>)
      %{pfx}c0 = vector.concat<0> %{pfx}x0, %{pfx}x1 : vector<16xi8>, vector<16xi8> -> vector<32xi8>
      %{pfx}c1 = vector.concat<0> %{pfx}x2, %{pfx}x3 : vector<16xi8>, vector<16xi8> -> vector<32xi8>
      %{pfx}c2 = vector.concat<0> %{pfx}x4, %{pfx}x5 : vector<16xi8>, vector<16xi8> -> vector<32xi8>
      %{pfx}c3 = vector.concat<0> %{pfx}x6, %{pfx}x7 : vector<16xi8>, vector<16xi8> -> vector<32xi8>
"""


_TP = """  %ti1 = scalar.constant 1 : i16
  %ti3 = scalar.constant 3 : i16
  %ti9 = scalar.constant 9 : i16
  %ti27 = scalar.constant 27 : i16
  %ti81 = scalar.constant 81 : i16
  %tp1 = vector.splat %ti1 : vector<8xi16>
  %tp3 = vector.splat %ti3 : vector<8xi16>
  %tp9 = vector.splat %ti9 : vector<8xi16>
  %tp27 = vector.splat %ti27 : vector<8xi16>
  %tp81 = vector.splat %ti81 : vector<8xi16>
  %tp3lo = vector.from_elements %ti9, %ti9, %ti9, %ti9, %ti27, %ti27, %ti27, %ti27 : vector<8xi16>
  %tp2hi = vector.from_elements %ti1, %ti1, %ti1, %ti1, %ti3, %ti3, %ti3, %ti3 : vector<8xi16>
  %tp3hi = vector.from_elements %ti81, %ti81, %ti81, %ti81, %ti1, %ti3, %ti9, %ti27 : vector<8xi16>
"""


def mv_block(name, K, R, TG, mode="store"):
    """A PTQ1_0 int8-activation matvec with the generic kernel's name, grid and arguments (gen_llm._mv, iq): each lane decodes
    whole 128-value blocks (one 28-byte load, no divergence, no re-read of the block by four lanes), W waves take the R rows in
    turn, the lanes of a wave stride over the row's blocks, one subgroup reduction per row and token. mode: store | swiglu
    (silu(x W^T) * (x W2^T)) | pair (y = x W^T, y2 = x W2^T). Every token's sum runs in the same order for any TG (verify ==
    decode)."""
    nb = K // 128
    groups = K // 32
    W = min(R, 4)
    nw = 1 if mode == "store" else 2
    wargs = "%w: buffer" if nw == 1 else "%w: buffer, %w2: buffer"
    yargs = "%y: buffer, %y2: buffer" if mode == "pair" else "%y: buffer"
    accs = [f"%a{wi}_{j}" for wi in range(nw) for j in range(TG)]
    init = ", ".join(f"{a} = %zero_f" for a in accs)
    types = ", ".join("f32" for _ in accs)
    body = []
    for wi, wn in enumerate(["w", "w2"][:nw]):
        body.append(f"""      %wv{wi} = buffer.view %{wn}[%zero] : buffer -> view<[%n]x{nb}x7xi32>
      %b{wi}blk = vector.load %wv{wi}[%row, %bk, %c0] : view<[%n]x{nb}x7xi32> -> vector<7xi32>
""" + _block_codes(f"b{wi}"))
    for j in range(TG):
        body.append(f"""      %tr{j} = index.add %token0, %cj{j} : index
      %tl{j} = index.cmp ult, %tr{j}, %tokens : index
      %q{j}, %m{j} = scf.if %tl{j} -> (vector<32xi32>, vector<8xf32>) {{
        %tk{j} = index.assume %tr{j} [lt(%tr{j}, %tokens)] : index
        %qv = vector.load %xq_view[%tk{j}, %bk, %c0] : view<[%tokens]x{nb}x32xi32> -> vector<32xi32>
        %mv = vector.load %xm_view[%tk{j}, %bk, %c0] : view<[%tokens]x{nb}x8xf32> -> vector<8xf32>
        scf.yield %qv, %mv : vector<32xi32>, vector<8xf32>
      }} else {{
        scf.yield %zero32i, %zero8f : vector<32xi32>, vector<8xf32>
      }}
      %qb{j} = vector.bitcast %q{j} : vector<32xi32> to vector<128xi8>""")
        for g in range(4):
            body.append(f"""      %qg{j}_{g} = vector.slice %qb{j}[{32 * g}] : vector<128xi8> -> vector<32xi8>
      %da{j}_{g} = vector.extract %m{j}[{2 * g}] : vector<8xf32> -> f32
      %xs{j}_{g} = vector.extract %m{j}[{2 * g + 1}] : vector<8xf32> -> f32""")
        for wi in range(nw):
            terms = []
            for g in range(4):
                body.append(f"""      %dp{wi}_{j}_{g} = vector.dot4i<s8s8> %b{wi}c{g}, %qg{j}_{g}, %zero8i : vector<32xi8>, vector<32xi8>, vector<8xi32>
      %di{wi}_{j}_{g} = vector.reduce<addi> %dp{wi}_{j}_{g}, %zero_i : vector<8xi32>, i32
      %df{wi}_{j}_{g} = scalar.sitofp %di{wi}_{j}_{g} : i32 to f32
      %dc{wi}_{j}_{g} = scalar.subf %df{wi}_{j}_{g}, %xs{j}_{g} : f32
      %ds{wi}_{j}_{g} = scalar.mulf %dc{wi}_{j}_{g}, %da{j}_{g} : f32""")
                terms.append(f"%ds{wi}_{j}_{g}")
            body.append(f"""      %s01_{wi}_{j} = scalar.addf {terms[0]}, {terms[1]} : f32
      %s23_{wi}_{j} = scalar.addf {terms[2]}, {terms[3]} : f32
      %sb_{wi}_{j} = scalar.addf %s01_{wi}_{j}, %s23_{wi}_{j} : f32
      %na{wi}_{j} = scalar.fmaf %sb_{wi}_{j}, %b{wi}d, %a{wi}_{j} : f32""")
    yields = ", ".join(f"%na{wi}_{j}" for wi in range(nw) for j in range(TG))
    outs = []
    for j in range(TG):
        red = [f"""    %r{wi}_{j} = kernel.subgroup.reduce<addf> %res{wi}_{j} : f32""" for wi in range(nw)]
        if mode == "swiglu":
            val = f"""      %silu{j} = scalar.siluf<afn> %r0_{j} : f32
      %v{j} = scalar.mulf %silu{j}, %r1_{j} : f32
      func.call @g_store_row(%y, %tokens, %n, %tr{j}o, %row, %v{j}, %accumulate) : (buffer, index, index, index, index, f32, i32)"""
        elif mode == "pair":
            val = f"""      func.call @g_store_row(%y, %tokens, %n, %tr{j}o, %row, %r0_{j}, %accumulate) : (buffer, index, index, index, index, f32, i32)
      func.call @g_store_row(%y2, %tokens, %n, %tr{j}o, %row, %r1_{j}, %accumulate) : (buffer, index, index, index, index, f32, i32)"""
        else:
            val = f"""      func.call @g_store_row(%y, %tokens, %n, %tr{j}o, %row, %r0_{j}, %accumulate) : (buffer, index, index, index, index, f32, i32)"""
        outs.append("\n".join(red) + f"""
    %tr{j}o = index.add %token0, %cj{j} : index
    %tl{j}o = index.cmp ult, %tr{j}o, %tokens : index
    %st{j} = scalar.andi %tl{j}o, %first : i1
    scf.if %st{j} {{
{val}
    }}""")
    res = ", ".join(f"%res{wi}_{j}" for wi in range(nw) for j in range(TG))
    cj = "\n".join(f"  %cj{j} = index.constant {j} : index" for j in range(TG))
    return f"""
kernel.def export("{name}") @{name}(%n: index, %tokens: index) {{
  %c1 = index.constant 1 : index
  %lanes = index.constant {32 * W} : index
  %rows_per = index.constant {R} : index
  %tg = index.constant {TG} : index
  %tg_less = index.constant {TG - 1} : index
  %row_groups = index.div %n, %rows_per : index
  %tg_pad = index.add %tokens, %tg_less : index
  %token_groups = index.div %tg_pad, %tg : index
  kernel.launch.config workgroups(%token_groups, %row_groups, %c1) workgroup_size(%lanes, %c1, %c1) : index
}} launch(%n_arg: index, %tokens_arg: index, %accumulate: i32, %x: buffer, {wargs}, {yargs}) {{
  %n = index.assume %n_arg [range(%n_arg, {R}, 262144), multiple_of(%n_arg, {R})] : index
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %group_id = kernel.workgroup.id<y> : index
  %tgroup_id = kernel.workgroup.id<x> : index
  %wave_id = kernel.subgroup.id : index
  %wave = index.assume %wave_id [range(%wave_id, 0, {W - 1})] : index
  %lane_id = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c32 = index.constant 32 : index
  %cW = index.constant {W} : index
  %rows_per = index.constant {R} : index
  %tg = index.constant {TG} : index
  %nbk = index.constant {nb} : index
  %zero_f = scalar.constant 0.0 : f32
  %zero_i = scalar.constant 0 : i32
  %zero8i = vector.splati 0 : vector<8xi32>
  %zero32i = vector.splati 0 : vector<32xi32>
  %zero8f = vector.splatf 0.0 : vector<8xf32>
{_TP}{cj}
  %xq_view = buffer.view %x[%zero] : buffer -> view<[%tokens]x{nb}x32xi32>
  %xm_k = index.constant {K} : offset
  %xm_off = index.scale %tokens, %xm_k : index, offset -> offset
  %xm_view = buffer.view %x[%xm_off] : buffer -> view<[%tokens]x{nb}x8xf32>
  %row_base = index.mul %group_id, %rows_per : index
  %token0 = index.mul %tgroup_id, %tg : index
  %first = index.cmp eq, %lane, %c0 : index
  scf.for %rr = [%wave to %rows_per step %cW] {{
    %row_v = index.add %row_base, %rr : index
    %row = index.assume %row_v [lt(%row_v, %n)] : index
    {res} = scf.for %bk0 = [%lane to %nbk step %c32]({init} : {types}) -> ({types}) {{
      %bk = index.assume %bk0 [range(%bk0, 0, {nb - 1})] : index
{chr(10).join(body)}
      scf.yield {yields} : {types}
    }}
{chr(10).join(outs)}
  }}
  kernel.return
}}
"""


def use_block_mv(text):
    """Replace every generic PTQ1_0 int8 matvec definition (g_mv_ptq1_0_k<K>[_swiglu|_pair][_t<TG>]_q8[_r<R>], rdna3 and rdna4
    copies) by mv_block() of the same name. Returns (text, count)."""
    pat = re.compile(r'kernel\.def (target\(@\w+\) )?export\("(g_mv_ptq1_0_k(\d+)(_swiglu|_pair)?(?:_t(\d+))?_q8(?:_r(\d+))?(_rdna4)?)"\).*?\n  kernel\.return\n\}\n', re.S)
    n = [0]

    def sub(m):
        tgt, name, K, md, TG, R, _ = m.groups()
        K, TG = int(K), int(TG or 1)
        R = int(R or 16)
        mode = {"_swiglu": "swiglu", "_pair": "pair"}.get(md, "store")
        k = mv_block(name, K, R, TG, mode)
        if tgt:
            k = k.replace("\nkernel.def export(", f"\nkernel.def {tgt}export(", 1)
        n[0] += 1
        return k.lstrip("\n")
    return pat.sub(sub, text), n[0]


# ---- table decode (choice decode.ptq1_0.mv = "lut") ----
# A 256-entry workgroup table maps a byte to its trits 0..3 (one byte each, codes 0..2) plus a 256-byte table of trit 4: a 16-byte
# run takes two gathers, a 4x4 byte transpose per dword (vector.shuffle) puts the 4 bytes' trit n into one dword, about 1 VALU op
# per weight instead of the 16-bit lane arithmetic's 3 (mv_block).
LUT_INIT = """  %lut_bytes = index.constant 1024 : offset
  %lut = buffer.alloca<workgroup> align(16) %lut_bytes : buffer
  %lut4_bytes = index.constant 256 : offset
  %lut4 = buffer.alloca<workgroup> align(16) %lut4_bytes : buffer
  %lut_v = buffer.view %lut[%zero] : buffer -> view<256xi32>
  %lut4_v = buffer.view %lut4[%zero] : buffer -> view<256xi8>
  %lc256 = index.constant 256 : index
  %lc1 = index.constant 1 : index
  scf.for %le0 = [%ltid to %lc256 step %lstep] {
    %le = index.assume %le0 [range(%le0, 0, 255)] : index
    %lq = index.cast %le : index to i32
    %l255 = scalar.constant 255 : i32
    %l3 = scalar.constant 3 : i32
    %l8 = scalar.constant 8 : i32
    %lm1 = scalar.constant 3 : i32
    %lm2 = scalar.constant 9 : i32
    %lm3 = scalar.constant 27 : i32
    %lm4 = scalar.constant 81 : i32
    %lq1 = scalar.muli %lq, %lm1 : i32
    %lq2 = scalar.muli %lq, %lm2 : i32
    %lq3 = scalar.muli %lq, %lm3 : i32
    %lq4 = scalar.muli %lq, %lm4 : i32
    %lr0a = scalar.andi %lq, %l255 : i32
    %lr1a = scalar.andi %lq1, %l255 : i32
    %lr2a = scalar.andi %lq2, %l255 : i32
    %lr3a = scalar.andi %lq3, %l255 : i32
    %lr4a = scalar.andi %lq4, %l255 : i32
    %lr0b = scalar.muli %lr0a, %l3 : i32
    %lr1b = scalar.muli %lr1a, %l3 : i32
    %lr2b = scalar.muli %lr2a, %l3 : i32
    %lr3b = scalar.muli %lr3a, %l3 : i32
    %lr4b = scalar.muli %lr4a, %l3 : i32
    %lt0 = scalar.shrui %lr0b, %l8 : i32
    %lt1 = scalar.shrui %lr1b, %l8 : i32
    %lt2 = scalar.shrui %lr2b, %l8 : i32
    %lt3 = scalar.shrui %lr3b, %l8 : i32
    %lt4 = scalar.shrui %lr4b, %l8 : i32
    %l16 = scalar.constant 16 : i32
    %l24 = scalar.constant 24 : i32
    %lt1s = scalar.shli %lt1, %l8 : i32
    %lt2s = scalar.shli %lt2, %l16 : i32
    %lt3s = scalar.shli %lt3, %l24 : i32
    %lp01 = scalar.ori %lt0, %lt1s : i32
    %lp23 = scalar.ori %lt2s, %lt3s : i32
    %lp = scalar.ori %lp01, %lp23 : i32
    view.store %lp, %lut_v[%le] : i32, view<256xi32>
    %lt4b = scalar.trunci %lt4 : i32 to i8
    view.store %lt4b, %lut4_v[%le] : i8, view<256xi8>
  }
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
"""

_T44 = "[0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]"


def _lut_codes(pfx):
    """IR: the 4 groups' codes (0..2) %{pfx}c0..3 and scale %{pfx}d of the block vector<7xi32> %{pfx}blk via the tables."""
    return f"""      %{pfx}by = vector.bitcast %{pfx}blk : vector<7xi32> to vector<28xi8>
      %{pfx}a = vector.slice %{pfx}by[0] : vector<28xi8> -> vector<16xi8>
      %{pfx}b8 = vector.slice %{pfx}by[16] : vector<28xi8> -> vector<8xi8>
      %{pfx}h2 = vector.slice %{pfx}by[24] : vector<28xi8> -> vector<2xi8>
      %{pfx}dw = vector.extract %{pfx}blk[6] : vector<7xi32> -> i32
      %{pfx}c16 = scalar.constant 16 : i32
      %{pfx}dsh = scalar.shrui %{pfx}dw, %{pfx}c16 : i32
      %{pfx}d16 = scalar.trunci %{pfx}dsh : i32 to i16
      %{pfx}dh = scalar.bitcast %{pfx}d16 : i16 to f16
      %{pfx}d = scalar.extf %{pfx}dh : f16 to f32
      %{pfx}ai = vector.extui %{pfx}a : vector<16xi8> to vector<16xi32>
      %{pfx}bi = vector.extui %{pfx}b8 : vector<8xi8> to vector<8xi32>
      %{pfx}hi = vector.extui %{pfx}h2 : vector<2xi8> to vector<2xi32>
      %{pfx}al = vector.gather %lut_v[0][%{pfx}ai] : view<256xi32>, vector<16xi32> -> vector<16xi32>
      %{pfx}a4 = vector.gather %lut4_v[0][%{pfx}ai] : view<256xi8>, vector<16xi32> -> vector<16xi8>
      %{pfx}bl = vector.gather %lut_v[0][%{pfx}bi] : view<256xi32>, vector<8xi32> -> vector<8xi32>
      %{pfx}b4 = vector.gather %lut4_v[0][%{pfx}bi] : view<256xi8>, vector<8xi32> -> vector<8xi8>
      %{pfx}hl = vector.gather %lut_v[0][%{pfx}hi] : view<256xi32>, vector<2xi32> -> vector<2xi32>
      %{pfx}ab = vector.bitcast %{pfx}al : vector<16xi32> to vector<64xi8>
      %{pfx}bbb = vector.bitcast %{pfx}bl : vector<8xi32> to vector<32xi8>
      %{pfx}hb = vector.bitcast %{pfx}hl : vector<2xi32> to vector<8xi8>
""" + "".join(f"""      %{pfx}ak{k} = vector.slice %{pfx}ab[{16 * k}] : vector<64xi8> -> vector<16xi8>
      %{pfx}at{k} = vector.shuffle<{_T44}> %{pfx}ak{k} : vector<16xi8>
""" for k in range(4)) + "".join(f"""      %{pfx}bk{k} = vector.slice %{pfx}bbb[{16 * k}] : vector<32xi8> -> vector<16xi8>
      %{pfx}bt{k} = vector.shuffle<{_T44}> %{pfx}bk{k} : vector<16xi8>
""" for k in range(2)) + f"""      %{pfx}ht = vector.shuffle<[0, 4, 1, 5, 2, 6, 3, 7]> %{pfx}hb : vector<8xi8>
""" + "".join(f"""      %{pfx}aw{k} = vector.bitcast %{pfx}at{k} : vector<16xi8> to vector<4xi32>
""" for k in range(4)) + "".join(f"""      %{pfx}bw{k} = vector.bitcast %{pfx}bt{k} : vector<16xi8> to vector<4xi32>
""" for k in range(2)) + f"""      %{pfx}hw = vector.bitcast %{pfx}ht : vector<8xi8> to vector<2xi32>
      %{pfx}a4w = vector.bitcast %{pfx}a4 : vector<16xi8> to vector<4xi32>
      %{pfx}b4w = vector.bitcast %{pfx}b4 : vector<8xi8> to vector<2xi32>
""" + _lut_groups(pfx)


def _lut_groups(pfx):
    """Assemble the 4 groups (8 dwords each) from the transposed dwords: at{k}[n] = trit n of A bytes 4k..4k+3, bt{k}[n] = trit n of
    B bytes 4k..4k+3, a4w / b4w = trit 4, hw = the qh values (120..127)."""
    def d(src, i):
        return f"%{pfx}{src}_e{i}"
    lines, names = [], {}
    for k in range(4):
        for n in range(4):
            nm = f"%{pfx}A{n}_{k}"
            lines.append(f"      {nm} = vector.extract %{pfx}aw{k}[{n}] : vector<4xi32> -> i32")
            names[("A", n, k)] = nm
        nm = f"%{pfx}A4_{k}"
        lines.append(f"      {nm} = vector.extract %{pfx}a4w[{k}] : vector<4xi32> -> i32")
        names[("A", 4, k)] = nm
    for k in range(2):
        for n in range(4):
            nm = f"%{pfx}B{n}_{k}"
            lines.append(f"      {nm} = vector.extract %{pfx}bw{k}[{n}] : vector<4xi32> -> i32")
            names[("B", n, k)] = nm
        nm = f"%{pfx}B4_{k}"
        lines.append(f"      {nm} = vector.extract %{pfx}b4w[{k}] : vector<2xi32> -> i32")
        names[("B", 4, k)] = nm
    for k in range(2):
        nm = f"%{pfx}Q_{k}"
        lines.append(f"      {nm} = vector.extract %{pfx}hw[{k}] : vector<2xi32> -> i32")
        names[("Q", 0, k)] = nm
    order = [[("A", 0, k) for k in range(4)] + [("A", 1, k) for k in range(4)],
             [("A", 2, k) for k in range(4)] + [("A", 3, k) for k in range(4)],
             [("A", 4, k) for k in range(4)] + [("B", 0, k) for k in range(2)] + [("B", 1, k) for k in range(2)],
             [("B", 2, k) for k in range(2)] + [("B", 3, k) for k in range(2)] + [("B", 4, k) for k in range(2)] + [("Q", 0, k) for k in range(2)]]
    for g, ks in enumerate(order):
        lines.append(f"      %{pfx}cw{g} = vector.from_elements {', '.join(names[x] for x in ks)} : vector<8xi32>")
        lines.append(f"      %{pfx}c{g} = vector.bitcast %{pfx}cw{g} : vector<8xi32> to vector<32xi8>")
    return "\n".join(lines) + "\n"


def use_lut_mv(text):
    """mv_block kernels with the table decode: same names (see use_block_mv), the tables filled once per workgroup."""
    text, n = use_block_mv(text)
    pat = re.compile(r'(kernel\.def (?:target\(@\w+\) )?export\("g_mv_ptq1_0_k\d+[^"]*"\).*?\n  kernel\.return\n\}\n)', re.S)

    def sub(m):
        k = m.group(1)
        k = re.sub(r"(      %b(\d)by = vector\.bitcast %b\2blk.*?%b\2c3 = vector\.concat<0>[^\n]*\n)",
                   lambda mm: _lut_codes(f"b{mm.group(2)}"), k, flags=re.S)
        k = k.replace("  %first = index.cmp eq, %lane, %c0 : index\n",
                      "  %first = index.cmp eq, %lane, %c0 : index\n  %lw32 = index.mul %wave, %c32 : index\n  %ltid = index.add %lw32, %lane : index\n"
                      f"  %lstep = index.mul %cW, %c32 : index\n{LUT_INIT}", 1)
        return k
    return pat.sub(sub, text), n


# ---- load-time repack Q2_0 -> PTQ1_0 (weight policy pair ("q2_0x", "ptq1_0"), choice decode.ternary.repack) ----
def prep_kernel():
    """g_prep_q2_0x_ptq1_0 + its stage program: one lane per 128 values (two Q2_0 blocks, 36 bytes) -> one PTQ1_0 block (28 bytes),
    quantize_row_ptq1_0_ref's packing (gguf_ext.q2_0_to_ptq1_0 is the oracle); the scale is the first half's unless that half is all 0."""
    def trit(v):   # IR: %t{v} (i32, trit + 1) of value v (static) of the block
        h, j = divmod(v, 64)
        return (f"    %tb{v} = view.load %iv[{2 + 18 * h + j // 4}] : view<36xi8> -> i8\n"
                f"    %tu{v} = scalar.extui %tb{v} : i8 to i32\n"
                f"    %ts{v} = scalar.constant {2 * (j % 4)} : i32\n"
                f"    %tr{v} = scalar.shrui %tu{v}, %ts{v} : i32\n"
                f"    %t{v} = scalar.andi %tr{v}, %c3i : i32\n")
    lines = [trit(v) for v in range(128)]
    def byte(o, vals, extra):
        acc = "%z"
        out = []
        for k, v in enumerate(vals):
            out.append(f"    %p{o}_{k}a = scalar.muli {acc}, %c3i : i32\n    %p{o}_{k} = scalar.addi %p{o}_{k}a, %t{v} : i32\n")
            acc = f"%p{o}_{k}"
        if extra:
            out.append(f"    %p{o}_x = scalar.muli {acc}, %c3i : i32\n"); acc = f"%p{o}_x"
        out.append(f"    %q{o}a = scalar.muli {acc}, %c256 : i32\n    %q{o}b = scalar.addi %q{o}a, %c242 : i32\n"
                   f"    %q{o}c = scalar.divui %q{o}b, %c243 : i32\n    %q{o} = scalar.trunci %q{o}c : i32 to i8\n"
                   f"    view.store %q{o}, %ov[{o}] : i8, view<28xi8>\n")
        return "".join(out)
    for m in range(16):
        lines.append(byte(m, [m + 16 * n for n in range(5)], False))
    for m in range(8):
        lines.append(byte(16 + m, [80 + m + 8 * n for n in range(5)], False))
    for h in range(2):
        lines.append(byte(24 + h, [120 + h + 2 * n for n in range(4)], True))
    nz = ["%nz_init"]
    for v in range(64):
        lines.append(f"    %ne{v} = scalar.cmpi ne, %t{v}, %c1i : i32\n    %nz{v} = scalar.ori {nz[-1]}, %ne{v} : i1\n")
        nz.append(f"%nz{v}")
    return f"""
// Load-time repack (ternary.prep_kernel): staged Q2_0 ternary tensor -> PTQ1_0 blocks, one lane per 128 values.
kernel.def export("g_prep_q2_0x_ptq1_0") @g_prep_q2_0x_ptq1_0(%blocks: index) {{
  %c1 = index.constant 1 : index
  %c63 = index.constant 63 : index
  %c64 = index.constant 64 : index
  %rounded = index.add %blocks, %c63 : index
  %groups = index.div %rounded, %c64 : index
  kernel.launch.config workgroups(%groups, %c1, %c1) workgroup_size(%c64, %c1, %c1) : index
}} launch(%blocks_arg: index, %in64: i64, %output: buffer, %input: buffer) {{
  %blocks = index.assume %blocks_arg [range(%blocks_arg, 1, 1073741824)] : index
  %in_bound = scalar.assume %in64 [range(%in64, 0, 9223372036854775807)] : i64
  %in_origin = index.cast %in_bound : i64 to offset
  %c64 = index.constant 64 : index
  %wg = kernel.workgroup.id<x> : index
  %lane = kernel.workitem.id<x> : index
  %base = index.mul %wg, %c64 : index
  %g = index.add %base, %lane : index
  %live = index.cmp ult, %g, %blocks : index
  scf.if %live {{
    %ib = index.constant 36 : offset
    %io = index.scale %g, %ib : index, offset -> offset
    %ia = index.add %in_origin, %io : offset
    %iv = buffer.view %input[%ia] : buffer -> view<36xi8>
    %ob = index.constant 28 : offset
    %oo = index.scale %g, %ob : index, offset -> offset
    %ov = buffer.view %output[%oo] : buffer -> view<28xi8>
    %z = scalar.constant 0 : i32
    %c1i = scalar.constant 1 : i32
    %c3i = scalar.constant 3 : i32
    %c256 = scalar.constant 256 : i32
    %c242 = scalar.constant 242 : i32
    %c243 = scalar.constant 243 : i32
    %nz_init = scalar.constant false : i1
{"".join(lines)}    %d0a = view.load %iv[0] : view<36xi8> -> i8
    %d0b = view.load %iv[1] : view<36xi8> -> i8
    %d1a = view.load %iv[18] : view<36xi8> -> i8
    %d1b = view.load %iv[19] : view<36xi8> -> i8
    %da = scf.select {nz[-1]}, %d0a, %d1a : i8
    %db = scf.select {nz[-1]}, %d0b, %d1b : i8
    view.store %da, %ov[26] : i8, view<28xi8>
    view.store %db, %ov[27] : i8, view<28xi8>
  }}
  kernel.return
}}

command.def public @llm_prep_q2_0x_ptq1_0() launch(%output: buffer, %input: buffer) {{
  %b64 = config.get @llm.prep_blocks : i64
  %blocks = index.cast %b64 : i64 to index
  %in64 = config.get @llm.prep_input : i64
  kernel.launch @g_prep_q2_0x_ptq1_0[%blocks](%blocks, %in64, %output, %input) : [index](index, i64, buffer, buffer)
  command.return
}}
"""


# ---- register / subgroup FWHT (choice decode.hadamard.fast, env HAD_FAST=1) ----
def fwht_fast(pfx="ff"):
    """IR: normalized-free 1024-point FWHT of the block held as 4 consecutive values per lane (%{pfx}x0..3 = elements 4l..4l+3,
    256 lanes = 8 waves of 32): stages h = 1, 2 in registers, h = 4..64 by xor shuffles inside the wave, h = 128..512 through the
    workgroup buffer %buf (3 store / barrier / load rounds). Leaves %{pfx}f0..3."""
    t = []
    x = [f"%{pfx}x{k}" for k in range(4)]
    def bf(a, b, n1, n2):
        t.append(f"  {n1} = scalar.addf {a}, {b} : f32\n  {n2} = scalar.subf {a}, {b} : f32\n")
    bf(x[0], x[1], f"%{pfx}a0", f"%{pfx}a1"); bf(x[2], x[3], f"%{pfx}a2", f"%{pfx}a3")
    bf(f"%{pfx}a0", f"%{pfx}a2", f"%{pfx}b0", f"%{pfx}b2"); bf(f"%{pfx}a1", f"%{pfx}a3", f"%{pfx}b1", f"%{pfx}b3")
    cur = [f"%{pfx}b{k}" for k in range(4)]
    t.append(f"  %{pfx}w32 = scalar.constant 32 : i32\n  %{pfx}lane32 = index.rem %lane, %{pfx}c32 : index\n  %{pfx}li = index.cast %{pfx}lane32 : index to i32\n")
    for s, m in enumerate((1, 2, 4, 8, 16)):
        t.append(f"  %{pfx}m{s} = scalar.constant {m} : i32\n  %{pfx}hb{s}0 = scalar.andi %{pfx}li, %{pfx}m{s} : i32\n"
                 f"  %{pfx}hb{s} = scalar.cmpi ne, %{pfx}hb{s}0, %zero_i : i32\n")
        nxt = []
        for k in range(4):
            c = cur[k]
            t.append(f"  %{pfx}i{s}_{k} = scalar.bitcast {c} : f32 to i32\n"
                     f"  %{pfx}o{s}_{k}, %{pfx}u{s}_{k} = kernel.subgroup.shuffle<xor> %{pfx}i{s}_{k}, %{pfx}m{s}, %{pfx}w32 : i32, i32, i32\n"
                     f"  %{pfx}p{s}_{k} = scalar.bitcast %{pfx}o{s}_{k} : i32 to f32\n"
                     f"  %{pfx}s{s}_{k} = scalar.addf {c}, %{pfx}p{s}_{k} : f32\n"
                     f"  %{pfx}d{s}_{k} = scalar.subf %{pfx}p{s}_{k}, {c} : f32\n"
                     f"  %{pfx}n{s}_{k} = scf.select %{pfx}hb{s}, %{pfx}d{s}_{k}, %{pfx}s{s}_{k} : f32\n")
            nxt.append(f"%{pfx}n{s}_{k}")
        cur = nxt
    t.append(f"  %{pfx}bv = buffer.view %buf[%zero] : buffer -> view<256x4xf32>\n  %{pfx}wv = index.div %lane, %{pfx}c32 : index\n")
    for s, m in enumerate((1, 2, 4)):   # partner wave = wave ^ m (elements i ^ 128 m)
        st = "\n".join(f"  view.store {cur[k]}, %{pfx}bv[%lane, %c{k}x] : f32, view<256x4xf32>" for k in range(4))
        t.append(f"""  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
{st}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %{pfx}W{s} = index.constant {m} : index
  %{pfx}wq{s} = index.div %{pfx}wv, %{pfx}W{s} : index
  %{pfx}wr{s} = index.rem %{pfx}wq{s}, %{pfx}c2 : index
  %{pfx}up{s} = index.cmp ne, %{pfx}wr{s}, %c0 : index
  %{pfx}dl{s} = index.mul %{pfx}W{s}, %{pfx}c32 : index
  %{pfx}pl0{s} = scf.select %{pfx}up{s}, %{pfx}dl{s}, %c0 : index
  %{pfx}pl1{s} = index.add %lane, %{pfx}dl{s} : index
  %{pfx}pl2{s} = index.sub %lane, %{pfx}pl0{s} : index
  %{pfx}plr{s} = scf.select %{pfx}up{s}, %{pfx}pl2{s}, %{pfx}pl1{s} : index
  %{pfx}pl{s} = index.assume %{pfx}plr{s} [range(%{pfx}plr{s}, 0, 255)] : index
""")
        nxt = []
        for k in range(4):
            t.append(f"  %{pfx}q{s}_{k} = view.load %{pfx}bv[%{pfx}pl{s}, %c{k}x] : view<256x4xf32> -> f32\n"
                     f"  %{pfx}S{s}_{k} = scalar.addf {cur[k]}, %{pfx}q{s}_{k} : f32\n"
                     f"  %{pfx}D{s}_{k} = scalar.subf %{pfx}q{s}_{k}, {cur[k]} : f32\n"
                     f"  %{pfx}N{s}_{k} = scf.select %{pfx}up{s}, %{pfx}D{s}_{k}, %{pfx}S{s}_{k} : f32\n")
            nxt.append(f"%{pfx}N{s}_{k}")
        cur = nxt
    st = "\n".join(f"  view.store {cur[k]}, %{pfx}bv[%lane, %c{k}x] : f32, view<256x4xf32>" for k in range(4))
    pre = f"  %{pfx}c32 = index.constant 32 : index\n  %{pfx}c2 = index.constant 2 : index\n" + "".join(f"  %c{k}x = index.constant {k} : index\n" for k in range(4))
    return pre + "".join(t) + f"  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n{st}\n  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n"


def _gather_regs(width, perm, gdn, pfx="ff"):
    """As _gather, into registers %{pfx}x0..3 (this lane's 4 consecutive inputs of block %blk) instead of the buffer."""
    g = _gather(width, perm, gdn)
    g = g.replace(f"  %buf_v = buffer.view %buf[%zero] : buffer -> view<{BLK}xf32>\n", "")
    body = g[g.index("  scf.for %e"):]
    head = g[:g.index("  scf.for %e")]
    inner = body[body.index("{\n") + 2: body.rindex("  }\n")]
    inner = inner.replace(f"      view.store %xs, %buf_v[%ei] : f32, view<{BLK}xf32>\n", "")
    out = [head]
    for k in range(4):
        out.append(_uniq(inner.replace("%e4 = index.mul %lane, %c4 : index\n      %ei0 = index.add %e4, %e : index",
                                       f"%e4 = index.mul %lane, %c4 : index\n      %ek = index.constant {k} : index\n      %ei0 = index.add %e4, %ek : index"), f"g{k}")
                   + f"      %{pfx}x{k} = scalar.addf %g{k}xs, %zero_f : f32\n")
    return "".join(out)


def use_fast_q8(text, cfg, used):
    """Replace the g_had_q8_w* kernels by the register / shuffle FWHT form (same names and outputs)."""
    gdn = cfg.get("gdn")
    for k, w, perm in sorted(used):
        if k != "q8":
            continue
        name = f"g_had_q8_w{w}{'p' if perm else ''}"
        old = q8_kernel(w, perm, gdn)
        new = old.replace(_gather(w, perm, gdn) + fwht("buf", 1, 256, "fw"), _gather_regs(w, perm, gdn) + fwht_fast())
        assert new != old and old in text, name
        text = text.replace(old, new)
    return text
