# Model: bartowski-Ornith-1.5-122B-loom

Ornith-1.5-122B Q4_K_M (qwen35moe, 256 experts): expert layers such as layer 12 carry tiny Q4_K block scales (`d` <= 7.7e-5,
f16-subnormal). The grouped MoE GEMMs' default dequant rounds the group's scale `d * sc` and minimum `dmin * mn` to f16 and fuses
`code * scale - minimum` in f16; where the products are not f16-exact the group's 32 weights share both rounding errors and the
cancelling difference amplifies them: only 81 % (down) to 33 % (expert 255) of layer 12's weights equal ggml's f32 dequant rounded
to f16, prefill outputs drift, and the `moe_gemm_12` kernel test fails its f64 ggml oracle on 0.2.0 and 0.3.0 alike.
`prefill.gemm.kq_zero_point` (on for this package) dequantizes as `d * (code * sc - sc * c0) + f16(c0 * scale - minimum)` in f16
(exact integer `n`, one rounding per weight): 99.97 % equal, the rest one ulp away. Strix Halo only (the weights do not fit the R9700).
