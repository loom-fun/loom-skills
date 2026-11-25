# Model: bartowski-Ornith-1.5-122B-loom

Ornith-1.5-122B Q4_K_M (qwen35moe, 256 experts): several expert layers (layer 12 gate/up/down) carry Q4_K block scales `d` that are
f16-subnormal (<= 5.6e-5). The grouped MoE GEMMs' default dequant rounds the group's scale and minimum to f16 and fuses in f16; the
minimum's rounding error is shared by the group's 32 weights, so prefill outputs drift (the `moe_gemm_12` kernel test fails its f64
ggml oracle, 0.2.0 and 0.3.0 alike, and prefill logits move against llama.cpp). `prefill.gemm.kq_deq_f32` (on for this package)
dequantizes in f32 and rounds each weight once: the ideal f16 weights. Strix Halo only (the weights do not fit the R9700).
