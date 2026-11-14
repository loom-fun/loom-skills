# Strix Halo: prefill

- **...and through constant data.** Strix Halo's GPU is power-limited: WMMA on constant operands runs
  52–57 TOPS, on random operands ~40 (wave32) / ~43 (wave64), for f16 and int8 alike. A prefill GEMM fed
  `check.generate.fill` constants read 36 TFLOP/s; with the GGUF's real weights and random activations
  30 (the model measured 28). Feed compute-bound benchmarks real weights (`check.file.read.npy`) and
  random activations before trusting a kernel A/B.
