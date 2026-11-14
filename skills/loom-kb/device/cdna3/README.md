# CDNA3 (gfx9xx, MI300) - planned

Wave 64, MFMA instead of WMMA, different LDS and matrix layouts. A regex rewrite of the RDNA3 text cannot produce good MFMA kernels:
each matrix-kernel family (GEMM, flash attention, MoE, matvec) needs its own emitter. `dual_target.py` is written for one base and
one twin family; a third target needs N-target selection first. No device rows yet.
