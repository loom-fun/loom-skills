# Exact MXFP4 operand producer pilot

This experiment uses current generated H5120/no-postnorm producer templates and
the same reusable source helpers intended for any later product change. Default
program generation does not call the new helpers. It neither activates rejected
GEMM schedules nor changes weight/activation quantization.

Run only on the campaign-authorized single R9700 through DevLab, with
`.github/devlab/mxfp4-producers-kbench.json`, suite `mxfp4_producers`. One proposed
kernel round is estimated at <=15 runner minutes, excluding queue. The lead
reviews the frozen source and owns dispatch. No whole-model trial for a loser.

Five arms distinguish generic vs fixed-K row scale, moving existing F16
arithmetic into its existing flag guard, retaining input loads across max
reduction, and postnorm's dead F32 sidecar. `held` uses held row-scale inputs and
postnorm WG160+half guard; `h32` shares the held row kernel but uses postnorm
WG256+dead-h32+half guard. Do not infer one mechanism's gain from a combined
summary; compare the separate case identities. The postnorm base/fixed arms are
intentionally identical controls. A future compound WG160+h32 is not measured
here. WG160 requires bit equality despite a smaller reduction workgroup.

There are 24 positive whole-output cases: 12 row-scale (three real widths,
8/256 rows, both F16 flag values), and 12 postnorm (8/256/257 rows, both flags,
constant/random input). All five arms run for each case. Row payloads/scales and
F16 outputs use an independent RNE codebook/power-of-two oracle, with dead,
zero, tiny and large rows. Postnorm residual and normalized H use an independent
float64 reference; every functional output/payload also compares exactly to the
baseline. Constant postnorm cases additionally independently check quantized
payloads/scales. Random postnorm quantized payloads use exact old/new comparison
because independent rsqrt rounding can straddle an FP8 boundary. Every arm has
its own mutable residual input allocation; padding sentinels are checked.

The sidecar h32 may remain sentinel only at rows divisible by32 with F16 off.
All F16 and 257-row fallback cases require the actual h32 output. This isolated producer benchmark explicitly selects the paired-AB-FP8
precondition; it does not validate any whole-model reader graph. Product wiring
still requires a complete actual-reader audit, preserved
public/head overwrite, and numerical/model gates. Unknown reader configurations
must retain h32. Four separate strict EXPECT/006 controls detect a missing half
write, wrong FP8 scale, an unsafe h32 omission in F16 mode, and an unsafe tail
omission. Compilation errors or XPASS are not accepted negative proof.

Timing is 8 separate cases per arm: actual rows1024/2048 for widths5120,6144,
17408 and H5120 postnorm, four interleaved repeats. Analyze raw case JSON and
zero dropped records; retain all per-row host and summed device timing. The
real clean producer budget is0.026700ms/actual prompt token from accepted
profile44177367771 (row0.012816/post0.013884), about8.9% GPU busy. This is an
attribution ceiling, not a promised speedup. Postnorm dead-h32's ideal write
removal is2.60MB/token, ~0.00424ms/token at the separately measured bandwidth;
actual cache/transaction interaction and rowhold register pressure require
measurement. Numerical success alone is not adoption or support evidence.
