# Topic: kernels-ir (Loom IR cookbook for model ports)

The upstream language guide is in hrx-system at `loom/docs/src/guide/*.md`; read
`kernels-and-launch`, `buffers-views-memory`, `facts-and-specialization`,
`vectors-and-structured-compute`, and `command-programs`. This file collects what
mattered in practice when writing LLM kernels. Proof facts, diagnostics, kernel tests and the
compile errors hit most often (with fixes) are topic `ir-debug`.

## Shape of a kernel

```
kernel.def export("g_name") @g_name(%n: index, %tokens: index) {          // grid-shape args
  %c1 = index.constant 1 : index
  %c32 = index.constant 32 : index
  kernel.launch.config workgroups(%n, %tokens, %c1) workgroup_size(%c32, %c1, %c1) : index
} launch(%n_arg: index, %tokens_arg: index, %x: buffer, %w: buffer, %y: buffer) {   // runtime args
  %n = index.assume %n_arg [range(%n_arg, 1, 65536)] : index                       // FACTS: every bound you index with
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %row_id = kernel.workgroup.id<x> : index
  %row = index.assume %row_id [lt(%row_id, %n)] : index
  %lane = kernel.workitem.id<x> : index            // also kernel.subgroup.id / kernel.subgroup.lane.id
  %zero = index.constant 0 : offset
  %x_view = buffer.view %x[%zero] : buffer -> view<[%tokens]x[%groups]x32xf32>
  ...
  kernel.return
}
```

Launching from a command program:
```
kernel.launch @g_name[%n, %t](%n, %t, %x, %w, %y) : [index, index](index, index, buffer, buffer, buffer)
```
- The `[...]` operands feed the grid-shape args. The `(...)` types must match the
  `launch(...)` signature exactly; for weights that is
  `view<BYTESxi8>` from `command.parameter`.
- **Indirect dispatch**: declare the kernel as `@k(%count_buffer: buffer)`, read the
  count with `view.load` in the config region, and launch it as
  `kernel.launch @k[%count_buf](...) : [buffer](...)`. The fallback-attention token
  list uses this.
- **Weights**: `%w = command.parameter %weights, "blk.3.attn_q.weight" : view<Nxi8>`.
  N must equal the tensor's byte size exactly; `gen_llm.tensor_bytes()` computes it.
  Indexed names work as well (`"blk.{}.attn_q.weight"[%layer]`), but every layer then
  needs the same size, which mixed Q4_K_M breaks. Use a static per-layer unroll.
- `config.decl @llm.tokens : %value: index where [range(...)]` declares a stage
  constant that `prepare.loom` sets per packing shape (`config_i64`). Read it with
  `config.get`.

## Idioms that compile to fast code on RDNA3

- **Matvec, one lane per 32-value group.** `vector.load` 32 f32 of x into registers,
  dequantize the 32 weights, then
  `vector.dotf<reassoc|contract> %w, %x, %zero_f`, then
  `kernel.subgroup.reduce<addf>`. Fold Q4_K scale/min per group:
  `d·dot(codes,x) − m·sum(x)`.
- **WMMA.** Use `vector.fragment<lhs|rhs|init>`, `vector.fragment.load<lhs>`, and
  `vector.mma` on f16 operands with f32 accumulation (16×16×16).
  - RDNA3 result layout: element i of lane l is row `2i + l/16`, column `l % 16`.
  - The transposed form (Sᵀ = K·Qᵀ) makes per-query softmax lane-local.
- **Workgroup memory.** Use `buffer.alloca<workgroup> align(16) <const> : buffer` and
  `kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)`.
- **Atomics.** `view.atomic.rmw<addi> %v, %view[...] {ordering = relaxed, scope = workgroup}`.
- **Cross-lane.** `kernel.subgroup.reduce<addf|maxnumf>`, and shuffles with
  `shuffle<xor>` (an xor-16 shuffle swaps the two half-waves).
- **Math.**
  - `scalar.siluf<afn>`, `vector.logisticf<afn>`, `scalar.rsqrtf`, `scalar.expf<afn>`,
    `scalar.cosf<afn>`, `scalar.sinf<afn>`.
  - Range-reduce RoPE angles (subtract a multiple of 2π) before `cosf`/`sinf<afn>`.
  - Use `scalar.fptrunc` for f16 KV writes.
- **Loops.** `scf.for %i = [%a to %b step %s](%acc = %init : f32) -> (f32) { … scf.yield … }`
  handles loop-carried values. Add `unroll` for static trip counts.
  - `scf.if %c -> (T) { scf.yield … } else { … }` must yield in both branches.
  - `scf.select` is a select, not a branch.
