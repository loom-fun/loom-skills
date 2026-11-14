# Topic: ir-debug (Loom IR diagnostics, proof facts, kernel tests and compile-error fixes)

## Proof facts (SUBRANGE/*)

Every view access must be provably in bounds, and the compiler's prover only knows
what you tell it.
- `index.assume %v [range(%v, lo, hi)]`, `[lt(%v, %n)]`, `multiple_of(%v, 16)`. Facts
  attach to the *new* SSA value, so use that value afterwards.
- For scalars use `scalar.assume %i32 [range(...)]` and then `index.cast`.
- `SUBRANGE/010` ("full-vector footprint upper bound is not proven") on
  `vector.load`, `vector.store` or `fragment.store`: the prover needs
  `origin + vector_extent <= view_bound`. There are three fixes:
  - **Clamp.** `%c = index.min %i, %last` followed by
    `index.assume %c [range(%c, 0, last)]`.
  - **Guard.** Wrap the access in `scf.if %in_bounds { ... }`. For prefetch of
    step k+1, use `scf.if step+1 < k_steps`.
  - **Reshape.** Make the view's static dims match the access, e.g.
    `view<[%tokens]x16x32xf32>` loaded with `[t, g, 0] -> vector<32xf32>`.
- `SUBRANGE/023` ("origin lower bound … 0 <= origin"): use `scalar.maxsi %x, %zero`,
  or add a `range(…, 0, …)` fact, before the cast.
- Loop induction variables carry no facts. Re-`assume` them inside the body
  (`%bs = index.assume %s [lt(%s, %spans)]`).
- Values read from buffers (token ids, slots, positions) carry no facts at all.
  Clamp them (`maxsi`/`minsi`) and then `scalar.assume`.

## Other diagnostics

| Code | Meaning | Fix |
|---|---|---|
| PARSE/002 "SSA value '%x' is already defined" | Names are unique per function, **including nested regions**. | Suffix per unrolled copy (`%wq_{i}`). Rename in helpers (`%info_view` → `%token_info`). |
| PARSE/001 "undefined SSA value" | Usually an f-string or generator substitution miss (`{VAR}` left in the text, or a constant defined only in one branch). | `assert "{" not in line` after substitution. Define constants at the top of the kernel. |
| PARSE/003 "unexpected token '{'" | An unescaped `{` in an f-string, or a doubled `}}` in a non-f-string. | Keep `{{`/`}}` only inside f-strings. Generators that mix both should post-substitute placeholders. |
| TYPE/001 "type view<[%tokens]x64x32xf32> does not match contract argument" | `func.call` argument view types must match the callee *spelling*: dynamic `[%groups]` vs static `64`. | Build the view with the same dims the callee declares. |
| TYPE/003 "operand has type offset, expected index" | `offset` (bytes) and `index` (elements) are distinct. | `index.scale %i, %bytes : index, offset -> offset`. Use `index.add` on offsets. |
| TARGET/003 "rejected 'index.div' address-width 'u32'" | 32-bit address math failed because the prover could not bound the division. | Add range facts on the operands. |
| TARGET/092 "rejected 'buffer.alloca'" | `buffer.alloca<workgroup>` with a non-constant size, or inside a func that is not inlined. | Use constant sizes and do the alloca in the kernel body. |
| SYMBOL/002 "referenced but not defined" | A helper func lives in another file that is not concatenated or linked. | Include it in the library list (`model.json` `programs.source.libraries`) or the test's `cat`. |
| BACKEND/005 unspillable-register-exhausted | Too many live vectors (NPU especially). | Smaller unroll, fewer simultaneous accumulators. |

## Testing kernels (`check.case`, loom-test)

```
check.case public @mv_q6 {
  %n = check.literal value(1024) : index
  %t = check.literal value(4) : index
  %acc = check.literal value(0) : i32
  %w = check.file.read.npy path("w.npy") : tensor<860160xi8>        // raw GGUF bytes
  %x = check.file.read.npy path("x.npy") : tensor<4x1024xf32>
  %e = check.file.read.npy path("y.npy") : tensor<4x1024xf32>       // numpy reference
  %y = check.generate.fill value(0.0) : tensor<4x1024xf32>
  kernel.launch @g_mv_q6_k_k1024_t4[%n, %t](%n, %t, %acc, %x, %w, %y) : [index, index](index, index, i32, tensor<4x1024xf32>, tensor<860160xi8>, tensor<4x1024xf32>)
  check.expect.approximate actual(%y) expected(%e) atol(2.0e-3) rtol(2.0e-3) nan(different) : tensor<4x1024xf32>, tensor<4x1024xf32>
  check.return
}
```
- Concatenate the kernel library and the cases into one file, then run
  `loom-test file.loom-test --device=amdgpu > out.json`. Each sample in
  `out.json["samples"]` has `case`, `passed` and `expectations.failures[].detail`
  ("element at index 8 (0) is not close to …").
- `check.generate.iota offset(a) step(b)` gives synthetic inputs. To A/B a new kernel
  against the old one, launch both on identical inputs and use `check.expect.approximate`.
  This is how every optimization was verified before the end-to-end golden run.
- For references, use `gguf.quants.dequantize(np.asarray(t.data), t.tensor_type)`;
  shapes are reversed (`t.shape = [K, N]`).
- `loom-run file.loom --root=@kernel --device=amdgpu --kernel-input-buffer=…` runs a
  single kernel. `loom-compile --root=@kernel` compiles one kernel; command programs
  can't be roots.

## Compile errors seen in ports, and their fixes

- **`LOWERING/036` on `scalar.geluf<tanh>`.** The exact tanh doesn't lower. Write
  GELU-tanh as `x · logistic(2·√(2/π)·(x + 0.044715·x³))`, using
  `vector.logisticf<afn>` or exp. It is mathematically identical.
- **`TARGET/058`: a range assume on `workgroup.id` failed.** Do the bounds compare
  (`lt(%row_id, %n)`) and branch first, then use the id inside the branch, where the
  fact holds.
- **"stage parameter root N differs".** A parameter, such as a norm weight, is used
  only in one path (`scf.if` branch or stage). Bind it once in the command program's
  shared params section, so every stage binds the same set.
- **`SUBRANGE/024` in a store helper.** Pass the indices that carry the proven
  bound (the ones you compared against n) to the store, not a recomputed value.
