# Issue1145 split attention experiment

Three generated copies of the actual Qwen2.5 0.5B split kernel: legacy two-comparison mask (1c503ee3), direct unsigned mask (9b88b310), and a prototype uniform full-block branch over the unsigned mask. Shared KV-offset/split-count helpers are unchanged. The prototype only bypasses masks on wholly visible aligned blocks; edge blocks use the exact unsigned path.

Each benchmark case cycles through all24 layers of a384MiB KV arena, exceeding both GPUs' cache capacities. Inputs are uniform nonzero synthetic values; this is an isolation test, and only end-to-end timing can establish a model gain. Five interleaved repetitions, the existing dispatch-complete timing and whole-case aggregation are used.

Thirteen numerical cases span first tokens, unaligned ranges, block boundaries and 32k context. Each variant must match a nonzero analytic partial-accumulator reference, catching the two-no-op failure mode. Interval and full-block variants also match legacy partial accumulators and softmax state exactly (zero tolerance). The generator writes only small origin/info/reference NPY files; the arena is generated at runtime.

Regenerate checked-in fixtures from the CPU-generated legacy and interval programs:

    python generate_benches.py LEGACY/program/llm.loom INTERVAL/program/llm.loom OUTPUT

`generate_data.py` is invoked through the existing DevLab generator hook. The kernel fixtures are experimental test data, not shipping package programs.
