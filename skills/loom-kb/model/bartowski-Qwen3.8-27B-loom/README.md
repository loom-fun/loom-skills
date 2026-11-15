# Model: bartowski-Qwen3.8-27B-loom

Radeon AI PRO R9700 serving settings of `bartowski-Qwen3.8-27B-loom` (prefill_capacity 1024); resolved per device by `resolve.py`, written into the `<first>-r9700` preset of model.json by `make_info.py build --fit`.

MTP proposals (depth 3) read a Q4_0 copy of the head's first 81920 rows plus the added-token rows (`draft.propose.head_rows`, `draft/prep_weights.py`; load-time conversion of the published Q6_K head, verification keeps the Q6_K head), 0.2 GB per proposal round instead of 1.04 GB. Same package on both GPUs.
