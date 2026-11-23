import re
from pathlib import Path

def replace(text, old, new, count=1):
    assert text.count(old) == count, (old, text.count(old), count)
    return text.replace(old, new)

def sub(text, pattern, replacement, count=1):
    text, actual = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    assert actual == count, (pattern, actual, count)
    return text

for source in ("gufo/src/models/loom/embed/loom_text_embed.c",
               "extension/loom-embed/loom_text_embed.c"):
    file = Path(source)
    text = file.read_text()
    text = replace(text, '#include "loom/serve/service/text/schedule.h"\n', '')
    text = replace(text, '  iree_host_size_t shape_count;\n  loom_serve_packing_shape_t* shapes;\n', '')
    text = sub(text, r'  embed->shape_count =\n.*?loom_serve_text_default_shapes\(text->prefill_capacity, embed->shapes\);\n'.replace('.*?', r'[\s\S]*?'), '')
    text = replace(text, '      .epoch_count = embed->shape_count,\n      .epoch_shapes = embed->shapes,',
                   '      .span_capacity = iree_min(text->prefill_capacity, embed->rows),')
    text = replace(text, '  iree_allocator_free(embed->allocator, embed->shapes);\n', '')
    text = sub(text, r"// Widest epoch shape's span capacity[\s\S]*?^}\n\n", '')
    text = sub(text, r'  iree_host_size_t spans = 0;\n  for \(iree_host_size_t i = 0; i < embed->shape_count; \+\+i\) \{\n    spans = iree_max\(spans, embed->shapes\[i\]\.span_capacity\);\n  \}\n  return iree_min\(spans, \(iree_host_size_t\)embed->rows\);',
               '  return loom_serve_text_model_span_capacity(embed->model);')
    text = replace(text, 'iree_min(loom_text_embed_span_capacity(embed), embed->rows)',
                   'loom_serve_text_model_span_capacity(embed->model)')
    text = replace(text, "// min(widest epoch's spans, resident rows) slots", "// admitted span slots")
    text = sub(text, r'// Smallest prepared shape admitting the spans and tokens\.[\s\S]*?^}\n\n', '')
    text = sub(text, r'  iree_host_size_t shape = 0;\n(?:  iree_host_size_t total = 0;\n  for \(size_t i = 0; i < count; \+\+i\) total \+= limits\[i\] \? width : 1;\n)?  if \(!loom_text_embed_shape\(embed, [^\n]+\n[\s\S]*?^  }\n', '', count=4)
    text = replace(text, "loom_serve_text_model_epoch(embed->model, shape,", "loom_serve_text_model_epoch(embed->model,", count=3)
    text = replace(text, "embed->model, shape, count, spans, limits, NULL, results",
                   "embed->model, count, spans, limits, NULL, results")
    text = replace(text, '  IREE_RETURN_IF_ERROR(loom_serve_text_storage_plan_snapshot(\n'
                        '      storage, state->row_count + checkpoint_index, views, true, block_count,\n'
                        '      blocks, state->host_allocator, &merged_count, &merged));',
                   '  const loom_serve_text_storage_blocks_t span = {\n'
                   '      .count = block_count, .values = blocks};\n'
                   '  IREE_RETURN_IF_ERROR(loom_serve_text_storage_plan_snapshot(\n'
                   '      storage, state->row_count + checkpoint_index, views, true, 1,\n'
                   '      &span, state->host_allocator, &merged_count, &merged));')
    for length, slot in (("embed->logits_stride", "pending"), ("device->length", "operation_count")):
        assert text.count(f'.length = {length}}}}}') == (1 if slot == "pending" else 2)
        text = replace(text, f'.length = {length}}}}}', f'.region = &regions[{slot}]}}}}',
                       count=1 if slot == "pending" else 2)
    text = sub(text, r'(  iree_hal_transfer_operation_t operations\[[^\n]+\];\n)',
               r'\1  iree_hal_transfer_region_t regions[IREE_ARRAYSIZE(operations)];\n', count=2)
    text = replace(text, '    operations[pending] = (iree_hal_transfer_operation_t){',
                   '    regions[pending] = (iree_hal_transfer_region_t){\n'
                   '        .row_length = embed->logits_stride, .row_count = 1, .slice_count = 1};\n'
                   '    operations[pending] = (iree_hal_transfer_operation_t){')
    text = replace(text, '    operations[operation_count++] =\n',
                   '    regions[operation_count] = (iree_hal_transfer_region_t){\n'
                   '        .row_length = device->length, .row_count = 1, .slice_count = 1};\n'
                   '    operations[operation_count] =\n')
    text = replace(text, '                                  .region = &regions[operation_count]}};',
                   '                                  .region = &regions[operation_count]}};\n'
                   '    ++operation_count;')
    file.write_text(text)
