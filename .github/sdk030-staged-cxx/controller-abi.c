#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "loom_text_embed.h"

static void require(int status, loom_text_embed_t* embed, const char* action) {
  if (status) {
    fprintf(stderr, "%s: %s\n", action, loom_text_embed_error(embed));
    exit(1);
  }
}

typedef struct {
  loom_text_embed_checkpoint_info_t info;
  size_t size;
  int32_t next;
} checkpoint_record_t;

static void fail(const char* message) {
  fprintf(stderr, "%s\n", message);
  exit(1);
}

static void reset_row(loom_text_embed_t* embed) {
  loom_text_embed_row_end(embed, 0);
  require(loom_text_embed_row_reset(embed, 0), embed, "reset row");
}

static void restore_checkpoint(loom_text_embed_t* embed, loom_text_embed_request_t* request,
                               const checkpoint_record_t* record, const uint8_t* image,
                               size_t extent) {
  fprintf(stderr, "checkpoint: upload and restore\n");
  loom_text_embed_checkpoint_t* checkpoint = NULL;
  require(loom_text_embed_checkpoint_import(embed, &record->info, image, record->size, &checkpoint),
          embed, "upload checkpoint");
  int restored = 0;
  require(loom_text_embed_row_restore(embed, 0, checkpoint, &restored), embed, "restore checkpoint");
  loom_text_embed_checkpoint_release(embed, checkpoint);
  if (!restored || loom_text_embed_row_token(embed, 0) != record->info.token)
    fail("Restoring the image changed the pending token");
  require(loom_text_embed_row_begin(embed, 0, request, extent, 0), embed, "continue row");
  require(loom_text_embed_row_decode(embed, 0), embed, "decode after restore");
  if (loom_text_embed_row_token(embed, 0) != record->next)
    fail("The continuation changed after checkpoint restore");
}

static int32_t check_checkpoint(loom_text_embed_t* embed, loom_text_embed_request_t* request,
                                size_t extent, const char* image_path) {
  fprintf(stderr, "checkpoint: pin and size\n");
  loom_text_embed_checkpoint_t* checkpoint = NULL;
  require(loom_text_embed_row_pin(embed, 0, &checkpoint), embed, "pin checkpoint");
  checkpoint_record_t record = {0};
  if (!loom_text_embed_checkpoint_export(embed, checkpoint, NULL, 0, &record.size, &record.info) ||
      !record.size || !record.info.has_prediction)
    fail("The checkpoint does not expose a pending prediction and image");
  uint8_t* image = malloc(record.size);
  if (!image) fail("Cannot allocate the checkpoint image");
  fprintf(stderr, "checkpoint: download %zu bytes\n", record.size);
  require(loom_text_embed_checkpoint_export(embed, checkpoint, image, record.size,
                                            &record.size, &record.info), embed, "download checkpoint");
  loom_text_embed_checkpoint_release(embed, checkpoint);
  require(loom_text_embed_row_decode(embed, 0), embed, "decode before restore");
  record.next = loom_text_embed_row_token(embed, 0);
  FILE* file = fopen(image_path, "wb");
  if (!file || fwrite(&record, sizeof(record), 1, file) != 1 ||
      fwrite(image, 1, record.size, file) != record.size || fclose(file))
    fail("Cannot write the checkpoint for a fresh process");
  reset_row(embed);
  restore_checkpoint(embed, request, &record, image, extent);
  free(image);
  printf("checkpoint_roundtrip=pass bytes=%zu position=%u\n", record.size, record.info.position);
  return record.info.token;
}

static int32_t check_cold_checkpoint(loom_text_embed_t* embed, loom_text_embed_request_t* request,
                                     size_t prompt_tokens, size_t extent, const char* image_path) {
  loom_text_embed_image_layout_t layout = {0};
  loom_text_embed_image_layout(embed, &layout);
  if (!layout.supported || !layout.fixed_bytes || !layout.block_size || !layout.page_bytes)
    fail("Flash Next has no recurrent checkpoint geometry on a fresh runtime");
  FILE* file = fopen(image_path, "rb");
  checkpoint_record_t record = {0};
  if (!file || fread(&record, sizeof(record), 1, file) != 1 || !record.info.has_prediction ||
      record.info.position != prompt_tokens)
    fail("Cannot read the checkpoint metadata from the warm runtime");
  const uint64_t pages = (record.info.position + layout.block_size - 1) / layout.block_size;
  if (record.size != layout.fixed_bytes + pages * layout.page_bytes)
    fail("The cold checkpoint layout differs from the exported image");
  uint8_t* image = malloc(record.size);
  if (!image || fread(image, 1, record.size, file) != record.size || fclose(file))
    fail("Cannot read the checkpoint image");
  restore_checkpoint(embed, request, &record, image, extent);
  printf("cold_checkpoint=pass bytes=%zu position=%u\n", record.size, record.info.position);

  reset_row(embed);
#ifdef GUFO_HAS_MEMORY_INFO
  loom_text_embed_memory_t before, after;
  loom_text_embed_memory(embed, &before);
#endif
  loom_text_embed_checkpoint_t* invalid = NULL;
  if (!loom_text_embed_checkpoint_import(embed, &record.info, image, record.size - 1, &invalid) ||
      invalid || !strstr(loom_text_embed_error(embed), "checkpoint image is"))
    fail("Import accepted a truncated checkpoint image or returned the wrong error");
#ifdef GUFO_HAS_MEMORY_INFO
  loom_text_embed_memory(embed, &after);
  if (before.free_tokens != after.free_tokens)
    fail("Importing a truncated checkpoint leaked page ownership");
  printf("checkpoint_free_tokens=%llu\n", (unsigned long long)after.free_tokens);
#endif
  restore_checkpoint(embed, request, &record, image, extent);
  puts("checkpoint_rollback=pass");
  free(image);
  return record.info.token;
}

int main(int argc, char** argv) {
  const int tiny = argc == 4 && !strcmp(argv[3], "tiny");
  const int warm = argc == 5 && !strcmp(argv[3], "flash");
  const int cold = argc == 5 && !strcmp(argv[3], "flash-cold");
  if (!tiny && !warm && !cold) {
    fprintf(stderr, "Usage: controller-abi PACKAGE PRESET tiny|{flash,flash-cold} [IMAGE]\n");
    return 2;
  }
  const size_t max_tokens = tiny ? 2 : 64;
  setvbuf(stdout, NULL, _IONBF, 0);
  loom_text_embed_options_ex_t options = {0};
  options.struct_size = sizeof(options);
  options.base.model_path = argv[1];
  options.base.preset = argv[2];
  options.base.rows = 1;
  options.elastic = 1;
  loom_text_embed_t* embed = NULL;
  char error[4096] = {0};
  fprintf(stderr, "opening package %s preset %s\n", argv[1], argv[2]);
  if (loom_text_embed_open_ex(&options, &embed, error, sizeof(error))) {
    fprintf(stderr, "open: %s\n", error);
    return 1;
  }

  loom_text_embed_info_t info;
  loom_text_embed_info(embed, &info);
  fprintf(stderr, "opened %s with context capacity %u\n", info.model_id, info.context_capacity);
  char body[1024];
  snprintf(body, sizeof(body),
           "{\"model\":\"%s\",\"messages\":[{\"role\":\"user\",\"content\":\"%s\"}],"
           "\"max_tokens\":%zu%s}",
           info.model_id, tiny ? "abxa" : "What is the capital of France? Answer in one word.",
           max_tokens, tiny ? "" : ",\"temperature\":0,\"enable_thinking\":false");
  loom_text_embed_request_t* request = NULL;
  int32_t tokens[4096];
  size_t count = 0;
  require(loom_text_embed_chat_prepare(embed, body, strlen(body), tokens,
                                      sizeof(tokens) / sizeof(tokens[0]), &count, &request),
          embed, "prepare");
  const size_t prompt_tokens = count;
  const size_t extent = prompt_tokens + max_tokens;
  if (!cold) {
    require(loom_text_embed_row_begin(embed, 0, request, extent, 1), embed, "begin");
    fprintf(stderr, "prefill: %zu tokens\n", prompt_tokens);
    require(loom_text_embed_row_prefill(embed, 0, tokens, prompt_tokens), embed, "prefill");
    fprintf(stderr, "prefill complete\n");
  }
  count = 0;
  if (warm)
    tokens[count++] = check_checkpoint(embed, request, extent, argv[4]);
  if (cold)
    tokens[count++] = check_cold_checkpoint(embed, request, prompt_tokens, extent, argv[4]);
  while (count < max_tokens && !loom_text_embed_row_is_eos(embed, 0)) {
    tokens[count++] = loom_text_embed_row_token(embed, 0);
    if (count < max_tokens)
      require(loom_text_embed_row_decode(embed, 0), embed, "decode");
  }
  fprintf(stderr, "decode complete: %zu tokens\n", count);
  char decoded[4096];
  size_t decoded_size = 0;
  require(loom_text_embed_detokenize(embed, tokens, count, decoded, sizeof(decoded), &decoded_size),
          embed, "detokenize");
  printf("model=%s prompt_tokens=%zu completion_tokens=%zu text=%.*s\n",
         info.model_id, prompt_tokens, count, (int)decoded_size, decoded);
  char answer[4096];
  size_t answer_size = 0;
  require(loom_text_embed_output_feed(request, decoded, decoded_size, 1, answer,
                                      sizeof(answer), &answer_size), embed, "read output");
  printf("answer=%.*s\n", (int)answer_size, answer);
  loom_text_embed_request_free(request);
  fprintf(stderr, "closing model\n");
  loom_text_embed_close(embed);
  if (tiny)
    return prompt_tokens == 4 && count == 2 && decoded_size == 2 && !memcmp(decoded, "bx", 2) ? 0 : 1;
  size_t start = 0;
  while (start < answer_size && isspace((unsigned char)answer[start])) ++start;
  while (answer_size > start &&
         (isspace((unsigned char)answer[answer_size - 1]) || answer[answer_size - 1] == '.'))
    --answer_size;
  return count && answer_size - start == 5 && !memcmp(answer + start, "Paris", 5) ? 0 : 1;
}
