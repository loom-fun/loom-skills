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

static void check_checkpoint(loom_text_embed_t* embed, loom_text_embed_request_t* request,
                             size_t extent) {
  fprintf(stderr, "checkpoint: pin and size\n");
  loom_text_embed_checkpoint_t* checkpoint = NULL;
  require(loom_text_embed_row_pin(embed, 0, &checkpoint), embed, "pin checkpoint");
  loom_text_embed_checkpoint_info_t info;
  size_t size = 0;
  if (loom_text_embed_checkpoint_export(embed, checkpoint, NULL, 0, &size, &info) == 0 ||
      size == 0 || !info.has_prediction) {
    fprintf(stderr, "The checkpoint does not expose a pending prediction and image\n");
    exit(1);
  }
  uint8_t* image = malloc(size);
  if (!image) exit(1);
  fprintf(stderr, "checkpoint: download %zu bytes\n", size);
  require(loom_text_embed_checkpoint_export(embed, checkpoint, image, size, &size, &info),
          embed, "download checkpoint");
  loom_text_embed_checkpoint_release(embed, checkpoint);
  require(loom_text_embed_row_decode(embed, 0), embed, "decode before restore");
  const int32_t next = loom_text_embed_row_token(embed, 0);

  loom_text_embed_row_end(embed, 0);
  require(loom_text_embed_row_reset(embed, 0), embed, "reset row");
  fprintf(stderr, "checkpoint: upload and restore\n");
  require(loom_text_embed_checkpoint_import(embed, &info, image, size, &checkpoint),
          embed, "upload checkpoint");
  free(image);
  int restored = 0;
  require(loom_text_embed_row_restore(embed, 0, checkpoint, &restored), embed, "restore checkpoint");
  loom_text_embed_checkpoint_release(embed, checkpoint);
  if (!restored || loom_text_embed_row_token(embed, 0) != info.token) {
    fprintf(stderr, "Restoring the image changed the pending token\n");
    exit(1);
  }
  require(loom_text_embed_row_begin(embed, 0, request, extent, 0), embed, "continue row");
  require(loom_text_embed_row_decode(embed, 0), embed, "decode after restore");
  if (loom_text_embed_row_token(embed, 0) != next) {
    fprintf(stderr, "The continuation changed after checkpoint restore\n");
    exit(1);
  }
  printf("checkpoint_roundtrip=pass bytes=%zu position=%u\n", size, info.position);
}

int main(int argc, char** argv) {
  if (argc != 4 || (strcmp(argv[3], "tiny") && strcmp(argv[3], "flash"))) {
    fprintf(stderr, "Usage: controller-abi PACKAGE PRESET tiny|flash\n");
    return 2;
  }
  const int tiny = !strcmp(argv[3], "tiny");
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
  require(loom_text_embed_row_begin(embed, 0, request, count + max_tokens, 1),
          embed, "begin");
  fprintf(stderr, "prefill: %zu tokens\n", count);
  require(loom_text_embed_row_prefill(embed, 0, tokens, count), embed, "prefill");
  fprintf(stderr, "prefill complete\n");
  const size_t prompt_tokens = count;
  count = 0;
  const int32_t initial = loom_text_embed_row_token(embed, 0);
  if (!tiny && !loom_text_embed_row_is_eos(embed, 0)) {
    check_checkpoint(embed, request, prompt_tokens + max_tokens);
    tokens[count++] = initial;
  }
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
