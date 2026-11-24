/* Capture the exact package's first logit row after a visual prompt. */
#include "loom_text_embed.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static char *read_request(const char *path, size_t *size) {
  FILE *input = fopen(path, "rb");
  if (!input) return NULL;
  if (fseek(input, 0, SEEK_END) != 0) {
    fclose(input);
    return NULL;
  }
  long length = ftell(input);
  if (length <= 0 || fseek(input, 0, SEEK_SET) != 0) {
    fclose(input);
    return NULL;
  }
  char *body = malloc((size_t)length);
  if (!body || fread(body, 1, (size_t)length, input) != (size_t)length) {
    free(body);
    fclose(input);
    return NULL;
  }
  fclose(input);
  *size = (size_t)length;
  return body;
}

int main(int argc, char **argv) {
  if (argc != 5) {
    fprintf(stderr, "usage: %s MODEL PRESET REQUEST.json LOGITS.f32\n", argv[0]);
    return 2;
  }
  size_t body_size = 0;
  char *body = read_request(argv[3], &body_size);
  if (!body) {
    fprintf(stderr, "cannot read request: %s\n", argv[3]);
    return 2;
  }

  loom_text_embed_t *embed = NULL;
  loom_text_embed_request_t *request = NULL;
  float *logits = NULL;
  int row_open = 0;
  int result = 1;
  const char *operation = "open";
  char error[2048] = {0};
  loom_text_embed_options_ex_t options = {
      .struct_size = sizeof(options),
      .base = {.model_path = argv[1], .preset = argv[2],
               .device_uri = "amdgpu://?completion_spin_ns=4294967295",
               .memory_bytes = UINT64_C(34000000000), .rows = 1},
      .logits = 1,
      .elastic = 1,
      .row_context = 4096,
  };
  if (loom_text_embed_open_ex(&options, &embed, error, sizeof(error))) {
    fprintf(stderr, "open: %s\n", error);
    goto cleanup;
  }
  int32_t tokens[4096];
  size_t count = 0;
  operation = "prepare chat";
  if (loom_text_embed_chat_prepare(embed, body, body_size, tokens,
                                   sizeof(tokens) / sizeof(tokens[0]),
                                   &count, &request))
    goto failed;
  uint32_t offset = 0, image_rows = 0;
  operation = "verify image and logit request";
  if (loom_text_embed_request_media(request, &offset, &image_rows, 1) != 1 ||
      image_rows == 0 || !loom_text_embed_request_logits(request))
    goto failed;
  operation = "reserve row";
  if (loom_text_embed_row_begin(embed, 0, request, count + 1, 1)) goto failed;
  row_open = 1;
  operation = "project image";
  if (loom_text_embed_request_project(request)) goto failed;
  operation = "prefill";
  if (loom_text_embed_row_prefill_request(embed, 0, request, 0, count, 1))
    goto failed;
  loom_text_embed_logits_info_t info = {0};
  loom_text_embed_logits_info(embed, &info);
  operation = "read first logit row";
  if (!info.supported || !info.enabled || info.width == 0) goto failed;
  logits = malloc((size_t)info.width * sizeof(float));
  if (!logits) goto failed;
  size_t width = 0;
  if (loom_text_embed_row_logits(embed, 0, logits, info.width, &width) ||
      width != info.width)
    goto failed;
  operation = "write first logit row";
  FILE *output = fopen(argv[4], "wb");
  if (!output) goto failed;
  size_t written = fwrite(logits, sizeof(float), width, output);
  int close_status = fclose(output);
  if (written != width || close_status) goto failed;
  printf("{\"prompt_tokens\":%zu,\"image_offset\":%u,\"image_rows\":%u,"
         "\"device_token\":%d,\"vocabulary\":%zu,\"prompt_token_ids\":[",
         count, offset, image_rows, loom_text_embed_row_token(embed, 0), width);
  for (size_t i = 0; i < count; ++i)
    printf("%s%d", i ? "," : "", tokens[i]);
  puts("]}");
  result = 0;
  goto cleanup;

failed:
  fprintf(stderr, "%s: %s\n", operation, loom_text_embed_error(embed));
cleanup:
  if (row_open) loom_text_embed_row_end(embed, 0);
  if (request) loom_text_embed_request_free(request);
  if (embed) loom_text_embed_close(embed);
  free(logits);
  free(body);
  return result;
}
