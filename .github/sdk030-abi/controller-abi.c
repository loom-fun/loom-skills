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

int main(int argc, char** argv) {
  if (argc != 4) {
    fprintf(stderr, "Usage: controller-abi PACKAGE PRESET tiny|flash\n");
    return 2;
  }
  const int tiny = !strcmp(argv[3], "tiny");
  loom_text_embed_options_ex_t options = {0};
  options.struct_size = sizeof(options);
  options.base.model_path = argv[1];
  options.base.preset = argv[2];
  options.base.rows = 1;
  options.row_context = tiny ? 96 : 4096;
  options.elastic = 1;
  loom_text_embed_t* embed = NULL;
  char error[4096] = {0};
  if (loom_text_embed_open_ex(&options, &embed, error, sizeof(error))) {
    fprintf(stderr, "open: %s\n", error);
    return 1;
  }

  loom_text_embed_info_t info;
  loom_text_embed_info(embed, &info);
  char body[1024];
  snprintf(body, sizeof(body),
           "{\"model\":\"%s\",\"messages\":[{\"role\":\"user\",\"content\":\"%s\"}],"
           "\"max_tokens\":%d%s}",
           info.model_id, tiny ? "abxa" : "What is the capital of France? Reply with one word.",
           tiny ? 2 : 64, tiny ? "" : ",\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}");
  loom_text_embed_request_t* request = NULL;
  int32_t tokens[4096];
  size_t count = 0;
  require(loom_text_embed_chat_prepare(embed, body, strlen(body), tokens,
                                      sizeof(tokens) / sizeof(tokens[0]), &count, &request),
          embed, "prepare");
  require(loom_text_embed_row_begin(embed, 0, request, count + (tiny ? 2 : 64), 1),
          embed, "begin");
  require(loom_text_embed_row_prefill_request(embed, 0, request, 0, count, 1), embed, "prefill");
  const size_t prompt_tokens = count;
  count = 0;
  while (count < (tiny ? 2 : 64) && !loom_text_embed_row_is_eos(embed, 0)) {
    tokens[count++] = loom_text_embed_row_token(embed, 0);
    if (count < (tiny ? 2 : 64))
      require(loom_text_embed_row_decode(embed, 0), embed, "decode");
  }
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
  loom_text_embed_close(embed);
  if (tiny)
    return prompt_tokens == 4 && count == 2 && decoded_size == 2 && !memcmp(decoded, "bx", 2) ? 0 : 1;
  return count && memmem(answer, answer_size, "Paris", 5) ? 0 : 1;
}
