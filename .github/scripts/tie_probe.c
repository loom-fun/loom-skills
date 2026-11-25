/* Read the shipped package's logits after the exact native-ratchet prefix. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "loom_text_embed.h"

int main(int argc, char **argv) {
    if (argc != 7) return 2;
    FILE *body = fopen(argv[3], "rb"), *prefix = fopen(argv[4], "r");
    if (!body || !prefix) return 2;
    fseek(body, 0, SEEK_END);
    long len = ftell(body);
    if (len <= 0) return 2;
    rewind(body);
    char *json = calloc(len + 1, 1);
    if (!json || fread(json, 1, len, body) != (size_t)len) return 2;
    fclose(body);
    loom_text_embed_options_ex_t opts = {0};
    opts.struct_size = sizeof(opts);
    opts.base.model_path = argv[1];
    opts.base.preset = argv[2];
    opts.logits = 1;
    loom_text_embed_t *e = NULL;
    char error[2048] = {0};
    if (loom_text_embed_open_ex(&opts, &e, error, sizeof(error))) {
        fprintf(stderr, "open: %s\n", error);
        return 1;
    }
#define TRY(x) do { if ((x)) { fprintf(stderr,"%s: %s\n", #x, loom_text_embed_error(e)); return 1; } } while (0)
    loom_text_embed_info_t info;
    loom_text_embed_info(e, &info);
    int32_t *ids = calloc(info.context_capacity, sizeof(*ids));
    loom_text_embed_request_t *req = NULL;
    size_t count = 0;
    TRY(loom_text_embed_chat_prepare(e,json,len,ids,info.context_capacity,&count,&req));
    if (!loom_text_embed_request_logits(req)) return 1;
    TRY(loom_text_embed_row_begin(e,0,req,count+64,1));
    TRY(loom_text_embed_row_prefill(e,0,ids,count));
    loom_text_embed_request_free(req);
    printf("{\"prompt_ids\":[");
    for (size_t i=0;i<count;i++) printf("%s%d",i?",":"",ids[i]);
    printf("],\"prefix_device_choices\":[");
    uint32_t row=0;
    int forced, step=0;
    while (fscanf(prefix,"%d",&forced)==1) {
        printf("%s%d",step++?",":"",loom_text_embed_row_token(e,0));
        int32_t token=forced;
        TRY(loom_text_embed_decode_batch_tokens(e,&row,1,&token));
    }
    fclose(prefix);
    loom_text_embed_logits_info_t li;
    loom_text_embed_logits_info(e,&li);
    float *logits=calloc(li.width,sizeof(*logits));
    size_t n=0;
    TRY(loom_text_embed_row_logits(e,0,logits,li.width,&n));
    int vocab=loom_text_embed_vocabulary_size(e);
    if (vocab<10 || n < (size_t)vocab) return 1;
    int best[10]; for(int k=0;k<10;k++) best[k]=-1;
    for(int i=0;i<vocab;i++) {
        if(!isfinite(logits[i])) return 1;
        for(int k=0;k<10;k++) if(best[k]<0 || logits[i]>logits[best[k]]) {
            for(int j=9;j>k;j--) best[j]=best[j-1];
            best[k]=i; break;
        }
    }
    double peak=logits[best[0]], sum=0;
    for(int i=0;i<vocab;i++) sum+=exp((double)logits[i]-peak);
    double denom=peak+log(sum);
    printf("],\"device\":%d,\"top\":[",loom_text_embed_row_token(e,0));
    for(int k=0;k<10;k++) printf("%s[%d,%.9f]",k?",":"",best[k],logits[best[k]]-denom);
    int a=atoi(argv[5]),b=atoi(argv[6]);
    if(a<0 || b<0 || a>=vocab || b>=vocab) return 1;
    printf("],\"candidate_lp\":[%.9f,%.9f]}\n",logits[a]-denom,logits[b]-denom);
    loom_text_embed_close(e);
    return 0;
}
