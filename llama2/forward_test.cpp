#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <string>
#include <cstdint>

#include "firmware/config.h"
#include "firmware/typedefs.h"
#include "firmware/forward.h"              
#include "firmware/tokenizer.h"            
#include "firmware/sampling.h"             
#include "firmware/transformer_loader.h"   

int main() {
    // 一进 main 就打印
    fprintf(stderr, ">>> CSIM: entered main()\n");
    fflush(stderr);

    // Model and tokenizer files
    std::string checkpoint_path = "modelq.bin";
    std::string tokenizer_path  = "tokenizer.bin";
    const char *prompt          = "Long time ago, ";
    int   steps                 = 64;

    // 1) load model
    fprintf(stderr, ">>> CSIM: before build_transformer\n");
    fflush(stderr);
    static Transformer<
        dim, hidden_dim, n_layers, n_heads,
        n_kv_heads, vocab_size, seq_len, GS
    > transformer;
    build_transformer(&transformer, checkpoint_path);
    fprintf(stderr, ">>> CSIM: after build_transformer\n");
    fflush(stderr);

    // 2) initial tokenizer
    fprintf(stderr, ">>> CSIM: before build_tokenizer\n");
    fflush(stderr);
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);
    fprintf(stderr, ">>> CSIM: after build_tokenizer\n");
    fflush(stderr);

    // 3) tokenize the prompt
    fprintf(stderr, ">>> CSIM: before encode, prompt=\"%s\"\n", prompt);
    fflush(stderr);
    int prompt_len         = std::strlen(prompt);
    int *prompt_tokens     = new int[prompt_len + 3];
    int  num_prompt_tokens = 0;
    encode(&tokenizer,
           const_cast<char*>(prompt),
           /*bos=*/1, /*eos=*/0,
           prompt_tokens,
           &num_prompt_tokens);
    fprintf(stderr, ">>> CSIM: after encode, num_tokens=%d\n", num_prompt_tokens);
    fflush(stderr);

    // 4) Sampling parameters
    fprintf(stderr, ">>> CSIM: setting sampling params\n");
    fflush(stderr);
    unsigned long long rng_seed = (unsigned long long)std::time(nullptr);
    float temperature = 1.0f;
    float topp        = 1.0f;

    // 5) Allocating cache and output buffer
    fprintf(stderr, ">>> CSIM: before buffer allocation\n");
    fflush(stderr);
    float *logits      = new float[vocab_size];
    constexpr int kv_dim = (dim * n_kv_heads) / n_heads;
    float *key_cache   = new float[n_layers * seq_len * kv_dim]();
    float *value_cache = new float[n_layers * seq_len * kv_dim]();
    fprintf(stderr, ">>> CSIM: after buffer allocation\n");
    fflush(stderr);
    
    // 6) generation loop
    int token_id = prompt_tokens[0];
    int pos      = 0;
    while (pos < steps) {
        // HLS kernel 
        printf("[CSIM] Calling forward, pos=%d, token=%d\n", pos, token_id);
        std::fflush(stdout);
        forward(&transformer,
                token_id,
                pos,
                key_cache,
                value_cache,
                logits);
        printf("[CSIM] Returned from forward, pos=%d\n", pos);
        std::fflush(stdout);

        // The first few tokens are forced to use prompt
        int next = (pos < num_prompt_tokens - 1)
                 ? prompt_tokens[pos + 1]
                 : sample(logits,
                          transformer.config.vocab_size,
                          temperature,
                          topp,
                          &rng_seed);
        printf("[CSIM] Sampled next=%d\n", next);
        std::fflush(stdout);

        // decode and print
        char *piece = decode(&tokenizer, token_id, next);
        printf("[CSIM] Decoded piece=%s\n", piece);
        std::fflush(stdout);    

        safe_printf(piece);
        std::fflush(stdout);

        if (next == 1) break;  // when BOS(token=1) end
        token_id = next;
        pos++;
    }
    std::printf("\n");

    // 7) delete
    delete[] logits;
    delete[] key_cache;
    delete[] value_cache;
    delete[] prompt_tokens;
    free_tokenizer(&tokenizer);

    return 0;
}
