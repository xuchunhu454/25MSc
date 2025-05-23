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
    // Model and tokenizer files
    std::string checkpoint_path = "modelq.bin";
    std::string tokenizer_path  = "tokenizer.bin";
    char *prompt                = "Hello";
    int   steps                 = 64;

    // 1) load model
    static Transformer<
        dim, hidden_dim, n_layers, n_heads,
        n_kv_heads, vocab_size, seq_len, GS
    > transformer;
    build_transformer(&transformer, checkpoint_path);

    // 2) initial tokenizer
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // 3) tokenize the prompt
    int prompt_len        = std::strlen(prompt);
    int *prompt_tokens    = new int[prompt_len + 3];
    int  num_prompt_tokens= 0;
    encode(&tokenizer,
           prompt,
           /*bos=*/1, /*eos=*/0,
           prompt_tokens,
           &num_prompt_tokens);

    // 4) Sampling parameters
    unsigned long long rng_seed = (unsigned long long)std::time(nullptr);
    float temperature = 1.0f;
    float topp        = 1.0f;

    // 5) Allocating cache and output buffer
    float *logits      = new float[vocab_size];
    constexpr int kv_dim = (dim * n_kv_heads) / n_heads;
    float *key_cache   = new float[n_layers * seq_len * kv_dim]();
    float *value_cache = new float[n_layers * seq_len * kv_dim]();

    // 6) generation loop
    int token_id = prompt_tokens[0];
    int pos      = 0;
    while (pos < steps) {
        // HLS kernel 
        forward(&transformer,
                token_id,
                pos,
                key_cache,
                value_cache,
                logits);

        // The first few tokens are forced to use prompt
        int next = (pos < num_prompt_tokens - 1)
                 ? prompt_tokens[pos + 1]
                 : sample(logits,
                          transformer.config.vocab_size,
                          temperature,
                          topp,
                          &rng_seed);

        // decode and print
        char *piece = decode(&tokenizer, token_id, next);
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
