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
    // 1) Initialize the model and tokenizer
    std::string checkpoint_path = "modelq.bin";
    std::string tokenizer_path  = "tokenizer.bin";
    static Transformer<
        dim, hidden_dim, n_layers, n_heads,
        n_kv_heads, vocab_size, seq_len, GS
    > transformer;
    build_transformer(&transformer, checkpoint_path);

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // 2) parameters
    unsigned long long rng_seed = (unsigned long long)std::time(nullptr);
    const float temperature = 1.0f;
    const float topp        = 1.0f;
    const int   max_steps   = 64;

    // 3) Global cache (conversation state is accumulated over multiple rounds)
    constexpr int kv_dim = (dim * n_kv_heads) / n_heads;
    float *key_cache   = new float[n_layers * seq_len * kv_dim]();
    float *value_cache = new float[n_layers * seq_len * kv_dim]();
    float *logits      = new float[vocab_size];

    // 4) main loop
    while (true) {
        std::string prompt;
        std::cout << "\nUser> ";
        if (!std::getline(std::cin, prompt)) break;
        if (prompt == "exit" || prompt == "quit") break;

        // 4.1) encode input words
        int prompt_len        = prompt.size();
        int *prompt_tokens    = new int[prompt_len * 4 + 3]; // worst-case UTF-8，4 bytes per char
        int  num_prompt_tokens = 0;
        encode(&tokenizer,
               const_cast<char*>(prompt.c_str()),
               /*bos=*/1, /*eos=*/0,
               prompt_tokens,
               &num_prompt_tokens);

        // 4.2) Burying prompt tokens in cache
        int pos = 0;
        int token_id;
        // Feed each token in the prompt to the forward function, only once, without sampling
        for (int i = 0; i < num_prompt_tokens; i++) {
            token_id = prompt_tokens[i];
            std::memset(logits, 0, vocab_size * sizeof(float));
            forward(&transformer,
                    token_id,
                    pos,
                    key_cache,
                    value_cache,
                    logits);
            pos++;
        }

        // 4.3) Model response generation
        std::cout << "Model> ";
        for (int step = 0; step < max_steps; step++) {
            // Sample the next token
            int next = sample(logits,
                              transformer.config.vocab_size,
                              temperature,
                              topp,
                              &rng_seed);
            // If it is EOS, end this round
            if (next == 2) break;

            // Output this token
            char *piece = decode(&tokenizer, token_id, next);
            safe_printf(piece);

            // Use this token to continue forwarding
            token_id = next;
            std::memset(logits, 0, vocab_size * sizeof(float));
            forward(&transformer,
                    token_id,
                    pos,
                    key_cache,
                    value_cache,
                    logits);
            pos++;
            if (pos >= seq_len) break; // Avoid exceeding the maximum seq length
        }
        std::cout << std::endl;

        delete[] prompt_tokens;
        // Continue to the next round, do not clear key_cache/value_cache, let the model retain the conversation context
    }

    // 5) clean
    delete[] logits;
    delete[] key_cache;
    delete[] value_cache;
    free_tokenizer(&tokenizer);
    return 0;
}
