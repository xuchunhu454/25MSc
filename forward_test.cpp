#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include "firmware/config.h"
#include "firmware/typedefs.h"
#include "firmware/forward.h"

// tokenizer functions
extern void build_tokenizer(Tokenizer *t, std::string tokenizer_path, int vocab_size);
extern void free_tokenizer(Tokenizer *t);
extern void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens);
extern char *decode(Tokenizer *t, int prev_token, int token);
extern void safe_printf(char *piece);

// transformer loader
extern void build_transformer(Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *t, std::string checkpoint_path);

// sampling
extern int sample(float *logits, int vocab_size, float temperature, float topp, unsigned long long *rng_state);

int main() {
    std::string checkpoint_path = "stories110M.bin";
    std::string tokenizer_path = "tokenizer.bin";
    char prompt[] = "Hello";
    int steps = 64;

    // build transformer model
    static Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> transformer;
    build_transformer(&transformer, checkpoint_path);

    // build tokenizer
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // encode prompt
    int *prompt_tokens = new int[strlen(prompt) + 3];
    int num_prompt_tokens = 0;
    encode(&tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);

    // sampling params
    unsigned long long rng_seed = (unsigned long long)time(NULL);
    float temperature = 1.0f;
    float topp = 1.0f;

    // allocate buffers
    float *logits = new float[vocab_size];
    int kv_dim = (dim * n_kv_heads) / n_heads;
    float *key_cache = new float[n_layers * seq_len * kv_dim]();
    float *value_cache = new float[n_layers * seq_len * kv_dim]();

    int token = prompt_tokens[0];
    int pos = 0;
    int next;

    while (pos < steps) {
        forward(&transformer, token, pos, key_cache, value_cache, logits);

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(logits, vocab_size, temperature, topp, &rng_seed);
        }

        char *piece = decode(&tokenizer, token, next);
        safe_printf(piece);
        fflush(stdout);

        if (next == 1) break;
        token = next;
        pos++;
    }

    printf("\n");

    // clean up
    delete[] logits;
    delete[] key_cache;
    delete[] value_cache;
    delete[] prompt_tokens;
    free_tokenizer(&tokenizer);
    return 0;
}
