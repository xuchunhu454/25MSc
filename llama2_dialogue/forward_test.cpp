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

// 从 stdin 读一行，带提示
static void read_stdin(const char *prompt, char *buf, size_t buf_sz) {
    std::cout << prompt << std::flush;
    if (!std::cin.getline(buf, buf_sz)) {
        buf[0] = '\0';
    }
}

// 完整的 chat 函数，几乎照搬官方版本
void chat(Transformer<dim,hidden_dim,n_layers,n_heads,n_kv_heads,vocab_size,seq_len,GS> *transformer,
          Tokenizer *tokenizer,
          int steps, float temperature, float topp) {
    char system_prompt[512], user_prompt[512], rendered_prompt[1152];
    int  *prompt_tokens = (int*)malloc(1152*sizeof(int));
    int   num_prompt_tokens, user_idx;
    int8_t user_turn = 1;
    int    next=0, token=0, pos=0;
    unsigned long long rng_seed = (unsigned long long)std::time(nullptr);

    constexpr int KV_DIM = (dim * n_kv_heads) / n_heads;
    float *key_cache   = new float[n_layers*seq_len*KV_DIM]();
    float *value_cache = new float[n_layers*seq_len*KV_DIM]();
    float *logits      = new float[vocab_size];

    while (pos < steps) {
        if (user_turn) {
            // system/user prompt 同前面一样读取并 render、encode
            if (pos==0) {
                read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
            }
            read_stdin("User> ", user_prompt, sizeof(user_prompt));
            if (pos==0 && system_prompt[0]) {
                sprintf(rendered_prompt,
                        "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]",
                        system_prompt, user_prompt);
            } else {
                sprintf(rendered_prompt, "[INST] %s [/INST]", user_prompt);
            }
            encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0; user_turn = 0;
            std::cout << "Assistant> " << std::flush;
        }

        // 取下一个 token
        if (user_idx < num_prompt_tokens) {
            token = prompt_tokens[user_idx++];
        } else {
            token = next;
        }
        if (token == 2) user_turn = 1;  // EOS 回到用户

        // forward
        std::memset(logits, 0, vocab_size*sizeof(float));
        forward(transformer, token, pos, key_cache, value_cache, logits);

        // sample：注意这里调用签名
        next = sample(logits,
                      transformer->config.vocab_size,
                      temperature,
                      topp,
                      &rng_seed);
        pos++;

        // 打印模型输出
        if (user_idx >= num_prompt_tokens && next != 2) {
            char *piece = decode(tokenizer, token, next);
            safe_printf(piece);
            std::fflush(stdout);
        }
        if (next == 2) std::cout << std::endl;
    }

    delete[] logits;
    delete[] key_cache;
    delete[] value_cache;
    free(prompt_tokens);
}

int main() {
    std::string ckpt = "modelq.bin";
    std::string tokf = "tokenizer.bin";
    static Transformer<dim,hidden_dim,n_layers,n_heads,n_kv_heads,vocab_size,seq_len,GS> transformer;
    build_transformer(&transformer, ckpt);

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokf, transformer.config.vocab_size);

    // 直接调用 chat，不用 build_sampler
    chat(&transformer, &tokenizer,
         /*steps=*/64,
         /*temperature=*/1.0f,
         /*topp=*/1.0f);

    free_tokenizer(&tokenizer);
    return 0;
}