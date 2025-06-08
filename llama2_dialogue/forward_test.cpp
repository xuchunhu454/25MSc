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
          Tokenizer *tokenizer, Sampler *sampler,
          const char *cli_user_prompt, const char *cli_system_prompt,
          int steps) {
    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int  *prompt_tokens = (int*)malloc(1152 * sizeof(int));
    int   num_prompt_tokens;
    int   user_idx;
    int8_t user_turn = 1;
    int    next = 0, token = 0;
    int    pos = 0;

    // 全局 cache
    constexpr int KV_DIM = (dim * n_kv_heads) / n_heads;
    float *key_cache   = new float[n_layers * seq_len * KV_DIM]();
    float *value_cache = new float[n_layers * seq_len * KV_DIM]();
    float *logits      = new float[vocab_size];

    while (pos < steps) {
        // 用户回合：读取 system/user prompt 并 encode
        if (user_turn) {
            if (pos == 0) {
                if (cli_system_prompt) strcpy(system_prompt, cli_system_prompt);
                else                   read_stdin("Enter system prompt (optional): ",
                                                  system_prompt, sizeof(system_prompt));
            }
            if (pos == 0 && cli_user_prompt) strcpy(user_prompt, cli_user_prompt);
            else                              read_stdin("User> ", user_prompt, sizeof(user_prompt));

            if (pos == 0 && system_prompt[0] != '\0') {
                sprintf(rendered_prompt,
                        "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]",
                        system_prompt, user_prompt);
            } else {
                sprintf(rendered_prompt,
                        "[INST] %s [/INST]",
                        user_prompt);
            }

            encode(tokenizer, rendered_prompt, /*bos=*/1, /*eos=*/0,
                   prompt_tokens, &num_prompt_tokens);
            user_idx = 0;
            user_turn = 0;
            std::cout << "Assistant> " << std::flush;
        }

        // 决定下一个输入 transformer 的 token
        if (user_idx < num_prompt_tokens) {
            token = prompt_tokens[user_idx++];
        } else {
            token = next;
        }
        if (token == 2) user_turn = 1;  // EOS 切回用户

        // forward 得到 logits
        std::memset(logits, 0, vocab_size * sizeof(float));
        forward(transformer, token, pos, key_cache, value_cache, logits);
        next = sample(sampler, logits);
        pos++;

        // 输出模型回复
        if (user_idx >= num_prompt_tokens && next != 2) {
            char *piece = decode(tokenizer, token, next);
            safe_printf(piece);
            std::fflush(stdout);
        }
        if (next == 2) {
            std::cout << std::endl;
        }
    }

    // 收尾
    delete[] logits;
    delete[] key_cache;
    delete[] value_cache;
    free(prompt_tokens);
}

int main() {
    // 1) 初始化模型 & tokenizer & sampler
    std::string checkpoint_path = "modelq.bin";
    std::string tokenizer_path  = "tokenizer.bin";
    static Transformer<dim,hidden_dim,n_layers,n_heads,n_kv_heads,vocab_size,seq_len,GS> transformer;
    build_transformer(&transformer, checkpoint_path);

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    Sampler sampler;
    // 采样参数根据需要调整
    unsigned long long rng_seed = (unsigned long long)std::time(nullptr);
    const float temperature = 1.0f;
    const float topp        = 1.0f;
    build_sampler(&sampler, transformer.config.vocab_size,
                  temperature, topp, rng_seed);

    // 2) 直接进入 chat loop
    const int max_steps = 64;
    chat(&transformer, &tokenizer, &sampler,
         /*cli_user_prompt=*/nullptr, /*cli_system_prompt=*/nullptr,
         max_steps);

    // 3) 清理
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
