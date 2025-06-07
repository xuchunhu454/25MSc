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
    // 1) 初始化模型和 tokenizer
    std::string checkpoint_path = "modelq.bin";
    std::string tokenizer_path  = "tokenizer.bin";
    static Transformer<
        dim, hidden_dim, n_layers, n_heads,
        n_kv_heads, vocab_size, seq_len, GS
    > transformer;
    build_transformer(&transformer, checkpoint_path);

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // 2) 采样参数
    unsigned long long rng_seed = (unsigned long long)std::time(nullptr);
    const float temperature = 1.0f;
    const float topp        = 1.0f;
    const int   max_steps   = 64;

    // 3) 全局缓存（对话状态在多轮中累积）
    constexpr int kv_dim = (dim * n_kv_heads) / n_heads;
    float *key_cache   = new float[n_layers * seq_len * kv_dim]();
    float *value_cache = new float[n_layers * seq_len * kv_dim]();
    float *logits      = new float[vocab_size];

    // 4) 对话主循环
    while (true) {
        std::string prompt;
        std::cout << "\nUser> ";
        if (!std::getline(std::cin, prompt)) break;
        if (prompt == "exit" || prompt == "quit") break;

        // 4.1) encode 用户输入
        int prompt_len        = prompt.size();
        int *prompt_tokens    = new int[prompt_len * 4 + 3]; // worst-case UTF-8，4 bytes per char
        int  num_prompt_tokens = 0;
        encode(&tokenizer,
               const_cast<char*>(prompt.c_str()),
               /*bos=*/1, /*eos=*/0,
               prompt_tokens,
               &num_prompt_tokens);

        // 4.2) 把 prompt tokens “埋进” cache
        int pos = 0;
        int token_id;
        // 用 prompt 里的每个 token 喂给 forward，只做一次，不 sample
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

        // 4.3) 模型回复生成
        std::cout << "Model> ";
        for (int step = 0; step < max_steps; step++) {
            // 采样下一个 token
            int next = sample(logits,
                              transformer.config.vocab_size,
                              temperature,
                              topp,
                              &rng_seed);
            // 如果是 EOS，结束本轮
            if (next == 2) break;

            // 输出这个 token
            char *piece = decode(&tokenizer, token_id, next);
            safe_printf(piece);

            // 用这个 token 继续 forward
            token_id = next;
            std::memset(logits, 0, vocab_size * sizeof(float));
            forward(&transformer,
                    token_id,
                    pos,
                    key_cache,
                    value_cache,
                    logits);
            pos++;
            if (pos >= seq_len) break; // 避免超过最大 seq 长度
        }
        std::cout << std::endl;

        delete[] prompt_tokens;
        // 继续下一轮，不要清空 key_cache/value_cache，让模型保留对话上下文
    }

    // 5) 清理
    delete[] logits;
    delete[] key_cache;
    delete[] value_cache;
    free_tokenizer(&tokenizer);
    return 0;
}
