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
    // 模型与分词器文件
    std::string checkpoint_path = "stories110M.bin";
    std::string tokenizer_path  = "tokenizer.bin";
    char *prompt                = "Hello";
    int   steps                 = 64;

    // 1) 加载模型
    static Transformer<
        dim, hidden_dim, n_layers, n_heads,
        n_kv_heads, vocab_size, seq_len, GS
    > transformer;
    build_transformer(&transformer, checkpoint_path);

    // 2) 初始化分词器
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // 3) 对 prompt 做 tokenize
    int prompt_len        = std::strlen(prompt);
    int *prompt_tokens    = new int[prompt_len + 3];
    int  num_prompt_tokens= 0;
    encode(&tokenizer,
           prompt,
           /*bos=*/1, /*eos=*/0,
           prompt_tokens,
           &num_prompt_tokens);

    // 4) 采样参数
    unsigned long long rng_seed = (unsigned long long)std::time(nullptr);
    float temperature = 1.0f;
    float topp        = 1.0f;

    // 5) 分配缓存与输出 buffer
    float *logits      = new float[vocab_size];
    constexpr int kv_dim = (dim * n_kv_heads) / n_heads;
    float *key_cache   = new float[n_layers * seq_len * kv_dim]();
    float *value_cache = new float[n_layers * seq_len * kv_dim]();

    // 6) generation loop
    int token_id = prompt_tokens[0];
    int pos      = 0;
    while (pos < steps) {
        // HLS kernel 调用
        forward(&transformer,
                token_id,
                pos,
                key_cache,
                value_cache,
                logits);

        // 前几个 token 强制用 prompt 的
        int next = (pos < num_prompt_tokens - 1)
                 ? prompt_tokens[pos + 1]
                 : sample(logits,
                          transformer.config.vocab_size,
                          temperature,
                          topp,
                          &rng_seed);

        // 解码并打印
        char *piece = decode(&tokenizer, token_id, next);
        safe_printf(piece);
        std::fflush(stdout);

        if (next == 1) break;  // 遇到 BOS(token=1) 就结束
        token_id = next;
        pos++;
    }
    std::printf("\n");

    // 7) 清理
    delete[] logits;
    delete[] key_cache;
    delete[] value_cache;
    delete[] prompt_tokens;
    free_tokenizer(&tokenizer);

    return 0;
}
