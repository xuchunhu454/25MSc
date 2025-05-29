// firmware/tokenizer.h
#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <cstdint>

// 最大词表大小与最大 token 长度，请与实际模型保持一致
static constexpr int MAX_VOCAB     = 32000;
static constexpr int MAX_TOKEN_LEN = 32;  // 根据 tokenizer.bin 中 max_token_length=27 设置为 32

typedef struct {
    char *str;
    int   id;
} TokenIndex;

typedef struct {
    int    vocab_size;
    int    max_token_length;
    char   vocab[MAX_VOCAB][MAX_TOKEN_LEN+1];
    float  vocab_scores[MAX_VOCAB];
    TokenIndex sorted_vocab[MAX_VOCAB];
    unsigned char byte_pieces[256*2];
} Tokenizer;

void build_tokenizer(Tokenizer *t, const std::string &tokenizer_path, int vocab_size);
void free_tokenizer(Tokenizer *t);

void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens);
char *decode(Tokenizer *t, int prev_token, int token);
void safe_printf(char *piece);

#endif // TOKENIZER_H
