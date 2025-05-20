#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <cstdint>

// 用于二分合并搜索的词表条目
typedef struct {
    char *str;
    int   id;
} TokenIndex;

// 分词器状态
typedef struct {
    char       **vocab;
    float      *vocab_scores;
    TokenIndex *sorted_vocab;
    int         vocab_size;
    unsigned    max_token_length;
    unsigned char byte_pieces[512];
} Tokenizer;

// 构建/销毁
// 注意：和源码保持一致，第二个参数按值传递
void build_tokenizer(Tokenizer *t, const std::string &tokenizer_path, int vocab_size);
void free_tokenizer(Tokenizer *t);

// 编码／解码
// 注意：text 和 piece 都非 const，以匹配实现
void encode    (Tokenizer *t, const char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens);
char *decode   (Tokenizer *t, int prev_token, int token);
void safe_printf(const char *piece);

#endif // TOKENIZER_H
