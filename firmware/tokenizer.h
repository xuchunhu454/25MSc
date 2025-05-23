#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <cstdint>

// Wordlist entries for binary merge search
typedef struct {
    char *str;
    int   id;
} TokenIndex;

// Tokenizer Status
typedef struct {
    char       **vocab;
    float      *vocab_scores;
    TokenIndex *sorted_vocab;
    int         vocab_size;
    unsigned    max_token_length;
    unsigned char byte_pieces[512];
} Tokenizer;


void build_tokenizer(Tokenizer *t, const std::string &tokenizer_path, int vocab_size);
void free_tokenizer(Tokenizer *t);


void encode    (Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens);
char *decode   (Tokenizer *t, int prev_token, int token);
void safe_printf(char *piece);

#endif // TOKENIZER_H
