// firmware/tokenizer.h
#pragma once
#include <string>
#include <cstdint>

// Maximum vocabulary size and token length (adjust if needed)
static constexpr int MAX_VOCAB = 32000;
static constexpr int MAX_TOKEN_LEN = 8;

// Tokenizer struct with static arrays to avoid dynamic allocations in CSIM
struct Tokenizer {
    int vocab_size;
    int max_token_length;
    char vocab[MAX_VOCAB][MAX_TOKEN_LEN+1];    // vocabulary strings
    float vocab_scores[MAX_VOCAB];             // vocabulary scores
    struct { char *str; int id; } sorted_vocab[MAX_VOCAB];  // sorted pointers for binary search
    char byte_pieces[256*2];                   // raw byte fallback tokens
};

// Build tokenizer from binary file
void build_tokenizer(Tokenizer *t, const std::string &tokenizer_path, int vocab_size);

// Encode text into token IDs
void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens);

// Decode a token ID into a piece of text (handles raw bytes and leading spaces)
char *decode(Tokenizer *t, int prev_token, int token);

// Print a decoded piece safely
void safe_printf(char *piece);


// firmware/tokenizer.cpp
#include "tokenizer.h"
#include <cstdio>
#include <cstring>
#include <cctype>
#include <cstdlib>

// Helper: build sorted_vocab using insertion sort
static void build_sorted_vocab(Tokenizer *t) {
    for (int i = 0; i < t->vocab_size; i++) {
        t->sorted_vocab[i].str = t->vocab[i];
        t->sorted_vocab[i].id  = i;
    }
    for (int i = 1; i < t->vocab_size; i++) {
        auto key = t->sorted_vocab[i];
        int j = i - 1;
        while (j >= 0 && std::strcmp(t->sorted_vocab[j].str, key.str) > 0) {
            t->sorted_vocab[j+1] = t->sorted_vocab[j];
            j--;
        }
        t->sorted_vocab[j+1] = key;
    }
}

// Helper: binary search in sorted_vocab
static int str_lookup(const char *str, Tokenizer *t) {
    int lo = 0, hi = t->vocab_size - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        int cmp = std::strcmp(str, t->sorted_vocab[mid].str);
        if (cmp == 0) return t->sorted_vocab[mid].id;
        if (cmp < 0) hi = mid - 1;
        else        lo = mid + 1;
    }
    return -1;
}

void build_tokenizer(Tokenizer *t, const std::string &tokenizer_path, int vocab_size) {
    t->vocab_size = vocab_size;
    FILE *file = std::fopen(tokenizer_path.c_str(), "rb");
    if (!file) { std::fprintf(stderr, "cannot open %s\n", tokenizer_path.c_str()); std::exit(1); }
    if (std::fread(&t->max_token_length, sizeof(int), 1, file) != 1) std::exit(1);

    for (int i = 0; i < vocab_size; i++) {
        if (std::fread(&t->vocab_scores[i], sizeof(float), 1, file) != 1) std::exit(1);
        int len; if (std::fread(&len, sizeof(int), 1, file) != 1) std::exit(1);
        if (len > MAX_TOKEN_LEN) len = MAX_TOKEN_LEN;
        std::fread(t->vocab[i], 1, len, file);
        t->vocab[i][len] = '\0';
    }
    std::fclose(file);

    // byte pieces initialization
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i*2]   = static_cast<char>(i);
        t->byte_pieces[i*2+1] = '\0';
    }

    build_sorted_vocab(t);
}

void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    if (!text) { std::fprintf(stderr, "encode NULL text\n"); std::exit(1); }
    static char str_buffer[MAX_TOKEN_LEN*2 + 3];
    *n_tokens = 0;
    if (bos) tokens[(*n_tokens)++] = 1;
    if (text[0]) {
        int dp = str_lookup(" ", t);
        tokens[(*n_tokens)++] = dp;
    }
    size_t str_len = 0;
    for (char *c = text; *c; c++) {
        if ((*c & 0xC0) != 0x80) str_len = 0;
        str_buffer[str_len++] = *c;
        str_buffer[str_len]   = '\0';
        if ((*(c+1)&0xC0)==0x80 && str_len < (size_t)t->max_token_length) continue;
        int id = str_lookup(str_buffer, t);
        if (id >= 0) {
            tokens[(*n_tokens)++] = id;
        } else {
            for (size_t i = 0; i < str_len; i++) {
                tokens[(*n_tokens)++] = static_cast<unsigned char>(str_buffer[i]) + 3;
            }
        }
        str_len = 0;
    }
    while (0) {} // no merging in CSIM
    if (eos) tokens[(*n_tokens)++] = 2;
}

char *decode(Tokenizer *t, int prev_token, int token) {
    char *piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') piece++;
    // raw byte token handling
    if (piece[0] == '<' && std::sscanf(piece, "<0x%02hhX>", (unsigned char*)&piece[0]) == 1) {
        unsigned char byte_val;
        std::sscanf(piece, "<0x%02hhX>", &byte_val);
        piece = t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    if (!piece || !piece[0]) return;
    if (!piece[1]) {
        unsigned char b = static_cast<unsigned char>(piece[0]);
        if (!(std::isprint(b) || std::isspace(b))) return;
    }
    std::printf("%s", piece);
}


// extern "C" int sample(
//     float *logits,
//     int    vocab_size,
//     float  temperature,
//     float  topp,
//     unsigned long long *rng_state
// ) {
//     // 用内部接口构造一个临时 Sampler
//     Sampler s;
//     build_sampler(&s, vocab_size, temperature, topp, *rng_state);
//     
//     int tok = sample(&s, logits);
//     
//     free_sampler(&s);
//     return tok;
// }
