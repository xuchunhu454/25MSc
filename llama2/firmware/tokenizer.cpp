// firmware/tokenizer.cpp
#include "tokenizer.h"
#include <cstdio>
#include <cstring>
#include <cctype>
#include <cstdlib>

// 插入排序构建 sorted_vocab
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

// 二分查找
static int str_lookup(const char *str, Tokenizer *t) {
    int lo = 0, hi = t->vocab_size - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        int cmp = std::strcmp(str, t->sorted_vocab[mid].str);
        if (cmp == 0)      return t->sorted_vocab[mid].id;
        else if (cmp < 0)  hi = mid - 1;
        else               lo = mid + 1;
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

    // 初始化 byte_pieces
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i*2]   = static_cast<unsigned char>(i);
        t->byte_pieces[i*2+1] = 0;
    }

    build_sorted_vocab(t);
}

void free_tokenizer(Tokenizer *t) {
    // no-op for static version
    (void)t;
}

void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    if (!text) { std::fprintf(stderr, "encode NULL text\n"); std::exit(1); }
    static char str_buffer[MAX_TOKEN_LEN*2 + 3];

    *n_tokens = 0;
    if (bos) tokens[(*n_tokens)++] = 1;
    if (*text) {
        int dp = str_lookup(" ", t);
        if (dp >= 0) tokens[(*n_tokens)++] = dp;
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

    // 合并最佳 pair
    while (true) {
        float best_score = -1e10f;
        int best_id = -1, best_idx = -1;
        for (int i = 0; i + 1 < *n_tokens; i++) {
            std::snprintf(str_buffer, sizeof(str_buffer), "%s%s",
                          t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int cid = str_lookup(str_buffer, t);
            if (cid >= 0 && t->vocab_scores[cid] > best_score) {
                best_score = t->vocab_scores[cid];
                best_id    = cid;
                best_idx   = i;
            }
        }
        if (best_idx < 0) break;
        tokens[best_idx] = best_id;
        for (int j = best_idx+1; j + 1 < *n_tokens; j++) {
            tokens[j] = tokens[j+1];
        }
        (*n_tokens)--;
    }

    if (eos) tokens[(*n_tokens)++] = 2;
}

char *decode(Tokenizer *t, int prev_token, int token) {
    char *piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') piece++;
    if (piece[0] == '<') {
        unsigned int b;
        if (std::sscanf(piece, "<0x%02X>", &b) == 1) {
            // 正确地将 unsigned char* 转为 char*
            piece = reinterpret_cast<char*>(t->byte_pieces + (b & 0xFF)*2);
        }
    }
    return piece;
}

void safe_printf(char *piece) {
    if (!piece || !*piece) return;
    if (!piece[1]) {
        unsigned char b = static_cast<unsigned char>(piece[0]);
        if (!(std::isprint(b) || std::isspace(b))) return;
    }
    std::printf("%s", piece);
}
