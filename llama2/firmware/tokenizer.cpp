// firmware/tokenizer.cpp

#include "tokenizer.h"
#include <cstdio>
#include <cstring>
#include <cctype>
#include <cstdlib>

// ---------------------------------------------------------------------------------
// 1. compare_tokens：用于 qsort / bsearch，对比两个 TokenIndex 字符串部分
// ---------------------------------------------------------------------------------
static int compare_tokens(const void *a, const void *b) {
    return std::strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
}

// ---------------------------------------------------------------------------------
// 2. str_lookup：在 sorted_vocab[0..vocab_size-1] 中二分查找给定字符串
//    返回其对应 id，否则返回 -1。
// ---------------------------------------------------------------------------------
static int str_lookup(const char *str, TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex key;
    key.str = const_cast<char*>(str);
    TokenIndex *res = (TokenIndex *)std::bsearch(
        &key,
        sorted_vocab,
        (size_t)vocab_size,
        sizeof(TokenIndex),
        compare_tokens
    );
    return (res == nullptr) ? -1 : res->id;
}

// ---------------------------------------------------------------------------------
// 3. 全局静态缓冲区声明：
//    - vocab[MAX_VOCAB][MAX_TOKEN_LEN+1]：保存从文件中读取的子词字符串
//    - vocab_scores[MAX_VOCAB]：保存每个子词的分数
//    - sorted_vocab[MAX_VOCAB]：用于二分查找的 (str, id) 对
//    - byte_pieces[256*2]：保存 byte-fallback 时用到的 "<0xXX>" 映射回单字节
//    - str_buffer[MAX_TOKEN_LEN*2+3]：encode 合并 candidate 时使用的临时缓冲
// ---------------------------------------------------------------------------------
static char           vocab[MAX_VOCAB][MAX_TOKEN_LEN + 1];
static float          vocab_scores[MAX_VOCAB];
static TokenIndex     sorted_vocab[MAX_VOCAB];
static unsigned char  byte_pieces[256 * 2];
static char           str_buffer[MAX_TOKEN_LEN * 2 + 3];  // 合并候选子词时临时使用

// ---------------------------------------------------------------------------------
// 4. build_tokenizer：
//    - 从 tokenizer_path 文件里读取 max_token_length, vocab_scores 及 vocab 数组
//    - 将它们分别存到 t->vocab_scores[i]、t->vocab[i]
//    - 再把所有 (vocab[i], i) 拷贝到 sorted_vocab，并 qsort 排序
//    - 初始化 byte_pieces 数组
// ---------------------------------------------------------------------------------
void build_tokenizer(Tokenizer *t, const std::string &tokenizer_path, int vocab_size) {
    // 4.1) 保存 vocab_size，并读入 max_token_length
    t->vocab_size       = vocab_size;
    t->max_token_length = 0;

    // 4.2) 初始化 byte_pieces：每个字节 b 对应 { (unsigned char)b, '\0' }
    for (int i = 0; i < 256; i++) {
        byte_pieces[i * 2]     = (unsigned char)i;
        byte_pieces[i * 2 + 1] = 0;
    }
    t->byte_pieces = byte_pieces;

    // 4.3) 打开 tokenizer 文件并读取
    FILE *file = std::fopen(tokenizer_path.c_str(), "rb");
    if (!file) {
        std::fprintf(stderr, "cannot open %s\n", tokenizer_path.c_str());
        std::exit(1);
    }

    // 4.4) 读取 max_token_length
    int max_len = 0;
    if (std::fread(&max_len, sizeof(int), 1, file) != 1) {
        std::fprintf(stderr, "failed to read max_token_length\n");
        std::exit(1);
    }
    if (max_len > MAX_TOKEN_LEN) max_len = MAX_TOKEN_LEN;
    t->max_token_length = max_len;

    // 4.5) 准备一个临时缓冲 tmp_buf，用于读取每个子词字符串
    char tmp_buf[MAX_TOKEN_LEN + 1];

    // 4.6) 依次读取 vocab_scores[i]、长度 len、字符内容
    for (int i = 0; i < vocab_size; i++) {
        // 读分数
        if (std::fread(&t->vocab_scores[i], sizeof(float), 1, file) != 1) {
            std::fprintf(stderr, "failed read vocab_scores[%d]\n", i);
            std::exit(1);
        }
        // 读长度 len
        int len = 0;
        if (std::fread(&len, sizeof(int), 1, file) != 1) {
            std::fprintf(stderr, "failed read len[%d]\n", i);
            std::exit(1);
        }
        if (len > MAX_TOKEN_LEN) len = MAX_TOKEN_LEN;
        // 读 len 字节至 tmp_buf
        if (std::fread(tmp_buf, 1, (size_t)len, file) != (size_t)len) {
            std::fprintf(stderr, "failed read vocab[%d]\n", i);
            std::exit(1);
        }
        tmp_buf[len] = '\0';
        std::strncpy(t->vocab[i], tmp_buf, (size_t)len + 1);
    }
    std::fclose(file);

    // 4.7) 构建 sorted_vocab 数组并 qsort 排序
    for (int i = 0; i < vocab_size; i++) {
        sorted_vocab[i].str = t->vocab[i];
        sorted_vocab[i].id  = i;
    }
    std::qsort(
        sorted_vocab,
        (size_t)vocab_size,
        sizeof(TokenIndex),
        compare_tokens
    );

    // 4.8) 把静态数组挂回到结构体里
    t->vocab         = (char (*)[MAX_TOKEN_LEN + 1])vocab;
    t->vocab_scores  = vocab_scores;
    t->sorted_vocab  = sorted_vocab;
}

// ---------------------------------------------------------------------------------
// 5. free_tokenizer：不执行任何操作，因为所有存储都在静态数组里
// ---------------------------------------------------------------------------------
void free_tokenizer(Tokenizer *t) {
    (void)t;
}

// ---------------------------------------------------------------------------------
// 6. encode：把文本拆分成若干子词 token ID
//    逻辑与官方仓库完全一致，只是使用静态 str_buffer 代替动态分配
// ---------------------------------------------------------------------------------
void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos,
            int *tokens, int *n_tokens) {
    if (text == nullptr) {
        std::fprintf(stderr, "encode NULL text\n");
        std::exit(1);
    }
    *n_tokens = 0;

    // 6.1) 可选添加 BOS(=1)
    if (bos) {
        tokens[(*n_tokens)++] = 1;
    }
    // 6.2) 如果 text 非空，先把空格当 dummy prefix
    if (*text != '\0') {
        int dp = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        if (dp >= 0) {
            tokens[(*n_tokens)++] = dp;
        }
    }

    // 6.3) 局部缓冲 local_buf 用于累积 UTF-8 码点
    char local_buf[MAX_TOKEN_LEN + 1];
    size_t str_len = 0;
    for (char *c = text; *c; c++) {
        if (((unsigned char)*c & 0xC0) != 0x80) {
            str_len = 0;
        }
        local_buf[str_len++] = *c;
        local_buf[str_len]   = '\0';

        if (((*(c + 1) & 0xC0) == 0x80) &&
            (str_len < (size_t)t->max_token_length)) {
            continue;
        }
        // 已读满一个 UTF-8 码点或到长度上限，尝试在词表中查找
        int id = str_lookup(local_buf, t->sorted_vocab, t->vocab_size);
        if (id >= 0) {
            tokens[(*n_tokens)++] = id;
        } else {
            // byte-level fallback
            for (size_t i = 0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)local_buf[i] + 3;
            }
        }
        str_len = 0;
    }

    // 6.4) 合并最佳相邻 pair，循环直到无可合并
    while (true) {
        float best_score = -1e10f;
        int best_id = -1, best_idx = -1;
        for (int i = 0; i + 1 < *n_tokens; i++) {
            std::snprintf(
                str_buffer,
                sizeof(str_buffer),
                "%s%s",
                t->vocab[tokens[i]],
                t->vocab[tokens[i + 1]]
            );
            int cid = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (cid >= 0 && t->vocab_scores[cid] > best_score) {
                best_score = t->vocab_scores[cid];
                best_id    = cid;
                best_idx   = i;
            }
        }
        if (best_idx < 0) {
            break;
        }
        tokens[best_idx] = best_id;
        for (int j = best_idx + 1; j + 1 < *n_tokens; j++) {
            tokens[j] = tokens[j + 1];
        }
        (*n_tokens)--;
    }

    // 6.5) 可选添加 EOS(=2)
    if (eos) {
        tokens[(*n_tokens)++] = 2;
    }
}

// ---------------------------------------------------------------------------------
// 7. decode：把单个 token ID 转换成一个子词片段 (piece)
//    - 若 prev_token == 1 且 piece 以空格开头，则删除前导空格
//    - 若 piece 形如 "<0xXX>"，则转换回对应单字节
//    - 否则直接返回 t->vocab[token]
// ---------------------------------------------------------------------------------
char *decode(Tokenizer *t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // 删除 BOS 之后的前导空格
    if (prev_token == 1 && piece[0] == ' ') {
        piece++;
    }
    // 解析 "<0xXX>" 格式
    unsigned int b;
    if (std::sscanf(piece, "<0x%02X>", &b) == 1) {
        piece = (char *)&t->byte_pieces[b * 2];
    }
    return piece;
}

// ---------------------------------------------------------------------------------
// 8. safe_printf：只打印“可打印字符”或空格，屏蔽其他控制字符
// ---------------------------------------------------------------------------------
void safe_printf(char *piece) {
    if (!piece || !*piece) return;
    if (!piece[1]) {
        unsigned char bv = (unsigned char)piece[0];
        if (!(std::isprint(bv) || std::isspace(bv))) {
            return;
        }
    }
    std::printf("%s", piece);
}
