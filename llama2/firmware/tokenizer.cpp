// firmware/tokenizer.cpp

#include "tokenizer.h"
#include <cstdio>
#include <cstring>
#include <cctype>
#include <cstdlib>

// ---------------------------------------------------------------------------
// compare_tokens：用于 qsort/bsearch，对比两个 TokenIndex.str
// ---------------------------------------------------------------------------
static int compare_tokens(const void *a, const void *b) {
    return std::strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
}

// ---------------------------------------------------------------------------
// str_lookup：在 sorted_vocab[0..vocab_size-1] 中二分查找字符串 str
// 返回对应 id，否则 -1
// ---------------------------------------------------------------------------
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

// 全局静态缓冲区声明：
// 1) 保存从文件读到的每个子词字符串
// 2) 保存对应的分数
// 3) sorted_vocab 用于二分查找；byte_pieces 用于 byte-fallback
// 4) str_buffer 用于 encode 时合并 candidate
static char           global_vocab[MAX_VOCAB][MAX_TOKEN_LEN + 1];
static float          global_vocab_scores[MAX_VOCAB];
static TokenIndex     global_sorted_vocab[MAX_VOCAB];
static unsigned char  global_byte_pieces[256 * 2];
static char           global_str_buffer[MAX_TOKEN_LEN * 2 + 3];

// ---------------------------------------------------------------------------
// build_tokenizer：
//   - 从二进制文件读取 max_token_length、vocab_scores、vocab
//   - 初始化 byte_pieces
//   - 构建并排序 sorted_vocab
// ---------------------------------------------------------------------------
void build_tokenizer(Tokenizer *t, const std::string &tokenizer_path, int vocab_size) {
    // 1) 保存 vocab_size，并读取 max_token_length
    t->vocab_size = vocab_size;
    t->max_token_length = 0;

    // 2) 初始化 byte_pieces：每个原始字节 b (0..255) 对应 {b, '\0'}
    for (int i = 0; i < 256; i++) {
        global_byte_pieces[i * 2]     = (unsigned char)i;
        global_byte_pieces[i * 2 + 1] = 0;
    }
    // 将全局缓冲的地址复制到结构体中
    std::memcpy(t->byte_pieces, global_byte_pieces, sizeof(global_byte_pieces));

    // 3) 打开 tokenizer 文件
    FILE *file = std::fopen(tokenizer_path.c_str(), "rb");
    if (!file) {
        std::fprintf(stderr, "cannot open %s\n", tokenizer_path.c_str());
        std::exit(1);
    }

    // 4) 读取 max_token_length
    int max_len = 0;
    if (std::fread(&max_len, sizeof(int), 1, file) != 1) {
        std::fprintf(stderr, "failed to read max_token_length\n");
        std::exit(1);
    }
    if (max_len > MAX_TOKEN_LEN) max_len = MAX_TOKEN_LEN;
    t->max_token_length = max_len;

    // 5) 临时缓冲区，用于读取每个 token 对应的字符串
    char tmp_buf[MAX_TOKEN_LEN + 1];

    // 6) 依次读取 vocab_scores[i]、长度 len、tmp_buf，再拷贝到 t->vocab[i]
    for (int i = 0; i < vocab_size; i++) {
        // 6.1) 读分数
        if (std::fread(&t->vocab_scores[i], sizeof(float), 1, file) != 1) {
            std::fprintf(stderr, "failed read vocab_scores[%d]\n", i);
            std::exit(1);
        }
        // 6.2) 读长度 len
        int len = 0;
        if (std::fread(&len, sizeof(int), 1, file) != 1) {
            std::fprintf(stderr, "failed read len[%d]\n", i);
            std::exit(1);
        }
        if (len > MAX_TOKEN_LEN) len = MAX_TOKEN_LEN;
        // 6.3) 读 len 字节到 tmp_buf
        if (std::fread(tmp_buf, 1, (size_t)len, file) != (size_t)len) {
            std::fprintf(stderr, "failed read vocab[%d]\n", i);
            std::exit(1);
        }
        tmp_buf[len] = '\0';  // 添加 NUL 终结符
        // 6.4) 拷贝 tmp_buf 到结构体内的 vocab[i]
        std::strncpy(t->vocab[i], tmp_buf, (size_t)len + 1);
    }
    std::fclose(file);

    // 7) 构建 sorted_vocab 并 qsort 排序
    for (int i = 0; i < vocab_size; i++) {
        global_sorted_vocab[i].str = t->vocab[i];
        global_sorted_vocab[i].id  = i;
    }
    std::qsort(
        global_sorted_vocab,
        (size_t)vocab_size,
        sizeof(TokenIndex),
        compare_tokens
    );
    // 拷贝到结构体内
    std::memcpy(t->sorted_vocab, global_sorted_vocab, sizeof(TokenIndex) * (size_t)vocab_size);
}

// ---------------------------------------------------------------------------
// free_tokenizer：由于所有缓冲区均为静态分配，无需释放
// ---------------------------------------------------------------------------
void free_tokenizer(Tokenizer *t) {
    (void)t;
}

// ---------------------------------------------------------------------------
// encode：拆分文本 text 为 token ID 序列
//   - 按官方仓库逻辑：BOS -> dummy prefix 空格 -> UTF-8 子串查找 & byte-fallback
//   - 合并最优相邻 pair -> 添加 EOS
// ---------------------------------------------------------------------------
void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    if (text == nullptr) {
        std::fprintf(stderr, "encode NULL text\n");
        std::exit(1);
    }

    *n_tokens = 0;

    // 1) 添加 BOS
    if (bos) {
        tokens[(*n_tokens)++] = 1;
    }
    // 2) 如果 text 非空，则先插入空格 token（dummy prefix）
    if (*text != '\0') {
        int dp = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        if (dp >= 0) {
            tokens[(*n_tokens)++] = dp;
        }
    }

    // 3) 准备用于累积 UTF-8 码点的临时缓冲 local_buf
    char local_buf[MAX_TOKEN_LEN + 1];
    size_t str_len = 0;

    for (char *c = text; *c; c++) {
        // 当当前字节不是 continuation byte 时，重置 str_len
        if (((unsigned char)*c & 0xC0) != 0x80) {
            str_len = 0;
        }
        local_buf[str_len++] = *c;
        local_buf[str_len]   = '\0';

        // 如果下一个字节仍是 continuation byte，且未超 max_token_length，则继续累积
        if (((*(c + 1) & 0xC0) == 0x80) &&
            (str_len < (size_t)t->max_token_length)) {
            continue;
        }
        // 此时 local_buf 存放一个完整的 UTF-8 码点或已经达到长度上限
        int id = str_lookup(local_buf, t->sorted_vocab, t->vocab_size);
        if (id >= 0) {
            tokens[(*n_tokens)++] = id;
        } else {
            // byte-fallback：把 local_buf 中每个字节作为单独 token
            for (size_t i = 0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)local_buf[i] + 3;
            }
        }
        str_len = 0;
    }

    // 4) 合并最优相邻 pair（重复执行，直到无法合并）
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

    // 5) 添加 EOS
    if (eos) {
        tokens[(*n_tokens)++] = 2;
    }
}

// ---------------------------------------------------------------------------
// decode：将单个 token ID 转换为子词片段 (piece)
//   - 如果 prev_token == BOS (1) 且 piece 以空格开头，需删去空格
//   - 如果 piece 形如 "<0xXX>"，解析为单原始字节字符串
//   - 返回指向 t->vocab[token] 或 byte_pieces[b*2] 的指针
// ---------------------------------------------------------------------------
char *decode(Tokenizer *t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // 删除 BOS 后的前导空格
    if (prev_token == 1 && piece[0] == ' ') {
        piece++;
    }
    // 如果形如 "<0xXX>"
    unsigned int b;
    if (std::sscanf(piece, "<0x%02X>", &b) == 1) {
        piece = (char *)&t->byte_pieces[b * 2];
    }
    return piece;
}

// ---------------------------------------------------------------------------
// safe_printf：仅打印可打印字符及空格，屏蔽其他控制字符
// ---------------------------------------------------------------------------
void safe_printf(char *piece) {
    if (!piece || !*piece) {
        return;
    }
    // 如果只有一个字符，就判断它是否 isprint 或 isspace
    if (!piece[1]) {
        unsigned char bv = (unsigned char)piece[0];
        if (!(std::isprint(bv) || std::isspace(bv))) {
            return;
        }
    }
    std::printf("%s", piece);
}
