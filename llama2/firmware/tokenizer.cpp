// firmware/tokenizer.cpp

#include "tokenizer.h"
#include <cstdio>
#include <cstring>
#include <cctype>
#include <cstdlib>

// ---------------------------------------------------------------------------------
// 1. 辅助比较函数，用于 qsort/bsearch，在 sorted_vocab 中按照 str 升序排列
// ---------------------------------------------------------------------------------
static int compare_tokens(const void *a, const void *b) {
    return std::strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
}

// ---------------------------------------------------------------------------------
// 2. 二分查找函数：在 sorted_vocab[0..vocab_size-1] 中查找 str
//    返回对应 TokenIndex.id，如果找不到则返回 -1。
// ---------------------------------------------------------------------------------
static int str_lookup(const char *str, TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex key;
    key.str = const_cast<char*>(str);
    // key.id 对搜索无影响，只查 key.str
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
// 3. build_tokenizer：
//    - 从二进制文件 “tokenizer_path” 中读取 max_token_length、vocab_scores、vocab[] 字符串
//    - 将它们填入 t->vocab_scores[i], t->vocab[i]
//    - 然后把 t->vocab[i]、i 一一拷贝到 t->sorted_vocab，最后调用 qsort 排序
//    - 初始化 t->byte_pieces 数组
// ---------------------------------------------------------------------------------
void build_tokenizer(Tokenizer *t, const std::string &tokenizer_path, int vocab_size) {
    // 3.1) 保存 vocab_size，并初始化 max_token_length
    t->vocab_size = vocab_size;
    t->max_token_length = 0;

    // 3.2) 初始化 byte_pieces：每个原始字节 b (0..255) 对应两个 char：{ (unsigned char)b, '\0' }
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2]     = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = 0;
    }

    // 3.3) 打开二进制 tokenizer 文件
    FILE *file = std::fopen(tokenizer_path.c_str(), "rb");
    if (!file) {
        std::fprintf(stderr, "cannot open %s\n", tokenizer_path.c_str());
        std::exit(1);
    }

    // 3.4) 读取 max_token_length（int）
    int max_len = 0;
    if (std::fread(&max_len, sizeof(int), 1, file) != 1) {
        std::fprintf(stderr, "failed to read max_token_length\n");
        std::exit(1);
    }
    // 文件里可能写了一个略大的值，我们限制到 MAX_TOKEN_LEN
    if (max_len > MAX_TOKEN_LEN) max_len = MAX_TOKEN_LEN;
    t->max_token_length = max_len;

    // 临时缓冲区，用于读取每个 token 的字符串
    // 大小 = MAX_TOKEN_LEN + 1（NUL 终结符）
    // 我们直接用栈上的数组，不需要动态分配
    char tmp_buf[MAX_TOKEN_LEN + 1];

    // 3.5) 按照官方格式依次读取每个词的 score 和字符串
    for (int i = 0; i < vocab_size; i++) {
        // 读取 vocab_scores[i]
        if (std::fread(&t->vocab_scores[i], sizeof(float), 1, file) != 1) {
            std::fprintf(stderr, "failed read vocab_scores[%d]\n", i);
            std::exit(1);
        }
        // 读取当前 token 的长度 len（int）
        int len = 0;
        if (std::fread(&len, sizeof(int), 1, file) != 1) {
            std::fprintf(stderr, "failed read len[%d]\n", i);
            std::exit(1);
        }
        if (len > MAX_TOKEN_LEN) len = MAX_TOKEN_LEN;
        // 读取 len 字节到 tmp_buf[0..len-1]
        if (std::fread(tmp_buf, 1, (size_t)len, file) != (size_t)len) {
            std::fprintf(stderr, "failed read vocab[%d]\n", i);
            std::exit(1);
        }
        // 添上终结符
        tmp_buf[len] = '\0';
        // 拷贝到 t->vocab[i][]
        std::strncpy(t->vocab[i], tmp_buf, (size_t)len + 1);
    }
    std::fclose(file);

    // 3.6) 构建并排序 sorted_vocab
    // 每个 sorted_vocab[i] = { t->vocab[i], i }
    for (int i = 0; i < vocab_size; i++) {
        t->sorted_vocab[i].str = t->vocab[i];
        t->sorted_vocab[i].id  = i;
    }
    std::qsort(
        t->sorted_vocab,
        (size_t)vocab_size,
        sizeof(TokenIndex),
        compare_tokens
    );
}

// ---------------------------------------------------------------------------------
// 4. free_tokenizer：
//    由于我们所有数据都放在 static 数组里，不需要做任何释放
// ---------------------------------------------------------------------------------
void free_tokenizer(Tokenizer *t) {
    (void)t;
}

// ---------------------------------------------------------------------------------
// 5. encode：把输入的 UTF-8 文本 text 拆解成 WordPiece/BPE token ID 序列
//    - bos != 0 时先加 BOS(=1)，eos != 0 时最后加 EOS(=2)
//    - 使用二分查找 str_lookup 在 sorted_vocab 中匹配最长字符串
//    - 若找不到，就对每个字节做 “byte-fallback”：令 token = (unsigned char)byte + 3
//    - 然后循环执行“合并最优相邻 pair”操作，直到无法再合并
// ---------------------------------------------------------------------------------
void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos,
            int *tokens, int *n_tokens) {
    if (text == nullptr) {
        std::fprintf(stderr, "encode NULL text\n");
        std::exit(1);
    }

    *n_tokens = 0;

    // 5.1) 可选地加 BOS token (id = 1)
    if (bos) {
        tokens[(*n_tokens)++] = 1;
    }

    // 5.2) 如果 text 非空，就先把一个“空格”当 dummy prefix
    if (*text != '\0') {
        int dp = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        if (dp >= 0) {
            tokens[(*n_tokens)++] = dp;
        }
    }

    // 5.3) 准备一个局部 str_buffer，用于累积 UTF-8 码点
    // 大小 = MAX_TOKEN_LEN + 1
    char local_buf[MAX_TOKEN_LEN + 1];
    size_t str_len = 0;

    for (char *c = text; *c; c++) {
        // 如果当前 byte 不是 continuation byte (0b10xxxxxx)，则重置 str_len
        if (((unsigned char)*c & 0xC0) != 0x80) {
            str_len = 0;
        }
        local_buf[str_len++] = *c;
        local_buf[str_len]   = '\0';

        // 如果下一个 byte 仍然是 continuation byte，且长度未超 max_token_length，则继续累积
        if (((*(c + 1) & 0xC0) == 0x80) &&
            (str_len < (size_t)t->max_token_length)) {
            continue;
        }
        // 走到这里，表示已累积了一个完整的 UTF-8 码点（或达到长度上限），尝试在词表中查找
        int id = str_lookup(local_buf, t->sorted_vocab, t->vocab_size);
        if (id >= 0) {
            tokens[(*n_tokens)++] = id;
        } else {
            // Byte-level fallback：对 local_buf[0..str_len-1] 中的每个 byte 单独编码
            for (size_t i = 0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)local_buf[i] + 3;
            }
        }
        str_len = 0;
    }

    // 5.4) 合并最优 consecutive pair，一直重复直到没法再合并
    //     每次找到一个能拼成词表中子词的 pair，且 score 最大的那个，然后合并
    while (true) {
        float best_score = -1e10f;
        int best_id = -1, best_idx = -1;

        // 遍历所有相邻 pair，构建串 local_buf = vocab[tokens[i]] + vocab[tokens[i+1]]
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
        // 合并最佳 pair：tokens[best_idx] = best_id；删除 tokens[best_idx+1]
        tokens[best_idx] = best_id;
        for (int j = best_idx + 1; j + 1 < *n_tokens; j++) {
            tokens[j] = tokens[j + 1];
        }
        (*n_tokens)--;
    }

    // 5.5) 可选地加 EOS (id = 2)
    if (eos) {
        tokens[(*n_tokens)++] = 2;
    }
}

// ---------------------------------------------------------------------------------
// 6. decode：将单个 token ID 转换为子词字符串 (piece)
//    - 如果 prev_token == BOS(1) 且 piece 以空格开头，就删掉空格
//    - 如果 piece 的格式为 "<0xXX>"，就把它解析为一个原始字节
//    - 返回一个指向 t->vocab[token] (或对应 byte_pieces) 的指针
// ---------------------------------------------------------------------------------
char *decode(Tokenizer *t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // 如果上一个 token 是 BOS(1) 且 piece 以' '开头，就删掉前导空格
    if (prev_token == 1 && piece[0] == ' ') {
        piece++;
    }
    // 检查是否形如 "<0xXX>"
    unsigned int b;
    if (std::sscanf(piece, "<0x%02X>", &b) == 1) {
        // b ∈ [0,255]，直接从 t->byte_pieces 拿对应两字节序列
        piece = (char *)&t->byte_pieces[b * 2];
    }
    return piece;
}

// ---------------------------------------------------------------------------------
// 7. safe_printf：只打印“可打印字符”或“空格”，屏蔽不想要的控制码
// ---------------------------------------------------------------------------------
void safe_printf(char *piece) {
    if (!piece || !(*piece)) return;
    // 如果 piece 长度 == 1，就检查该字符是否为 isprint/ispace
    if (!piece[1]) {
        unsigned char bv = (unsigned char)piece[0];
        if (!(std::isprint(bv) || std::isspace(bv))) {
            return;
        }
    }
    std::printf("%s", piece);
}
