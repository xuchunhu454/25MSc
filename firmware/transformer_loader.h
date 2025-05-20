#ifndef TRANSFORMER_LOADER_H
#define TRANSFORMER_LOADER_H

#include <string>
#include "config.h"
#include "forward.h"

// 把 checkpoint.bin 里的权重读到 Transformer 对象里
template <
    int dim, int hidden_dim,
    int n_layers, int n_heads,
    int n_kv_heads,
    int vocab_size, int seq_len,
    int GS
>
void build_transformer(
    Transformer<dim,hidden_dim,n_layers,n_heads,n_kv_heads,vocab_size,seq_len,GS> *t,
    std::string checkpoint_path   // <— 这里跟 cpp 保持一致
);

#endif // TRANSFORMER_LOADER_H
