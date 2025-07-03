#pragma once
#include "typedefs.h"

// 260K model
static constexpr int dim = 64;
static constexpr int hidden_dim = 172;
static constexpr int n_layers = 5;
static constexpr int n_heads = 8;
static constexpr int n_kv_heads = 4;
static constexpr int vocab_size = 512;
static constexpr int seq_len = 512;
static constexpr int GS = 64;


constexpr Config config = {
    .dim = dim,
    .hidden_dim = hidden_dim,
    .n_layers = n_layers,
    .n_heads = n_heads,
    .n_kv_heads = n_kv_heads,
    .vocab_size = vocab_size,
    .seq_len = seq_len,
    .GS = GS,
};
