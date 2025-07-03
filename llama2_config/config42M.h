#pragma once
#include "typedefs.h"

// 42M model
static constexpr int dim = 512;
static constexpr int hidden_dim = 1376;
static constexpr int n_layers = 8;
static constexpr int n_heads = 8;
static constexpr int n_kv_heads = 8;
static constexpr int vocab_size = 32000;
static constexpr int seq_len = 1024;
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
