#pragma once
#include "typedefs.h"

// 110M model
//static constexpr int dim = 768;
//static constexpr int hidden_dim = 2048;
//static constexpr int n_layers = 12;
//static constexpr int n_heads = 12;
//static constexpr int n_kv_heads = 12;
//static constexpr int vocab_size = 32000;
//static constexpr int seq_len = 1024;
//static constexpr int GS = 64;

// 42M model
//static constexpr int dim = 512;
//static constexpr int hidden_dim = 1376;
//static constexpr int n_layers = 8;
//static constexpr int n_heads = 8;
//static constexpr int n_kv_heads = 8;
//static constexpr int vocab_size = 32000;
//static constexpr int seq_len = 1024;
//static constexpr int GS = 64;

// 15M model
static constexpr int dim = 288;
static constexpr int hidden_dim = 768;
static constexpr int n_layers = 6;
static constexpr int n_heads = 6;
static constexpr int n_kv_heads = 6;
static constexpr int vocab_size = 32000;
static constexpr int seq_len = 256;
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
