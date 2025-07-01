#ifndef FIRMWARE_FORWARD_H
#define FIRMWARE_FORWARD_H
#include "typedefs.h"
#include "config.h"
#include <math.h>
#include <cstring>
extern "C" void forward(Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *transformer, int token, int pos, float key_cache[n_layers * seq_len * ((dim * n_kv_heads) / n_heads)], float value_cache[n_layers * seq_len * ((dim * n_kv_heads) / n_heads)], float *out);
template <int N, int D, int GS=128>
void matmul(
    float* xout,
    const int8_t* __restrict xq,
    const float* __restrict xs,
    const int8_t* __restrict wq,
    const float* __restrict ws)
{
    // 创新点1: 分组并行计算架构
    constexpr int GROUPS = N / GS;
    static_assert(N % GS == 0, "N must be divisible by GS");

    // 创新点2: 输入向量静态展开
    int8_t x_buffer[N];
    float xs_buffer[GROUPS];
    #pragma HLS ARRAY_PARTITION variable=x_buffer cyclic factor=32
    #pragma HLS ARRAY_PARTITION variable=xs_buffer complete

    // 阶段1: 并行加载输入
    load_input:
    for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL factor=32
        x_buffer[i] = xq[i];
        if (i % GS == 0) {
            xs_buffer[i/GS] = xs[i/GS];
        }
    }

    // 阶段2: 流水线化输出计算
    output_loop:
    for (int i = 0; i < D; i++) {
        #pragma HLS PIPELINE II=1
        
        // 创新点3: 权重行缓存优化
        int8_t w_row[N];
        #pragma HLS ARRAY_PARTITION variable=w_row cyclic factor=32
        
        load_weight_row:
        for (int j = 0; j < N; j++) {
            #pragma HLS UNROLL factor=4
            w_row[j] = wq[i * N + j];
        }

        // 创新点4: 分组点积并行化
        float acc = 0;
        group_dot:
        for (int g = 0; g < GROUPS; g++) {
            #pragma HLS UNROLL factor=4
            
            int32_t sum = 0;
            #pragma HLS BIND_OP variable=sum op=add impl=fabric
            dot_product:
            for (int j = 0; j < GS; j++) {
                #pragma HLS UNROLL
                sum += x_buffer[g*GS + j] * w_row[g*GS + j];
            }
            
            // 创新点5: 融合缩放计算
            acc += xs_buffer[g] * ws[i * GROUPS + g] * (float)sum;
        }
        
        xout[i] = acc;
    }
}

template <int S>
void dequantize(QuantizedTensor<S> *qx, float x[S], int GS)
{
  for (int i = 0; i < S; i++)
  {
    x[i] = qx->q[i] * qx->s[i / GS];
  }
}

template <int S>
void quantize(QuantizedTensor<S> *qx, float x[S], int GS)
{
  constexpr int num_groups = S / 64;
  constexpr float Q_MAX = 127.0f;
  float scale_buffer[num_groups];
  int8_t quantized_buffer[S];
//#pragma HLS ARRAY_PARTITION variable = x type=cyclic factor = 8
#pragma HLS ARRAY_PARTITION variable = quantized_buffer type=cyclic factor=64
#pragma HLS ARRAY_PARTITION variable = scale_buffer type=cyclic factor = 16


main_loop:
  for (int group = 0; group < num_groups; group++)
  {
#pragma HLS UNROLL factor = 8
#pragma HLS PIPELINE
    float wmax = 0.0;
    int base_idx = group * GS;

    // Calculate the max absolute value in the current group
    max:
    for (int i = 0; i < GS; i++)
    {
#pragma HLS PIPELINE
      float val = fabs(x[base_idx + i]);
      if (val > wmax)
      {
        wmax = val;
      }
    }

    // Calculate and write the scaling factor
    float scale = wmax / Q_MAX;
    scale_buffer[group] = scale;

    // Calculate and write the quantized values
    for (int i = 0; i < GS; i++)
    {
//#pragma HLS UNROLL factor=8 skip_exit_check
#pragma HLS PIPELINE
      float quant_value = x[base_idx + i] / scale;   // scale
      int8_t quantized = (int8_t)round(quant_value); // round and clamp
      quantized_buffer[base_idx + i] = quantized;
    }
  }

  std::memcpy(qx->q, quantized_buffer, S * sizeof(int8_t));
  std::memcpy(qx->s, scale_buffer, num_groups * sizeof(float));
}
#endif // FIRMWARE_FORWARD_H