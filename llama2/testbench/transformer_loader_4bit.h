#ifndef TRANSFORMER_LOADER_H
#define TRANSFORMER_LOADER_H

#include <string>
#include "../firmware/config.h"
#include "../firmware/forward.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>

// 使用线程局部存储避免多线程问题
thread_local int global_version = 3;

// 解码 int4（packed 中的低 4 位或高 4 位）
inline int8_t decode_int4(int8_t packed, int idx) {
  int val = (idx == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
  return (val >= 8) ? val - 16 : val;
}

template <
    int dim, int hidden_dim,
    int n_layers, int n_heads,
    int n_kv_heads,
    int vocab_size, int seq_len,
    int GS
>
void build_transformer(
    Transformer<dim,hidden_dim,n_layers,n_heads,n_kv_heads,vocab_size,seq_len,GS> *t,
    const std::string& checkpoint_path  
)
{
 read_checkpoint(checkpoint_path, &t->config, &t->weights);
}

template <int SIZE>
void init_quantized_tensors(void **ptr, QuantizedTensor<SIZE> *tensor, int n, int size_each, int group_size) {
  void *p = *ptr;
  for (int i = 0; i < n; i++) {
    tensor[i].group_size = group_size; // 设置实际分组大小
    
    if (global_version == 3) {
      // 使用 int4: 每个 byte 存两个权重
      const int packed_size = size_each / 2;
      for (int j = 0; j < packed_size; j++) {
        int8_t packed = ((int8_t*)p)[j];
        tensor[i].q[2*j] = decode_int4(packed, 0);
        tensor[i].q[2*j+1] = decode_int4(packed, 1);
      }
      p = (int8_t*)p + packed_size;
    } else {
      // 默认 int8 加载
      std::memcpy(tensor[i].q, p, size_each * sizeof(int8_t));
      p = (int8_t *)p + size_each;
    }
    
    // 读取缩放因子
    const int num_scales = size_each / group_size;
    std::memcpy(tensor[i].s, p, num_scales * sizeof(float));
    p = (float *)p + num_scales;
  }
  *ptr = p;
}

template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
void memory_map_weights(TransformerWeights<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *w, void *ptr, uint8_t shared_classifier, int group_size) {
  int head_size = dim / n_heads;

  float *fptr = (float *)ptr;
  std::memcpy(w->rms_att_weight, fptr, n_layers * dim * sizeof(float));
  fptr += n_layers * dim;
  std::memcpy(w->rms_ffn_weight, fptr, n_layers * dim * sizeof(float));
  fptr += n_layers * dim;
  std::memcpy(w->rms_final_weight, fptr, dim * sizeof(float));
  fptr += dim;

  ptr = (void *)fptr;

  // 使用实际分组大小初始化量化张量
  init_quantized_tensors(&ptr, w->q_tokens, 1, vocab_size * dim, group_size);
  dequantize<vocab_size * dim>(w->q_tokens, w->token_embedding_table);

  init_quantized_tensors(&ptr, w->wq, n_layers, dim * (n_heads * head_size), group_size);
  init_quantized_tensors(&ptr, w->wk, n_layers, dim * (n_kv_heads * head_size), group_size);
  init_quantized_tensors(&ptr, w->wv, n_layers, dim * (n_kv_heads * head_size), group_size);
  init_quantized_tensors(&ptr, w->wo, n_layers, (n_heads * head_size) * dim, group_size);

  init_quantized_tensors(&ptr, w->w1, n_layers, dim * hidden_dim, group_size);
  init_quantized_tensors(&ptr, w->w2, n_layers, hidden_dim * dim, group_size);
  init_quantized_tensors(&ptr, w->w3, n_layers, dim * hidden_dim, group_size);

  if (shared_classifier) {
    std::memcpy(w->wcls, w->q_tokens, sizeof(QuantizedTensor<vocab_size * dim>));
  } else {
    init_quantized_tensors(&ptr, w->wcls, 1, dim * vocab_size, group_size);
  }
}

template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
void read_checkpoint(std::string checkpoint, Config *config, TransformerWeights<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *weights) {
  FILE *file = fopen(checkpoint.c_str(), "rb");
  if (!file) {
    fprintf(stderr, "Couldn't open file %s\n", checkpoint.c_str());
    exit(EXIT_FAILURE);
  }

  uint32_t magic_number;
  if (fread(&magic_number, sizeof(uint32_t), 1, file) != 1) exit(EXIT_FAILURE);
  if (magic_number != 0x616b3432) {
    fprintf(stderr, "Bad magic number\n");
    exit(EXIT_FAILURE);
  }

  int version;
  if (fread(&version, sizeof(int), 1, file) != 1) exit(EXIT_FAILURE);
  if (version != 2 && version != 3) {
    fprintf(stderr, "Bad version %d, expected 2 or 3\n", version);
    exit(EXIT_FAILURE);
  }
  global_version = version;

  int header_size = 256;
  if (fread(config, sizeof(Config) - sizeof(int), 1, file) != 1) exit(EXIT_FAILURE);

  uint8_t shared_classifier;
  if (fread(&shared_classifier, sizeof(uint8_t), 1, file) != 1) exit(EXIT_FAILURE);

  int group_size;
  if (fread(&group_size, sizeof(int), 1, file) != 1) exit(EXIT_FAILURE);
  
  // 关键修改：验证并设置实际分组大小
  if (group_size != GS) {
    fprintf(stderr, "Error: Model group size (%d) does not match expected GS (%d)\n", 
            group_size, GS);
    exit(EXIT_FAILURE);
  }
  config->GS = group_size;  // 使用文件中的实际值

  fseek(file, 0, SEEK_END);
  auto file_size = ftell(file);
  fclose(file);

  int fd = open(checkpoint.c_str(), O_RDONLY);
  if (fd == -1) {
    fprintf(stderr, "open failed!\n");
    exit(EXIT_FAILURE);
  }

  auto data = (float *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (data == MAP_FAILED) {
    fprintf(stderr, "mmap failed!\n");
    close(fd);
    exit(EXIT_FAILURE);
  }

  void *weights_ptr = ((char *)data) + header_size;
  
  // 传递实际分组大小给权重映射函数
  memory_map_weights(weights, weights_ptr, shared_classifier, group_size);
  
  // 保存映射信息用于后续释放
  weights->mapped_data = data;
  weights->mapped_size = file_size;
  weights->mapped_fd = fd;
}

// 添加权重释放函数
template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
void free_weights(TransformerWeights<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *weights) {
  if (weights->mapped_data != MAP_FAILED) {
    munmap(weights->mapped_data, weights->mapped_size);
    weights->mapped_data = MAP_FAILED;
  }
  if (weights->mapped_fd != -1) {
    close(weights->mapped_fd);
    weights->mapped_fd = -1;
  }
}

#endif // TRANSFORMER_LOADER_H