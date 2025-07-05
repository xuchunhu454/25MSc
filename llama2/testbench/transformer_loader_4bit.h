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

// 判断使用 int4 模型（版本 3）
int global_version = 3;

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
    //const std::string& checkpoint_path  
    const std::string checkpoint_path  
)
{
 read_checkpoint(checkpoint_path, &t->config, &t->weights);
}
;

template <int SIZE>
void init_quantized_tensors(void **ptr, QuantizedTensor<SIZE> *tensor, int n, int size_each) {
  char *p = (char *)*ptr;
  for (int i = 0; i < n; i++) {
    if (global_version == 3) {
      // int4：每两个 weights 压缩成 1 byte
      int packed_bytes = size_each / 2;
      for (int j = 0; j < size_each; j++) {
        int byte_idx = j / 2;
        int is_high = j % 2;
        int8_t packed = ((int8_t*)p)[byte_idx];
        tensor[i].q[j] = decode_int4(packed, is_high);

        // debug 打印前 8 项
        if (i == 0 && j < 8) {
          std::cout << "[C++] tensor[" << i << "].q[" << j << "] = "
                    << static_cast<int>(tensor[i].q[j]) << std::endl;
        }
      }
      p += packed_bytes;
    } else {
      // int8 直接拷贝
      std::memcpy(tensor[i].q, p, size_each * sizeof(int8_t));
      p += size_each;
    }

    // scale 数量
    int num_scales = size_each / GS;
    std::memcpy(tensor[i].s, p, num_scales * sizeof(float));

    if (i == 0) {
      for (int s = 0; s < std::min(2, num_scales); s++) {
        std::cout << "[C++] tensor[" << i << "].s[" << s << "] = " << tensor[i].s[s] << std::endl;
      }
    }

    p += num_scales * sizeof(float);
  }

  *ptr = (void *)p;
}



template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
void memory_map_weights(TransformerWeights<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *w, void *ptr, uint8_t shared_classifier) {
  int head_size = dim / n_heads;

  float *fptr = (float *)ptr;
  std::memcpy(w->rms_att_weight, fptr, n_layers * dim * sizeof(float));
  fptr += n_layers * dim;
  std::memcpy(w->rms_ffn_weight, fptr, n_layers * dim * sizeof(float));
  fptr += n_layers * dim;
  std::memcpy(w->rms_final_weight, fptr, dim * sizeof(float));
  fptr += dim;

  ptr = (void *)fptr;

  init_quantized_tensors(&ptr, w->q_tokens, 1, vocab_size * dim);
  dequantize<vocab_size * dim>(w->q_tokens, w->token_embedding_table, GS);

  init_quantized_tensors(&ptr, w->wq, n_layers, dim * (n_heads * head_size));
  init_quantized_tensors(&ptr, w->wk, n_layers, dim * (n_kv_heads * head_size));
  init_quantized_tensors(&ptr, w->wv, n_layers, dim * (n_kv_heads * head_size));
  init_quantized_tensors(&ptr, w->wo, n_layers, (n_heads * head_size) * dim);

  init_quantized_tensors(&ptr, w->w1, n_layers, dim * hidden_dim);
  init_quantized_tensors(&ptr, w->w2, n_layers, hidden_dim * dim);
  init_quantized_tensors(&ptr, w->w3, n_layers, dim * hidden_dim);

  if (shared_classifier) {
    std::memcpy(w->wcls, w->q_tokens, sizeof(QuantizedTensor<vocab_size * dim>));
  } else {
    init_quantized_tensors(&ptr, w->wcls, 1, dim * vocab_size);
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
  config->GS = GS;

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
    exit(EXIT_FAILURE);
  }

  void *weights_ptr = ((char *)data) + header_size;
  memory_map_weights(weights, weights_ptr, shared_classifier);
  close(fd);
  if (data != MAP_FAILED) munmap(data, file_size);
}

#endif // TRANSFORMER_LOADER_H
