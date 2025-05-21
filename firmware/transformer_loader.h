#ifndef TRANSFORMER_LOADER_H
#define TRANSFORMER_LOADER_H

#include <string>
#include "config.h"
#include "forward.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

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
    const std::string& checkpoint_path  
);

template <int SIZE>
/* initialize `n` x quantized tensor (with `size_each` elements), starting from memory pointed at *ptr */
void init_quantized_tensors(void **ptr, QuantizedTensor<SIZE> *tensor, int n, int size_each)
{
  void *p = *ptr;
  for (int i = 0; i < n; i++)
  {
    /* map quantized int8 values*/
    std::memcpy(tensor[i].q, p, size_each * sizeof(int8_t));
    p = (int8_t *)p + size_each;
    /* map scale factors */
    std::memcpy(tensor[i].s, p, (size_each / GS) * sizeof(float));

    p = (float *)p + size_each / GS;
  }
  *ptr = p; // advance ptr to current position
}

template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
void memory_map_weights(TransformerWeights<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *w, void *ptr, uint8_t shared_classifier)
{
  int head_size = dim / n_heads;
  // first are the parameters that are kept in fp32 (the rmsnorm (1D) weights)
  float *fptr = (float *)ptr; // cast our pointer to float*
  std::memcpy(w->rms_att_weight, fptr, n_layers * dim * sizeof(float));
  fptr += n_layers * dim;
  std::memcpy(w->rms_ffn_weight, fptr, n_layers * dim * sizeof(float));
  fptr += n_layers * dim;
  std::memcpy(w->rms_final_weight, fptr, dim * sizeof(float));
  fptr += dim;

  // now read all the quantized weights
  ptr = (void *)fptr; // now cast the pointer back to void*
  init_quantized_tensors(&ptr, w->q_tokens, 1, vocab_size * dim);
  // dequantize token embedding table
  dequantize<vocab_size * dim>(w->q_tokens, w->token_embedding_table, GS);

  init_quantized_tensors(&ptr, w->wq, n_layers, dim * (n_heads * head_size));
  init_quantized_tensors(&ptr, w->wk, n_layers, dim * (n_kv_heads * head_size));
  init_quantized_tensors(&ptr, w->wv, n_layers, dim * (n_kv_heads * head_size));
  init_quantized_tensors(&ptr, w->wo, n_layers, (n_heads * head_size) * dim);

  init_quantized_tensors(&ptr, w->w1, n_layers, dim * hidden_dim);
  init_quantized_tensors(&ptr, w->w2, n_layers, hidden_dim * dim);
  init_quantized_tensors(&ptr, w->w3, n_layers, dim * hidden_dim);

  if (shared_classifier)
  {
    std::memcpy(w->wcls, w->q_tokens, sizeof(QuantizedTensor<vocab_size * dim>));
  }
  else
  {
    init_quantized_tensors(&ptr, w->wcls, 1, dim * vocab_size);
  }
}

template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
void read_checkpoint(std::string checkpoint, Config *config, TransformerWeights<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *weights)
{
  FILE *file = fopen(checkpoint.c_str(), "rb");
  if (!file)
  {
    fprintf(stderr, "Couldn't open file %s\n", checkpoint.c_str());
    exit(EXIT_FAILURE);
  }
  // read in magic number (uint32), has to be 0x616b3432, i.e. "ak42" in ASCII
  uint32_t magic_number;
  if (fread(&magic_number, sizeof(uint32_t), 1, file) != 1)
  {
    exit(EXIT_FAILURE);
  }
  if (magic_number != 0x616b3432)
  {
    fprintf(stderr, "Bad magic number\n");
    exit(EXIT_FAILURE);
  }
  // read in the version number (uint32), has to be 1
  int version;
  if (fread(&version, sizeof(int), 1, file) != 1)
  {
    exit(EXIT_FAILURE);
  }
  if (version != 2)
  {
    fprintf(stderr, "Bad version %d, need version 2\n", version);
    exit(EXIT_FAILURE);
  }
  int header_size = 256; // the header size for version 2 in bytes
  // read in the Config
  if (fread(config, sizeof(Config) - sizeof(int), 1, file) != 1)
  {
    exit(EXIT_FAILURE);
  }
  // read in flags
  uint8_t shared_classifier; // a byte to indicate if the classifier is shared
  if (fread(&shared_classifier, sizeof(uint8_t), 1, file) != 1)
  {
    exit(EXIT_FAILURE);
  }
  int group_size; // the group size used in quantization
  if (fread(&group_size, sizeof(int), 1, file) != 1)
  {
    exit(EXIT_FAILURE);
  }
  config->GS = GS;
  // figure out the file size
  fseek(file, 0, SEEK_END);     // move file pointer to end of file
  auto file_size = ftell(file); // get the file size, in bytes
  fclose(file);
  // memory map the Transformer weights into the data pointer
  auto fd = open(checkpoint.c_str(), O_RDONLY); // open in read only mode
  if (fd == -1)
  {
    fprintf(stderr, "open failed!\n");
    exit(EXIT_FAILURE);
  }
  auto data = (float *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (data == MAP_FAILED)
  {
    fprintf(stderr, "mmap failed!\n");
    exit(EXIT_FAILURE);
  }
  void *weights_ptr = ((char *)data) + header_size; // skip header bytes. char is 1 byte
  memory_map_weights(weights, weights_ptr, shared_classifier);
  close(fd);
  if (data != MAP_FAILED)
  {
    munmap(data, file_size);
  }
}

#endif // TRANSFORMER_LOADER_H
