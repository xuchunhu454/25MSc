#ifndef FIRMWARE_SAMPLING_H
#define FIRMWARE_SAMPLING_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float prob;
    int   index;
} ProbIndex;

typedef struct {
    int                  vocab_size;
    ProbIndex           *probindex;
    float                temperature;
    float                topp;
    unsigned long long   rng_state;
} Sampler;

void build_sampler(
    Sampler             *sampler,
    int                  vocab_size,
    float                temperature,
    float                topp,
    unsigned long long   rng_seed
);

void free_sampler(Sampler *sampler);

int sample(Sampler *sampler, float *logits);

int sample(
    float               *logits,
    int                  vocab_size,
    float                temperature,
    float                topp,
    unsigned long long  *rng_state
);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // FIRMWARE_SAMPLING_H

