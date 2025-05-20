#ifndef SAMPLING_H
#define SAMPLING_H

#include <cstdint>

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

#endif // SAMPLING_H
