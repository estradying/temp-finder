#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include "cubiomes/biomenoise.h"
#include "cubiomes/finders.h"
#include "cubiomes/generator.h"

#define THREADS 12
#define BATCH_SIZE 16384

typedef struct {
  BiomeNoise *t, *h, *c, *e, *w;
} ClimateNoises;

typedef struct {
  double t, h, c, e, w;
} ClimateValues;

static atomic_uint_fast64_t next_seed = 0;

static void print_result(uint64_t seed, int x, int z, double value) {
  printf("%" PRId64 " %d %d %f\n", (int64_t)seed, x * 4, z * 4, value);
}

static int check_offsets(uint64_t seed) {
  Xoroshiro x1, x2, x3;
  xSetSeed(&x1, seed);
  uint64_t xlo = xNextLong(&x1);
  uint64_t xhi = xNextLong(&x1);
  x2.lo = xlo ^ 0x5c7e6b29735f0d7f;
  x2.hi = xhi ^ 0xf7d86f1bbc734988;
  xlo = xNextLong(&x2);
  xhi = xNextLong(&x2);
  x3.lo = xlo ^ 0x36d326eed40efeb2;
  x3.hi = xhi ^ 0x5be9ce18223c636a;
  xSkipN(&x3, 1);
  float off0a =
      fabsf((xNextLong(&x3) >> 32 & 0xFFFFFF) * (1.0f / 16777216.0f) - 0.5f);
  xlo = xNextLong(&x2);
  xhi = xNextLong(&x2);
  x3.lo = xlo ^ 0x36d326eed40efeb2;
  x3.hi = xhi ^ 0x5be9ce18223c636a;
  xSkipN(&x3, 1);
  float off0b =
      fabsf((xNextLong(&x3) >> 32 & 0xFFFFFF) * (1.0f / 16777216.0f) - 0.5f);
  return off0a + off0b < 0.05f;
}

static double sample(BiomeNoise *bn, int a, int b, int x, int z) {
  double v = 0;
  if (a) {
    OctaveNoise *on = &bn->climate[bn->nptype].octA;
    for (int i = 0; i < on->octcnt; i++) {
      if (a & (1 << i)) {
        PerlinNoise *pn = on->octaves + i;
        double l = pn->lacunarity;
        v += samplePerlin(pn, x * l, 0, z * l, 0, 0) * pn->amplitude;
      }
    }
  }
  if (b) {
    OctaveNoise *on = &bn->climate[bn->nptype].octB;
    for (int i = 0; i < on->octcnt; i++) {
      if (b & (1 << i)) {
        PerlinNoise *pn = on->octaves + i;
        double l = pn->lacunarity * (337.0 / 331.0);
        v += samplePerlin(pn, x * l, 0, z * l, 0, 0) * pn->amplitude;
      }
    }
  }
  return v * bn->climate[bn->nptype].amplitude;
}

static void lattice(ClimateNoises *n, uint64_t seed, double max_a, int sign,
                    int x, int z) {
  setClimateParaSeed(n->t, seed, 0, NP_TEMPERATURE, 4);
  for (int x1 = x - 7340032; x1 <= x + 7340032; x1 += 262144)
    for (int z1 = z - 7340032; z1 <= z + 7340032; z1 += 262144) {
      if (sample(n->t, 0b00, 0b01, x1, z1) * sign + max_a < 1.85) {
        continue;
      }
      double max = 0;
      int max_x = 0;
      int max_z = 0;
      for (int x2 = x1 - 128; x2 <= x1 + 128; x2 += 32)
        for (int z2 = z1 - 128; z2 <= z1 + 128; z2 += 32) {
          if (sample(n->t, 0b11, 0b11, x2, z2) * sign < 1.95) {
            continue;
          }
          for (int x3 = x2 - 16; x3 <= x2 + 16; x3 += 1)
            for (int z3 = z2 - 16; z3 <= z2 + 16; z3 += 1) {
              double value = sample(n->t, 0b11, 0b11, x3, z3) * sign;
              if (value > max) {
                max = value;
                max_x = x3;
                max_z = z3;
              }
            }
        }
      if (max > 2.15) {
        print_result(seed, max_x, max_z, max * sign);
      }
    }
}

static void check(ClimateNoises *n, uint64_t seed) {
  if (!check_offsets(seed)) {
    return;
  }
  setClimateParaSeed(n->t, seed, 0, NP_TEMPERATURE, 1);
  for (int x1 = -8192; x1 <= 8192; x1 += 2048)
    for (int z1 = -8192; z1 <= 8192; z1 += 2048) {
      double value = sample(n->t, 0b1, 0b0, x1, z1);
      int sign = (value > 0.6) - (value < -0.6);
      if (sign == 0) {
        continue;
      }
      for (int x2 = x1 - 512; x2 <= x1 + 512; x2 += 128)
        for (int z2 = z1 - 512; z2 <= z1 + 512; z2 += 128) {
          if (sample(n->t, 0b1, 0b0, x2, z2) * sign < 0.8) {
            continue;
          }
          setClimateParaSeed(n->t, seed, 0, NP_TEMPERATURE, 3);
          double max = 0;
          int max_x = 0;
          int max_z = 0;
          for (int x3 = x2 - 256; x3 <= x2 + 256; x3 += 64)
            for (int z3 = z2 - 256; z3 <= z2 + 256; z3 += 64) {
              value = sample(n->t, 0b11, 0b00, x3, z3) * sign;
              if (value > max) {
                max = value;
                max_x = x3;
                max_z = z3;
              }
            }
          if (max > 1.0) {
            lattice(n, seed, max, sign, max_x, max_z);
          }
          return;
        }
    }
}

static void *worker(void *arg) {
  Generator g;
  setupGenerator(&g, MC_NEWEST, 0);
  BiomeNoise t, h, c, e, w;
  t = h = c = e = w = g.bn;
  ClimateNoises n = {&t, &h, &c, &e, &w};
  while (1) {
    uint64_t start = atomic_fetch_add(&next_seed, BATCH_SIZE);
    uint64_t end = start + BATCH_SIZE;
    for (uint64_t seed = start; seed < end; seed++) {
      check(&n, seed);
    }
  }
  return NULL;
}

int main(void) {
  srand((unsigned int)time(NULL));
  uint64_t seed = 0;
  for (int i = 0; i < 4; i++) {
    seed = (seed << 16) | (uint64_t)(rand() & 0xFFFF);
  }
  atomic_store(&next_seed, seed);
  pthread_t threads[THREADS];
  for (int i = 0; i < THREADS; i++) {
    pthread_create(&threads[i], NULL, worker, NULL);
  }
  for (int i = 0; i < THREADS; i++) {
    pthread_join(threads[i], NULL);
  }
  return 0;
}
