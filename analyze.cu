#include "utilities.h"
#include <cuda_runtime.h>
#include <random>
#include <utility>
#include <vector>

std::vector<void *> sattoloChain(void **d_base, size_t n, size_t stride,
                                 unsigned seed = 42) {
  std::vector<size_t> order(n);
  for (size_t i = 0; i < n; i++)
    order[i] = i;
  std::mt19937_64 rng(seed);
  for (size_t i = n - 1; i > 0; i--) {
    std::uniform_int_distribution<unsigned> dist(0, i - 1);
    std::swap(order[i], order[dist(rng)]);
  }

  size_t step = (stride) / sizeof(void *);
  std::vector<void *> host(n * step, nullptr);
  for (size_t i = 0; i < n; i++) {
    host[i * step] = (void *)(d_base + order[i] * step);
  }
  return host;
}

std::vector<void *> sequentialChain(void **d_base, size_t n, size_t loopback) {
  std::vector<void *> chain(n, nullptr);
  for (size_t i = 0; i < n - 1; i++) {
    chain[i] = (void *)(d_base + i + 1);
  }
  chain[n - 1] = (void *)(d_base + loopback);
  return chain;
}

__global__ void pointerChasing(void **pointers, size_t n, size_t iterations,
                               long long *timeCycles, size_t *out) {
  void **cur = (void **)pointers[0];
  long long t0 = clock64();
  for (size_t i = 0; i < iterations; i++) {
    cur = (void **)*cur;
  }
  long long t1 = clock64();
  *out = (size_t)*cur;
  *timeCycles = t1 - t0;
}

__global__ void pointerChaseRecLastAccess(void **pointers, size_t n,
                                          long long *timeCycles, size_t *out) {
  void **cur = (void **)pointers[0];
  for (size_t i = 0; i < n - 1; i++) {
    cur = (void **)*cur;
  }
  long long t0 = clock64();
  cur = (void **)*cur; // load value at last pointer's address
  *out = (size_t)*cur; // put this value in *out so that it doesn't get
                       // optimized away
  long long t1 = clock64();
  *timeCycles = t1 - t0;
}

std::vector<size_t> cacheSizeBenchmark() {
  printf("Cache size:\n");
  std::vector<long long> cycles;
  std::vector<size_t> sizes;
  long long prev = INT_MAX / 2;
  long long cycle;
  size_t level = 1;

  size_t stride = 8;
  size_t numElements = 1; // set per loop iteration below
  const size_t iterations = 1'000'000;

  cudaFuncSetAttribute(pointerChasing,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxL1);

  auto setupPtrs = [&](void *&data) {
    CUDA_CHECK(cudaMalloc((void **)&data, numElements * stride));
    std::vector<void *> ptrArr =
        sattoloChain((void **)data, numElements, stride);
    CUDA_CHECK(cudaMemcpy(data, ptrArr.data(), numElements * stride,
                          cudaMemcpyHostToDevice));
  };
  auto launch = [&](void *data, long long *timeCycles, size_t *out) {
    pointerChasing<<<1, 1>>>((void **)data, numElements, iterations, timeCycles,
                             out);
  };

  for (size_t i = 1; i < (1 << 17); i += 256) {
    numElements = i;
    cycle = timeKernel(setupPtrs, launch) / iterations;
    if (cycle > prev * 1.5) {
      printf(ANSI_GREEN "L%lu Cache is %luKB\n" ANSI_RESET, sizes.size() + 1,
             i * stride / 1024);
      sizes.push_back(i * 8);
      break;
    }
    prev = cycle;
  }

  return sizes;
}

void cacheLineSizeBenchmark(std::vector<size_t> cacheSizes) {
  long long prev = INT_MAX / 2;
  long long cycle;
  size_t level = 1;
  size_t loopback = 0;
  size_t stride = 8;
  size_t numElements = 0;

  cudaFuncSetAttribute(pointerChaseRecLastAccess,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxL1);

  auto setupPtrs = [&](void *&data) {
    CUDA_CHECK(cudaMalloc((void **)&data, numElements * stride));
    std::vector<void *> ptrArr =
        sequentialChain((void **)data, numElements, loopback);
    CUDA_CHECK(cudaMemcpy(data, ptrArr.data(), numElements * stride,
                          cudaMemcpyHostToDevice));
  };
  auto launch = [&](void *data, long long *timeCycles, size_t *out) {
    pointerChaseRecLastAccess<<<1, 1>>>((void **)data, numElements, timeCycles,
                                        out);
  };

  printf("Cache line size:\n");
  for (auto cacheSize : cacheSizes) {
    numElements = cacheSize / 16 + 1;
    // Test cache line sizes from 8-256 bytes
    printf("L%lu cache, array size: %luB\n", level, cacheSize + 1);
    for (size_t i = 1; i < (1 << 9); i *= 2) {
      loopback = i;
      cycle = timeKernel(setupPtrs, launch);
      printf("Offset: %lu, cycles: %llu\n", i, cycle);
      if (cycle < prev - 10) {
        printf(ANSI_GREEN "Level %lu cache line is %luB\n" ANSI_RESET, level,
               i);
        level++;
      }
      prev = cycle;
    }
  }
}

int main(int argc, char **argv) {
  std::vector<size_t> cacheSizes = cacheSizeBenchmark();
  cacheLineSizeBenchmark(cacheSizes);
}
