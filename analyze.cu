#include <cuda_runtime.h>
#include "utilities.h"
#include <queue>
#include <random>
#include <span>
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

__global__ void pointerChasing(void **pointers, size_t n, size_t iterations,
                               long long *timeCycles, void **out) {
  void **cur = (void **)pointers[0];
  long long t0 = clock64();
  for (size_t i = 0; i < iterations; i++) {
    cur = (void **)*cur;
  }
  long long t1 = clock64();
  *out = (void *)cur;
  *timeCycles = t1 - t0;
}




long long pointerChase(size_t numElements, size_t stride, size_t iterations) {
  cudaFuncSetAttribute(pointerChasing,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxL1);

  auto setupPtrs = [&]() {
    CUDA_CHECK(cudaMalloc((void **)&data, numElements * stride));
    std::vector<void *> ptrArr = sattoloChain(data, numElements, stride);
    CUDA_CHECK(cudaMemcpy(data, ptrArr.data(), numElements * stride,
                          cudaMemcpyHostToDevice));
  };
  auto launch = [&](long long *timeCycles, void **out) {
    pointerChasing<<<1, 1>>>(data, numElements, iterations, timeCycles, out);
  };
  return timeKernel(setupPtrs, launch, iterations);
}

void cacheSizeBenchmark() {
  printf("# of Pointers: [");
  bool comma = false;
  std::vector<long long> cycles;
  for (size_t i = 1; i < (1 << 17); i *= 2) {
    if (comma) {
      printf(", ");
    } else {
      comma = true;
    }
    printf("%zu", i);
    cycles.push_back(pointerChase(i, 8, 1'000'000) / 1'000'000);
  }
  printf("]\nCycles:        ");
  printVec<long long>(cycles);

  long long prev = cycles[0];
  size_t level = 1;
  for (size_t i = 1; i < cycles.size(); i++) {
    if (cycles[i] > prev * 2) {
      printf(ANSI_GREEN "L%lu Cache is %uKB\n" ANSI_RESET, level,
             (1 << i) / 1024 * 8);
      level++;
    }
    prev = cycles[i];
  }
}

void cacheLineSizeBenchmark() {
  std::vector<long long> cycles;
  printf("Cache line size:");
  bool comma = false;
  for (size_t i = 8; i < (1 << 9); i *= 2) {
    if (comma) {
      printf(", ");
    } else {
      comma = true;
    }
    printf("%zu", i);
    cycles.push_back(timeKernel(2, i, 2));
  }
  printf("]\nCycles:        ");
  printVec<long long>(cycles);

  level = 0;
  prev = cycles[0];
  for (size_t i = 1; i < cycles.size(); i++) {
    if (cycles[i] > prev * 1.5) {
      printf(ANSI_GREEN "Cache %s is %uB\n" ANSI_RESET,
             level ? "line" : "sector", (1 << i) * 8);
      level++;
      break;
    }
    prev = cycles[i];
  }
  return 0;
}

int main(int argc, char **argv) {
   cacheSizeBenchmark();
   cacheLineSizeBenchmark();
}