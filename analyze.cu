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

__global__ void cacheLineMeasurer(uint8_t *data, size_t n, size_t lastAccess,
                                  long long *timeCycles, size_t *out) {
  size_t sum = 0;
  for (size_t i = 0; i < n; i++) {
    sum += data[i];
  }
  uint32_t v; // "r" constraint requires a 32-bit register; ld.u8 zero-extends
  long long t0 = clock64();
  asm volatile("ld.volatile.global.u8 %0, [%1];\n\t" // 8 bit load from %1 into
                                                     // %0 (in this case v)
               "add.u32 %0, %0, 0;"                  // v = v + 0
               : "=r"(v)
               : "l"(data + lastAccess)
               : "memory");
  long long t1 = clock64();
  *out = sum + v;
  *timeCycles = t1 - t0;
}

long long pointerChase(size_t numElements, size_t stride, size_t iterations) {
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
  return timeKernel(setupPtrs, launch);
}

long long scan(size_t numBytes, size_t lastAccess) {
  cudaFuncSetAttribute(cacheLineMeasurer,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxL1);
  auto setupPtrs = [&](void *&data) {
    CUDA_CHECK(cudaMalloc((void **)&data, numBytes));
    std::vector<uint8_t> host(numBytes, 0);
    CUDA_CHECK(cudaMemcpy(data, host.data(), numBytes, cudaMemcpyHostToDevice));
  };
  auto launch = [&](void *data, long long *timeCycles, size_t *out) {
    cacheLineMeasurer<<<1, 1>>>((uint8_t *)data, numBytes, lastAccess,
                                timeCycles, out);
  };
  return timeKernel(setupPtrs, launch);
}

std::vector<size_t> cacheSizeBenchmark() {
  printf("Cache size:\n");
  std::vector<long long> cycles;
  std::vector<size_t> sizes;
  long long prev = INT_MAX / 2;
  long long cycle;
  size_t level = 1;

  for (size_t i = 1; i < (1 << 17); i *= 2) {
    cycle = pointerChase(i, 8, 1'000'000) / 1'000'000;
    printf("Pointers: %lu, cycles: %llu\n", i, cycle);
    if (cycle > prev * 2) {
      printf(ANSI_GREEN "L%lu Cache is %luKB\n" ANSI_RESET, sizes.size() + 1,
             i / 1024 * 8);
      sizes.push_back(i * 8);
      level++;
    }
    prev = cycle;
  }

  return sizes;
}

void cacheLineSizeBenchmark(std::vector<size_t> cacheSizes) {
  long long prev = INT_MAX / 2;
  long long cycle;
  size_t level = 1;
  printf("Cache line size:\n");
  for (auto cacheSize : cacheSizes) {
    // Test cache line sizes from 8-256 bytes
    printf("L%lu cache, array size: %luB\n", level, cacheSize + 1);
    for (size_t i = 1; i < (1 << 9); i *= 2) {
      cycle = scan(cacheSize + 1, i);
      printf("Offset: %lu, cycles: %llu\n", i, cycle);
      if (cycle > prev * 2) {
        printf(ANSI_GREEN "Level %lu cache line is %luB" ANSI_RESET, level, i);
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
