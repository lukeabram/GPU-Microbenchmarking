#pragma once
// Fail fast on any CUDA error.
#define CUDA_CHECK(stmt)                                                       \
  do {                                                                         \
    cudaError_t err = (stmt);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA ERROR %s (%d): %s at %s:%d\n", #stmt, int(err),    \
              cudaGetErrorString(err), __FILE__, __LINE__);                    \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// Simple ANSI colors for PASS/FAIL printing.
#define ANSI_GREEN "\x1b[32m"
#define ANSI_RED "\x1b[31m"
#define ANSI_RESET "\x1b[0m"

template <typename T> void printVec(std::span<T> vec) {
  printf("[");
  bool comma = false;
  for (auto e : vec) {
    if (comma) {
      printf(", ");
    } else {
      comma = true;
    }
    if constexpr (std::is_arithmetic_v<T>) {
      printf("%lld", e);
    } else
      printf("%p", e);
  }
  printf("]");
  printf("\n");
}

template <typename Setup, typename Launch, typename Cleanup>
long long timeKernel(Setup &setup, Launch &launch size_t iterations) {
  size_t repeat = 10;
  long long *timeCycles = nullptr;
  long long cyclesOut = 0;
  // General purpose data used for benchmark timings
  void **data = nullptr;
  // Used to prevent compiler from optimizing out the kernel
  void **out = nullptr;

  // data
  void **pointers = nullptr;

  // allocate memory
  CUDA_CHECK(cudaMalloc((void **)&timeCycles, sizeof(long long)));
  CUDA_CHECK(cudaMalloc((void **)&out, sizeof(void *)));

  // setup
  setup();

  // run/time kernel
  launch(timeCycles, out);

  CUDA_CHECK(cudaMemcpy(&cyclesOut, timeCycles, sizeof(long long),
                        cudaMemcpyDeviceToHost));

  // free mem
  CUDA_CHECK(cudaFree(data));
  CUDA_CHECK(cudaFree(timeCycles));
  CUDA_CHECK(cudaFree(out));
  return cyclesOut;
}