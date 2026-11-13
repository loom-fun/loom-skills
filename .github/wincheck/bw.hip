// Streaming-read bandwidth of the GPU (the roofline's "bw"): each thread sums float4s over a 4 GiB buffer.
#include <hip/hip_runtime.h>
#include <cstdio>
__global__ void rd(const float4* __restrict__ a, size_t n, float* out) {
  float s = 0.f;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    float4 v = a[i]; s += v.x + v.y + v.z + v.w;
  }
  if (s == 1234.5f) out[0] = s;
}
__global__ void cp(const float4* __restrict__ a, float4* __restrict__ b, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) b[i] = a[i];
}
int main() {
  size_t bytes = 4ull << 30, n = bytes / 16; float4 *a, *b; float* o;
  hipMalloc(&a, bytes); hipMalloc(&b, bytes / 2); hipMalloc(&o, 4); hipMemset(a, 0, bytes);
  hipEvent_t t0, t1; hipEventCreate(&t0); hipEventCreate(&t1);
  int grids[] = {256, 1024, 4096, 16384}; double best = 0, bestc = 0;
  for (int g : grids) for (int r = 0; r < 6; r++) {
    hipEventRecord(t0); rd<<<g, 256>>>(a, n, o); hipEventRecord(t1); hipEventSynchronize(t1);
    float ms; hipEventElapsedTime(&ms, t0, t1); if (r) best = fmax(best, bytes / (ms * 1e-3) / 1e9);
    hipEventRecord(t0); cp<<<g, 256>>>(a, b, n / 2); hipEventRecord(t1); hipEventSynchronize(t1);
    hipEventElapsedTime(&ms, t0, t1); if (r) bestc = fmax(bestc, bytes / (ms * 1e-3) / 1e9);
  }
  printf("{\"read_gbs\": %.1f, \"copy_gbs\": %.1f}\n", best, bestc);
}
