#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d -> %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)


__global__ void vecAdd(const float *A, const float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}


int main() {
    int n = 1 << 24; // ~16 million (big enough to matter)
    size_t bytes = n * sizeof(float);

    float *hA = (float*)malloc(bytes);
    float *hB = (float*)malloc(bytes);
    float *hC = (float*)malloc(bytes);

    for(int i=0;i<n;i++) {
        hA[i] = (float)i;
        hB[i] = (float)(2*i);
    }

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, bytes));
    CHECK_CUDA(cudaMalloc(&dB, bytes));
    CHECK_CUDA(cudaMalloc(&dC, bytes));

    // ---------------- GPU TIMING ----------------
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    CHECK_CUDA(cudaEventRecord(start));
    vecAdd<<<blocks, threads>>>(dA, dB, dC, n);
    CHECK_CUDA(cudaEventRecord(stop));

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventSynchronize(stop));

    float gpu_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&gpu_ms, start, stop));

    CHECK_CUDA(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));

    // ---------------- CPU TIMING ----------------
    // Simple CPU loop for comparison
    clock_t cpu_start = clock();
    for(int i=0;i<n;i++) {
        hC[i] = hA[i] + hB[i];
    }
    clock_t cpu_end = clock();

    double cpu_ms = 1000.0 * (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC;

    printf("GPU kernel time: %f ms\n", gpu_ms);
    printf("CPU time:        %f ms\n", cpu_ms);

    printf("C[0] = %f\n", hC[0]);
    printf("C[10] = %f\n", hC[10]);
    printf("C[n-1] = %f\n", hC[n-1]);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    free(hA);
    free(hB);
    free(hC);

    return 0;
}

