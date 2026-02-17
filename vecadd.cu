#include <stdio.h>

__global__ void vecAdd(const float *A, const float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

int main() {
    int n = 1000;
    size_t bytes = n * sizeof(float);

    float *hA = (float*)malloc(bytes);
    float *hB = (float*)malloc(bytes);
    float *hC = (float*)malloc(bytes);

    for(int i=0;i<n;i++) {
        hA[i] = i;
        hB[i] = 2*i;
    }

    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    vecAdd<<<blocks, threads>>>(dA, dB, dC, n);

    cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);

    printf("C[0] = %f\n", hC[0]);
    printf("C[10] = %f\n", hC[10]);
    printf("C[999] = %f\n", hC[999]);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    free(hA);
    free(hB);
    free(hC);

    return 0;
}

