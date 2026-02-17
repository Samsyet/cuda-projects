#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d -> %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

__global__ void matmul_naive(const float *A, const float *B, float *C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N) return;

    float sum = 0.0f;
    for(int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
}

#define TILE 16

__global__ void matmul_tiled(const float *A, const float *B, float *C, int N)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for(int t = 0; t < (N + TILE - 1) / TILE; t++) {

        int Acol = t * TILE + threadIdx.x;
        int Brow = t * TILE + threadIdx.y;

        // Load tile of A into shared memory
        if (row < N && Acol < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + Acol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile of B into shared memory
        if (Brow < N && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[Brow * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for(int k = 0; k < TILE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

void fill_matrix(float *M, int N)
{
    for(int i = 0; i < N*N; i++) {
        M[i] = (float)(rand() % 100) / 100.0f;
    }
}

float max_abs_diff(const float *A, const float *B, int N)
{
    float m = 0.0f;
    for(int i = 0; i < N*N; i++) {
        float d = fabsf(A[i] - B[i]);
        if (d > m) m = d;
    }
    return m;
}

float run_kernel(void (*kernel)(const float*, const float*, float*, int),
                 const float *dA, const float *dB, float *dC, int N,
                 dim3 blocks, dim3 threads)
{
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    // launch
    kernel<<<blocks, threads>>>(dA, dB, dC, N);

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms;
}

int main()
{
    int N = 1024; // 1024x1024
    size_t bytes = (size_t)N * N * sizeof(float);

    printf("Matrix size: %d x %d\n", N, N);

    float *hA = (float*)malloc(bytes);
    float *hB = (float*)malloc(bytes);
    float *hC_naive = (float*)malloc(bytes);
    float *hC_tiled = (float*)malloc(bytes);

    srand(0);
    fill_matrix(hA, N);
    fill_matrix(hB, N);

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, bytes));
    CHECK_CUDA(cudaMalloc(&dB, bytes));
    CHECK_CUDA(cudaMalloc(&dC, bytes));

    CHECK_CUDA(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    // Warmup
    matmul_naive<<<blocks, threads>>>(dA, dB, dC, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    float naive_ms = run_kernel((void(*)(const float*, const float*, float*, int))matmul_naive,
                               dA, dB, dC, N, blocks, threads);

    CHECK_CUDA(cudaMemcpy(hC_naive, dC, bytes, cudaMemcpyDeviceToHost));

    float tiled_ms = run_kernel((void(*)(const float*, const float*, float*, int))matmul_tiled,
                               dA, dB, dC, N, blocks, threads);

    CHECK_CUDA(cudaMemcpy(hC_tiled, dC, bytes, cudaMemcpyDeviceToHost));

    float diff = max_abs_diff(hC_naive, hC_tiled, N);

    printf("Naive kernel time: %f ms\n", naive_ms);
    printf("Tiled kernel time: %f ms\n", tiled_ms);
    printf("Max abs diff:      %e\n", diff);

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    free(hA);
    free(hB);
    free(hC_naive);
    free(hC_tiled);

    return 0;
}

