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

// -------------------- NAIVE KERNEL --------------------
// 1 thread computes 1 output element C[row, col]
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

// -------------------- TILED KERNEL --------------------
// Shared tile is 32x32
// Thread block is 16x16
// Each thread computes a 2x2 block of C

#define TILE  32
#define BLOCK 16

__global__ void matmul_tiled_32x32_16x16(const float *A, const float *B, float *C, int N)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x; // 0..15
    int ty = threadIdx.y; // 0..15

    // This block computes a 32x32 tile of C
    int row0 = blockIdx.y * TILE;
    int col0 = blockIdx.x * TILE;

    // Each thread computes a 2x2 sub-block of C
    int row = row0 + ty * 2;
    int col = col0 + tx * 2;

    float c00 = 0.0f;
    float c01 = 0.0f;
    float c10 = 0.0f;
    float c11 = 0.0f;

    int numTiles = (N + TILE - 1) / TILE;

    for(int t = 0; t < numTiles; t++)
    {
        // Tile coordinates in A and B
        int Acol0 = t * TILE;
        int Brow0 = t * TILE;

        // Each thread loads 2x2 into As and Bs
        // ---- Load A ----
        int Arow = row0 + ty * 2;
        int Acol = Acol0 + tx * 2;

        As[ty*2 + 0][tx*2 + 0] = (Arow + 0 < N && Acol + 0 < N) ? A[(Arow + 0)*N + (Acol + 0)] : 0.0f;
        As[ty*2 + 0][tx*2 + 1] = (Arow + 0 < N && Acol + 1 < N) ? A[(Arow + 0)*N + (Acol + 1)] : 0.0f;
        As[ty*2 + 1][tx*2 + 0] = (Arow + 1 < N && Acol + 0 < N) ? A[(Arow + 1)*N + (Acol + 0)] : 0.0f;
        As[ty*2 + 1][tx*2 + 1] = (Arow + 1 < N && Acol + 1 < N) ? A[(Arow + 1)*N + (Acol + 1)] : 0.0f;

        // ---- Load B ----
        int Brow = Brow0 + ty * 2;
        int Bcol = col0 + tx * 2;

        Bs[ty*2 + 0][tx*2 + 0] = (Brow + 0 < N && Bcol + 0 < N) ? B[(Brow + 0)*N + (Bcol + 0)] : 0.0f;
        Bs[ty*2 + 0][tx*2 + 1] = (Brow + 0 < N && Bcol + 1 < N) ? B[(Brow + 0)*N + (Bcol + 1)] : 0.0f;
        Bs[ty*2 + 1][tx*2 + 0] = (Brow + 1 < N && Bcol + 0 < N) ? B[(Brow + 1)*N + (Bcol + 0)] : 0.0f;
        Bs[ty*2 + 1][tx*2 + 1] = (Brow + 1 < N && Bcol + 1 < N) ? B[(Brow + 1)*N + (Bcol + 1)] : 0.0f;

        __syncthreads();

        // Multiply the two shared tiles
        #pragma unroll
        for(int k = 0; k < TILE; k++)
        {
            float a0 = As[ty*2 + 0][k];
            float a1 = As[ty*2 + 1][k];

            float b0 = Bs[k][tx*2 + 0];
            float b1 = Bs[k][tx*2 + 1];

            c00 += a0 * b0;
            c01 += a0 * b1;
            c10 += a1 * b0;
            c11 += a1 * b1;
        }

        __syncthreads();
    }

    // Store results
    if (row + 0 < N && col + 0 < N) C[(row + 0) * N + (col + 0)] = c00;
    if (row + 0 < N && col + 1 < N) C[(row + 0) * N + (col + 1)] = c01;
    if (row + 1 < N && col + 0 < N) C[(row + 1) * N + (col + 0)] = c10;
    if (row + 1 < N && col + 1 < N) C[(row + 1) * N + (col + 1)] = c11;
}

// -------------------- UTILS --------------------

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

float run_kernel_naive(const float *dA, const float *dB, float *dC, int N,
                       dim3 blocks, dim3 threads)
{
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    matmul_naive<<<blocks, threads>>>(dA, dB, dC, N);
    CHECK_CUDA(cudaEventRecord(stop));

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms;
}

float run_kernel_tiled(const float *dA, const float *dB, float *dC, int N,
                       dim3 blocks, dim3 threads)
{
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    matmul_tiled_32x32_16x16<<<blocks, threads>>>(dA, dB, dC, N);
    CHECK_CUDA(cudaEventRecord(stop));

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms;
}

// -------------------- MAIN --------------------

int main()
{
    int N = 1024;
    size_t bytes = (size_t)N * N * sizeof(float);

    printf("Matrix size: %d x %d\n", N, N);
    printf("Tiled kernel: TILE=%d, BLOCK=%d (each thread computes 2x2)\n", TILE, BLOCK);

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

    // Naive: 16x16 threads compute 16x16 outputs
    dim3 threads_naive(16, 16);
    dim3 blocks_naive((N + 16 - 1) / 16, (N + 16 - 1) / 16);

    // Tiled: 16x16 threads compute 32x32 outputs
    dim3 threads_tiled(BLOCK, BLOCK);
    dim3 blocks_tiled((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    // Warmup
    matmul_naive<<<blocks_naive, threads_naive>>>(dA, dB, dC, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    float naive_ms = run_kernel_naive(dA, dB, dC, N, blocks_naive, threads_naive);
    CHECK_CUDA(cudaMemcpy(hC_naive, dC, bytes, cudaMemcpyDeviceToHost));

    float tiled_ms = run_kernel_tiled(dA, dB, dC, N, blocks_tiled, threads_tiled);
    CHECK_CUDA(cudaMemcpy(hC_tiled, dC, bytes, cudaMemcpyDeviceToHost));

    float diff = max_abs_diff(hC_naive, hC_tiled, N);

    printf("Naive kernel time: %f ms\n", naive_ms);
    printf("Tiled kernel time: %f ms\n", tiled_ms);
    printf("Speedup:           %fx\n", naive_ms / tiled_ms);
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

