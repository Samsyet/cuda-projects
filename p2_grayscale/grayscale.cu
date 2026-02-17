#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d -> %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

__global__ void rgb_to_grayscale(const unsigned char *rgb,
                                unsigned char *gray,
                                int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row

    if (x >= width || y >= height) return;

    int rgb_idx  = (y * width + x) * 3;
    int gray_idx = (y * width + x);

    unsigned char r = rgb[rgb_idx + 0];
    unsigned char g = rgb[rgb_idx + 1];
    unsigned char b = rgb[rgb_idx + 2];

    float gray_f = 0.299f * r + 0.587f * g + 0.114f * b;
    gray[gray_idx] = (unsigned char)gray_f;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        printf("Usage: %s input_image output.png\n", argv[0]);
        return 1;
    }

    const char *input_path  = argv[1];
    const char *output_path = argv[2];

    int width, height, channels;

    // Force RGB (3 channels)
    unsigned char *h_rgb = stbi_load(input_path, &width, &height, &channels, 3);
    if (!h_rgb) {
        printf("Failed to load image: %s\n", input_path);
        return 1;
    }

    printf("Loaded: %s (%dx%d)\n", input_path, width, height);

    size_t rgb_bytes  = (size_t)width * height * 3;
    size_t gray_bytes = (size_t)width * height;

    unsigned char *h_gray = (unsigned char*)malloc(gray_bytes);

    unsigned char *d_rgb, *d_gray;
    CHECK_CUDA(cudaMalloc(&d_rgb, rgb_bytes));
    CHECK_CUDA(cudaMalloc(&d_gray, gray_bytes));

    CHECK_CUDA(cudaMemcpy(d_rgb, h_rgb, rgb_bytes, cudaMemcpyHostToDevice));

    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x,
                (height + threads.y - 1) / threads.y);

    rgb_to_grayscale<<<blocks, threads>>>(d_rgb, d_gray, width, height);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_gray, d_gray, gray_bytes, cudaMemcpyDeviceToHost));

    int ok = stbi_write_png(output_path, width, height, 1, h_gray, width);

    if (!ok) printf("Failed to write output: %s\n", output_path);
    else     printf("Wrote grayscale image: %s\n", output_path);

    CHECK_CUDA(cudaFree(d_rgb));
    CHECK_CUDA(cudaFree(d_gray));

    stbi_image_free(h_rgb);
    free(h_gray);

    return 0;
}

