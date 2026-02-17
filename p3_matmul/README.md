P3: MatMul :fire:

Implements square matrix multiplication in CUDA:

- Naive kernel (global memory)
- Tiled kernel (shared memory)

## Build
```bash
nvcc matmul.cu -O3 -o matmul
