#include <stdio.h>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void mat_mul_kernel(const int *a, const int *b, int *c, const int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    c[row * N + col] = 0;
    for (int k = 0; k < N; k++) {
        c[row * N + col] += a[row * N + k] * b[k * N + col];
    }
}

int main(){
    // 512 * 512
    int N = 1 << 9;
    int size = N * N * sizeof(int);

    // Allocate memory on Host 
    int *a = new int[N * N];
    int *b = new int[N * N];
    int *c = new int[N * N];

    // Initialising arrays
    std::iota(a, a + N*N, 0);
    std::iota(b, b + N*N, 1);

    int *dev_a, *dev_b, *dev_c;
    // Allocating memory on device
    cudaMalloc(dev_a, size);
    cudaMalloc(dev_b, size);
    cudaMalloc(dev_c, size);

    // Copy data from host to device
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, size, cudaMemcpyHostToDevice);

    // lauching a kernel of 512 blocks each containing 512 threads
    mat_mul_kernel<<<512, 512>>> (dev_a, dev_b, dev_c, N);

    // copy result from device to Host
    cudaMemcpy(a, dev_a, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(b, dev_b, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

    // Free the allocated memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    delete[] a;
    delete[] b;
    delete[] c;
}