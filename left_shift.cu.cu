#include <stdio.h>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void left_shift_kernel(int *a, const int N){
    int idx = threadIdx.x;
    if (idx < N - 1){
        int temp = a[idx + 1];
        __syncthreads();
        a[idx] = temp;
        __syncthreads();
    }
}

int main(){
    int N = 1 << 10;
    size_t size = N * sizeof(int);

    int *a = new int[N];
    std::iota(a, a + N, 0);

    for (int i = 0; i < N; i++){
        printf("%d ", a[i]);
    }
    printf("\n");

    int *dev_a;
    cudaMalloc(&dev_a, size);
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);

    left_shift_kernel<<<4, 256>>> (dev_a, N);

    cudaMemcpy(a, dev_a, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++){
        printf("%d ", a[i]);
    }
    printf("\n");

    cudaFree(dev_a);
    delete[] a;
}