#include <stdio.h>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#define SHARDED_MEM_SIZE 256

__global__ void sum_reduce_kernel(int *a, int * res, const int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int parital_sum[SHARDED_MEM_SIZE];
    parital_sum[threadIdx.x] = a[idx];
    
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2){
        if (threadIdx.x % (2 * stride == 0)){
            parital_sum[threadIdx.x] += parital_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0){
        res[blockIdx.x] = parital_sum[0];
    }

}

int main(){
    const int N = 1 << 16;
    size_t size = N * sizeof(int);

    int *a = new int[N];
    int *res = new int[N];
    memset(a, 1, N);
    memset(a, 0, N);

    int *dev_a, *dev_res;
    cudaMalloc(&dev_a, size);
    cudaMalloc(&dev_res, size);

    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);

    const int BLOCK_SIZE = 256;
    const int GRID_SIZE = N / BLOCK_SIZE;

    sum_reduce_kernel<<<GRID_SIZE, BLOCK_SIZE>>> (dev_a, dev_res);
    sum_reduce_kernel<<<GRID_SIZE, BLOCK_SIZE>>> (dev_res, dev_res);

    cudaMemcpy(res, dev_res, size, cudaMemcpyDeviceToHost);

    std::cout << "Sum = " << res[0]  << std::endl;
    return 0;
}