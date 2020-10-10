#include <stdio.h>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

void left_shift_kernel(int *a, const int N){
    int idx = thread.Idx;
    if (idx < N - 1){
        int temp = a[idx + 1];
        __syncthread();
        a[idx] = temp;
        __syncthread();
    }
}

int main(){
    int N = 1 << 10;
    size_t size = N * sizeof(int);

    std::vector<int> a(N, 0), res(N, 0);
    std::iota(a.begin(), a.end(), 0);

    for (int i : a) std::cout << i << " ";
    int *dev_a;
    cudaMalloc(&dev_a, size);
    cudaMemcpy(dev_a, a.data(), size, cudaMemcpyHostToDevice);

    left_shift_kernel<<<4, 256>>> (dev_a, N);

    cudaMemcpy(a.data(), dev_a, size, cudaMemcpyDeviceToHost);

    for (int i : a) std::cout << i << " ";
    std::cout << '\n';
}