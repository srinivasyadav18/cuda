#include <iostream>
#include <fstream>
#include <string>

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

__global__ void index_kernel( int* a, int N){
    int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    printf("%d\n", threadId);
}

void print_to_file(const char* file_name, const int* a, int N){
    std::ofstream fout(file_name);
    if (fout.is_open()){
        for (int i = 0; i < N; i++){
            fout << a[i] << "\n";
        }
        fout.close();
    }
    else {
        std::cout << "Unable to open file\n";
    }
}

int main(){
    int N = 128;
    size_t size = N * sizeof(int);

    int *a = new int[N];

    print_to_file("input.txt", a, N);

    int *dev_a;
    cudaMalloc((void **)&dev_a, size);

    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    index_kernel<<<dim3(2, 2, 2), dim3(4, 2, 2)>>> (dev_a, N);
    cudaMemcpy(a, dev_a, size, cudaMemcpyDeviceToHost);
    print_to_file("output1.txt", a, N);

    /*
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    vector_add_kernel<<<dim3(2, 1, 1), dim3(64, 1, 1)>>> (dev_a, N);
    cudaMemcpy(a, dev_a, size, cudaMemcpyDeviceToHost);
    print_to_file("output2.txt", a, N);
    
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    vector_add_kernel<<<dim3(2, 2, 1), dim3(32, 1, 1)>>> (dev_a, N);
    cudaMemcpy(a, dev_a, size, cudaMemcpyDeviceToHost);
    print_to_file("output3.txt", a, N);
    */

    cudaFree(dev_a);
    delete[] a;
    return 0;
}