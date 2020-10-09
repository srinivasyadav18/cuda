%%writefile vec_add.cu

#include <iostream>
#include <fstream>
#include <string>

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

__global__ void vector_add_kernel(const int* a, const int* b, int *c, int N){
    
    int thread_index = threadIdx.x;
    
    if (thread_index < N)
        c[thread_index] = a[thread_index] + b[thread_index];
}

void print_to_file(const char* file_name, const int* a, const int *b, const int *c, int N){
    std::ofstream fout(file_name);
    if (fout.is_open()){
        for (int i = 0; i < N; i++){
            fout << a[i] << " " << b[i] << " " << c[i] << "\n";
        }
        fout.close();
    }
    else {
        std::cout << "Unable to open file\n";
    }
}

int main(){
    cudaError_t err = cudaSuccess;
    int N = 512;
    size_t size = N * sizeof(int);

    int *a = new int[N];
    int *b = new int[N];
    int *c = new int[N];

    for (int i = 0; i < N; i++) {
        a[i] = rand() % 10 + 1;
        b[i] = rand() % 10 + 1;
        c[i] = 0;
    }

    print_to_file("input.txt", a, b, c, N);

    int *dev_a, *dev_b, *dev_c;
    cudaMalloc((void **)&dev_a, size);
    cudaMalloc((void **)&dev_b, size);
    cudaMalloc((void **)&dev_c, size);

    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, size, cudaMemcpyHostToDevice);
    
    vector_add_kernel<<<1, N>>> (dev_a, dev_b, dev_c, N);

    cudaMemcpy(a, dev_a, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(b, dev_b, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

    print_to_file("output.txt", a, b, c, N);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    delete[] a;
    delete[] b;
    delete[] c;
    return 0;
}