#include <iostream>
#include <iomanip>

__global__ void VecAdd(int* A,
                       int* B,
                       int* C) {
    int i = threadIdx.x;
    printf("tIdx=%d\n", i);
    C[i] = A[i] + B[i];
}

int main() {
    const constexpr int N = 3;
    const constexpr int size = N * sizeof(int);
    int A[] = {1, 2, 3};
    int B[] = {10, 10, 10};
    int C[] = {0, 0, 0};

    int* d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = N;
    int blocksPerGrid = 1;

    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    std::cout << std::setprecision(2);
    std::cout << C[0] << '\n'
    << C[1] << '\n'
    << C[2] << '\n';
}