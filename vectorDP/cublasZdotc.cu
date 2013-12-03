#include "cublas_v2.h"
#include <cuComplex.h>
#include <math.h>
#include <stdio.h>
#include <iostream>


int main() {

  int numElements = pow (2,21);
  size_t size = numElements * sizeof(cuDoubleComplex);

  cuDoubleComplex *h_A, *h_B;
  cuDoubleComplex *d_A, *d_B;

  cudaMallocHost((cuDoubleComplex**) &h_A, size);
  cudaMallocHost((cuDoubleComplex**) &h_B, size);

  cudaMalloc((void **) &d_A, size);
  cudaMalloc((void **) &d_B, size);

  for (int i = 0; i < numElements; i++) {  
    h_A[i] = make_cuDoubleComplex(2.25,2.25);
    h_B[i] = make_cuDoubleComplex(2.25,2.25);
  }

  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  cublasHandle_t handle;
  cublasCreate(&handle);

  cuDoubleComplex *result;
  cudaMallocHost((cuDoubleComplex**) &result, sizeof(cuDoubleComplex));

  cublasStatus_t stat = cublasZdotc(handle, numElements, d_A, 1, d_B, 1, result);

  if (stat == CUBLAS_STATUS_SUCCESS) {
    std::cout << "The operation completed successfully\n";
  }

  cudaFree(d_A);
  cudaFree(d_B);

  cudaFreeHost(h_A);
  cudaFreeHost(h_B);

  cudaDeviceReset();
}
