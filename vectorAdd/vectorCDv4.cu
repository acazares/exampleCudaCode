#include <iostream>
#include <stdio.h>
#include <math.h>
#include <cuComplex.h>
#include "cublas_v2.h"

__global__ void vectorAdd(const cuDoubleComplex *A, cuDoubleComplex *B, cuDoubleComplex cDouble, int numElements)
{
  extern __shared__ cuDoubleComplex SM[];

  int i = threadIdx.x / 2;
  int off = 64 * blockIdx.x;

  if(threadIdx.x % 2 == 0) {
    SM[2*i] = A[off+i];
  }
  else {
    SM[2*i+1] = B[off+i];
  }
  
  __syncthreads();

  if(threadIdx.x % 2 == 0){ 
    B[off+i] = cuCadd(cuCmul(SM[2*i],cDouble),SM[2*i+1]);
  }
}

int main()
{
  int numElements = pow(2,20);
  size_t size = numElements * sizeof(cuDoubleComplex);
    
  cuDoubleComplex *h_A, *h_B; // host pointers
  cuDoubleComplex *d_A, *d_B = NULL; // device pointers
    
  // Allocate space on the host
  cudaError_t status_HA = cudaMallocHost((cuDoubleComplex **)&h_A, size);
  cudaError_t status_HB = cudaMallocHost((cuDoubleComplex **)&h_B, size);
    
  // Make sure memory was allocated properly on the host
  if((status_HA || status_HB) != cudaSuccess) {
    printf("Memory Allocation Error");
    return 0;
  }
    
  // Allocate space on the device (GPU)
  cudaMalloc((void **)&d_A, size);
  cudaMalloc((void **)&d_B, size);
  
  // Initialize the host input vectors
  for (int b = 0; b < numElements; b++)
    {
      h_A[b] = make_cuDoubleComplex(2.25,2.25);
      h_B[b] = make_cuDoubleComplex(2.25,2.25);
    }
    
  // Copying vectors from host to device
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
  
  cudaError_t error = cudaSuccess;
  
  int blocksize = 128;
  int blocksPerGrid = numElements / 64;

  int SM_size = (blocksize * 2) * (sizeof(cuDoubleComplex));
  
  cuDoubleComplex CDouble = make_cuDoubleComplex(2.25,2.25);

  vectorAdd <<< blocksPerGrid, blocksize, SM_size >>> (d_A, d_B, CDouble, numElements);

  error = cudaGetLastError();
  if (error != cudaSuccess) {
    fprintf(stderr, "Kernel didn't launch: %d %d %s \n", blocksize, blocksPerGrid, cudaGetErrorString(error)); 
  }

  // Copy back the results of vector C  
  cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);
    
  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  
  // Free host memory
  cudaFreeHost(h_A);
  cudaFreeHost(h_B);
  
  cudaDeviceReset();
  return 0;
}
