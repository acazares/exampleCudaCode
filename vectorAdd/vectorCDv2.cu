#include <iostream>
#include <stdio.h>
#include <math.h>
#include <cuComplex.h>
#include "cublas_v2.h"

__global__ void vectorAdd(const cuDoubleComplex *A, cuDoubleComplex *B, cuDoubleComplex cDouble, int numElements)
{
  int blockfst = blockIdx.x * 256;
  int fstglb = blockfst + (threadIdx.x * 2);
  int sndglb = fstglb + 1;

  cuDoubleComplex Afst = A[fstglb];
  cuDoubleComplex Asnd = A[sndglb];
  cuDoubleComplex Bfst = B[fstglb];
  cuDoubleComplex Bsnd = B[sndglb];

  if (fstglb < numElements && sndglb < numElements) {
    B[fstglb] = cuCadd(cuCmul(Afst,cDouble),Bfst);
    B[sndglb] = cuCadd(cuCmul(Asnd,cDouble),Bsnd);
  }
  else if (fstglb < numElements) {
    B[fstglb] = cuCadd(cuCmul(Afst,cDouble),Bfst);
  }
  else if (sndglb < numElements) {
    B[sndglb] = cuCadd(cuCmul(Asnd,cDouble),Bsnd);
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
  
  cuDoubleComplex CDouble = make_cuDoubleComplex(2.25,2.25);
  int blocksize = 128;
  int gridsize = numElements / 256;
  
  vectorAdd <<< gridsize, blocksize >>> (d_A, d_B, CDouble, numElements);
 
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    fprintf(stderr, "Kernel didn't launch: %d %d %s \n", blocksize, gridsize, cudaGetErrorString(error)); 
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
